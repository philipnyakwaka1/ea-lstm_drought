"""
src/engineer/engineer.py

Data engineering module updated for GEE preprocessed data and Lees et al. (2022).
"""

import calendar
import pickle
import warnings
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr


def minus_months(year: int, month: int, diff_months: int) -> Tuple[int, int, date]:
    """Compute (year, month, last_day_date) shifted backwards by diff_months."""
    target_month = month - diff_months
    target_year = year

    while target_month <= 0:
        target_month += 12
        target_year -= 1

    last_day = calendar.monthrange(target_year, target_month)[1]
    return target_year, target_month, date(target_year, target_month, last_day)


class EALSTMDataEngineer:
    name = "one_month_forecast"

    def __init__(
        self,
        data_folder: Union[str, Path] = Path("data"),
        process_static: bool = True,
    ) -> None:
        self.data_folder = Path(data_folder)
        self.process_static = process_static

        self.interim_folder = self.data_folder / "interim"
        assert self.interim_folder.exists(), (
            f"{self.interim_folder} does not exist. Run preprocessors first."
        )

        self.output_folder = self.data_folder / "features" / self.name
        self.output_folder.mkdir(parents=True, exist_ok=True)

        if self.process_static:
            self.static_output_folder = self.data_folder / "features" / "static"
            self.static_output_folder.mkdir(parents=True, exist_ok=True)

    def engineer(
        self,
        test_years: Union[int, List[int]] = [2024, 2025],
        val_years: Union[int, List[int]] = [2023],
        target_variable: str = "modis_VCI3M",
        pred_months: int = 3,
        expected_length: Optional[int] = 3,
        global_means: bool = False,
        pixel_means: bool = True,
    ) -> None:
        if isinstance(test_years, int):
            test_years = [test_years]
        if isinstance(val_years, int):
            val_years = [val_years]

        print(f"--- Engineering Dynamic Data (Target: {target_variable}) ---")
        self._process_dynamic(
            test_years=test_years,
            val_years=val_years,
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        if self.process_static:
            print("--- Engineering Static & Climatology Signatures ---")
            self._process_static(
                test_years=test_years,
                val_years=val_years,
                global_means=global_means,
                pixel_means=pixel_means,
            )

    def _get_preprocessed_files(self, static: bool = False) -> List[Path]:
        interim = self.interim_folder / "static" if static else self.interim_folder
        nc_files: List[Path] = []
        for folder in sorted(interim.iterdir()):
            if folder.is_dir() and folder.name.endswith("_preprocessed"):
                nc_files.extend(sorted(list(folder.glob("*.nc"))))
        return nc_files

    def _make_dataset(self, static: bool = False) -> xr.Dataset:
        files = sorted(self._get_preprocessed_files(static=static))
        if not files:
            raise FileNotFoundError(
                f"No NetCDF files found in interim (static={static})"
            )

        datasets = []
        for f in files:
            ds = xr.open_dataset(f)
            # Drop spatial_ref CRS variable present in GEE exports
            if "spatial_ref" in ds.data_vars or "spatial_ref" in ds.coords:
                ds = ds.drop_vars("spatial_ref", errors="ignore")
            if "ndvi" in ds.data_vars or "ndvi" in ds.coords:
                ds = ds.drop_vars("ndvi", errors="ignore")
            if "modis_VCI" in ds.data_vars or "modis_VCI" in ds.coords:
                ds = ds.drop_vars("modis_VCI", errors="ignore")
            datasets.append(ds)

        base_ds = datasets[0]
        for ds in datasets[1:]:
            base_ds = base_ds.merge(ds, join="inner")
        return base_ds

    def calculate_static_means_ds(
        self,
        test_years: List[int],
        val_years: List[int],
        global_means_bool: bool = False,
        pixel_means_bool: bool = True,
    ) -> xr.Dataset:
        dynamic_ds = self._make_dataset(static=False)

        # Exclude both validation and test years from climatological normal calculations
        holdout_years = test_years + val_years
        min_year = int(dynamic_ds["time.year"].min())
        max_train_year = min(holdout_years) - 1

        train_slice = dynamic_ds.sel(
            time=slice(f"{min_year}-01-01", f"{max_train_year}-12-31")
        )

        static_mean_vars = {}
        if pixel_means_bool:
            pixel_means = train_slice.mean(dim="time", skipna=True)
            for var in pixel_means.data_vars:
                static_mean_vars[f"{var}_pixel_mean"] = pixel_means[var]

        if global_means_bool:
            global_means = train_slice.mean(dim=["lat", "lon", "time"], skipna=True)
            for var in global_means.data_vars:
                static_mean_vars[f"{var}_global_mean"] = global_means[var]

        return xr.Dataset(static_mean_vars)

    def _process_static(
        self,
        test_years: List[int],
        val_years: List[int],
        global_means: bool = False,
        pixel_means: bool = True,
    ) -> None:
        output_file = self.static_output_folder / "data.nc"

        try:
            static_ds = self._make_dataset(static=True)
        except (FileNotFoundError, ValueError):
            static_ds = xr.Dataset()

        static_means = self.calculate_static_means_ds(
            test_years=test_years,
            val_years=val_years,
            global_means_bool=global_means,
            pixel_means_bool=pixel_means,
        )

        if len(static_ds.data_vars) > 0:
            static_ds = xr.merge([static_ds, static_means],join="exact")
        else:
            static_ds = static_means

        normalization_dict: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        for var in static_ds.data_vars:
            if "one_hot" in var or var == "soil_type":
                mean_val, std_val = 0.0, 1.0
            else:
                mean_val = float(static_ds[var].mean(skipna=True).values)
                std_val = float(static_ds[var].std(skipna=True).values)
                if std_val == 0.0 or np.isnan(std_val):
                    std_val = 1.0
            normalization_dict[var]["mean"] = mean_val
            normalization_dict[var]["std"] = std_val

        static_ds.to_netcdf(output_file)
        with (self.static_output_folder / "normalizing_dict.pkl").open("wb") as f:
            pickle.dump(normalization_dict, f)

    def _process_dynamic(
        self,
        test_years: List[int],
        val_years: List[int],
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
    ) -> None:
        data = self._make_dataset(static=False)
        assert target_variable in data.data_vars, (
            f"Target variable {target_variable} not found. Available: {list(data.data_vars)}"
        )

        train_ds = self._train_test_split(
            ds=data,
            test_years=test_years,
            val_years=val_years,
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        # Dynamic normalization strictly over training periods
        normalization_dict: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        for var in train_ds.data_vars:
            mean_val = float(
                train_ds[var].mean(dim=["lat", "lon", "time"], skipna=True).values
            )
            std_val = float(
                train_ds[var].std(dim=["lat", "lon", "time"], skipna=True).values
            )
            if std_val == 0.0 or np.isnan(std_val):
                std_val = 1.0
            normalization_dict[var]["mean"] = mean_val
            normalization_dict[var]["std"] = std_val

        with (self.output_folder / "normalizing_dict.pkl").open("wb") as f:
            pickle.dump(normalization_dict, f)

        # Stratify training pairs
        self._stratify_training_data(
            train_ds=train_ds,
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
        )

    def _train_test_split(
        self,
        ds: xr.Dataset,
        test_years: List[int],
        val_years: List[int],
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
    ) -> xr.Dataset:
        earliest_holdout_year = min(test_years + val_years)

        # Training dataset cutoff strictly before holdout years
        train_cutoff = np.datetime64(f"{earliest_holdout_year}-01-01")
        train_mask = ds.time.values < train_cutoff
        train_ds = ds.isel(time=train_mask)

        # Generate holdout datasets (Validation and Testing)
        split_plan = [("val", sorted(val_years)), ("test", sorted(test_years))]
        for split_name, years in split_plan:
            for yr in years:
                for mo in range(1, 13):
                    xy, _ = self._stratify_xy(
                        ds=ds,
                        year=yr,
                        target_month=mo,
                        target_variable=target_variable,
                        pred_months=pred_months,
                        expected_length=expected_length,
                    )
                    if xy is not None:
                        self._save(xy, year=yr, month=mo, dataset_type=split_name)

        return train_ds

    def _stratify_training_data(
        self,
        train_ds: xr.Dataset,
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
    ) -> None:
        min_date = datetime.strptime(
            str(train_ds.time.values.min())[:10], "%Y-%m-%d"
        ).date()
        max_date = datetime.strptime(
            str(train_ds.time.values.max())[:10], "%Y-%m-%d"
        ).date()

        cur_year, cur_month = max_date.year, max_date.month
        cur_min_date = max_date

        while cur_min_date >= min_date:
            arrays, cur_min_date = self._stratify_xy(
                ds=train_ds,
                year=cur_year,
                target_month=cur_month,
                target_variable=target_variable,
                pred_months=pred_months,
                expected_length=expected_length,
            )
            if arrays is not None:
                self._save(
                    arrays,
                    year=cur_year,
                    month=cur_month,
                    dataset_type="train",
                )
            cur_year, cur_month = cur_min_date.year, cur_min_date.month

    @staticmethod
    def _stratify_xy(
        ds: xr.Dataset,
        year: int,
        target_month: int,
        target_variable: str,
        pred_months: int = 3,
        expected_length: Optional[int] = 3,
    ) -> Tuple[Optional[Dict[str, xr.Dataset]], date]:
        """
        Extracts antecedent sequence [t-pred_months, t-1] for target month t.
        Uses diff_months = pred_months + 1 to correctly span 1st-of-month timestamps.
        """
        max_target_date = date(
            year, target_month, calendar.monthrange(year, target_month)[1]
        )
        _, _, max_train_date = minus_months(year, target_month, diff_months=1)
        _, _, min_train_date = minus_months(
            year, target_month, diff_months=pred_months + 1
        )

        min_np = np.datetime64(str(min_train_date))
        max_train_np = np.datetime64(str(max_train_date))
        target_np = np.datetime64(str(max_target_date))

        x_cond = (ds.time.values > min_np) & (ds.time.values <= max_train_np)
        y_cond = (ds.time.values > max_train_np) & (ds.time.values <= target_np)

        if np.sum(y_cond) != 1:
            return None, max_train_date

        if expected_length is not None and np.sum(x_cond) != expected_length:
            return None, max_train_date

        x_ds = ds.isel(time=x_cond)
        y_ds = ds.isel(time=y_cond)[[target_variable]]

        return {"x": x_ds, "y": y_ds}, max_train_date

    def _save(
        self,
        ds_dict: Dict[str, xr.Dataset],
        year: int,
        month: int,
        dataset_type: str,
    ) -> None:
        save_folder = self.output_folder / dataset_type / f"{year}_{month}"
        save_folder.mkdir(parents=True, exist_ok=True)
        for name, ds in ds_dict.items():
            ds.to_netcdf(save_folder / f"{name}.nc")