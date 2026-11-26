from pathlib import Path
import xarray as xr
import numpy as np
from typing import Tuple, Optional
from functools import partial
import pandas as pd
from multiprocessing import Pool
from .base import BasePreProcessor


class ERA5LandPreprocessor(BasePreProcessor):

    dataset = "era5-land"

    # Original GEE band names and their expected units.
    band_units = {
        "volumetric_soil_water_layer_1": "m3 m-3",
        "volumetric_soil_water_layer_2": "m3 m-3",
        "volumetric_soil_water_layer_3": "m3 m-3",
        "volumetric_soil_water_layer_4": "m3 m-3",
        "potential_evaporation": "mm month-1",
        "total_evaporation": "mm month-1",
    }

    @classmethod
    def create_new_dataset(
        cls,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Convert the six-band ERA5-Land NetCDF structure into
        six individually named data variables.

        Each source file represents one monthly observation.
        """

        ds = BasePreProcessor.standardize_coordinates(ds)

        data_vars = [
            name for name in ds.data_vars
            if name != "spatial_ref"
        ]

        if len(data_vars) != 1:
            raise ValueError(
                "Expected exactly one multi-band data variable. "
                f"Found: {data_vars}"
            )

        source_name = data_vars[0]
        data = ds[source_name]

        if "band" not in data.dims:
            raise ValueError(
                f"Expected a 'band' dimension in {source_name}. "
                f"Found dimensions: {data.dims}"
            )

        band_names = data.attrs.get("long_name")

        if band_names is None:
            raise ValueError(
                "The source data variable has no 'long_name' "
                "attribute identifying its bands."
            )

        band_names = list(band_names)
        band_ids = data["band"].values

        if len(band_names) != len(band_ids):
            raise ValueError(
                "The number of band names does not match "
                "the number of band coordinates."
            )

        if set(band_names) != set(cls.band_units):
            raise ValueError(
                "Unexpected ERA5-Land band names. "
                f"Expected: {list(cls.band_units)}; "
                f"Found: {band_names}"
            )

        # Extract each band as an individual DataArray.
        variables = {}

        for band_id, band_name in zip(band_ids, band_names):

            band = data.sel(band=band_id, drop=True)

            band.attrs = {
                "long_name": band_name,
                "units": cls.band_units[band_name],
            }

            variables[band_name] = band

        # Construct a Dataset containing six named variables.
        new_ds = xr.Dataset(
            variables,
            attrs={
                key: value
                for key, value in ds.attrs.items()
                if key != "time_period"
            },
        )

        # Preserve spatial reference metadata.
        if "spatial_ref" in ds:
            new_ds["spatial_ref"] = ds["spatial_ref"]

        # Obtain monthly timestamp from the GEE export metadata.
        time_period = ds.attrs.get("time_period")

        if not isinstance(time_period, str):
            raise ValueError(
                "Missing or invalid 'time_period' global attribute."
            )

        try:
            timestamp = np.datetime64(
                f"{time_period}-01", "ns"
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid time_period: {time_period}"
            ) from exc

        new_ds = new_ds.expand_dims(
            time=[timestamp]
        )

        return new_ds

    def _preprocess_single(
        self,
        netcdf_filepath: Path,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[xr.Dataset] = None,
        regrid_method: str = "bilinear",
    ) -> Path:
        """
        Process one monthly ERA5-Land file.
        """

        print(f"Processing {netcdf_filepath.name}")

        with xr.open_dataset(netcdf_filepath) as source:
            ds = self.create_new_dataset(source)

            # Subset the region of interest.
            ds = self.chop_roi(ds, subset_str)

            # Optionally regrid to the reference grid.
            if regrid is not None:
                ds = self.regrid(ds, regrid, method=regrid_method)

            ds.load()

        filename = self.create_filename(
            netcdf_filepath,
            subset_name=subset_str,
        )

        output_path = self.interim / filename

        print(f"Saving to {output_path}")

        ds.to_netcdf(output_path)

        print(f"Completed {netcdf_filepath.name}")

        return output_path

    # def merge_files(
    #     self,
    #     files: List[Path],
    #     subset_str: Optional[str] = "kenya",
    #     filename: Optional[str] = None,
    # ) -> Tuple[Path, Path]:
    #     """
    #     Merge only the files generated during the current run.

    #     Also derive spatial temporal means for all six continuous
    #     ERA5-Land variables.
    #     """

    #     if not files:
    #         raise FileNotFoundError(
    #             "No processed ERA5-Land files were supplied."
    #         )

    #     files = sorted(files)

    #     print(f"Merging {len(files)} monthly files...")

    #     with xr.open_mfdataset(
    #         files,
    #         combine="by_coords",
    #         join="exact",
    #         compat="equals",
    #         chunks={"time": 12},
    #     ) as ds:

    #         ds = ds.sortby("time")

    #         self.validate_monthly_time(ds)

    #         if filename is None:
    #             filename = (
    #                 f"data_{subset_str}.nc"
    #                 if subset_str is not None
    #                 else "data.nc"
    #             )

    #         dynamic_path = self.out_dir / filename

    #         print("Writing merged dynamic dataset...")

    #         ds.to_netcdf(dynamic_path)

    #         print(f"Created: {dynamic_path}")

    #         # Derive monthly climatology (temporal means) for each variable.
    #         static_ds = ds.groupby("time.month").mean(
    #             dim="time",
    #             skipna=True,
    #             keep_attrs=True,
    #         )

    #         static_ds.attrs.update({
    #             "description": (
    #                 "Pixel-wise temporal means of monthly "
    #                 "ERA5-Land variables."
    #             ),
    #             "temporal_period": (
    #                 f"{str(ds.time.values[0])[:7]} to "
    #                 f"{str(ds.time.values[-1])[:7]}"
    #             ),
    #         })

    #         # Save the derived static dataset separately.
    #         static_folder = (
    #             self.preprocessed_folder
    #             / "static"
    #             / f"{self.dataset}_preprocessed"
    #         )

    #         static_folder.mkdir(
    #             parents=True,
    #             exist_ok=True,
    #         )

    #         static_filename = (
    #             f"data_{subset_str}_mean.nc"
    #             if subset_str is not None
    #             else "data_mean.nc"
    #         )

    #         static_path = static_folder / static_filename

    #         print("Writing spatial temporal means...")

    #         static_ds.to_netcdf(static_path)

    #         print(f"Created: {static_path}")

    #     return dynamic_path, static_path

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        regrid_method: str = "bilinear",
        n_processes: int = 1,
        cleanup: bool = True,
    ) -> Tuple[Path, Path]:
        """
        Preprocess monthly ERA5-Land NetCDF files.

        Temporal aggregation has already been performed in GEE.
        No additional temporal resampling is performed here.
        """

        print(
            f"Reading data from {self.raw_folder}. "
            f"Writing to {self.interim}"
        )

        nc_files = self.get_filepaths()

        if not nc_files:
            raise FileNotFoundError(
                f"No files found for {self.dataset}."
            )

        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        n_processes = max(1, min(n_processes, len(nc_files)))

        if n_processes > 1 and regrid is not None:
            raise ValueError(
                "Multiprocessing with regridding is disabled "
                "to avoid concurrent access to xESMF weight files."
            )

        if n_processes > 1:

            with Pool(processes=n_processes) as pool:
                outputs = pool.map(
                    partial(
                        self._preprocess_single,
                        subset_str=subset_str,
                        regrid=regrid,
                        regrid_method=regrid_method,
                    ),
                    nc_files,
                )
        else:
            outputs = [
                self._preprocess_single(
                    file,
                    subset_str=subset_str,
                    regrid=regrid,
                    regrid_method=regrid_method,
                )
                for file in nc_files
            ]

        output = self.merge_files(
            subset_str=subset_str,
        )

        if cleanup:
            for filepath in outputs:
                filepath.unlink(missing_ok=True)

        return output


class ERA5LandSoilTypePreprocessor(BasePreProcessor):
    """
    Preprocess static ERA5-Land soil-type data."""

    dataset = "era5-land_slt"
    static = True

    soil_type_legend = {
        1: "slt_coarse",
        2: "slt_medium",
        3: "slt_medium_fine",
        4: "slt_fine",
        5: "slt_very_fine",
        6: "slt_organic",
        7: "slt_tropical_organic",
    }

    @classmethod
    def create_new_dataset(
        cls,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Extract and standardize the static soil-type variable.

        The source dataset contains one band with soil_type
        stored as a categorical raster.
        """

        ds = BasePreProcessor.standardize_coordinates(ds)

        # Identify the soil-type data variable.
        candidates = [
            name
            for name, var in ds.data_vars.items()
            if var.attrs.get("long_name") == "soil_type"
        ]

        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one variable identified as "
                f"'soil_type'. Found: {candidates}"
            )

        soil = ds[candidates[0]]

        # Remove the singleton band dimension.
        if "band" in soil.dims:
            if soil.sizes["band"] != 1:
                raise ValueError(
                    "Expected a single soil-type band. "
                    f"Found {soil.sizes['band']} bands."
                )

            soil = soil.squeeze("band", drop=True)

        if not {"lat", "lon"}.issubset(soil.dims):
            raise ValueError(
                f"Unexpected soil-type dimensions: {soil.dims}"
            )

        # Validate categorical values without modifying missing pixels.
        values = soil.values
        valid_values = values[np.isfinite(values)]

        if not np.all(np.isclose(valid_values, np.round(valid_values))):
            raise ValueError(
                "Soil-type data contain non-integer class values."
            )

        observed_classes = np.unique(valid_values).astype(int)

        unexpected = set(observed_classes) - set(cls.soil_type_legend)

        if unexpected:
            raise ValueError(
                f"Unexpected soil-type classes: {sorted(unexpected)}"
            )

        soil = soil.astype("float32")
        soil.name = "slt"

        soil.attrs = {
            "long_name": "ERA5-Land soil type classification",
            "description": (
                "Categorical soil type classification "
                "with seven possible classes."
            ),
        }

        new_ds = xr.Dataset(
            {"slt": soil},
            attrs=ds.attrs,
        )

        # Preserve spatial reference information.
        if "spatial_ref" in ds:
            new_ds["spatial_ref"] = ds["spatial_ref"]

        return new_ds

    @classmethod
    def one_hot_encode(
        cls,
        ds: xr.Dataset,
        variable: str = "slt",
        ) -> xr.Dataset:
        """
        One-hot encode the categorical soil-type variable.

        Missing pixels remain NaN in all output channels.
        """

        if variable not in ds.data_vars:
            raise ValueError(
                f"Expected '{variable}' in dataset. "
                f"Found: {list(ds.data_vars)}"
            )

        soil = ds[variable]

        output_vars = {}

        for code, label in cls.soil_type_legend.items():

            encoded = xr.where(
                soil.notnull(),
                (soil == code).astype("float32"),
                np.nan,
            )

            encoded.name = f"{label}_one_hot"

            encoded.attrs = {
                "long_name": f"One-hot encoded {label}",
                "soil_type_code": code,
                "encoding": "one-hot",
            }

            output_vars[encoded.name] = encoded

        encoded_ds = xr.Dataset(
            output_vars,
            attrs=ds.attrs,
        )

        # Preserve spatial reference information.
        if "spatial_ref" in ds:
            encoded_ds["spatial_ref"] = ds["spatial_ref"]

        encoded_ds.attrs.update({
            "description": (
                "One-hot encoded ERA5-Land static soil-type "
                "classification. Missing pixels are preserved as NaN."
            ),
            "soil_type_classes": "1, 2, 3, 4, 5, 6, 7",
        })

        return encoded_ds

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        regrid_method: str = "nearest_s2d",
    ) -> Path:
        """
        Preprocess static ERA5-Land soil-type data.

        If a subset is specified, only that region will be processed.
        Regridding, when requested, uses nearest-neighbour
        interpolation to preserve categorical class values.
        """

        print(
            f"Reading static soil-type data from "
            f"{self.raw_folder / self.dataset}"
        )

        nc_files = self.get_filepaths()

        if not nc_files:
            raise FileNotFoundError(
                f"No NetCDF files found for {self.dataset}."
            )

        if len(nc_files) != 1:
            raise ValueError(
                "Expected exactly one static soil-type file. "
                f"Found {len(nc_files)} files: {nc_files}"
            )

        reference_grid = None

        if regrid is not None:
            reference_grid = self.load_reference_grid(regrid)

        source_path = nc_files[0]

        print(f"Processing {source_path.name}")

        with xr.open_dataset(source_path) as source:
            ds = self.create_new_dataset(source)
            ds = self.chop_roi(ds, subset_str)
            if reference_grid is not None:
                ds = self.regrid(
                    ds,
                    reference_grid,
                    method=regrid_method,
                )

            # Generate the seven one-hot channels.
            ds = self.one_hot_encode(ds)

            ds.load()

        if subset_str is None:
            filename = f"data_{self.dataset}.nc"
        else:
            filename = f"data_{self.dataset}_{subset_str}.nc"

        output_path = self.out_dir / filename

        print(f"Saving processed soil-type data to {output_path}")

        ds.to_netcdf(output_path)

        # Save the class legend alongside the processed dataset.
        legend = pd.DataFrame({
            "code": list(self.soil_type_legend.keys()),
            "label_text": list(self.soil_type_legend.values()),
        })

        legend.to_csv(
            self.out_dir / "soil_type_legend.csv",
            index=False,
        )

        print(f"\n**** {output_path} Created! ****\n")

        return output_path