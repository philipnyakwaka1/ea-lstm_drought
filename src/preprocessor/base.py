from pathlib import Path
import xarray as xr
from typing import List
import numpy as np
import pandas as pd
from typing import List, Optional
from datetime import datetime
import re
from .utils import select_bounding_box
from .regions import region_lookup


class BasePreProcessor:
    """Base for all pre-processor classes. The preprocessing classes
    are responsible for taking the raw data exports and normalizing them
    so that they can be ingested by the feature engineering class.
    This involves:
    - subsetting the ROI (default is Kenya)
    - regridding to a consistent spatial grid (pixel size / resolution)
    - resampling to a consistent time step (hourly, daily, monthly)
    - assigning coordinates to `.nc` files (latitude, longitude, time)

    Attributes:
    ----------
    data_folder: Path, default: Path('data')
        The location of the data folder.
    """

    dataset: str
    static: bool = False
    analysis: bool = False

    def __init__(
        self, data_folder: Path = Path("data"), output_name: Optional[str] = None
    ) -> None:

        self.data_folder = data_folder
        self.raw_folder = self.data_folder / "raw"
        self.preprocessed_folder = self.data_folder / "interim"

        if not self.preprocessed_folder.exists():
            self.preprocessed_folder.mkdir(exist_ok=True, parents=True)

        if not getattr(self, "dataset", None):
            raise AttributeError(
                "Each preprocessing subclass must define a dataset attribute."
            )

        try:
            if output_name is None:
                output_name = self.dataset

            if self.static:
                folder_prefix = f"static/{output_name}"
            else:
                folder_prefix = output_name

            if self.analysis:
                self.out_dir = (
                    self.data_folder / "analysis" / f"{folder_prefix}_preprocessed"
                )
            else:
                self.out_dir = (
                    self.preprocessed_folder / f"{folder_prefix}_preprocessed"
                )

            if not self.out_dir.exists():
                self.out_dir.mkdir(parents=True)

            self.interim = self.preprocessed_folder / f"{folder_prefix}_interim"
            if not self.interim.exists():
                self.interim.mkdir(parents=True)
        except AttributeError:
            print(
                "A dataset attribute must be added for "
                "the interim and out directories to be created"
            )

    def get_filepaths(self, folder: str = "raw") -> List[Path]:
        if folder == "raw":
            target_folder = self.raw_folder / self.dataset
        else:
            target_folder = self.interim
        outfiles = list(target_folder.glob("**/*.nc"))
        outfiles.sort()
        return outfiles

    def _parse_time_from_filename(self, filename: str) -> datetime:
            """
            Extract year and month from filenames such as:
            ndvi_modis_kenya_201001.nc
            """
            match = re.search(r"_(\d{4})(\d{2})\.nc$", filename)

            if not match:
                raise ValueError(f"Could not extract date from {filename}")

            year, month = map(int, match.groups())

            return datetime(year, month, 1)

    @staticmethod
    def standardize_coordinates(
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Rename spatial dimensions to lat/lon.

        Supports:
        - lat/lon
        - y/x

        This only renames dimensions. It does not transform
        projected coordinates into geographic coordinates.
        """

        rename = {}

        if "y" in ds.dims:
            rename["y"] = "lat"

        if "x" in ds.dims:
            rename["x"] = "lon"

        if rename:
            ds = ds.rename(rename)

        if not {"lat", "lon"}.issubset(ds.dims):
            raise ValueError(
                "Dataset must contain spatial dimensions "
                f"(lat, lon) or (y, x). Found: {ds.dims}"
            )

        return ds

    @staticmethod
    def create_filename(
        netcdf_filepath: str,
        subset_name: Optional[str] = None,
    ) -> str:
        """
        Example:
            chirps_2010_01.nc
            =>
            chirps_2010_01_kenya.nc
        """

        filename_stem = Path(netcdf_filepath).stem

        if subset_name is not None:
            filename_stem = f"{filename_stem}_{subset_name}"

        return f"{filename_stem}.nc"


    def regrid(
        self,
        ds: xr.Dataset,
        reference_ds: xr.Dataset,
        method: str = None,
        reuse_weights: bool = True,
        clean: bool = False,
    ) -> xr.Dataset:
        """
        Regrid ds to the spatial grid of reference_ds using xESMF.

        Preserve variable attributes and reconstruct spatial reference
        metadata for the destination grid.
        """

        try:
            import xesmf
            import esmpy
            import hashlib
            import rioxarray
        except ImportError as exc:
            raise ImportError(
                "Regridding requires xESMF, its ESMF backend, "
                "and rioxarray."
            ) from exc

        ds = self.standardize_coordinates(ds)
        reference_ds = self.standardize_coordinates(reference_ds)

        regridding_methods = [
            "bilinear",
            "conservative",
            "nearest_s2d",
            "nearest_d2s",
            "patch",
        ]

        if method not in regridding_methods:
            raise ValueError(
                f"{method} is not an acceptable regridding method. "
                f"Choose from {regridding_methods}"
            )

        # Create source and destination grids
        ds_in = xr.Dataset(
            coords={
                "lat": ds["lat"],
                "lon": ds["lon"],
            }
        )

        ds_out = xr.Dataset(
            coords={
                "lat": reference_ds["lat"],
                "lon": reference_ds["lon"],
            }
        )

        # Build a weight filename using the actual grid coordinates
        grid_hash = hashlib.sha256()

        for grid in (ds_in, ds_out):
            for coord in ("lat", "lon"):
                arr = np.ascontiguousarray(
                    grid[coord].values,
                    dtype="<f8",
                )
                grid_hash.update(coord.encode("utf-8"))
                grid_hash.update(np.asarray(arr.shape, dtype="<i8").tobytes())
                grid_hash.update(arr.tobytes())

        grid_id = grid_hash.hexdigest()[:12]

        filename = f"{method}_{grid_id}.nc"
        weight_path = self.preprocessed_folder / filename

        # Reuse weights only when they actually exist
        use_existing_weights = (
            reuse_weights and weight_path.exists()
        )

        regridder = xesmf.Regridder(
            ds_in,
            ds_out,
            method,
            filename=str(weight_path),
            reuse_weights=use_existing_weights,
        )

        output_dict = {}

        # Regrid spatial variables and preserve their attributes.
        # Do not copy the original spatial reference variable.
        for var in ds.data_vars:

            if var == "spatial_ref":
                continue

            if {"lat", "lon"}.issubset(ds[var].dims):

                print(f"- Regridding variable: {var}")

                output_dict[var] = regridder(ds[var])

                output_dict[var].attrs = ds[var].attrs.copy()
                output_dict[var].attrs["regrid_method"] = method

            else:
                output_dict[var] = ds[var]

        ds_regridded = xr.Dataset(
            output_dict,
            attrs=ds.attrs,
        )

        # Recover the source CRS definition.
        # xESMF uses geographic latitude/longitude coordinates,
        # so this assumes the source and destination use the same CRS.
        crs_attrs = {}

        if "spatial_ref" in ds:
            crs_attrs = ds["spatial_ref"].attrs.copy()

        crs = (
            crs_attrs.get("crs_wkt")
            or crs_attrs.get("spatial_ref")
        )

        if crs:

            # Register the destination spatial dimensions
            ds_regridded = ds_regridded.rio.set_spatial_dims(
                x_dim="lon",
                y_dim="lat",
            )

            # Reconstruct CRS metadata
            ds_regridded = ds_regridded.rio.write_crs(crs)

            # Restore additional CRS attributes, excluding the
            # obsolete source-grid GeoTransform.
            for key, value in crs_attrs.items():
                if key not in {"GeoTransform", "crs_wkt", "spatial_ref"}:
                    ds_regridded["spatial_ref"].attrs[key] = value

            # Calculate the transform from the destination coordinates
            ds_regridded = ds_regridded.rio.write_transform()

        if clean:
            regridder.clean_weight_file()

        return ds_regridded


    @staticmethod
    def load_reference_grid(
        path_to_grid: Path,
    ) -> xr.Dataset:
        """
        Load spatial coordinates from a reference NetCDF dataset.
        """

        with xr.open_dataset(path_to_grid) as full_dataset:

            grid = BasePreProcessor.standardize_coordinates(
                full_dataset
            )

            return xr.Dataset(
                coords={
                    "lat": grid["lat"].load(),
                    "lon": grid["lon"].load(),
                }
            )

    @staticmethod
    def chop_roi(
        ds: xr.Dataset,
        subset_str: Optional[str] = "kenya",
        ) -> xr.Dataset:

        """
        Subset a dataset using a predefined geographic region.

        Coordinate direction is handled by select_bounding_box().
        """

        if subset_str is None:
            return ds

        if subset_str not in region_lookup:
            raise ValueError(
                f"Unknown region: {subset_str}. "
                f"Available regions: {list(region_lookup)}"
            )

        region = region_lookup[subset_str]

        return select_bounding_box(ds, region)

    @staticmethod
    def validate_monthly_time(
        ds: xr.Dataset,
    ) -> None:
        """
        Validate monthly timestamps.

        Assumes all monthly timestamps are represented
        by the first day of each month.
        """

        if "time" not in ds.coords:
            raise ValueError(
                "Dataset must have a time coordinate before merging."
            )

        if ds["time"].ndim != 1:
            raise ValueError(
                "The time coordinate must be one-dimensional."
            )

        time_index = pd.DatetimeIndex(ds["time"].values)

        if not time_index.is_month_start.all():
            raise ValueError(
                "All timestamps must correspond to the first "
                "day of their respective months."
            )

        if time_index.has_duplicates:
            raise ValueError(
                "Duplicate monthly timestamps found."
            )

        if not time_index.is_monotonic_increasing:
            raise ValueError(
                "Monthly timestamps must be chronologically ordered."
            )

        expected = pd.date_range(
            start=time_index[0],
            end=time_index[-1],
            freq="MS",
        )

        if not time_index.equals(expected):
            missing = expected.difference(time_index)

            raise ValueError(
                "Missing monthly timestamps detected: "
                f"{missing.strftime('%Y-%m').tolist()}"
            )

    def merge_files(
        self,
        subset_str: Optional[str] = "kenya",
        filename: Optional[str] = None,
        ) -> Path:
        """
        Merge monthly NetCDF files along the time dimension.

        No temporal resampling or aggregation is performed.
        """

        files = self.get_filepaths("interim")

        if not files:
            raise FileNotFoundError(
                f"No NetCDF files found in {self.interim}"
            )

        print(f"Merging {len(files)} files...")

        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            join="exact",
            compat="equals",
            chunks={"time": 12},
        )

        try:
            ds = ds.sortby("time")

            self.validate_monthly_time(ds)

            if filename is None:
                filename = (
                    f"data_{self.dataset}_{subset_str}.nc"
                    if subset_str is not None
                    else f"data_{self.dataset}.nc"
                )

            output_path = self.out_dir / filename

            print("Writing merged dataset...")

            ds.to_netcdf(output_path)

        finally:
            ds.close()

        print(f"\n**** {output_path} Created! ****\n")

        return output_path