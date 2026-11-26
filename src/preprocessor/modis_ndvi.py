from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .base import BasePreProcessor
from ..analysis.indices.condition_index import ConditionIndex


class MODISNDVIPreprocessor(BasePreProcessor):
    """
    Preprocess monthly MODIS MOD13Q1 NDVI data exported from GEE.

    Expected input filenames:
        ndvi_modis_kenya_201001.nc
        ndvi_modis_kenya_201002.nc
        ...

    Expected input:
        Variable: ndvi
        Dimensions: band, y, x
        Spatial resolution: 250 m
        Temporal resolution: Monthly mean

    Processing:
        - Standardize spatial dimensions.
        - Assign monthly timestamps.
        - Mask invalid MODIS NDVI values.
        - Apply the MODIS scale factor.
        - Subset the study area.
        - Regrid to the reference grid.
        - Merge monthly observations.
        - Calculate monthly VCI.
        - Optionally calculate a trailing rolling VCI.

    MODIS NDVI scaling:
        NDVI = DN * 0.0001

    MOD13Q1 valid NDVI range:
        -2000 to 10000 (before scaling).
    """

    dataset = "modis_ndvi"

    SCALE_FACTOR = 0.0001
    VALID_MIN = -2000
    VALID_MAX = 10000

    def __init__(
        self,
        data_folder: Path = Path("data"),
        output_name: Optional[str] = None,
    ) -> None:

        self.static = False

        super().__init__(
            data_folder=data_folder,
            output_name=output_name,
        )

    @staticmethod
    def create_new_dataset(
        ds: xr.Dataset,
        timestamp: pd.Timestamp,
    ) -> xr.Dataset:
        """
        Standardize an individual monthly MODIS NDVI dataset.

        Expected input:
            ndvi(band, y, x)

        Expected output:
            ndvi(time, lat, lon)

        Spatial reference metadata is not copied here.
        The base class reconstructs spatial metadata during regridding.
        """

        # Standardize spatial dimensions.
        ds = BasePreProcessor.standardize_coordinates(ds)

        # Identify the NDVI variable.
        variable = next(
            (
                name
                for name in ds.data_vars
                if name.lower() == "ndvi"
            ),
            None,
        )

        if variable is None:
            raise ValueError(
                "Expected an NDVI variable. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = ds[variable]

        # Remove singleton band dimension, if present.
        if "band" in da.dims:

            if da.sizes["band"] != 1:
                raise ValueError(
                    "Expected a single NDVI band, "
                    f"found {da.sizes['band']} bands."
                )

            da = da.squeeze(
                dim="band",
                drop=True,
            )

        # Validate spatial dimensions.
        if not {"lat", "lon"}.issubset(da.dims):
            raise ValueError(
                "NDVI must contain lat and lon dimensions. "
                f"Found: {da.dims}"
            )

        # Add the monthly timestamp.
        da = da.expand_dims(time=[timestamp])

        output = da.to_dataset(name="ndvi")

        # Preserve source CRS metadata for the base regridding method.
        if "spatial_ref" in ds:
            output = output.assign_coords(
                spatial_ref=ds["spatial_ref"]
            )

        return output

    def _preprocess_single(
        self,
        netcdf_filepath: Path,
        reference_grid: xr.Dataset,
        subset_str: Optional[str] = "kenya",
        regrid_method: str = "bilinear",
    ) -> Path:
        """
        Preprocess one monthly MODIS NDVI NetCDF file.

        Regridding is mandatory.
        """

        print(f"Starting work on {netcdf_filepath.name}")

        timestamp = pd.Timestamp(
            self._parse_time_from_filename(
                netcdf_filepath.name
            )
        )

        with xr.open_dataset(netcdf_filepath) as source:

            ds = self.create_new_dataset(
                source,
                timestamp,
            )

            ds.load()

        # Mask values outside the MODIS valid range.
        da = ds["ndvi"]

        da = da.where(
            (da >= self.VALID_MIN)
            & (da <= self.VALID_MAX)
        )

        # Convert MODIS DN to actual NDVI.
        da = (
            da * self.SCALE_FACTOR
        ).astype("float32")

        da.attrs = {
            "long_name": "Normalized Difference Vegetation Index",
            "units": "1",
            "source": "MODIS/061/MOD13Q1",
            "scale_factor_applied": self.SCALE_FACTOR,
            "temporal_aggregation": "Monthly mean",
            "spatial_resolution": "250 m",
        }

        ds["ndvi"] = da

        # Subset the study area.
        if subset_str is not None:
            ds = self.chop_roi(
                ds,
                subset_str,
            )

        # Regrid to the reference grid.
        ds = self.regrid(
            ds,
            reference_grid,
            method=regrid_method,
        )

        ds.load()

        output_path = self.interim / netcdf_filepath.name

        print(f"Saving to {output_path}")

        ds.to_netcdf(
            output_path,
            encoding={
                "ndvi": {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 4,
                    "_FillValue": np.float32(np.nan),
                }
            },
        )

        print(
            f"** Done for MODIS NDVI "
            f"{netcdf_filepath.name} **"
        )

        return output_path

    def _convert_to_VCI(
        self,
        ds: xr.Dataset,
        variable: str = "ndvi",
        rolling_window: int = 1,
        reference_period: Optional[Tuple[str, str]] = None,
    ) -> xr.Dataset:
        """
        Calculate monthly VCI and an optional rolling VCI product.

        Returns
        -------
        xr.Dataset
            Dataset containing raw VCI and, when requested,
            the rolling VCI product.
        """

        index = ConditionIndex(ds=ds)

        index.fit(
            variable=variable,
            rolling_window=rolling_window,
            reference_period=reference_period,
        )

        result = index.index

        # Rename raw VCI to the MODIS-specific variable name.
        rename = {
            f"{variable}_condition_index": "modis_VCI"
        }

        # Dynamically name the rolling VCI product.
        if rolling_window > 1:

            rolling_name = (
                f"{variable}_condition_index_{rolling_window}"
            )

            rename[rolling_name] = (
                f"modis_VCI{rolling_window}M"
            )

        result = result.rename(rename)

        # Add product-specific metadata.
        result["modis_VCI"].attrs.update({
            "long_name": "MODIS Vegetation Condition Index",
            "units": "%",
        })

        if rolling_window > 1:

            rolling_variable = (
                f"modis_VCI{rolling_window}M"
            )

            result[rolling_variable].attrs.update({
                "long_name": (
                    f"{rolling_window}-month "
                    "MODIS Vegetation Condition Index"
                ),
                "units": "%",
                "rolling_window": rolling_window,
                "rolling_alignment": "trailing",
            })

        return result

    def merge_files(
        self,
        subset_str: Optional[str] = "kenya",
        filename: Optional[str] = None,
        reference_period: Optional[Tuple[str, str]] = None,
        rolling_window: int = 3,
    ) -> Path:
        """
        Merge monthly NDVI files, calculate VCI and optional
        rolling VCI, and save the combined dataset.

        Parameters
        ----------
        subset_str : Optional[str]
            Study area used during preprocessing.

        filename : Optional[str]
            Output filename. If None, a default name is generated.

        reference_period : Optional[Tuple[str, str]]
            Climatological reference period.

        rolling_window : int
            Number of consecutive months used for the trailing
            rolling VCI calculation. Default is 3.
        """

        if rolling_window < 1:
            raise ValueError(
                "rolling_window must be greater than or equal to 1."
            )

        files = self.get_filepaths("interim")

        if not files:
            raise FileNotFoundError(
                f"No interim NetCDF files found in {self.interim}"
            )

        print(f"Merging {len(files)} monthly NDVI files...")

        with xr.open_mfdataset(
            files,
            combine="by_coords",
            join="exact",
            compat="equals",
            chunks={"time": 12},
        ) as source:

            ds = source.sortby("time")

            self.validate_monthly_time(ds)

            # Calculate raw VCI and optional rolling VCI.
            vci = self._convert_to_VCI(
                ds,
                variable="ndvi",
                rolling_window=rolling_window,
                reference_period=reference_period,
            )

            # Merge NDVI and VCI products.
            ds = xr.merge(
                [ds, vci],
                join="exact",
                compat="no_conflicts",
            )

            if filename is None:

                filename = (
                    f"data_{self.dataset}_{subset_str}.nc"
                    if subset_str is not None
                    else f"data_{self.dataset}.nc"
                )

            output_path = self.out_dir / filename

            # Construct output encoding.
            output_variables = [
                "ndvi",
                "modis_VCI",
            ]

            if rolling_window > 1:
                output_variables.append(
                    f"modis_VCI{rolling_window}M"
                )

            encoding = {
                variable: {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 4,
                    "_FillValue": np.float32(np.nan),
                }
                for variable in output_variables
            }

            print("Writing merged NDVI and VCI dataset...")

            ds.to_netcdf(
                output_path,
                encoding=encoding,
            )

        print(
            f"\n**** {output_path} Created! ****\n"
        )

        return output_path

    def preprocess(
        self,
        regrid: Path,
        subset_str: Optional[str] = "kenya",
        regrid_method: str = "bilinear",
        reference_period: Optional[Tuple[str, str]] = None,
        rolling_window: int = 3,
        n_processes: int = 1,
        cleanup: bool = False,
    ) -> Path:
        """
        Preprocess monthly MODIS NDVI data.

        Parameters
        ----------
        regrid : Path
            Mandatory path to the reference NetCDF grid.

        subset_str : Optional[str]
            Region to subset, e.g. 'kenya'.
            None disables spatial subsetting.

        regrid_method : str
            xESMF regridding method.
            Default is 'bilinear'.

        reference_period : Optional[Tuple[str, str]]
            Start and end dates defining the VCI climatological
            reference period.

        rolling_window : int
            Number of consecutive months used for the trailing
            rolling VCI calculation. Default is 3.
            A value of 1 produces raw VCI only.

        n_processes : int
            Number of processing workers.
            Must be 1 when regridding.

        cleanup : bool
            Whether to delete intermediate files generated
            during this run after successful processing.
        """

        print(
            f"Reading data from {self.raw_folder}. "
            f"Writing to {self.interim}"
        )

        if rolling_window < 1:
            raise ValueError(
                "rolling_window must be greater than or equal to 1."
            )

        if n_processes < 1:
            raise ValueError(
                "n_processes must be greater than or equal to 1."
            )

        if n_processes > 1:
            raise ValueError(
                "MODIS NDVI regridding must use n_processes=1. "
                "Concurrent access to xESMF weight files can cause "
                "file access conflicts."
            )

        if regrid is None:
            raise ValueError(
                "A reference grid is required. "
                "Provide the path using the 'regrid' argument."
            )

        if not regrid.exists():
            raise FileNotFoundError(
                f"Reference grid not found: {regrid}"
            )

        nc_files = self.get_filepaths("raw")

        if not nc_files:
            raise FileNotFoundError(
                f"No MODIS NDVI NetCDF files found in "
                f"{self.raw_folder / self.dataset}"
            )

        reference_grid = self.load_reference_grid(regrid)

        # Process individual monthly files.
        outputs = []

        for ix, filepath in enumerate(nc_files):

            print(
                f"\nProcessing file {ix + 1}/{len(nc_files)}"
            )

            output = self._preprocess_single(
                filepath,
                reference_grid=reference_grid,
                subset_str=subset_str,
                regrid_method=regrid_method,
            )

            outputs.append(output)

        # Merge monthly files and calculate VCI products.
        outpath = self.merge_files(
            subset_str=subset_str,
            reference_period=reference_period,
            rolling_window=rolling_window,
        )

        if cleanup:

            for filepath in outputs:
                filepath.unlink(missing_ok=True)

        return outpath
