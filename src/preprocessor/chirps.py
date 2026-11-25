from pathlib import Path
import xarray as xr
import numpy as np
from typing import Optional
import re
from functools import partial
from multiprocessing import Pool
from .base import BasePreProcessor




class CHIRPSPreprocessor(BasePreProcessor):
    """
    Preprocessor for monthly CHIRPS precipitation data.

    GEE performs the temporal aggregation before export.
    CHIRPS values represent monthly accumulated precipitation.

    This class performs:
    - Time coordinate assignment.
    - Spatial coordinate standardization.
    - ROI selection.
    - Optional spatial regridding.
    - Monthly file merging.
    """

    dataset = "chirps"

    @staticmethod
    def create_new_dataset(
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Standardize an individual monthly CHIRPS dataset.

        Expected input:
            chirps(band, y, x)

        Expected output:
            chirps(time, lat, lon)
        """

        # Standardize spatial dimensions
        ds = BasePreProcessor.standardize_coordinates(ds)

        # Remove singleton band dimension
        if "band" in ds.dims:
            if ds.sizes["band"] != 1:
                raise ValueError(
                    "Expected a singleton band dimension, "
                    f"found band={ds.sizes['band']}."
                )

            ds = ds.squeeze(
                dim="band",
                drop=True,
            )

        # Validate expected variable
        if "chirps" not in ds.data_vars:
            raise KeyError(
                "Expected a 'chirps' data variable in the input dataset."
            )

        # Extract monthly timestamp from GEE metadata
        time_period = ds.attrs.get("time_period")

        if time_period is None:
            raise ValueError(
                "Missing 'time_period' attribute in CHIRPS dataset."
            )

        if not re.fullmatch(r"\d{4}-\d{2}", str(time_period)):
            raise ValueError(
                f"Invalid time_period format: {time_period}. "
                "Expected YYYY-MM."
            )

        timestamp = np.datetime64(
            f"{time_period}-01",
            "ns",
        )

        # Remove the file-specific attribute.
        # It would be misleading as a global attribute after merging.
        ds.attrs.pop("time_period", None)

        # Add monthly time dimension
        ds = ds.expand_dims(
            time=[timestamp]
        )

        return ds

    def _preprocess_single(
        self,
        netcdf_filepath: Path,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[xr.Dataset] = None,
        regrid_method: str = 'bilinear',
    ) -> Path:
        """
        Preprocess one monthly CHIRPS NetCDF file.
        """

        print(f"Starting work on {netcdf_filepath.name}")

        # Read and standardize the source dataset.
        with xr.open_dataset(netcdf_filepath) as source:
            ds = self.create_new_dataset(source)

            # Spatial subset
            if subset_str is not None:
                ds = self.chop_roi(
                    ds,
                    subset_str,
                )

            # Optional spatial regridding
            if regrid is not None:
                ds = self.regrid(
                    ds,
                    regrid,
                    method=regrid_method,
                )

            # Load data before closing the source file.
            ds.load()

        filename = self.create_filename(
            netcdf_filepath.name,
            subset_name=subset_str,
        )

        output_path = self.interim / filename
        ds.to_netcdf(output_path)
        print(f"Saved: {output_path}")
        return output_path

    def preprocess(
        self,
        subset_str: Optional[str] = "kenya",
        regrid: Optional[Path] = None,
        regrid_method: str = 'bilinear',
        n_processes: int = 1,
        cleanup: bool = False,
    ) -> Path:
        """
        Preprocess all monthly CHIRPS NetCDF files.

        Parameters
        ----------
        subset_str : Optional[str]
            Region to subset. Default is Kenya.

        regrid : Optional[Path]
            Reference NetCDF grid. If None, no regridding occurs.

        regrid_method : str
            xESMF regridding method. Default is 'bilinear'.

        n_processes : int
            Number of multiprocessing workers.

        cleanup : bool
            Whether to delete intermediate files after successful merging.
        """

        print(
            f"Reading data from {self.raw_folder}. "
            f"Writing to {self.interim}"
        )

        nc_files = self.get_filepaths("raw")

        if not nc_files:
            raise FileNotFoundError(
                f"No CHIRPS NetCDF files found in "
                f"{self.raw_folder / self.dataset}"
            )

        # Load reference grid once
        if regrid is not None:
            regrid = self.load_reference_grid(regrid)

        n_processes = max(1, int(n_processes))
        n_processes = min(n_processes, len(nc_files))

        # Concurrent generation/reuse of xESMF weight files
        # can cause file access conflicts.
        if regrid is not None and n_processes > 1:
            raise ValueError(
                "Use n_processes=1 when regridding. "
                "This avoids concurrent access to xESMF weight files."
            )

        if n_processes > 1:
            print(
                f"Processing {len(nc_files)} files "
                f"using {n_processes} workers."
            )
            with Pool(processes=n_processes) as pool:
                outputs = pool.map(
                    partial(
                        self._preprocess_single,
                        subset_str=subset_str,
                        regrid=regrid,
                    ),
                    nc_files,
                )
        else:
            outputs = []
            for filepath in nc_files:
                output = self._preprocess_single(
                    filepath,
                    subset_str=subset_str,
                    regrid=regrid,
                    regrid_method=regrid_method,
                )
                outputs.append(output)

        merged_file = self.merge_files(
            subset_str=subset_str,
        )

        # Cleanup only the files generated in this run.
        if cleanup:
            for filepath in outputs:
                filepath.unlink(missing_ok=True)
        return merged_file
