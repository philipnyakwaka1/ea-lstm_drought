from pathlib import Path
import xarray as xr
from typing import Optional
from .base import BasePreProcessor



class SRTMPreprocessor(BasePreProcessor):
    dataset = "srtm"
    static = True

    @staticmethod
    def create_new_dataset(ds: xr.Dataset) -> xr.Dataset:
        """
        Standardize the exported SRTM NetCDF dataset.

        Expected source:
            srtm(band, y, x)

        Output:
            elevation(lat, lon)
        """

        ds = BasePreProcessor.standardize_coordinates(ds)

        # Identify the elevation variable using its long_name.
        elevation_vars = [
            var
            for var in ds.data_vars
            if ds[var].attrs.get("long_name") == "elevation"
        ]

        if len(elevation_vars) != 1:
            raise ValueError(
                "Expected exactly one variable with long_name='elevation'. "
                f"Found: {elevation_vars}"
            )

        elevation = ds[elevation_vars[0]]

        # Remove the singleton band dimension.
        if "band" in elevation.dims:
            if elevation.sizes["band"] != 1:
                raise ValueError(
                    "SRTM should contain exactly one elevation band. "
                    f"Found {elevation.sizes['band']} bands."
                )

            elevation = elevation.squeeze("band", drop=True)

        if set(elevation.dims) != {"lat", "lon"}:
            raise ValueError(
                "Elevation must have exactly two spatial dimensions "
                f"(lat, lon). Found: {elevation.dims}"
            )

        elevation = elevation.rename("elevation")

        elevation.attrs = {
            **elevation.attrs,
            "long_name": "Elevation",
            "units": "m",
        }

        output_ds = xr.Dataset(
            {"elevation": elevation},
            attrs=ds.attrs,
        )

        if "spatial_ref" in ds:
            output_ds["spatial_ref"] = ds["spatial_ref"]
            output_ds["elevation"].attrs["grid_mapping"] = "spatial_ref"

        return output_ds

    @staticmethod
    def coarsen_elevation(
        ds: xr.Dataset,
        factor: int = 10,
    ) -> xr.Dataset:
        """
        Aggregate high-resolution SRTM elevation into coarser
        spatial cells using the mean elevation.

        Dask-backed arrays are processed lazily.
        """

        if factor < 1:
            raise ValueError("Coarsening factor must be >= 1.")

        if factor == 1:
            return ds

        ds = ds.copy()

        # Reduce memory consumption.
        ds["elevation"] = ds["elevation"].astype("float32")

        # Aggregate elevation and spatial coordinates.
        ds = ds.coarsen(
            lat=factor,
            lon=factor,
            boundary="trim",
        ).mean(keep_attrs=True)

        return ds

    def preprocess(
        self,
        subset_str: str = "kenya",
        regrid: Optional[Path] = None,
        regrid_method: str = "bilinear",
        reuse_weights: bool = True,
        clean_weights: bool = False,
    ) -> Path:
        """
        Preprocess the static SRTM elevation dataset.

        Parameters
        ----------
        subset_str : str
            Region identifier defined in region_lookup.

        regrid : Optional[Path]
            Optional path to the reference NetCDF grid.
            If None, the original SRTM resolution is retained.

        regrid_method : str
            xESMF regridding method. Bilinear is recommended
            for the continuous elevation variable.

        reuse_weights : bool
            Reuse existing xESMF weights when available.

        clean_weights : bool
            Remove the generated xESMF weight file after regridding.

        Returns
        -------
        Path
            Path to the processed NetCDF file.
        """

        print(
            f"Reading data from {self.raw_folder / self.dataset}. "
            f"Writing to {self.out_dir}"
        )

        raw_files = self.get_filepaths()

        if len(raw_files) != 1:
            raise ValueError(
                f"Expected exactly one SRTM NetCDF file in "
                f"{self.raw_folder / self.dataset}. "
                f"Found {len(raw_files)}: {raw_files}"
            )

        netcdf_filepath = raw_files[0]

        print(f"Starting work on {netcdf_filepath.name}")

        # Open lazily with spatial chunks to avoid loading the
        # entire high-resolution SRTM raster into memory.
        with xr.open_dataset(
            netcdf_filepath,
            chunks={"band": 1, "y": 2048, "x": 2048},
        ) as source_ds:

            ds = self.create_new_dataset(source_ds)

            ds = self.chop_roi(
                ds,
                subset_str=subset_str,
            )

            print(
                f"Spatial dimensions after cropping: {dict(ds.sizes)}"
            )

            # Optionally regrid to the reference dataset.
            if regrid is not None:

                print("Coarsening SRTM before regridding...")

                ds = self.coarsen_elevation(
                    ds,
                    factor=10,
                )

                print(f"Dimensions after coarsening: {dict(ds.sizes)}")

                reference_ds = self.load_reference_grid(regrid)

                print(f"Regridding SRTM using xESMF: {regrid_method}")

                ds = self.regrid(
                    ds=ds,
                    reference_ds=reference_ds,
                    method=regrid_method,
                    reuse_weights=reuse_weights,
                    clean=clean_weights,
                )

            output_path = self.out_dir / f"{self.dataset}_{subset_str}.nc"

            print(f"Saving processed dataset to {output_path}")

            ds.to_netcdf(output_path)

        print(f"\n**** {output_path} Created! ****\n")

        return output_path
