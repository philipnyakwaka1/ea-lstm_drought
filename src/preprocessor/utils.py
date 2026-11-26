import os
import xarray as xr
from typing import Union
from .regions import Region
import re
import rioxarray


def select_bounding_box(ds: Union[xr.Dataset, xr.DataArray],region: Region,) -> Union[xr.Dataset, xr.DataArray]:
    """Select a bounding box from an xarray Dataset or DataArray."""

    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise TypeError(
            f"Expected an xarray Dataset or DataArray, got {type(ds)}"
        )

    coordinate_pairs = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("y", "x"),
    ]

    coordinate_names = next(
        (
            (lat_name, lon_name)
            for lat_name, lon_name in coordinate_pairs
            if lat_name in ds.coords and lon_name in ds.coords
        ),
        None,
    )

    if coordinate_names is None:
        raise ValueError(
            f"Could not identify latitude/longitude coordinates. "
            f"Available coordinates: {list(ds.coords)}"
        )

    lat_name, lon_name = coordinate_names
    latmin, latmax = region.latmin, region.latmax
    lonmin, lonmax = region.lonmin, region.lonmax
    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values

    if lat_values.ndim != 1 or lon_values.ndim != 1:
        raise ValueError(
            "This function currently supports only 1D latitude and longitude coordinates."
        )

    inverse_lat = lat_values[0] > lat_values[-1]
    inverse_lon = lon_values[0] > lon_values[-1]

    ds_slice = ds.sel(
        {
            lat_name: slice(
                latmax, latmin
            ) if inverse_lat else slice(latmin, latmax),

            lon_name: slice(
                lonmax, lonmin
            ) if inverse_lon else slice(lonmin, lonmax),
        }
    )
    if ds_slice.sizes[lat_name] == 0 or ds_slice.sizes[lon_name] == 0:
        raise ValueError(
            f"Spatial selection returned no values for region {region.name}. "
            f"Check the region boundaries and coordinate reference system."
        )
    return ds_slice

def extract_date_from_filename(filename):
  match = re.search(r"(\d{4})(\d{2})\.tif", filename, re.IGNORECASE)
  if match:
    return f'{match.group(1)}-{match.group(2)}'
  return 'unknown_date'

def convert_tif_to_netcdf(dataset_name, input_tif_folder, output_nc_folder):
	"""Convert a GeoTIFF files to NetCDF format."""
	for filename in os.listdir(input_tif_folder):
		if filename.lower().endswith(('.tif', '.tiff')):
			file_path = os.path.join(input_tif_folder, filename)

			da = rioxarray.open_rasterio(file_path)
			ds = da.to_dataset(name=dataset_name)

			date_str = extract_date_from_filename(filename)
			ds.attrs['time_period'] = date_str

			output_filename = filename.rsplit('.', 1)[0] + '.nc'
			output_path = os.path.join(output_nc_folder, output_filename)

			ds.to_netcdf(output_path)
			print(f"Successfully converted: {filename} -> {output_filename} ({date_str})")

def check_raw_files(dataset_name, raw_folder, GEE_exports_folder):
	"""check if raw folder exists and contains netcdf files otherwise convert
	tif files in the GEE_Exports folder to netcdf and save in the raw folder."""
	if not raw_folder.exists() or not list(raw_folder.glob("*.nc")):
		print(f"Raw folder {raw_folder} does not exist or contains no NetCDF files.")
		print(f"Converting GeoTIFF files from {GEE_exports_folder} to NetCDF format...")
		raw_folder.mkdir(parents=True, exist_ok=True)
	if GEE_exports_folder.exists() and list(GEE_exports_folder.glob("*.tif")):
		convert_tif_to_netcdf(
			dataset_name=dataset_name,
			input_tif_folder=GEE_exports_folder,
			output_nc_folder=raw_folder,
		)
	else:
		print(f"""No GeoTIFF files found in {GEE_exports_folder}. Please check the that folder
              exists and contains GeoTIFF files.""")