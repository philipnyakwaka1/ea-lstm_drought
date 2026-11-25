import xarray as xr
from typing import Union
from .regions import Region

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
