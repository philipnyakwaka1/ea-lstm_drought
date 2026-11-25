from collections import namedtuple

Region = namedtuple("Region", ["name", "lonmin", "lonmax", "latmin", "latmax"])

def get_kenya() -> Region:
    """
    Get the bounding box for Kenya.
    This function allows Kenya's bounding box to be easily accessed
    by all exporters.
    """
    return Region(
        name="kenya", lonmin=33.501, lonmax=42.283, latmin=-5.202, latmax=6.002
    )

def get_kenya_asal() -> Region:
    """Get the bounding box for Kenya's ASAL region."""
    return Region(
        name="kenya_asal", lonmin=33.9921, lonmax=41.9060, latmin=0.1836, latmax=5.4141
    )

# dictionary lookup of regions
region_lookup = {
    "kenya": get_kenya(),
    "kenya_asal": get_kenya_asal(),
}