"""County geometry helpers (raster aggregation support).

Until real county polygons are integrated (roadmap M2), each county is
approximated by a disk of ``county_buffer_km`` around its registry centroid.
The disk is built in a local azimuthal-equidistant projection (metres) and
reprojected to WGS84, so the radius is a true ground distance.

This is an *approximation* and is labelled as such wherever it surfaces; it
must not be presented as an administrative boundary.
"""

from __future__ import annotations

from .. import counties
from ..logging import get_logger

LOGGER = get_logger("ingestion.geo")


def county_polygons(buffer_km: float) -> dict[str, object]:
    """Return ``{county_code: shapely Polygon}`` in EPSG:4326."""
    from pyproj import Transformer
    from shapely.geometry import Point
    from shapely.ops import transform

    polys: dict[str, object] = {}
    for c in counties.all_counties():
        aeqd = (
            f"+proj=aeqd +lat_0={c.lat} +lon_0={c.lon} "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        disk = Point(0.0, 0.0).buffer(buffer_km * 1000.0)
        to_wgs = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)
        polys[c.code] = transform(to_wgs.transform, disk)
    LOGGER.debug("built %d county approximations (buffer=%.0f km)", len(polys), buffer_km)
    return polys
