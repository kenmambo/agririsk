"""Zonal statistics over raster files (GeoTIFF / GDAL subdatasets).

Providers use :func:`zonal_mean` to aggregate a raster to one number per
county approximation. Datasets in any CRS are warped on the fly through a
:class:`rasterio.vrt.WarpedVRT`, so both CHIRPS (EPSG:4326 GeoTIFF) and MODIS
(sinusoidal HDF4 subdatasets) share one code path.
"""

from __future__ import annotations

import math

import numpy as np

from ..logging import get_logger

LOGGER = get_logger("ingestion.raster_stats")


def zonal_mean(
    path: str,
    polygon,
    *,
    band: int = 1,
    nodata_values: tuple[float, ...] = (),
    valid_range: tuple[float | None, float | None] = (None, None),
    scale: float = 1.0,
    offset: float = 0.0,
) -> float:
    """Mean of ``band`` over ``polygon`` (EPSG:4326 geometry), or NaN if empty.

    Sentinel/fill handling is explicit because several feeds ship no nodata
    flag in the header (e.g. CHIRPS uses a -9999 sentinel): pass the values to
    exclude in ``nodata_values`` and/or a physical ``valid_range``.
    """
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds

    with rasterio.open(path) as src:
        with WarpedVRT(src, crs="EPSG:4326") as vrt:
            left, bottom, right, top = polygon.bounds
            b = vrt.bounds
            left, bottom = max(left, b.left), max(bottom, b.bottom)
            right, top = min(right, b.right), min(top, b.top)
            if left >= right or bottom >= top:
                return math.nan
            window = from_bounds(left, bottom, right, top, transform=vrt.transform)
            if window.width <= 0 or window.height <= 0:
                return math.nan
            # NB: read() rounds a fractional window outward, so polygons
            # smaller than one pixel still sample the covering pixel(s).
            data = vrt.read(band, window=window)
            inside = geometry_mask(
                [polygon], out_shape=data.shape,
                transform=vrt.window_transform(window),
                all_touched=True, invert=True,
            )
            vals = data[inside].astype(float)

    lo, hi = valid_range
    ok = np.isfinite(vals)
    for nv in nodata_values:
        ok &= vals != nv
    if lo is not None:
        ok &= vals >= lo
    if hi is not None:
        ok &= vals <= hi
    vals = vals[ok] * scale + offset
    if vals.size == 0:
        return math.nan
    return float(vals.mean())
