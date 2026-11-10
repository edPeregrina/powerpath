"""Optional speed-ups for the ra2ce hazard overlay (tested against ra2ce 1.2.2).

Additive and opt-in: nothing here runs unless :func:`patch_ra2ce_hazard_overlay`
is called. The multi-map preprocessing patch in
``book/preprocessing/prep_data_from_extent.py`` is deliberately left untouched
so that existing preprocessing stays reproducible.

ra2ce's ``HazardIntersectBuilderForTif._from_networkx`` spends its time in two
per-edge computations:

* ``EV{n}_ma`` (max depth along the edge): one ``rasterstats.zonal_stats`` call
  per edge (``all_touched=True``, ``stats="max"``). This is the column the road
  island computation depends on.
* ``EV{n}_fr`` (flooded fraction): ``_fraction_flooded_array``. Nothing in
  ``src/`` or ``book/`` reads it.

This module replaces the first with an exact, prefiltered equivalent and lets
the second be fast (same result as the existing book patch) or skipped.
"""

from __future__ import annotations

import numpy as np
from rasterio.features import rasterize

_TIF_MODULE = "ra2ce.network.hazard.hazard_intersect.hazard_intersect_builder_for_tif"

_originals: dict = {}
_prefilter_state: dict = {"arr": None, "nodata": None, "cum": None}


def _nonzero_integral_image(arr, nodata):
    """Summed-area table of valid, non-zero pixels, shape (H + 1, W + 1)."""
    valid = (arr != 0) & ~np.isnan(arr)
    if nodata is not None:
        valid &= arr != nodata
    cum = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.int64)
    cum[1:, 1:] = valid.cumsum(axis=0).cumsum(axis=1)
    return cum


def _get_integral_image(arr, nodata):
    # Identity check: ra2ce passes the same array object for every edge of a
    # given hazard file, so this is built once per file, not once per edge.
    st = _prefilter_state
    if st["arr"] is not arr or st["nodata"] != nodata:
        st["arr"], st["nodata"] = arr, nodata
        st["cum"] = _nonzero_integral_image(arr, nodata)
    return st["cum"]


def max_all_touched(geom, arr, transform, nodata=None, *, prefilter=True):
    """Max raster value over pixels touched by ``geom`` (rasterstats semantics).

    Returns ``None`` if the geometry's window lies outside the raster or has no
    valid pixels, ``0.0`` if the touched pixels are all zero.

    With ``prefilter=True`` a window containing only zero/NaN/nodata pixels
    returns ``0.0`` without rasterizing. This is exact for the "max" statistic
    and is what makes dry edges (the vast majority) nearly free.
    """
    height, width = arr.shape
    minx, miny, maxx, maxy = geom.bounds
    inv = ~transform
    c0, r0 = inv * (minx, maxy)
    c1, r1 = inv * (maxx, miny)
    ca = max(int(np.floor(min(c0, c1))) - 1, 0)
    cb = min(int(np.ceil(max(c0, c1))) + 1, width)
    ra = max(int(np.floor(min(r0, r1))) - 1, 0)
    rb = min(int(np.ceil(max(r0, r1))) + 1, height)
    if cb <= ca or rb <= ra:
        return None

    if prefilter:
        cum = _get_integral_image(arr, nodata)
        if cum[rb, cb] - cum[ra, cb] - cum[rb, ca] + cum[ra, ca] == 0:
            return 0.0

    sub = arr[ra:rb, ca:cb]
    touched = rasterize(
        [(geom, 1)],
        out_shape=sub.shape,
        transform=transform * transform.translation(ca, ra),
        all_touched=True,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    values = sub[touched]
    values = values[~np.isnan(values)]
    if nodata is not None:
        values = values[values != nodata]
    if values.size == 0:
        return None
    return float(values.max())


def _make_fast_zonal_stats(original):
    def fast_zonal_stats(vectors, raster, *args, **kwargs):
        supported = (
            not args
            and isinstance(raster, np.ndarray)
            and raster.ndim == 2
            and hasattr(vectors, "bounds")
            and kwargs.get("affine") is not None
            and kwargs.get("all_touched") is True
            and kwargs.get("stats") == "max"
            and not kwargs.get("add_stats")
            and set(kwargs) <= {"affine", "all_touched", "stats", "nodata"}
        )
        if not supported:
            return original(vectors, raster, *args, **kwargs)
        value = max_all_touched(
            vectors, raster, kwargs["affine"], kwargs.get("nodata")
        )
        return [{"max": value}]

    return fast_zonal_stats


def _make_fast_fraction_flooded():
    from rasterio.features import shapes
    from shapely import STRtree
    from shapely.geometry import shape

    def _fast_fraction_flooded_array(self, line, src):
        cache = getattr(self, "_flood_index_cache", None)
        if cache is None:
            cache = self._flood_index_cache = {}
        entry = cache.get(src.name)
        if entry is None:
            flood_bool = src.read(1) > 0
            polys = [
                shape(geom)
                for geom, _ in shapes(
                    flood_bool.astype("uint8"), mask=flood_bool, transform=src.transform
                )
            ]
            entry = cache[src.name] = (polys, STRtree(polys) if polys else None)
        polys, tree = entry
        if tree is None:
            return 0.0
        idxs = tree.query(line, predicate="intersects")
        if len(idxs) == 0:
            return 0.0
        return sum(polys[i].intersection(line).length for i in idxs) / line.length

    return _fast_fraction_flooded_array


def _skip_fraction_flooded(self, line, src):
    return 0.0


def patch_ra2ce_hazard_overlay(*, fraction_flooded="skip", fast_max=True):
    """Patch ra2ce's tif hazard overlay in-process. Idempotent; see :func:`unpatch`.

    Args:
        fraction_flooded: ``"skip"`` (``EV*_fr`` is all 0.0, unused downstream),
            ``"fast"`` (STRtree lookup, same values as the book patch) or
            ``"stock"`` (leave ra2ce's implementation).
        fast_max: replace the per-edge ``zonal_stats`` max with the prefiltered
            exact equivalent.
    """
    import importlib

    if fraction_flooded not in ("skip", "fast", "stock"):
        raise ValueError(f"Invalid fraction_flooded: {fraction_flooded!r}")
    unpatch()
    module = importlib.import_module(_TIF_MODULE)
    cls = module.HazardIntersectBuilderForTif

    _originals["fraction"] = cls.__dict__["_fraction_flooded_array"]
    _originals["zonal_stats"] = module.zonal_stats
    _originals["module"], _originals["cls"] = module, cls

    if fraction_flooded == "skip":
        cls._fraction_flooded_array = _skip_fraction_flooded
    elif fraction_flooded == "fast":
        cls._fraction_flooded_array = _make_fast_fraction_flooded()
    if fast_max:
        module.zonal_stats = _make_fast_zonal_stats(_originals["zonal_stats"])


def unpatch():
    """Restore ra2ce's original functions (no-op if not patched)."""
    if not _originals:
        return
    _originals["cls"]._fraction_flooded_array = _originals["fraction"]
    _originals["module"].zonal_stats = _originals["zonal_stats"]
    _originals.clear()
    _prefilter_state.update(arr=None, nodata=None, cum=None)
