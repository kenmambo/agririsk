"""Small HTTP/download helpers shared by the real-feed providers.

Everything here caches aggressively: a file that already exists on disk is
never re-downloaded, so repeated pipeline runs stay offline and cheap.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

from ..logging import get_logger

LOGGER = get_logger("ingestion.utils")


def download_cached(
    url: str,
    dest: Path,
    *,
    timeout: int,
    session=None,
    auth=None,
    chunk_size: int = 1 << 20,
) -> Path:
    """Fetch ``url`` to ``dest`` unless ``dest`` already exists (cache hit).

    Writes to a ``.part`` temp file and renames, so an interrupted download
    never leaves a seemingly-valid cached artefact. ``session`` may be any
    ``requests``-like object; one is created per call when omitted.
    """
    if dest.exists() and dest.stat().st_size > 0:
        LOGGER.debug("cache hit: %s", dest)
        return dest
    import requests  # local import: only external providers need it

    sess = session or requests.Session()
    tmp = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("downloading %s -> %s", url, dest)
    try:
        with sess.get(url, timeout=timeout, stream=True, auth=auth) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def ensure_gunzip(gz_path: Path, out_path: Path | None = None) -> Path:
    """Decompress a ``.gz`` raster once; the plain file acts as the cache key."""
    out_path = out_path or gz_path.with_suffix("")  # foo.tif.gz -> foo.tif
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    LOGGER.debug("unpacked %s -> %s", gz_path, out_path)
    return out_path
