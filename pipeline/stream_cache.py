"""
stream_cache.py — Results computed before the last photo of a scan arrives (plans/PLAN_STREAMING_DAG.md §2), read back
by the pipeline instead of computing them again. Only what depends on the photos alone, so a cached value is exactly
what the pipeline would compute:

- sift/<photo>.npz: SIFT keypoints (full-resolution pixel centres) and descriptors of rig_refine (the normalisation
  with K and the distortion of the final poses is done afterwards; SIFT on one photo gives the same result each run);
- median/<key>.npy: the 1/4-resolution background median of a (camera, level) group, keyed on the exact list of its
  photos (median of the same images = same median, whatever their order);
- ai/: the model outputs of mask_fusion (keyed on photo + crop box: reused only when the final box is the same).

Every entry carries the size and mtime of its photo(s): a photo replaced by another one is a miss, never a stale hit.
Layout under one directory (job["stream_cache"], runpod_job.py streaming mode: work/stream/).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("TurntableStudio.StreamCache")

_STATS: Dict[str, int] = {}
_LOCK = threading.Lock()


def _count(name: str) -> None:
    with _LOCK:
        _STATS[name] = _STATS.get(name, 0) + 1


def stats(reset: bool = False) -> Dict[str, int]:
    """Hits / misses / writes counted in this process since the last reset."""
    with _LOCK:
        out = dict(_STATS)
        if reset:
            _STATS.clear()
    return out


def file_key(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    return int(st.st_size), int(st.st_mtime_ns)


def photo_name(path: str) -> str:
    """rig/0/f_1000.jpg -> 0_f_1000 (unique per photo of a scan: the camera folder plus the stem)."""
    p = Path(path)
    return f"{p.parent.name}_{p.stem}"


# ----------------------------------------------------------------------------- SIFT (rig_refine)

def _sift_path(cache_dir: str, path: str) -> Path:
    return Path(cache_dir) / "sift" / f"{photo_name(path)}.npz"


def load_sift(cache_dir: Optional[str], path: str, scale: float, max_features: int
              ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(uv, desc) saved for this exact photo and SIFT settings, or None."""
    if not cache_dir:
        return None
    f = _sift_path(cache_dir, path)
    if not f.is_file():
        _count("sift_miss")
        return None
    try:
        d = np.load(str(f))
        size, mtime = file_key(path)
        if (int(d["size"]) != size or int(d["mtime_ns"]) != mtime or float(d["scale"]) != float(scale)
                or int(d["max_features"]) != int(max_features)):
            _count("sift_stale")
            return None
        _count("sift_hit")
        return d["uv"], d["desc"]
    except Exception as e:
        logger.debug(f"SIFT cache {f.name} unreadable: {e}")
        _count("sift_miss")
        return None


def save_sift(cache_dir: str, path: str, scale: float, max_features: int, uv: np.ndarray, desc: np.ndarray) -> None:
    f = _sift_path(cache_dir, path)
    f.parent.mkdir(parents=True, exist_ok=True)
    size, mtime = file_key(path)
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez(str(tmp), uv=uv, desc=desc, size=size, mtime_ns=mtime, scale=float(scale),
             max_features=int(max_features))
    os.replace(str(tmp), str(f))
    _count("sift_write")


# ----------------------------------------------------------------------------- background medians (mask_fusion)

def median_key(paths: Sequence[str]) -> str:
    """Same photos (whatever the order) -> same key; a photo changed on disk -> another key."""
    items = sorted((str(Path(p).resolve()), *file_key(p)) for p in paths)
    return hashlib.sha1(repr(items).encode()).hexdigest()[:20]


def median_path(cache_dir: Optional[str], paths: Sequence[str]) -> Optional[Path]:
    """The cached median of exactly these photos, or None."""
    if not cache_dir:
        return None
    try:
        f = Path(cache_dir) / "median" / f"{median_key(paths)}.npy"
    except OSError:
        return None
    if f.is_file():
        _count("median_hit")
        return f
    _count("median_miss")
    return None


def save_median(cache_dir: str, paths: Sequence[str], med: np.ndarray) -> Path:
    f = Path(cache_dir) / "median" / f"{median_key(paths)}.npy"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + ".tmp.npy")
    np.save(str(tmp), med)
    os.replace(str(tmp), str(f))
    _count("median_write")
    return f


def ai_dir(cache_dir: Optional[str]) -> Optional[str]:
    return str(Path(cache_dir) / "ai") if cache_dir else None


def summary(cache_dir: Optional[str]) -> Dict[str, Any]:
    """What the cache holds (stats.json "stream")."""
    if not cache_dir or not Path(cache_dir).is_dir():
        return {}
    root = Path(cache_dir)
    return {k: len(list((root / k).glob("*"))) if (root / k).is_dir() else 0 for k in ("sift", "median", "ai")}
