"""
resources.py — Real CPU/RAM limits of the container, RAM estimate for DensifyPointCloud,
and the number of parallel workers.

Inside a container psutil reports the RAM and CPUs of the whole host, not what the
container may use. The cgroup limits (v2: cpu.max / memory.max, v1: cfs quota /
memory.limit_in_bytes) are read first; psutil / os are only the fallback.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

logger = logging.getLogger("TurntableStudio.Resources")

CGROUP_ROOT = Path("/sys/fs/cgroup")

# DensifyPointCloud RAM model: RAM_GB = A + B * N * P, P = working megapixels per image.
# Fitted on two runs at --resolution-level 1 (80 photos -> 9.8 GB, 120 photos -> 11.3 GB).
# tools/fit_ram_model.py refits A and B from the stats.json of more runs.
RAM_MODEL_A_GB = 6.8
RAM_MODEL_B_BYTES_PER_PX = 12.0
# Other steps never exceed this on the measured sets (ReconstructMesh 4.7 GB, texrecon 4.6 GB)
RAM_FLOOR_GB = 5.0
# Python workers (masks, undistortion, texturing images): one 12 MP image decoded plus copies
RAM_PER_WORKER_GB = 0.5
RAM_BASE_PYTHON_GB = 1.0

WORKERS_ENV = "TURNTABLE_WORKERS"


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (OSError, ValueError):
        return None


def _cgroup_cpu_limit(root: Path = CGROUP_ROOT) -> Optional[float]:
    """CPU quota of the cgroup in CPUs (None when unlimited or unreadable)."""
    v2 = _read_text(root / "cpu.max")
    if v2:
        parts = v2.split()
        if parts and parts[0] != "max" and len(parts) >= 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
        return None
    quota = _read_text(root / "cpu" / "cpu.cfs_quota_us") or _read_text(root / "cpu.cfs_quota_us")
    period = _read_text(root / "cpu" / "cpu.cfs_period_us") or _read_text(root / "cpu.cfs_period_us")
    try:
        if quota and period and float(quota) > 0:
            return float(quota) / float(period)
    except (ValueError, ZeroDivisionError):
        pass
    return None


def _cgroup_memory_limit_bytes(root: Path = CGROUP_ROOT) -> Optional[int]:
    """Memory limit of the cgroup in bytes (None when unlimited or unreadable)."""
    v2 = _read_text(root / "memory.max")
    if v2 is not None:
        if v2 == "max":
            return None
        try:
            return int(v2)
        except ValueError:
            return None
    v1 = _read_text(root / "memory" / "memory.limit_in_bytes") or _read_text(root / "memory.limit_in_bytes")
    try:
        value = int(v1) if v1 else None
    except ValueError:
        return None
    # cgroup v1 reports "unlimited" as a huge number (close to 2^63)
    if value is None or value >= (1 << 60):
        return None
    return value


def _cgroup_memory_used_bytes(root: Path = CGROUP_ROOT) -> Optional[int]:
    for path in (root / "memory.current", root / "memory" / "memory.usage_in_bytes", root / "memory.usage_in_bytes"):
        text = _read_text(path)
        if text:
            try:
                return int(text)
            except ValueError:
                continue
    return None


def available_cpus(root: Path = CGROUP_ROOT) -> int:
    """CPUs this process may really use: affinity mask, capped by the cgroup quota."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    quota = _cgroup_cpu_limit(root)
    if quota is not None:
        cpus = min(cpus, max(1, math.floor(quota)))
    return max(1, cpus)


def total_ram_gb(root: Path = CGROUP_ROOT) -> float:
    """RAM the container may use: min(cgroup limit, physical RAM)."""
    physical = psutil.virtual_memory().total
    limit = _cgroup_memory_limit_bytes(root)
    return (min(physical, limit) if limit else physical) / (1024 ** 3)


def free_ram_gb(root: Path = CGROUP_ROOT) -> float:
    """RAM still available: inside a limited cgroup limit - usage, otherwise psutil's available."""
    limit = _cgroup_memory_limit_bytes(root)
    available = psutil.virtual_memory().available
    if limit:
        used = _cgroup_memory_used_bytes(root)
        if used is not None:
            available = min(available, max(0, limit - used))
    return available / (1024 ** 3)


def working_megapixels(width: int, height: int, resolution_level: int) -> float:
    """Megapixels DensifyPointCloud works on: every resolution level halves both sides."""
    return (float(width) * float(height)) / (4 ** max(0, int(resolution_level))) / 1e6


def estimate_ram_gb(
    n_images: int,
    width: int = 4056,
    height: int = 3040,
    resolution_level: int = 1,
    a_gb: float = RAM_MODEL_A_GB,
    b_bytes_per_px: float = RAM_MODEL_B_BYTES_PER_PX,
) -> float:
    """
    Peak RAM (GB) of the pipeline, dominated by DensifyPointCloud:
    A + B * N * P, at least RAM_FLOOR_GB. Without margin: callers add their own.
    """
    p = working_megapixels(width, height, resolution_level)
    densify = a_gb + b_bytes_per_px * n_images * p * 1e6 / 1e9
    return max(RAM_FLOOR_GB, densify)


def resolve_workers(
    requested: Optional[int] = None,
    ram_per_worker_gb: float = RAM_PER_WORKER_GB,
    root: Path = CGROUP_ROOT,
) -> int:
    """
    Number of parallel workers. requested > 0 wins; otherwise TURNTABLE_WORKERS; otherwise
    automatic: min(CPUs - 1, (free RAM - base) / RAM per worker), at least 1.
    """
    if requested is None or int(requested) <= 0:
        env = os.environ.get(WORKERS_ENV, "").strip()
        try:
            requested = int(env) if env else 0
        except ValueError:
            logger.warning(f"Ignoring {WORKERS_ENV}={env!r}: not an integer.")
            requested = 0
    if requested and requested > 0:
        return int(requested)

    cpus = available_cpus(root)
    by_cpu = max(1, cpus - 1)
    by_ram = int(max(0.0, free_ram_gb(root) - RAM_BASE_PYTHON_GB) // max(ram_per_worker_gb, 0.01))
    return max(1, min(by_cpu, by_ram))


def describe_resources(root: Path = CGROUP_ROOT) -> Dict[str, Any]:
    """Snapshot written to the log and to stats.json."""
    limit = _cgroup_memory_limit_bytes(root)
    return {
        "cpus": available_cpus(root),
        "host_cpus": os.cpu_count(),
        "ram_total_gb": round(total_ram_gb(root), 2),
        "ram_free_gb": round(free_ram_gb(root), 2),
        "ram_cgroup_limit_gb": round(limit / (1024 ** 3), 2) if limit else None,
        "host_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }


class InsufficientRamError(RuntimeError):
    """Raised before DensifyPointCloud when the estimated RAM does not fit the container."""

    def __init__(self, required_gb: float, available_gb: float):
        self.required_gb = required_gb
        self.available_gb = available_gb
        super().__init__(
            f"Not enough RAM: DensifyPointCloud needs about {required_gb:.1f} GB, "
            f"the container has {available_gb:.1f} GB."
        )


def check_ram(
    n_images: int,
    width: int,
    height: int,
    resolution_level: int,
    margin: float = 1.0,
    warn_fraction: float = 0.85,
    root: Path = CGROUP_ROOT,
) -> Dict[str, Any]:
    """
    Compares the RAM estimate with the container RAM. Returns a dict with
    status "ok" | "tight" and raises InsufficientRamError when it does not fit.
    """
    required = estimate_ram_gb(n_images, width, height, resolution_level) * margin
    total = total_ram_gb(root)
    result = {
        "required_gb": round(required, 2),
        "available_gb": round(total, 2),
        "n_images": n_images,
        "resolution_level": resolution_level,
        "work_megapixels": round(working_megapixels(width, height, resolution_level), 3),
    }
    if required > total:
        raise InsufficientRamError(required, total)
    result["status"] = "tight" if required > warn_fraction * total else "ok"
    return result
