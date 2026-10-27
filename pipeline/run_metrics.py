"""
run_metrics.py — Features and outcomes saved in stats.json for predicting RAM, time, disk and quality
of a scan (tools/collect_stats.py turns many stats.json into one CSV row each).

Every function is best effort: it returns None / {} instead of raising, a failing metric must never
fail the scan.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("TurntableStudio.Metrics")

STATS_SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parent.parent


def _safe(fn):
    """Decorator: metrics never raise."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - a metric must not stop the pipeline
            logger.debug(f"Metric {fn.__name__} failed: {e}")
            return None
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _num(text: str) -> float:
    return float(text.replace(",", ""))


# ── Code version and hardware ───────────────────────────────────────────────────────────────────────────────────────

@_safe
def code_version() -> Dict[str, Any]:
    """git commit (local runs) or the values baked into the image (TURNTABLE_GIT_COMMIT / TURNTABLE_IMAGE)."""
    info: Dict[str, Any] = {
        "git_commit": os.environ.get("TURNTABLE_GIT_COMMIT"),
        "image": os.environ.get("TURNTABLE_IMAGE"),
    }
    if not info["git_commit"] and (REPO_ROOT / ".git").exists() and shutil.which("git"):
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        info["git_commit"] = out.stdout.strip() or None
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5)
        info["git_dirty"] = bool(dirty.stdout.strip())
    return info


@_safe
def cpu_info() -> Dict[str, Any]:
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        max_mhz = round(freq.max, 0) if freq and freq.max else None
    except Exception:
        physical, max_mhz = None, None
    return {"model": model or platform.processor() or None, "physical_cores": physical,
            "logical_cpus": os.cpu_count(), "max_mhz": max_mhz}


@_safe
def gpu_info() -> List[Dict[str, Any]]:
    """name, VRAM, compute capability, driver of every visible NVIDIA GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    fields = "name,memory.total,compute_cap,driver_version,clocks.max.sm"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5)
    if out.returncode != 0:  # older drivers do not know compute_cap
        fields = "name,memory.total,driver_version"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
    gpus = []
    for line in out.stdout.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        gpus.append(dict(zip(fields.split(","), values)))
    return gpus


@_safe
def host_info() -> Dict[str, Any]:
    """Where the run happens: RunPod pod, WSL (Docker Desktop on Windows), container, plain Linux."""
    version = Path("/proc/version").read_text(errors="ignore").lower() if Path("/proc/version").is_file() else ""
    in_container = Path("/.dockerenv").exists() or "docker" in (Path("/proc/1/cgroup").read_text(errors="ignore")
                                                                if Path("/proc/1/cgroup").is_file() else "")
    return {
        "platform": "runpod" if os.environ.get("RUNPOD_POD_ID") else ("wsl" if "microsoft" in version else platform.system().lower()),
        "in_container": in_container,
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "runpod_gpu_count": os.environ.get("RUNPOD_GPU_COUNT"),
        "datacenter": os.environ.get("RUNPOD_DC_ID"),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }


@_safe
def disk_benchmark(directory: str, size_mb: int = 128) -> Dict[str, Any]:
    """
    Sequential write (with fsync) and read speed of `directory`, ~1-2 s. Explains most of the
    DensifyPointCloud time difference between a bind-mounted Windows folder and a fast disk.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    probe = d / f".disk_probe_{os.getpid()}"
    block = os.urandom(4 * 1024 * 1024)
    n_blocks = max(1, size_mb // 4)
    try:
        t0 = time.perf_counter()
        with open(probe, "wb") as f:
            for _ in range(n_blocks):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
        write_s = time.perf_counter() - t0
        try:  # drop it from the page cache so the read hits the disk
            fd = os.open(probe, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except (AttributeError, OSError):
            pass
        t0 = time.perf_counter()
        with open(probe, "rb") as f:
            while f.read(4 * 1024 * 1024):
                pass
        read_s = time.perf_counter() - t0
    finally:
        probe.unlink(missing_ok=True)
    usage = shutil.disk_usage(d)
    mb = n_blocks * 4
    return {"path": str(d), "write_mb_s": round(mb / max(write_s, 1e-6), 1),
            "read_mb_s": round(mb / max(read_s, 1e-6), 1), "free_gb": round(usage.free / 1024 ** 3, 1)}


# ── Inputs ──────────────────────────────────────────────────────────────────────────────────────────────────────────

@_safe
def input_stats(sfm_path: str, name_regex: Optional[str] = None) -> Dict[str, Any]:
    """Photo count per level / camera, resolution and JPEG size, from cameraInit.sfm / cameras_known.sfm."""
    sfm = json.loads(Path(sfm_path).read_text(encoding="utf-8"))
    views = sfm.get("views", [])
    regex = re.compile(name_regex or r"(?P<level>\d)(?P<index>\d{3})$")
    levels: Dict[str, int] = {}
    cameras: Dict[str, int] = {}
    sizes = []
    for v in views:
        p = Path(v.get("path", ""))
        m = regex.search(p.stem)
        lvl = m.group("level") if m and "level" in m.groupdict() else "?"
        levels[lvl] = levels.get(lvl, 0) + 1
        cam = str(v.get("intrinsicId", "?"))
        cameras[cam] = cameras.get(cam, 0) + 1
        if p.is_file():
            sizes.append(p.stat().st_size)
    intr = (sfm.get("intrinsics") or [{}])[0]
    return {
        "n_images": len(views),
        "n_levels": len(levels),
        "images_per_level": dict(sorted(levels.items())),
        "n_cameras": len(cameras),
        "width": int(intr.get("width", 0) or 0),
        "height": int(intr.get("height", 0) or 0),
        "megapixels": round(int(intr.get("width", 0) or 0) * int(intr.get("height", 0) or 0) / 1e6, 2),
        "jpeg_total_mb": round(sum(sizes) / 1024 ** 2, 1) if sizes else None,
        "jpeg_mean_kb": round(statistics.mean(sizes) / 1024, 1) if sizes else None,
    }


@_safe
def pose_quality(report_path: str) -> Dict[str, Any]:
    """RMS, tags per photo, lift and per-level free-pose difference from turntable_report.txt."""
    text = Path(report_path).read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, Any] = {}
    m = re.search(r"tag-uri pe poză: min (\d+), median (\d+), max (\d+)", text)
    if m:
        out.update(tags_min=int(m[1]), tags_median=int(m[2]), tags_max=int(m[3]))
    m = re.search(r"Total: (\d+) colțuri, RMS ([\d.]+) px, maxim ([\d.]+) px", text)
    if m:
        out.update(corners=int(m[1]), rms_px=float(m[2]), max_err_px=float(m[3]))
    m = re.search(r"Urcare: ([\d.]+) mm/nivel, unghi față de axă ([\d.]+)", text)
    if m:
        out.update(lift_mm=float(m[1]), lift_angle_deg=float(m[2]))
    m = re.search(r"Pas: ([\d.]+)°", text)
    if m:
        out["step_deg"] = float(m[1])
    per_level: Dict[str, List[float]] = {}
    for lm in re.finditer(r"^\d+/\S*?(\d)\d{3}\.\w+\s+(\d+)\s+([\d.]+)\s+([\d.]+)", text, re.M):
        per_level.setdefault(lm[1], []).append(float(lm[4]))
    if per_level:
        out["free_diff_mm_per_level"] = {k: round(statistics.mean(v), 2) for k, v in sorted(per_level.items())}
    return out


@_safe
def object_image_stats(undistorted_dir: str, max_images: int = 24, scale: float = 0.25) -> Dict[str, Any]:
    """
    From the undistorted images and their masks (every k-th view, downscaled): how much of the frame the object
    fills, how textured it is inside the mask (gradient energy: predicts dense points and holes), brightness,
    saturation and the share of bright, colourless pixels (white labels / specular highlights) on the object.
    """
    import cv2
    d = Path(undistorted_dir)
    masks = sorted(p for p in d.glob("*.mask.png") if not p.name.endswith(".jpg.mask.png"))
    if not masks:
        return {}
    step = max(1, len(masks) // max_images)
    fg, grad, bright, sat, white, bbox_h, bbox_w = [], [], [], [], [], [], []
    for mp in masks[::step]:
        img_p = mp.with_name(mp.name.replace(".mask.png", ".jpg"))
        if not img_p.is_file():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if m is None or img is None:
            continue
        m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 127
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if not m.any():
            fg.append(0.0)
            continue
        inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # ignore the silhouette edge
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sel = inner if inner.any() else m
        fg.append(float(m.mean()))
        grad.append(float(g[sel].mean()))
        bright.append(float(hsv[..., 2][sel].mean()))
        sat.append(float(hsv[..., 1][sel].mean()))
        white.append(float(((hsv[..., 2] > 200) & (hsv[..., 1] < 40))[sel].mean()))
        ys, xs = np.where(m)
        bbox_h.append((ys.max() - ys.min() + 1) / m.shape[0])
        bbox_w.append((xs.max() - xs.min() + 1) / m.shape[1])

    def stats(v):
        return {"mean": round(statistics.mean(v), 4), "min": round(min(v), 4), "max": round(max(v), 4)} if v else None

    return {
        "sampled_views": len(fg),
        "foreground_fraction": stats(fg),
        "texture_gradient": stats(grad),
        "brightness": stats(bright),
        "saturation": stats(sat),
        "white_or_specular_fraction": stats(white),
        "bbox_height_fraction": stats(bbox_h),
        "bbox_width_fraction": stats(bbox_w),
    }


# ── Intermediate results ────────────────────────────────────────────────────────────────────────────────────────────

_TIME = r"\((?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?(?:(\d+)ms)?\)"


def _seconds(match_groups) -> float:
    h, m, s, ms = (int(x) if x else 0 for x in match_groups)
    return h * 3600 + m * 60 + s + ms / 1000.0


@_safe
def openmvs_log_stats(openmvs_dir: str) -> Dict[str, Any]:
    """Dense points, fused depths, GPU time of the depth maps, mesh size after Delaunay/graph-cut."""
    d = Path(openmvs_dir)
    out: Dict[str, Any] = {}
    dens = sorted(d.glob("DensifyPointCloud-*.log"))
    if dens:
        text = dens[-1].read_text(errors="ignore")
        est = [_seconds(m.groups()[-4:]) for m in re.finditer(
            r"Depth-map for image\s+\d+ estimated using\s+\d+ images: (\d+)x(\d+) " + _TIME, text)]
        res = re.search(r"Depth-map for image\s+\d+ estimated using\s+\d+ images: (\d+)x(\d+)", text)
        filt = [(int(a), int(b)) for a, b in re.findall(r"filtered using \d+ other images: (\d+)/(\d+) depths discarded", text)]
        fused = re.search(r"Depth-maps fused and filtered: (\d+) depth-maps, (\d+) depths, (\d+) points \((\d+)%", text)
        done = re.search(r"Densifying point-cloud completed: (\d+) points " + _TIME, text)
        out["densify"] = {
            "depth_map_estimations": len(est),
            "depth_map_gpu_seconds": round(sum(est), 1),
            "depth_map_resolution": f"{res[1]}x{res[2]}" if res else None,
            "depths_discarded_fraction": round(sum(a for a, _ in filt) / max(1, sum(b for _, b in filt)), 4) if filt else None,
            "fused_depth_maps": int(fused[1]) if fused else None,
            "fused_depths": int(fused[2]) if fused else None,
            "fused_points": int(fused[3]) if fused else None,
            "dense_points": int(done[1]) if done else None,
            "openmvs_seconds": round(_seconds(done.groups()[-4:]), 1) if done else None,
        }
    rec = sorted(d.glob("ReconstructMesh-*.log"))
    if rec:
        text = rec[-1].read_text(errors="ignore")
        tet = re.search(r"Delaunay tetrahedralization completed: (\d+) points -> (\d+) vertices, (\d+)", text)
        mesh = re.search(r"Mesh reconstruction completed: (\d+) vertices, (\d+) faces", text)
        out["reconstruct"] = {
            "input_points": int(tet[1]) if tet else None,
            "tetra_cells": int(tet[3]) if tet else None,
            "mesh_vertices": int(mesh[1]) if mesh else None,
            "mesh_faces": int(mesh[2]) if mesh else None,
        }
    return out


@_safe
def mesh_quality(mesh_path: str, label: str = "") -> Dict[str, Any]:
    """Connected components, boundary edges (holes), watertight, size of a mesh."""
    import trimesh
    m = trimesh.load(mesh_path, process=False, force="mesh")
    m.merge_vertices()
    comps = m.split(only_watertight=False)
    boundary = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1))
    ext = m.extents * 1000.0
    return {
        "label": label,
        "vertices": int(len(m.vertices)),
        "faces": int(len(m.faces)),
        "components": len(comps),
        "largest_component_face_fraction": round(max(len(c.faces) for c in comps) / max(1, len(m.faces)), 4) if comps else None,
        "boundary_edges": int(boundary),
        "watertight": bool(m.is_watertight),
        "bbox_mm": [round(float(x), 1) for x in ext],
    }


@_safe
def log_counters(run_log_path: str) -> Dict[str, Any]:
    """Counters only visible in run_log.txt (texrecon unseen faces, warnings, errors)."""
    text = Path(run_log_path).read_text(encoding="utf-8", errors="ignore")
    unseen = [int(_num(x)) for x in re.findall(r"Recoloured ([\d,]+) unseen faces", text)]
    return {
        "texrecon_unseen_faces": unseen,
        "warnings": len(re.findall(r"\[WARNING\]", text)),
        "errors": len(re.findall(r"\[ERROR\]", text)),
    }


@_safe
def directory_size_gb(*directories: str) -> float:
    total = 0
    for d in directories:
        for root, _, files in os.walk(d):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return round(total / 1024 ** 3, 3)
