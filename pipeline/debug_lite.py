"""
debug_lite.py — A small archive of what is needed to debug a scan afterwards (plans/PLAN.md 2.7), uploaded by the pod
to the debug bucket (never the bucket the Pi takes its models from) and moved to the PC by tools/pull_debug.py.

Contents (~5-40 MB; each part is skipped when missing, the archive stops growing at `max_mb`):
- raw_mesh.drc: the ReconstructMesh output before any cleaning (Draco, 14-bit positions; holes, disc pieces, what
  the cleaning dropped);
- refine_points.npz: dense-cloud points the visual hull pruned / added (4.2b);
- visual_hull.npz: the carved volume;
- masks/: every mask at 1/4 resolution (PNG);
- the small files: cameras_known.sfm, rig_refine.json, turntable_report.txt, stats.json, run_log.txt, object.json.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("TurntableStudio.DebugLite")

SMALL_FILES = ("cameras_known.sfm", "cameras_known_tags.sfm", "rig_refine.json", "turntable_report.txt",
               "stats.json", "run_log.txt", "object.json")
RAW_MESHES = ("scene_mesh.ply", "scene_dense_mesh.ply", "scene_mesh.obj")


def _raw_mesh_draco(openmvs_dir: Path) -> Optional[bytes]:
    for name in RAW_MESHES:
        p = openmvs_dir / name
        if p.is_file():
            try:
                import DracoPy
                import trimesh

                m = trimesh.load(str(p), process=False, force="mesh")
                return DracoPy.encode(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32),
                                      quantization_bits=14, compression_level=7)
            except Exception as e:
                logger.warning(f"Debug: raw mesh {p.name} not encoded ({e}).")
                return None
    return None


def _small_mask(path: Path) -> Optional[bytes]:
    import cv2

    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    small = cv2.resize(m, (max(1, m.shape[1] // 4), max(1, m.shape[0] // 4)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", small, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return buf.tobytes() if ok else None


def _heavy_parts(out: Path, work: Path):
    """(name, bytes, compress) of the large parts, built lazily: hull, refinement points, raw mesh, masks."""
    for d in (work, out):
        if (d / "visual_hull.npz").is_file():
            yield "visual_hull.npz", (d / "visual_hull.npz").read_bytes(), False
            break
    if (work / "debug_lite" / "refine_points.npz").is_file():
        yield "refine_points.npz", (work / "debug_lite" / "refine_points.npz").read_bytes(), False
    for d in (work / "openmvs", out / "openmvs"):
        if d.is_dir():
            yield "raw_mesh.drc", _raw_mesh_draco(d), False
            break
    masks = sorted((out / "masks").glob("*.mask.png")) if (out / "masks").is_dir() else []
    for m in masks:
        yield f"masks/{m.name}", _small_mask(m), False


def build_debug_parts(output_dir: str, work_dir: str, parts_path: str, max_mb: float = 60.0) -> Dict[str, Any]:
    """
    The large parts alone (raw mesh, masks, hull, refinement points), built by main.py in a thread right after
    ReconstructMesh while the mesh is cleaned and textured (B2: off the critical path). build_debug_lite(...,
    parts_zip=) then only adds the small files of the finished run.
    """
    t0 = time.time()
    zp = Path(parts_path)
    zp.parent.mkdir(parents=True, exist_ok=True)
    budget, total, n = int(max_mb * 1024 * 1024), 0, 0
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data, _ in _heavy_parts(Path(output_dir), Path(work_dir)):
            if data is None or total + len(data) > budget:
                continue
            zf.writestr(name, data)
            total += len(data)
            n += 1
    return {"path": str(zp), "parts": n, "seconds": round(time.time() - t0, 1)}


def build_debug_lite(output_dir: str, work_dir: str, zip_path: str, max_mb: float = 60.0,
                     parts_zip: Optional[str] = None) -> Dict[str, Any]:
    """
    Writes the archive; returns {"path", "size_mb", "parts": [...], "skipped": [...]}. `parts_zip`: the large parts
    already built by build_debug_parts (same content; built here when missing or unreadable).
    """
    out, work = Path(output_dir), Path(work_dir)
    budget = int(max_mb * 1024 * 1024)
    parts: List[str] = []
    skipped: List[str] = []
    zp = Path(zip_path)
    zp.parent.mkdir(parents=True, exist_ok=True)
    total = [0]
    prebuilt = None
    if parts_zip and Path(parts_zip).is_file():
        try:
            prebuilt = zipfile.ZipFile(parts_zip)
        except zipfile.BadZipFile:
            logger.warning("Debug: prebuilt parts unreadable, building them again.")
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        def add(name: str, data: Optional[bytes], compress: bool = True) -> None:
            if data is None:
                return
            if total[0] + len(data) > budget:
                skipped.append(name)
                return
            zf.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED)
            total[0] += len(data)
            parts.append(name)

        for name in SMALL_FILES:
            for base in (out, out / "logs"):
                if (base / name).is_file():
                    add(name, (base / name).read_bytes())
                    break
        if prebuilt is not None:
            with prebuilt:
                for name in prebuilt.namelist():
                    add(name, prebuilt.read(name), compress=False)
        else:
            for name, data, compress in _heavy_parts(out, work):
                add(name, data, compress)
    size_mb = round(zp.stat().st_size / (1024 * 1024), 2)
    if skipped:
        logger.warning(f"Debug archive over {max_mb:.0f} MB: {len(skipped)} part(s) left out.")
    logger.info(f"Debug archive {zp.name}: {len(parts)} part(s), {size_mb} MB.")
    return {"path": str(zp), "size_mb": size_mb, "parts": len(parts), "skipped": skipped[:20]}


def read_raw_mesh(drc_bytes: bytes):
    """(vertices, faces) of raw_mesh.drc (local side, tools)."""
    import DracoPy

    m = DracoPy.decode(drc_bytes)
    return np.asarray(m.points).reshape(-1, 3), np.asarray(m.faces).reshape(-1, 3)


def unzip_member(zip_path: str, name: str) -> bytes:
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(name)


def load_npz_member(zip_path: str, name: str) -> Dict[str, np.ndarray]:
    return dict(np.load(io.BytesIO(unzip_member(zip_path, name))))
