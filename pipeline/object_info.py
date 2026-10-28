"""
object_info.py — object.json: what the scanned object is and how big it is, for a catalogue / web page.

Title and description come from the job; dimensions, volume and area are measured on the cleaned watertight
mesh (model units: metres, Y up, disc at Y = 0), so they are the real size of the object whatever export_scale
the deliverables use.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("TurntableStudio.ObjectInfo")

OBJECT_INFO_FILE = "object.json"


def measure_mesh(mesh_path: str) -> Dict[str, Any]:
    """
    Real-size measurements of a mesh in metres (Y up):
    - height = Y extent;
    - length / width = sides of the smallest rectangle around the footprint (X-Z), so they do not depend on how
      the object was turned on the disc (length >= width);
    - volume only for a watertight mesh.
    """
    import trimesh

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64) * 1000.0      # mm
    height = float(v[:, 1].max() - v[:, 1].min())
    footprint = v[:, [0, 2]]
    try:
        import cv2
        (_, _), (a, b), angle = cv2.minAreaRect(footprint.astype(np.float32))
        length, width = max(a, b), min(a, b)
    except ImportError:
        ext = footprint.max(0) - footprint.min(0)
        length, width = float(ext.max()), float(ext.min())
        angle = 0.0
    watertight = bool(mesh.is_watertight)
    volume_cm3 = round(float(abs(mesh.volume)) * 1e6, 2) if watertight else None
    return {
        "dimensions_mm": {"length": round(float(length), 1), "width": round(float(width), 1),
                          "height": round(height, 1)},
        "bounding_box_mm": [round(float(x), 1) for x in (v.max(0) - v.min(0))],   # X, Y (height), Z of the model
        "volume_cm3": volume_cm3,
        "surface_area_cm2": round(float(mesh.area) * 1e4, 1),
        "watertight": watertight,
        "footprint_angle_deg": round(float(angle), 1),
    }


def write_object_info(
    mesh_path: str,
    output_dir: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Writes <output_dir>/object.json; returns its path, or None (with a warning) when the mesh cannot be measured.
    Without a title the object is named after the scan time ("Scan 2026-10-28 13:16"); the description stays "".
    """
    try:
        now = datetime.now(timezone.utc).astimezone()          # local time of the machine, with its UTC offset
        info: Dict[str, Any] = {
            "title": title or f"Scan {now:%Y-%m-%d %H:%M}",
            "description": description or "",
            "created_at": now.isoformat(timespec="seconds"),
            **measure_mesh(mesh_path),
            "units": "mm (dimensions), cm3 (volume), cm2 (area): real size of the object",
        }
        info.update(extra or {})
        out = Path(output_dir) / OBJECT_INFO_FILE
        out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        d = info["dimensions_mm"]
        logger.info(f"Object: {d['length']} x {d['width']} x {d['height']} mm, volume {info['volume_cm3']} cm3 "
                    f"-> {out.name}")
        return str(out)
    except Exception as e:
        logger.warning(f"Could not write {OBJECT_INFO_FILE}: {e}")
        return None
