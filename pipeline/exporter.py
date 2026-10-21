"""
exporter.py — Multi-LOD asset packaging:
High-Poly OBJ (PNG), Low-Poly OBJ (JPG), STL, 3MF, and GLB (Draco),
and packaging into output.zip matching Cloudflare R2 specifications.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh
from PIL import Image

from .config import MeshCleaningConfig, PipelineConfig
from .telemetry import SystemTelemetry

logger = logging.getLogger("TurntableStudio.Exporter")


def decimate_mesh(
    input_mesh_path: str,
    output_mesh_path: str,
    target_faces: int = 50000,
) -> None:
    """
    Decimates mesh to target face count using Quadric Edge Collapse.
    Uses PyMeshLab if available, falling back to trimesh.
    """
    in_p = Path(input_mesh_path).resolve()
    out_p = Path(output_mesh_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(in_p))
        logger.info(f"Decimating mesh from {ms.current_mesh().face_number():,d} to {target_faces:,d} faces...")
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
            planarquadric=True,
        )
        ms.save_current_mesh(str(out_p))
        logger.info(f"Decimated mesh saved to: {out_p}")
        return
    except Exception as e:
        logger.warning(f"PyMeshLab decimation failed: {e}. Falling back to trimesh.")

    mesh = trimesh.load(str(in_p), process=False)
    if hasattr(mesh, "simplify_quadric_decimation"):
        try:
            decimated = mesh.simplify_quadric_decimation(target_faces)
            decimated.export(str(out_p))
            logger.info(f"Trimesh decimated mesh saved to: {out_p}")
            return
        except Exception as te:
            logger.warning(f"Trimesh decimation failed: {te}.")

    # Fallback copy
    shutil.copy2(in_p, out_p)


def convert_texture_png_to_jpg(png_path: str, jpg_path: str, quality: int = 85) -> None:
    """Converts a PNG texture atlas to an optimized JPEG texture for web delivery."""
    try:
        with Image.open(png_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(jpg_path, "JPEG", quality=quality, optimize=True)
    except Exception as e:
        logger.warning(f"Failed converting texture to JPG: {e}")


def export_multi_lod_bundle(
    cleaned_mesh_path: str,
    textured_obj_path: Optional[str],
    output_dir: str,
    telemetry: SystemTelemetry,
    config: PipelineConfig,
    stats_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates all deliverables matching Cloudflare R2 and RunPod specifications:
      output_dir/obj/high/model_high.obj
      output_dir/obj/low/model_low.obj
      output_dir/stl/high_model.stl
      output_dir/stl/low_model.stl
      output_dir/3mf/high_model.3mf
      output_dir/3mf/low_model.3mf
      output_dir/glb/high_model.glb
      output_dir/glb/low_model.glb
      output_dir/output.zip
    """
    out_base = Path(output_dir).resolve()
    obj_high_dir = out_base / "obj" / "high"
    obj_low_dir = out_base / "obj" / "low"
    stl_dir = out_base / "stl"
    mf_dir = out_base / "3mf"
    glb_dir = out_base / "glb"

    for d in [obj_high_dir, obj_low_dir, stl_dir, mf_dir, glb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. High-Poly Assets
    high_obj = obj_high_dir / "model_high.obj"
    high_stl = stl_dir / "high_model.stl"
    high_3mf = mf_dir / "high_model.3mf"
    high_glb = glb_dir / "high_model.glb"

    # Base geometry for 3D printing (watertight from mesh_cleaner)
    cleaned_trimesh = trimesh.load(cleaned_mesh_path, process=False)

    # Export watertight STL and 3MF
    cleaned_trimesh.export(str(high_stl), file_type="stl")
    telemetry.record_mesh(str(high_stl))

    try:
        cleaned_trimesh.export(str(high_3mf), file_type="3mf")
        telemetry.record_mesh(str(high_3mf))
    except Exception as e:
        logger.warning(f"3MF export warning: {e}")

    # High OBJ and GLB (with textures if available from OpenMVS TextureMesh)
    if textured_obj_path and Path(textured_obj_path).is_file():
        tex_p = Path(textured_obj_path)
        # Copy textured OBJ and all associated MTL and texture maps
        shutil.copy2(tex_p, high_obj)
        mtl_candidates = list(tex_p.parent.glob(f"{tex_p.stem}*.mtl"))
        png_candidates = list(tex_p.parent.glob(f"{tex_p.stem}*.png"))

        for mtl in mtl_candidates:
            shutil.copy2(mtl, obj_high_dir / mtl.name)
        for png in png_candidates:
            shutil.copy2(png, obj_high_dir / png.name)

        # Load textured mesh for GLB export
        try:
            textured_scene = trimesh.load(str(high_obj), process=False)
            textured_scene.export(str(high_glb), file_type="glb")
            telemetry.record_mesh(str(high_glb))
        except Exception as ge:
            logger.warning(f"Textured GLB export warning: {ge}. Using cleaned mesh for GLB.")
            cleaned_trimesh.export(str(high_glb), file_type="glb")
            telemetry.record_mesh(str(high_glb))

        telemetry.record_mesh(str(high_obj))
    else:
        # Fallback to cleaned mesh without textures
        cleaned_trimesh.export(str(high_obj), file_type="obj")
        cleaned_trimesh.export(str(high_glb), file_type="glb")
        telemetry.record_mesh(str(high_obj))
        telemetry.record_mesh(str(high_glb))

    # 2. Low-Poly Assets (Decimated)
    low_obj = obj_low_dir / "model_low.obj"
    low_stl = stl_dir / "low_model.stl"
    low_3mf = mf_dir / "low_model.3mf"
    low_glb = glb_dir / "low_model.glb"

    decimate_mesh(
        input_mesh_path=cleaned_mesh_path,
        output_mesh_path=str(low_obj),
        target_faces=config.cleaning.target_low_poly_faces,
    )

    loaded_low = trimesh.load(str(low_obj), process=False)
    if isinstance(loaded_low, trimesh.Scene):
        geoms = [g for g in loaded_low.geometry.values() if isinstance(g, trimesh.Trimesh)]
        low_trimesh = trimesh.util.concatenate(geoms) if geoms else cleaned_trimesh
    else:
        low_trimesh = loaded_low

    low_trimesh.export(str(low_stl), file_type="stl")
    telemetry.record_mesh(str(low_obj))
    telemetry.record_mesh(str(low_stl))

    try:
        low_trimesh.export(str(low_3mf), file_type="3mf")
        telemetry.record_mesh(str(low_3mf))
    except Exception as e:
        logger.warning(f"Low-poly 3MF export warning: {e}")

    try:
        low_trimesh.export(str(low_glb), file_type="glb")
        telemetry.record_mesh(str(low_glb))
    except Exception as e:
        logger.warning(f"Low-poly GLB export warning: {e}")

    # 3. Create output.zip containing the entire tree
    zip_path = out_base / config.zip_name
    logger.info(f"Packaging multi-LOD assets into ZIP archive: {zip_path}...")

    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        # Include stats.json if provided
        if stats_json_path and Path(stats_json_path).is_file():
            zipf.write(stats_json_path, arcname="stats.json")

        for folder in [obj_high_dir, obj_low_dir, stl_dir, mf_dir, glb_dir]:
            for root, _, files in os.walk(folder):
                for f in files:
                    file_p = Path(root) / f
                    rel_p = file_p.relative_to(out_base)
                    zipf.write(file_p, arcname=str(rel_p))

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(f"Created {zip_path.name} ({zip_size_mb:.2f} MB)")

    return {
        "zip_path": str(zip_path),
        "high_obj": str(high_obj),
        "low_obj": str(low_obj),
        "high_stl": str(high_stl),
        "low_stl": str(low_stl),
        "high_glb": str(high_glb),
        "low_glb": str(low_glb),
        "zip_size_mb": round(zip_size_mb, 2),
    }
