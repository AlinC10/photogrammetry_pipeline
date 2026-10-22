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
        curr_faces = ms.current_mesh().face_number()
        logger.info(f"Decimating mesh from {curr_faces:,d} to {target_faces:,d} faces...")

        # 1. Try texture-preserving quadric decimation to preserve UV texture coordinates (vt)
        try:
            ms.meshing_decimation_quadric_edge_collapse_with_texture(
                targetfacenum=target_faces,
                preserveboundary=True,
                optimalplacement=True,
            )
            logger.info(f"Texture-preserving decimation succeeded: {ms.current_mesh().face_number():,d} faces.")
        except Exception as tex_e:
            logger.warning(f"Texture-preserving decimation failed ({tex_e}), trying standard decimation.")
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target_faces,
                preserveboundary=True,
                preservenormal=True,
                preservetopology=True,
                planarquadric=True,
            )

        ms.save_current_mesh(str(out_p), save_wedge_texcoord=True)
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


def find_source_mtl_and_textures(tex_p: Path) -> Tuple[Optional[Path], List[Path]]:
    """
    Finds the source MTL file and all associated texture images from OpenMVS output.
    """
    tex_dir = tex_p.parent
    mtl_file = None

    if tex_p.is_file():
        try:
            with open(tex_p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("mtllib "):
                        cand_name = line.split(maxsplit=1)[1].strip()
                        cand = tex_dir / Path(cand_name).name
                        if cand.is_file():
                            mtl_file = cand
                            break
        except Exception:
            pass

    if not mtl_file:
        cand_mtls = list(tex_dir.glob(f"{tex_p.stem}*.mtl")) or list(tex_dir.glob("*.mtl"))
        if cand_mtls:
            mtl_file = cand_mtls[0]

    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".tga", ".bmp", ".tif", ".tiff"}
    tex_images: List[Path] = []

    if mtl_file and mtl_file.is_file():
        try:
            with open(mtl_file, "r", encoding="utf-8", errors="ignore") as mf:
                for line in mf:
                    line = line.strip()
                    if any(line.startswith(prefix) for prefix in ["map_Kd", "map_Ka", "map_bump", "bump"]):
                        parts = line.split(maxsplit=1)
                        if len(parts) > 1:
                            ref_name = Path(parts[1].strip()).name
                            cand_img = tex_dir / ref_name
                            if cand_img.is_file() and cand_img not in tex_images:
                                tex_images.append(cand_img)
        except Exception as e:
            logger.warning(f"Error reading MTL {mtl_file}: {e}")

    # Fallback to search any images in directory matching stem or material
    if not tex_images:
        for f in tex_dir.iterdir():
            if f.is_file() and f.suffix.lower() in img_exts:
                if tex_p.stem in f.stem or "material" in f.name.lower() or "texture" in f.name.lower():
                    if f not in tex_images:
                        tex_images.append(f)

    return mtl_file, tex_images


def inpaint_texture_atlas(img_path: Path, max_black_thresh: int = 5, inpaint_radius: int = 5) -> None:
    """
    Detects unmapped black/empty pixels on the texture atlas and fills them
    seamlessly using Navier-Stokes/Telea inpainting from neighboring valid pixels.
    """
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(img_path))
        if img is None:
            return
        # Find pixels where all BGR channels are below threshold
        black_mask = np.all(img <= max_black_thresh, axis=-1).astype(np.uint8) * 255
        # Only inpaint if there is a mix of painted and black pixels
        if 0 < black_mask.sum() < black_mask.size * 255 * 0.90:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_mask = cv2.dilate(black_mask, kernel)
            inpainted = cv2.inpaint(img, dilated_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
            cv2.imwrite(str(img_path), inpainted)
            logger.info(f"Auto-inpainting filled empty seam gaps on {img_path.name}")
    except Exception as e:
        logger.warning(f"Inpainting warning on {img_path.name}: {e}")


def package_textured_model(
    src_obj_path: Path,
    dst_obj_path: Path,
    dst_mtl_path: Path,
    src_mtl_path: Optional[Path],
    src_texture_images: List[Path],
    target_format: str,            # "png" for high-poly, "jpg" for low-poly
    downscale_max_dim: Optional[int] = None, # e.g. 2048 for low-poly
    jpg_quality: int = 85,
) -> None:
    """
    Packages a textured OBJ model with an explicitly named MTL and converted/downscaled textures.
    - For high-poly: converts textures to PNG (lossless).
    - For low-poly: converts textures to downscaled JPG (web optimized) and ensures MTL is present.
    """
    dst_dir = dst_obj_path.parent
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert/downscale texture maps
    tex_map_replacements = {}
    for idx, src_img in enumerate(src_texture_images):
        suffix = f".{target_format.lower()}"
        out_tex_name = f"{dst_obj_path.stem}_tex_{idx}{suffix}" if len(src_texture_images) > 1 else f"{dst_obj_path.stem}_texture{suffix}"
        out_tex_path = dst_dir / out_tex_name

        try:
            with Image.open(src_img) as img:
                w, h = img.size
                if downscale_max_dim and max(w, h) > downscale_max_dim:
                    scale = downscale_max_dim / max(w, h)
                    nw = max(1, int(w * scale))
                    nh = max(1, int(h * scale))
                    img_out = img.resize((nw, nh), Image.Resampling.LANCZOS)
                else:
                    img_out = img.copy()

                if target_format.lower() in ("jpg", "jpeg"):
                    img_out.convert("RGB").save(str(out_tex_path), format="JPEG", quality=jpg_quality, optimize=True)
                else:
                    img_out.save(str(out_tex_path), format="PNG", optimize=True)

            # Apply auto-inpainting to fill any residual black seam pixels
            inpaint_texture_atlas(out_tex_path)

            tex_map_replacements[src_img.name] = out_tex_name
            logger.info(f"Generated {target_format.upper()} texture: {out_tex_name} ({img_out.size[0]}x{img_out.size[1]})")
        except Exception as e:
            logger.warning(f"Failed converting texture {src_img}: {e}")

    # 2. Generate destination MTL file
    mtl_lines = []
    if src_mtl_path and src_mtl_path.is_file():
        try:
            with open(src_mtl_path, "r", encoding="utf-8", errors="ignore") as mf:
                for line in mf:
                    clean_l = line.rstrip("\r\n")
                    # Replace texture image references
                    replaced = False
                    for old_name, new_name in tex_map_replacements.items():
                        if old_name in clean_l:
                            clean_l = clean_l.replace(old_name, new_name)
                            replaced = True
                    # Fallback if map_Kd had path or different name
                    if not replaced and any(clean_l.strip().startswith(p) for p in ["map_Kd", "map_Ka"]):
                        if tex_map_replacements:
                            first_new = list(tex_map_replacements.values())[0]
                            parts = clean_l.split(maxsplit=1)
                            clean_l = f"{parts[0]} {first_new}"
                    mtl_lines.append(clean_l + "\n")
        except Exception as e:
            logger.warning(f"Error reading source MTL: {e}")

    # Default MTL template if none was found
    if not mtl_lines:
        first_tex = list(tex_map_replacements.values())[0] if tex_map_replacements else f"{dst_obj_path.stem}_texture.{target_format.lower()}"
        mtl_lines = [
            f"# Material file for {dst_obj_path.name}\n",
            "newmtl material_0\n",
            "Ka 1.000000 1.000000 1.000000\n",
            "Kd 1.000000 1.000000 1.000000\n",
            "Ks 0.000000 0.000000 0.000000\n",
            "d 1.000000\n",
            "illum 1\n",
            f"map_Kd {first_tex}\n",
        ]

    with open(dst_mtl_path, "w", encoding="utf-8") as f:
        f.writelines(mtl_lines)
    logger.info(f"Saved material definition: {dst_mtl_path.name}")

    # 3. Write destination OBJ file with correct mtllib directive
    obj_lines = []
    has_mtllib = False
    has_usemtl = False

    with open(src_obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("mtllib "):
                obj_lines.append(f"mtllib {dst_mtl_path.name}\n")
                has_mtllib = True
            elif line.startswith("usemtl "):
                obj_lines.append(line)
                has_usemtl = True
            else:
                obj_lines.append(line)

    if not has_mtllib:
        obj_lines.insert(0, f"mtllib {dst_mtl_path.name}\n")
    if not has_usemtl:
        first_face_idx = next((i for i, l in enumerate(obj_lines) if l.startswith("f ")), len(obj_lines))
        obj_lines.insert(first_face_idx, "usemtl material_0\n")

    with open(dst_obj_path, "w", encoding="utf-8") as f:
        f.writelines(obj_lines)
    logger.info(f"Saved packaged OBJ model: {dst_obj_path.name}")

    # Clean up any extraneous temporary files like *.obj.mtl
    for extra in dst_dir.glob("*.obj.mtl"):
        try:
            extra.unlink()
        except Exception:
            pass

    # Clean up conflicting texture formats (e.g. remove unwanted PNG from low-poly JPG folder)
    if target_format.lower() in ("jpg", "jpeg"):
        for png_leftover in dst_dir.glob("*.png"):
            if png_leftover.name != dst_mtl_path.name:
                try:
                    png_leftover.unlink()
                    logger.debug(f"Removed extraneous PNG from JPG folder: {png_leftover.name}")
                except Exception:
                    pass
    elif target_format.lower() == "png":
        for jpg_leftover in list(dst_dir.glob("*.jpg")) + list(dst_dir.glob("*.jpeg")):
            try:
                jpg_leftover.unlink()
                logger.debug(f"Removed extraneous JPG from PNG folder: {jpg_leftover.name}")
            except Exception:
                pass


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
      output_dir/obj/high/model_high.obj (+ model_high.mtl + PNG texture)
      output_dir/obj/low/model_low.obj (+ model_low.mtl + downscaled JPG texture)
      output_dir/stl/high_model.stl
      output_dir/stl/low_model.stl
      output_dir/3mf/high_model.3mf
      output_dir/3mf/low_model.3mf
      output_dir/glb/high_model.glb (embedded PNG texture)
      output_dir/glb/low_model.glb (embedded JPG texture)
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
    high_mtl = obj_high_dir / "model_high.mtl"
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

    # Low-Poly Assets paths
    low_obj = obj_low_dir / "model_low.obj"
    low_mtl = obj_low_dir / "model_low.mtl"
    low_stl = stl_dir / "low_model.stl"
    low_3mf = mf_dir / "low_model.3mf"
    low_glb = glb_dir / "low_model.glb"

    has_textures = bool(textured_obj_path and Path(textured_obj_path).is_file())

    if has_textures:
        tex_p = Path(textured_obj_path)
        src_mtl, src_tex_imgs = find_source_mtl_and_textures(tex_p)

        # 1. Package High-Poly OBJ with PNG textures (lossless)
        package_textured_model(
            src_obj_path=tex_p,
            dst_obj_path=high_obj,
            dst_mtl_path=high_mtl,
            src_mtl_path=src_mtl,
            src_texture_images=src_tex_imgs,
            target_format="png",
            downscale_max_dim=None,
        )

        # 2. Decimate high-poly mesh to low-poly OBJ
        decimate_mesh(
            input_mesh_path=str(high_obj),
            output_mesh_path=str(low_obj),
            target_faces=config.cleaning.target_low_poly_faces,
        )

        # 3. Package Low-Poly OBJ with downscaled JPG textures + model_low.mtl
        package_textured_model(
            src_obj_path=low_obj,
            dst_obj_path=low_obj,
            dst_mtl_path=low_mtl,
            src_mtl_path=high_mtl,
            src_texture_images=src_tex_imgs,
            target_format="jpg",
            downscale_max_dim=2048,
            jpg_quality=85,
        )

        # Helper: GLB postprocessor for unlit, double-sided photogrammetry materials
        def gltf_unlit_postproc(tree):
            tree.setdefault('extensionsUsed', []).append('KHR_materials_unlit')
            for mat in tree.get('materials', []):
                mat['doubleSided'] = True
                mat.setdefault('extensions', {})['KHR_materials_unlit'] = {}

        # 4. Generate High-Poly GLB from model_high.obj (contains embedded PNG texture)
        try:
            textured_high_scene = trimesh.load(str(high_obj), process=False)
            glb_data = trimesh.exchange.gltf.export_glb(textured_high_scene, tree_postprocessor=gltf_unlit_postproc)
            with open(str(high_glb), "wb") as gf:
                gf.write(glb_data)
            telemetry.record_mesh(str(high_glb))
            logger.info(f"Exported textured High-Poly GLB (Unlit/PNG): {high_glb.name}")
        except Exception as ge:
            logger.warning(f"High GLB export warning: {ge}. Using standard GLB export.")
            cleaned_trimesh.export(str(high_glb), file_type="glb")
            telemetry.record_mesh(str(high_glb))

        # 5. Generate Low-Poly GLB from model_low.obj (contains embedded downscaled JPG texture)
        try:
            textured_low_scene = trimesh.load(str(low_obj), process=False)
            glb_data = trimesh.exchange.gltf.export_glb(textured_low_scene, tree_postprocessor=gltf_unlit_postproc)
            with open(str(low_glb), "wb") as gf:
                gf.write(glb_data)
            telemetry.record_mesh(str(low_glb))
            logger.info(f"Exported textured Low-Poly GLB (Unlit/JPG): {low_glb.name}")
        except Exception as ge:
            logger.warning(f"Low GLB export warning: {ge}. Using standard GLB export.")
            cleaned_trimesh.export(str(low_glb), file_type="glb")
            telemetry.record_mesh(str(low_glb))

        telemetry.record_mesh(str(high_obj))
        telemetry.record_mesh(str(low_obj))

    else:
        # Fallback to untextured mesh
        cleaned_trimesh.export(str(high_obj), file_type="obj")
        cleaned_trimesh.export(str(high_glb), file_type="glb")
        telemetry.record_mesh(str(high_obj))
        telemetry.record_mesh(str(high_glb))

        decimate_mesh(
            input_mesh_path=cleaned_mesh_path,
            output_mesh_path=str(low_obj),
            target_faces=config.cleaning.target_low_poly_faces,
        )
        loaded_low = trimesh.load(str(low_obj), process=False)
        loaded_low.export(str(low_glb), file_type="glb")
        telemetry.record_mesh(str(low_obj))
        telemetry.record_mesh(str(low_glb))

    # Low-poly STL and 3MF
    loaded_low = trimesh.load(str(low_obj), process=False)
    if isinstance(loaded_low, trimesh.Scene):
        geoms = [g for g in loaded_low.geometry.values() if isinstance(g, trimesh.Trimesh)]
        low_trimesh = trimesh.util.concatenate(geoms) if geoms else cleaned_trimesh
    else:
        low_trimesh = loaded_low

    low_trimesh.export(str(low_stl), file_type="stl")
    telemetry.record_mesh(str(low_stl))

    try:
        low_trimesh.export(str(low_3mf), file_type="3mf")
        telemetry.record_mesh(str(low_3mf))
    except Exception as e:
        logger.warning(f"Low-poly 3MF export warning: {e}")

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
