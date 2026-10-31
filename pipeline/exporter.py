"""
exporter.py — Multi-LOD asset packaging:
High-Poly OBJ (PNG), Low-Poly OBJ (JPG), STL, 3MF, and GLB (Draco),
and packaging into output.zip matching Cloudflare R2 specifications.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from PIL import Image

from .config import MeshCleaningConfig, PipelineConfig
from .telemetry import SystemTelemetry

logger = logging.getLogger("TurntableStudio.Exporter")


# The pipeline works Y-up (turntable plane Y = 0, as glTF / OBJ). STL and 3MF are Z-up by convention (3MF spec,
# slicers, Blender's STL import does not convert axes), so the print copies are rotated +90 deg about X:
# (x, y, z) -> (x, -z, y). A proper rotation (det +1): face winding and outward normals stay valid.
Y_UP_TO_Z_UP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def to_print_frame(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Copy of a Y-up mesh standing upright for STL / 3MF (Z-up), base on Z = 0."""
    out = mesh.copy()
    out.apply_transform(Y_UP_TO_Z_UP)
    return out


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

        # 1. Texture-preserving quadric decimation keeps the UV coordinates (vt), but only exists for meshes that
        #    have them: the untextured cleaned mesh goes straight to the standard decimation.
        textured = ms.current_mesh().has_wedge_tex_coord()
        try:
            if not textured:
                raise ValueError("mesh has no texture coordinates")
            ms.meshing_decimation_quadric_edge_collapse_with_texture(
                targetfacenum=target_faces,
                preserveboundary=True,
                optimalplacement=True,
            )
            logger.info(f"Texture-preserving decimation succeeded: {ms.current_mesh().face_number():,d} faces.")
        except Exception as tex_e:
            if textured:
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


_GLTF_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_GLTF_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
DRACO_EXT = "KHR_draco_mesh_compression"


def _read_glb(path: Path) -> Tuple[dict, bytes]:
    data = path.read_bytes()
    import struct
    magic, _, total = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:
        raise ValueError(f"{path.name} is not a GLB")
    off, js, binc = 12, None, b""
    while off < total:
        length, kind = struct.unpack("<II", data[off:off + 8])
        chunk = data[off + 8:off + 8 + length]
        if kind == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif kind == 0x004E4942:
            binc = chunk
        off += 8 + length
    return js, binc


def _write_glb(path: Path, js: dict, binc: bytes) -> None:
    import struct
    jb = json.dumps(js, separators=(",", ":")).encode("utf-8")
    jb += b" " * (-len(jb) % 4)
    binc += b"\0" * (-len(binc) % 4)
    total = 12 + 8 + len(jb) + 8 + len(binc)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(jb), 0x4E4F534A) + jb)
        f.write(struct.pack("<II", len(binc), 0x004E4942) + binc)


def draco_compress_glb(glb_path: Union[str, Path], position_bits: int = 14, texcoord_bits: int = 14,
                       normal_bits: int = 10, level: int = 7) -> Optional[Dict[str, float]]:
    """
    Compresses the geometry of every triangle primitive of a GLB with Draco (KHR_draco_mesh_compression), in place.
    Geometry is ~95 % of the GLB (positions, normals, UVs, indices); textures are untouched.
    Position quantization of 14 bits over a ~0.25 m object is ~0.015 mm; UVs at 14 bits are 1/4 texel on a
    4096 atlas. Viewers need a Draco decoder (Blender, three.js DRACOLoader, model-viewer, Babylon have one).
    Returns {"before_mb", "after_mb"} or None when DracoPy is missing or the GLB has nothing to compress.
    """
    try:
        import DracoPy
    except ImportError:
        logger.warning("[GLB] DracoPy not installed: GLB left without Draco compression.")
        return None
    path = Path(glb_path)
    js, binc = _read_glb(path)
    before = path.stat().st_size
    views = js.get("bufferViews", [])
    accessors = js.get("accessors", [])

    def read(acc_idx: int) -> np.ndarray:
        acc = accessors[acc_idx]
        bv = views[acc["bufferView"]]
        dtype = _GLTF_COMPONENT[acc["componentType"]]
        width = _GLTF_WIDTH[acc["type"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride")
        if stride and stride != np.dtype(dtype).itemsize * width:
            raise ValueError("interleaved GLB attributes are not supported")
        return np.frombuffer(binc, dtype, acc["count"] * width, start).reshape(acc["count"], width)

    draco_blobs: List[bytes] = []
    geometry_views = set()
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            if prim.get("mode", 4) != 4 or "indices" not in prim or "POSITION" not in attrs:
                continue
            if DRACO_EXT in prim.get("extensions", {}):
                continue
            pos = read(attrs["POSITION"]).astype(np.float64)
            faces = read(prim["indices"]).reshape(-1, 3).astype(np.int64)
            kwargs = {}
            if "TEXCOORD_0" in attrs:
                kwargs["tex_coord"] = read(attrs["TEXCOORD_0"]).astype(np.float64)
                kwargs["tex_coord_quantization_bits"] = texcoord_bits
            if "NORMAL" in attrs:
                kwargs["normals"] = read(attrs["NORMAL"]).astype(np.float64)
                kwargs["normal_quantization_bits"] = normal_bits
            blob = DracoPy.encode(pos, faces, quantization_bits=position_bits, compression_level=level,
                                  preserve_order=False, **kwargs)
            # Attribute ids and counts as the decoder will see them
            dec = DracoPy.decode(blob)
            ids = {a["attribute_type"]: a["unique_id"] for a in dec.attributes}
            draco_attrs = {"POSITION": ids[0]}
            if "NORMAL" in attrs:
                draco_attrs["NORMAL"] = ids[1]
            if "TEXCOORD_0" in attrs:
                draco_attrs["TEXCOORD_0"] = ids[3]
            n_points = len(dec.points)
            for key, acc_idx in [(k, attrs[k]) for k in draco_attrs] + [("indices", prim["indices"])]:
                acc = accessors[acc_idx]
                geometry_views.add(acc.pop("bufferView", None))
                acc.pop("byteOffset", None)
                acc["count"] = int(len(dec.faces) * 3 if key == "indices" else n_points)
                if key == "POSITION":
                    pts = np.asarray(dec.points, dtype=np.float64)
                    acc["min"], acc["max"] = pts.min(axis=0).tolist(), pts.max(axis=0).tolist()
                if key == "indices":
                    acc["componentType"] = 5125
            prim.setdefault("extensions", {})[DRACO_EXT] = {"bufferView": None, "attributes": draco_attrs}
            draco_blobs.append(blob)
            prim["_draco_index"] = len(draco_blobs) - 1
    if not draco_blobs:
        return None

    # New binary chunk: the views still in use (textures) + one view per Draco blob, re-indexed
    keep = [i for i in range(len(views)) if i not in geometry_views]
    remap, new_views, chunks, offset = {}, [], [], 0
    for i in keep:
        bv = views[i]
        data = binc[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        pad = -offset % 4
        chunks.append(b"\0" * pad)
        offset += pad
        remap[i] = len(new_views)
        new_views.append({k: v for k, v in bv.items() if k not in ("byteOffset", "byteStride", "target")} |
                         {"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        chunks.append(data)
        offset += len(data)
    blob_view = []
    for blob in draco_blobs:
        pad = -offset % 4
        chunks.append(b"\0" * pad)
        offset += pad
        blob_view.append(len(new_views))
        new_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        chunks.append(blob)
        offset += len(blob)
    for img in js.get("images", []):
        if "bufferView" in img:
            img["bufferView"] = remap[img["bufferView"]]
    for acc in accessors:
        if "bufferView" in acc:
            acc["bufferView"] = remap[acc["bufferView"]]
    for mesh in js.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "_draco_index" in prim:
                prim["extensions"][DRACO_EXT]["bufferView"] = blob_view[prim.pop("_draco_index")]
    js["bufferViews"] = new_views
    js["buffers"] = [{"byteLength": offset}]
    for key in ("extensionsUsed", "extensionsRequired"):
        if DRACO_EXT not in js.setdefault(key, []):
            js[key].append(DRACO_EXT)
    _write_glb(path, js, b"".join(chunks))
    after = path.stat().st_size
    logger.info(f"[GLB] Draco: {path.name} {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    return {"before_mb": round(before / 1e6, 2), "after_mb": round(after / 1e6, 2)}


def merge_into_one_object(tree: dict, name: str) -> bool:
    """
    trimesh exports one glTF node + mesh per OBJ material, so a high poly textured on 8 atlases opened in Blender
    as 8 objects ("material0000" ...) that had to be selected together to move the model. Rewrites the tree into
    ONE node and ONE mesh holding one primitive per material (one object with several material slots); the
    textures keep their resolution. Skipped when a node carries a transform (not the case for an OBJ).
    Returns True when the tree was changed.
    """
    nodes = tree.get("nodes") or []
    meshes = tree.get("meshes") or []
    if not meshes:
        return False
    identity = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
    for node in nodes:
        m = node.get("matrix")
        if (m is not None and any(abs(a - b) > 1e-9 for a, b in zip(m, identity))) or any(
                k in node for k in ("translation", "rotation", "scale", "children")):
            return False
    primitives = [prim for mesh in meshes for prim in mesh.get("primitives", [])]
    primitives.sort(key=lambda prim: prim.get("material", 0))
    tree["meshes"] = [{"name": name, "primitives": primitives}]
    tree["nodes"] = [{"name": name, "mesh": 0}]
    tree["scenes"] = [{"nodes": [0]}]
    tree["scene"] = 0
    return True


def convert_obj_to_glb(
    obj_folder_or_path: Union[str, Path],
    glb_path: Union[str, Path],
    compress_textures: bool = True,
    jpeg_quality: int = 85,
    draco: bool = False,
    draco_level: int = 7,
    draco_bits: int = 14,
) -> None:
    """
    Packs an OBJ + MTL + texture images into a single GLB (binary glTF).

    Stage 1 (trimesh): loads the OBJ scene preserving all geometry and material bindings,
                       and optionally re-encodes texture images as optimized JPEG buffers.
    Stage 2 (glTF export): packs into a single binary GLB file with double-sided,
                           unlit photogrammetry material settings.

    :param obj_folder_or_path: Path to the folder containing the OBJ or the OBJ file directly.
    :param glb_path: Path where the output GLB file will be saved.
    :param compress_textures: Whether to compress textures to JPEG.
    :param jpeg_quality: Quality for JPEG texture compression (default 85).
    :param draco: Whether to compress the geometry with Draco (KHR_draco_mesh_compression, needs DracoPy).
    """
    p = Path(obj_folder_or_path).resolve()
    if p.is_dir():
        for cand_name in ["model_high.obj", "model_low.obj", "texturedMesh.obj", "model_textured.obj"]:
            if (p / cand_name).is_file():
                obj_path = p / cand_name
                break
        else:
            objs = sorted(list(p.glob("*.obj")))
            if objs:
                obj_path = objs[0]
            else:
                logger.error(f"[GLB] ERROR: No OBJ file found in folder: {p}")
                return
    else:
        obj_path = p

    if not obj_path.is_file():
        logger.error(f"[GLB] ERROR: OBJ not found: {obj_path}")
        return

    try:
        logger.info(f"[GLB] Converting OBJ -> GLB (compress_textures={compress_textures})...")
        logger.info(f"      Source: {obj_path}")

        # Load scene preserving all geometries and materials
        scene = trimesh.load(str(obj_path), process=False, force="scene")

        if hasattr(scene, "geometry"):
            if compress_textures:
                import io
                from PIL import Image
                for geom in scene.geometry.values():
                    if hasattr(geom, "visual") and hasattr(geom.visual, "material"):
                        mat = geom.visual.material
                        for attr_name in ['image', 'baseColorTexture']:
                            if hasattr(mat, attr_name) and getattr(mat, attr_name) is not None:
                                img = getattr(mat, attr_name)
                                if isinstance(img, Image.Image):
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    buffer = io.BytesIO()
                                    img.save(buffer, format="JPEG", quality=jpeg_quality)
                                    buffer.seek(0)
                                    setattr(mat, attr_name, Image.open(buffer))

        out_glb_p = Path(glb_path).resolve()
        out_glb_p.parent.mkdir(parents=True, exist_ok=True)

        # GLB postprocessor for unlit, double-sided photogrammetry materials
        def gltf_unlit_postproc(tree):
            tree.setdefault('extensionsUsed', []).append('KHR_materials_unlit')
            for mat in tree.get('materials', []):
                mat['doubleSided'] = True
                mat.setdefault('extensions', {})['KHR_materials_unlit'] = {}
            merge_into_one_object(tree, out_glb_p.stem)

        try:
            glb_data = trimesh.exchange.gltf.export_glb(scene, tree_postprocessor=gltf_unlit_postproc)
        except Exception as ge:
            logger.warning(f"[GLB] Unlit postprocessor export failed ({ge}), exporting standard GLB...")
            glb_data = scene.export(file_type="glb")

        with open(str(out_glb_p), "wb") as gf:
            gf.write(glb_data)
        if draco:
            try:
                draco_compress_glb(out_glb_p, position_bits=draco_bits, level=draco_level)
            except Exception as de:
                # The uncompressed GLB is still valid: rewrite it rather than leave a half-converted file
                logger.warning(f"[GLB] Draco compression failed ({de}); keeping the uncompressed GLB.")
                with open(str(out_glb_p), "wb") as gf:
                    gf.write(glb_data)

        logger.info(f"[GLB] Successfully exported {out_glb_p.name} ({len(glb_data) / 1024 / 1024:.2f} MB)")

    except Exception as e:
        logger.error(f"[GLB] Error converting OBJ to GLB: {e}", exc_info=True)
        raise


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

    # Fallback: images named after this exact OBJ (texrecon: <stem>_material0000_map_Kd.png,
    # OpenMVS: <stem>_material_0_map_Kd.jpg). A bare substring test would also pick up
    # model_textured_low_* atlases for model_textured.obj.
    if not tex_images:
        prefixes = (f"{tex_p.stem}_material", f"{tex_p.stem}_texture", f"{tex_p.stem}_tex_")
        for f in sorted(tex_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in img_exts and f.stem.startswith(prefixes):
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
    scale: float = 1.0,
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
                    scale_f = downscale_max_dim / max(w, h)
                    nw = max(1, int(w * scale_f))
                    nh = max(1, int(h * scale_f))
                    img_out = img.resize((nw, nh), Image.Resampling.LANCZOS)
                else:
                    img_out = img.copy()

                if target_format.lower() in ("jpg", "jpeg"):
                    img_out.convert("RGB").save(str(out_tex_path), format="JPEG", quality=jpg_quality, optimize=True)
                else:
                    img_out.save(str(out_tex_path), format="PNG", optimize=True)

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
                    # Ensure neutral ambient reflection for photogrammetry baked textures (prevents washed-out blacks in Blender)
                    if clean_l.strip().startswith("Ka "):
                        clean_l = "Ka 0.000000 0.000000 0.000000"
                    mtl_lines.append(clean_l + "\n")
        except Exception as e:
            logger.warning(f"Error reading source MTL: {e}")

    # Default MTL template if none was found
    if not mtl_lines:
        first_tex = list(tex_map_replacements.values())[0] if tex_map_replacements else f"{dst_obj_path.stem}_texture.{target_format.lower()}"
        mtl_lines = [
            f"# Material file for {dst_obj_path.name}\n",
            "newmtl material_0\n",
            "Ka 0.000000 0.000000 0.000000\n",
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
            if scale != 1.0 and line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        scaled_v = [float(p) * scale for p in parts[1:4]]
                        rest = parts[4:]
                        v_str = f"v {scaled_v[0]:.6f} {scaled_v[1]:.6f} {scaled_v[2]:.6f}"
                        if rest:
                            v_str += " " + " ".join(rest)
                        obj_lines.append(v_str + "\n")
                        continue
                    except ValueError:
                        pass
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


DELIVERABLE_DIRS = (("obj", "high"), ("obj", "low"), ("stl",), ("3mf",), ("glb",))
# Reports packed next to the deliverables (pose quality: RMS, lift, rig correction; poses for re-use)
REPORT_FILES = ("object.json", "turntable_report.txt", "turntable_calib.json", "rig_refine.json")


def collect_log_files(output_dir: str, extra_dirs: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Log files to pack under logs/ in output.zip: run_log.txt, the OpenMVS app logs and the texrecon logs.
    extra_dirs: OpenMVS work directories outside output_dir (--work_dir). Returns {arcname: path}.
    """
    out_base = Path(output_dir).resolve()
    logs: Dict[str, str] = {}
    run_log = out_base / "run_log.txt"
    if run_log.is_file():
        logs["logs/run_log.txt"] = str(run_log)
    openmvs_dirs = [out_base / "openmvs"] + [Path(d) for d in (extra_dirs or [])]
    for d in openmvs_dirs:
        for f in sorted(d.glob("*.log")) if d.is_dir() else []:
            logs[f"logs/openmvs/{f.name}"] = str(f)
    tr_dir = out_base / "texrecon"
    for f in sorted(tr_dir.glob("*.log")) if tr_dir.is_dir() else []:
        logs[f"logs/texrecon/{f.name}"] = str(f)
    return logs


def package_output_zip(
    output_dir: str,
    zip_name: str = "output.zip",
    stats_json_path: Optional[str] = None,
    log_files: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Builds output.zip: obj/, stl/, 3mf/, glb/, stats.json, turntable_report.txt,
    turntable_calib.json and logs/ (log_files = {arcname: path}).
    """
    out_base = Path(output_dir).resolve()
    zip_path = out_base / zip_name
    logger.info(f"Packaging multi-LOD assets into ZIP archive: {zip_path}...")

    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        if stats_json_path and Path(stats_json_path).is_file():
            zipf.write(stats_json_path, arcname="stats.json")
        for name in REPORT_FILES:
            if (out_base / name).is_file():
                zipf.write(out_base / name, arcname=name)
        for parts in DELIVERABLE_DIRS:
            folder = out_base.joinpath(*parts)
            for root, _, files in os.walk(folder):
                for f in sorted(files):
                    file_p = Path(root) / f
                    zipf.write(file_p, arcname=str(file_p.relative_to(out_base)))
        for arcname, path in sorted((log_files or {}).items()):
            if Path(path).is_file():
                zipf.write(path, arcname=arcname)

    logger.info(f"Created {zip_path.name} ({zip_path.stat().st_size / (1024 * 1024):.2f} MB)")
    return zip_path


def export_multi_lod_bundle(
    cleaned_mesh_path: str,
    textured_obj_path: Optional[str],
    output_dir: str,
    telemetry: SystemTelemetry,
    config: PipelineConfig,
    stats_json_path: Optional[str] = None,
    textured_low_obj_path: Optional[str] = None,
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

    # Start from empty deliverable folders: a previous run with more texture atlases would
    # otherwise leave its extra model_high_tex_N.png files here and inside output.zip.
    for d in [obj_high_dir, obj_low_dir, stl_dir, mf_dir, glb_dir]:
        if d.is_dir():
            shutil.rmtree(str(d))
        d.mkdir(parents=True, exist_ok=True)

    # 1. High-Poly Assets
    high_obj = obj_high_dir / "model_high.obj"
    high_mtl = obj_high_dir / "model_high.mtl"
    high_stl = stl_dir / "high_model.stl"
    high_3mf = mf_dir / "high_model.3mf"
    high_glb = glb_dir / "high_model.glb"

    # Base geometry for 3D printing (watertight from mesh_cleaner)
    cleaned_trimesh = trimesh.load(cleaned_mesh_path, process=False)
    exp_scale = float(getattr(config, "export_scale", 100.0))
    if exp_scale != 1.0:
        high_print_mesh = cleaned_trimesh.copy()
        high_print_mesh.apply_scale(exp_scale)
    else:
        high_print_mesh = cleaned_trimesh

    # Export watertight STL and 3MF at real-world scale, standing upright (Z-up)
    high_print_mesh = to_print_frame(high_print_mesh)
    high_print_mesh.export(str(high_stl), file_type="stl")
    telemetry.record_mesh(str(high_stl))

    try:
        high_print_mesh.export(str(high_3mf), file_type="3mf")
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
            scale=exp_scale,
        )

        # 2. Package Low-Poly OBJ:
        # If textured_low_obj_path is provided from branching pipeline, package it directly (native UVs!).
        # Otherwise, fall back to geometry decimation of high_obj.
        if textured_low_obj_path and Path(textured_low_obj_path).is_file():
            low_src_p = Path(textured_low_obj_path)
            low_src_mtl, low_src_tex_imgs = find_source_mtl_and_textures(low_src_p)
            package_textured_model(
                src_obj_path=low_src_p,
                dst_obj_path=low_obj,
                dst_mtl_path=low_mtl,
                src_mtl_path=low_src_mtl or high_mtl,
                src_texture_images=low_src_tex_imgs or src_tex_imgs,
                target_format="jpg",
                downscale_max_dim=2048,
                jpg_quality=85,
                scale=exp_scale,
            )
            logger.info("Successfully packaged Low-Poly OBJ from dedicated branching texturing!")
        else:
            decimate_mesh(
                input_mesh_path=str(high_obj),
                output_mesh_path=str(low_obj),
                target_faces=config.cleaning.target_low_poly_faces,
            )
            package_textured_model(
                src_obj_path=low_obj,
                dst_obj_path=low_obj,
                dst_mtl_path=low_mtl,
                src_mtl_path=high_mtl,
                src_texture_images=src_tex_imgs,
                target_format="jpg",
                downscale_max_dim=2048,
                jpg_quality=85,
                scale=1.0,  # high_obj is already scaled by exp_scale, so avoid double-scaling
            )

        # 4. Generate High-Poly GLB from model_high.obj (contains embedded PNG texture)
        try:
            convert_obj_to_glb(
                obj_folder_or_path=high_obj,
                glb_path=high_glb,
                compress_textures=False,  # High-poly retains lossless PNG texture
                draco=getattr(config, "glb_draco", True),
            )
            telemetry.record_mesh(str(high_glb))
        except Exception as ge:
            logger.warning(f"High GLB export warning: {ge}. Using standard GLB export.")
            cleaned_trimesh.export(str(high_glb), file_type="glb")
            telemetry.record_mesh(str(high_glb))

        # 5. Generate Low-Poly GLB from model_low.obj (contains embedded downscaled JPG texture)
        try:
            convert_obj_to_glb(
                obj_folder_or_path=low_obj,
                glb_path=low_glb,
                compress_textures=True,  # Low-poly compresses textures to web-optimized JPEG
                jpeg_quality=85,
                draco=getattr(config, "glb_draco", True),
            )
            telemetry.record_mesh(str(low_glb))
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

    # Low-poly STL and 3MF (loaded from low_obj, which is already scaled to real-world scale exp_scale)
    loaded_low = trimesh.load(str(low_obj), process=False)
    if isinstance(loaded_low, trimesh.Scene):
        geoms = [g for g in loaded_low.geometry.values() if isinstance(g, trimesh.Trimesh)]
        low_trimesh = trimesh.util.concatenate(geoms) if geoms else cleaned_trimesh.copy()
    else:
        low_trimesh = loaded_low.copy()
    low_trimesh = to_print_frame(low_trimesh)   # STL / 3MF are Z-up

    low_trimesh.export(str(low_stl), file_type="stl")
    telemetry.record_mesh(str(low_stl))

    try:
        low_trimesh.export(str(low_3mf), file_type="3mf")
        telemetry.record_mesh(str(low_3mf))
    except Exception as e:
        logger.warning(f"Low-poly 3MF export warning: {e}")

    # 3. Create output.zip containing the entire tree (main.py repackages it at the very end with the logs)
    zip_path = package_output_zip(str(out_base), config.zip_name, stats_json_path)
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)

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
