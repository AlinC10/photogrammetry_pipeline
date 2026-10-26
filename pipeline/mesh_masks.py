"""
mesh_masks.py — Texturing masks rendered from the cleaned mesh with the locked camera poses.

The photometric masks (mask_processor.py) cut light-grey parts of the object and specular
highlights, which leaves BLACK holes inside the silhouette. texrecon only invalidates pure
black pixels connected to the image corners, so those holes are sampled as valid black texture.
The silhouette of the watertight mesh, rendered with the same undistorted camera (K_new) used for
the texturing images, is exact for every part of the object, whatever its colour.

Also provides a small vectorised triangle rasterizer (face-id + depth buffer) used by the
diagnostics and by the per-camera colour normalisation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("TurntableStudio.MeshMasks")

# Bounding-box classes (px) for the vectorised rasterizer; larger triangles are drawn one by one.
_BBOX_CLASSES = (4, 8, 16, 32, 64)
_MAX_CANDIDATES_PER_CHUNK = 2_000_000
_NEAR_Z = 1e-4


def project_vertices(
    vertices: np.ndarray, R: np.ndarray, C: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pinhole projection of world points (AliceVision R, C conventions, RULE 3):
    X_cam = R (X_world - C), u = fx x/z + cx, v = fy y/z + cy (OpenCV pixel centres).
    Returns (uv [N,2], z [N], X_cam [N,3]).
    """
    Xc = (np.asarray(vertices, dtype=np.float64) - np.asarray(C, dtype=np.float64).reshape(1, 3)) @ R.T
    z = Xc[:, 2]
    safe_z = np.where(np.abs(z) < 1e-12, 1e-12, z)
    u = K[0, 0] * Xc[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / safe_z + K[1, 2]
    return np.stack([u, v], axis=1), z, Xc


def _triangle_setup(
    uv: np.ndarray,
    z: np.ndarray,
    Xc: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    cull_backfaces: bool,
) -> Dict[str, np.ndarray]:
    f = np.asarray(faces, dtype=np.int64)
    p0, p1, p2 = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    z0, z1, z2 = z[f[:, 0]], z[f[:, 1]], z[f[:, 2]]
    ok = (z0 > _NEAR_Z) & (z1 > _NEAR_Z) & (z2 > _NEAR_Z)

    if cull_backfaces:
        X0, X1, X2 = Xc[f[:, 0]], Xc[f[:, 1]], Xc[f[:, 2]]
        n = np.cross(X1 - X0, X2 - X0)
        # camera at the origin: front-facing when the normal points back to the camera
        ok &= np.einsum("ij,ij->i", n, X0) < 0.0

    area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    ok &= np.abs(area) > 1e-12

    xs = np.stack([p0[:, 0], p1[:, 0], p2[:, 0]], axis=1)
    ys = np.stack([p0[:, 1], p1[:, 1], p2[:, 1]], axis=1)
    with np.errstate(invalid="ignore"):
        xmin = np.clip(np.ceil(np.nanmin(xs, axis=1)), 0, width - 1)
        xmax = np.clip(np.floor(np.nanmax(xs, axis=1)), 0, width - 1)
        ymin = np.clip(np.ceil(np.nanmin(ys, axis=1)), 0, height - 1)
        ymax = np.clip(np.floor(np.nanmax(ys, axis=1)), 0, height - 1)
    ok &= np.isfinite(xs).all(axis=1) & np.isfinite(ys).all(axis=1)
    ok &= (np.ceil(xs.min(axis=1)) <= width - 1) & (np.floor(xs.max(axis=1)) >= 0)
    ok &= (np.ceil(ys.min(axis=1)) <= height - 1) & (np.floor(ys.max(axis=1)) >= 0)
    ok &= (xmax >= xmin) & (ymax >= ymin)

    idx = np.nonzero(ok)[0]
    return {
        "face": idx,
        "p0": p0[idx], "p1": p1[idx], "p2": p2[idx],
        "iz0": 1.0 / z0[idx], "iz1": 1.0 / z1[idx], "iz2": 1.0 / z2[idx],
        "area": area[idx],
        "xmin": xmin[idx].astype(np.int64), "ymin": ymin[idx].astype(np.int64),
        "bw": (xmax[idx] - xmin[idx]).astype(np.int64) + 1,
        "bh": (ymax[idx] - ymin[idx]).astype(np.int64) + 1,
    }


def _raster_block(tri: Dict[str, np.ndarray], sel: np.ndarray, tile: int, width: int):
    """Tests every pixel centre of a tile x tile window for the selected triangles."""
    dy, dx = np.divmod(np.arange(tile * tile, dtype=np.int64), tile)
    px = tri["xmin"][sel, None] + dx[None, :]
    py = tri["ymin"][sel, None] + dy[None, :]
    inside_box = (dx[None, :] < tri["bw"][sel, None]) & (dy[None, :] < tri["bh"][sel, None])

    p0, p1, p2 = tri["p0"][sel], tri["p1"][sel], tri["p2"][sel]
    inv_area = 1.0 / tri["area"][sel]
    fx = px.astype(np.float64)
    fy = py.astype(np.float64)
    # normalised barycentric coordinates (edge functions / area)
    l0 = ((p2[:, 0, None] - p1[:, 0, None]) * (fy - p1[:, 1, None])
          - (p2[:, 1, None] - p1[:, 1, None]) * (fx - p1[:, 0, None])) * inv_area[:, None]
    l1 = ((p0[:, 0, None] - p2[:, 0, None]) * (fy - p2[:, 1, None])
          - (p0[:, 1, None] - p2[:, 1, None]) * (fx - p2[:, 0, None])) * inv_area[:, None]
    l2 = 1.0 - l0 - l1
    eps = -1e-9
    inside = inside_box & (l0 >= eps) & (l1 >= eps) & (l2 >= eps)

    inv_z = l0 * tri["iz0"][sel, None] + l1 * tri["iz1"][sel, None] + l2 * tri["iz2"][sel, None]
    rows, cols = np.nonzero(inside)
    pix = py[rows, cols] * width + px[rows, cols]
    depth = 1.0 / inv_z[rows, cols]
    return pix, depth, sel[rows]


def _iterate_fragments(tri: Dict[str, np.ndarray], width: int):
    """Yields (pixel_index, depth, triangle_slot) fragments for all set-up triangles."""
    n = len(tri["face"])
    if n == 0:
        return
    size = np.maximum(tri["bw"], tri["bh"])
    lower = 0
    for tile in _BBOX_CLASSES:
        sel_all = np.nonzero((size > lower) & (size <= tile))[0]
        lower = tile
        chunk = max(1, _MAX_CANDIDATES_PER_CHUNK // (tile * tile))
        for s in range(0, len(sel_all), chunk):
            yield _raster_block(tri, sel_all[s:s + chunk], tile, width)

    # Triangles larger than the biggest class (e.g. ear-clipped base cap): split their bounding
    # box into windows of the biggest class.
    tile = _BBOX_CLASSES[-1]
    for slot in np.nonzero(size > tile)[0]:
        one = {k: v[slot:slot + 1].copy() for k, v in tri.items()}
        x0, y0 = int(one["xmin"][0]), int(one["ymin"][0])
        bw, bh = int(one["bw"][0]), int(one["bh"][0])
        for oy in range(0, bh, tile):
            for ox in range(0, bw, tile):
                one["xmin"][0], one["ymin"][0] = x0 + ox, y0 + oy
                one["bw"][0], one["bh"][0] = min(tile, bw - ox), min(tile, bh - oy)
                pix, dep, _ = _raster_block(one, np.array([0]), tile, width)
                yield pix, dep, np.full(len(pix), slot, dtype=np.int64)


def render_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Binary silhouette (uint8, 0/255) of the mesh: union of all projected triangles, sampled at
    pixel centres. No depth test is needed for a silhouette.
    """
    uv, z, Xc = project_vertices(vertices, R, C, K)
    tri = _triangle_setup(uv, z, Xc, faces, width, height, cull_backfaces=False)
    flat = np.zeros(width * height, dtype=np.uint8)
    for pix, _, _ in _iterate_fragments(tri, width):
        flat[pix] = 255
    return flat.reshape(height, width)


def render_face_ids(
    vertices: np.ndarray,
    faces: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    cull_backfaces: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z-buffered rasterization. Returns (face_id [H,W] int32 with -1 = background,
    depth [H,W] float32 with +inf = background).
    """
    uv, z, Xc = project_vertices(vertices, R, C, K)
    tri = _triangle_setup(uv, z, Xc, faces, width, height, cull_backfaces=cull_backfaces)
    pix_l, dep_l, slot_l = [], [], []
    for pix, dep, slot in _iterate_fragments(tri, width):
        pix_l.append(pix)
        dep_l.append(dep)
        slot_l.append(slot)

    face_id = np.full(width * height, -1, dtype=np.int32)
    depth = np.full(width * height, np.inf, dtype=np.float32)
    if pix_l:
        pix = np.concatenate(pix_l)
        dep = np.concatenate(dep_l)
        slot = np.concatenate(slot_l)
        order = np.lexsort((dep, pix))
        pix, dep, slot = pix[order], dep[order], slot[order]
        first = np.ones(len(pix), dtype=bool)
        first[1:] = pix[1:] != pix[:-1]
        face_id[pix[first]] = tri["face"][slot[first]].astype(np.int32)
        depth[pix[first]] = dep[first].astype(np.float32)
    return face_id.reshape(height, width), depth.reshape(height, width)


def adjust_mask_margin(mask: np.ndarray, margin_px: int) -> np.ndarray:
    """Positive margin dilates the silhouette, negative erodes it (elliptical kernel)."""
    if margin_px == 0:
        return mask
    k = 2 * abs(int(margin_px)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if margin_px > 0:
        return cv2.dilate(mask, kernel)
    return cv2.erode(mask, kernel)


def load_sfm_cameras(sfm_json_path: str) -> List[Dict[str, Any]]:
    """
    Reads cameras_known.sfm and returns, per registered view, the locked pose and the
    undistorted pinhole camera K_new (identical to the one used by bridge_sfm for the images).
    """
    from .bridge_sfm import build_camera_matrix_and_distortion

    with open(sfm_json_path, "r", encoding="utf-8") as f:
        sfm = json.load(f)

    intr = {}
    for it in sfm.get("intrinsics", []):
        K, dist, w, h = build_camera_matrix_and_distortion(it)
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
        intr[str(it["intrinsicId"])] = {"K": K, "dist": dist, "K_new": K_new, "w": w, "h": h}

    poses = {}
    for p in sfm.get("poses", []):
        R = np.array(p["pose"]["transform"]["rotation"], dtype=np.float64).reshape((3, 3), order="F")
        C = np.array(p["pose"]["transform"]["center"], dtype=np.float64)
        poses[str(p["poseId"])] = (R, C)

    cams = []
    for v in sfm.get("views", []):
        pid = str(v["poseId"])
        if pid not in poses:
            continue
        ci = intr[str(v["intrinsicId"])]
        R, C = poses[pid]
        cams.append({
            "view_id": str(v["viewId"]),
            "intrinsic_id": str(v["intrinsicId"]),
            "path": str(v["path"]),
            "R": R, "C": C,
            "K": ci["K"], "dist": ci["dist"], "K_new": ci["K_new"],
            "w": ci["w"], "h": ci["h"],
        })
    return cams


def generate_mesh_masks(
    sfm_json_path: str,
    mesh_path: str,
    output_dir: str,
    margin_px: int = 2,
    name_fn=None,
) -> Dict[str, str]:
    """
    Renders one texturing mask per view from the cleaned mesh, in the undistorted image space
    (K_new). Masks are written as `<viewId>_<stem>.mask.png` (0 = background, 255 = object).
    Returns {view_id: mask_path}.
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    masks: Dict[str, str] = {}
    cams = load_sfm_cameras(sfm_json_path)
    logger.info(f"Rendering {len(cams)} mesh-silhouette texturing masks ({len(F):,d} faces, margin {margin_px:+d} px)...")
    for i, cam in enumerate(cams):
        sil = render_silhouette(V, F, cam["R"], cam["C"], cam["K_new"], cam["w"], cam["h"])
        sil = adjust_mask_margin(sil, margin_px)
        stem = Path(cam["path"]).stem
        name = name_fn(cam) if name_fn else f"{cam['view_id']}_{stem}.mask.png"
        dst = out / name
        cv2.imwrite(str(dst), sil)
        masks[cam["view_id"]] = str(dst)
        if (i + 1) % 10 == 0 or (i + 1) == len(cams):
            logger.info(f"  Rendered {i + 1}/{len(cams)} mesh masks")
    return masks
