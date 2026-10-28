"""
visual_hull.py — Volume carved from the silhouette masks of all locked views (shape from silhouette).

Two uses (plans/AUDIT_RECONSTRUCTIE_GENERALA.md, G6 and G2):
1. Bounds of the object for the proxy 3D points of bridge_sfm: DensifyPointCloud derives its ROI and the depth range
   of every view from them, and the old fixed 220 mm volume was lower than tall objects (fixativ: 231-244 mm).
2. Correcting the dense cloud where PatchMatch fails: on uniform, untextured plastic (a single-colour cap) it fuses
   wrong depths INSIDE the object from 2-3 views and ReconstructMesh digs into the cap. scene_dense.ply (which
   ReconstructMesh loads by name, with per-point view_indices / view_weights) is rewritten: low-view points deep
   inside the hull are pruned, and hull surface points are appended only where no MVS point is left nearby, the
   views that see them show the object and not the white background, and the surface is not near horizontal.
   Textured areas keep the MVS detail; the hull is always >= the object and blind to concavities.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("TurntableStudio.VisualHull")

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a runtime dependency
    cv2 = None

# Record layout written by OpenMVS v2.3 DensifyPointCloud (checked on the fixativ and multimeter clouds)
DENSE_PLY_PROPERTIES = [
    "property float32 x", "property float32 y", "property float32 z",
    "property uint8 red", "property uint8 green", "property uint8 blue",
    "property float32 nx", "property float32 ny", "property float32 nz",
    "property list uint8 uint32 view_indices",
    "property list uint8 float32 view_weights",
]
_FIXED_BYTES = 27   # xyz (12) + rgb (3) + normal (12)


# -----------------------------------------------------------------------------
# 1. Carving
# -----------------------------------------------------------------------------

def _frame_limits(cam: Dict[str, Any], distorted: bool) -> Tuple[float, float, float, float]:
    """Normalized (x/z, y/z) box of the image frame; points outside it are not judged by this view."""
    w, h = cam["w"], cam["h"]
    if not distorted:
        K = cam["K_new"]
        return (-K[0, 2] / K[0, 0], (w - K[0, 2]) / K[0, 0], -K[1, 2] / K[1, 1], (h - K[1, 2]) / K[1, 1])
    # Border of the distorted image, undistorted to normalized coordinates (the distortion polynomial folds far
    # off-frame points back into the image, so they must be rejected before projecting)
    t = np.linspace(0.0, 1.0, 17)

    def edge(px: np.ndarray) -> np.ndarray:
        return cv2.undistortPoints(px.astype(np.float64).reshape(-1, 1, 2), cam["K"], cam["dist"]).reshape(-1, 2)

    left = edge(np.stack([np.zeros_like(t), t * (h - 1)], 1))
    right = edge(np.stack([np.full_like(t, w - 1), t * (h - 1)], 1))
    top = edge(np.stack([t * (w - 1), np.zeros_like(t)], 1))
    bottom = edge(np.stack([t * (w - 1), np.full_like(t, h - 1)], 1))
    # The inscribed box is conservative: only points certainly inside the frame vote
    return (float(left[:, 0].max()), float(right[:, 0].min()), float(top[:, 1].max()), float(bottom[:, 1].min()))


def project_points(cam: Dict[str, Any], P: np.ndarray, distorted: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> pixel (u, v), camera depth and a mask of the points certainly inside the frame."""
    Xc = (P - cam["C"]) @ cam["R"].T
    z = Xc[:, 2]
    front = z > 0.05
    xn = np.where(front, Xc[:, 0] / np.where(front, z, 1.0), 0.0)
    yn = np.where(front, Xc[:, 1] / np.where(front, z, 1.0), 0.0)
    x0, x1, y0, y1 = _frame_limits(cam, distorted)
    ok = front & (xn > x0) & (xn < x1) & (yn > y0) & (yn < y1)
    uv = np.zeros((len(P), 2))
    if np.any(ok):
        if distorted:
            pts = np.stack([xn[ok], yn[ok], np.ones(int(ok.sum()))], 1).reshape(-1, 1, 3)
            uv[ok] = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), cam["K"], cam["dist"])[0].reshape(-1, 2)
        else:
            K = cam["K_new"]
            uv[ok, 0] = K[0, 0] * xn[ok] + K[0, 2]
            uv[ok, 1] = K[1, 1] * yn[ok] + K[1, 2]
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] < cam["w"] - 0.5) & (uv[:, 1] >= 0) & (uv[:, 1] < cam["h"] - 0.5)
    return uv, z, ok


def carve_visual_hull(
    cameras: List[Dict[str, Any]],
    mask_paths: List[Optional[str]],
    radius_m: float = 0.080,
    y_min_m: float = 0.003,
    y_max_m: float = 0.350,
    voxel_m: float = 0.0015,
    min_bg_votes: int = 2,
    min_seen_views: int = 3,
    distorted: bool = True,
) -> Dict[str, Any]:
    """
    Carves a voxel grid inside the masking cylinder with the silhouette masks.
    A voxel is removed when at least `min_bg_votes` views see it as background (one bad mask cannot cut the
    object) or when fewer than `min_seen_views` views see it at all (nothing can reconstruct it).
    `distorted`: masks of the original photos (camera K + dist) or undistorted masks (K_new).
    Returns {occupancy (nx, ny, nz bool), origin (3,), voxel_m, views_used, seconds}.
    """
    t0 = time.time()
    xs = np.arange(-radius_m, radius_m + 1e-9, voxel_m)
    ys = np.arange(y_min_m, y_max_m + 1e-9, voxel_m)
    shape = (len(xs), len(ys), len(xs))
    X, Y, Z = np.meshgrid(xs, ys, xs, indexing="ij")
    in_cyl = (X ** 2 + Z ** 2) <= radius_m ** 2
    idx = np.flatnonzero(in_cyl.ravel())
    P = np.stack([X.ravel()[idx], Y.ravel()[idx], Z.ravel()[idx]], 1)
    del X, Y, Z

    bg_votes = np.zeros(len(P), np.uint8)
    seen = np.zeros(len(P), np.uint16)
    used = 0
    for cam, mpath in zip(cameras, mask_paths):
        if not mpath or not os.path.isfile(mpath):
            continue
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape[:2] != (cam["h"], cam["w"]):
            logger.warning(f"Visual hull: mask {mpath} unreadable or of another size, view skipped.")
            continue
        used += 1
        alive = np.flatnonzero(bg_votes < min_bg_votes)
        uv, _, ok = project_points(cam, P[alive], distorted)
        sel = alive[ok]
        ui = np.rint(uv[ok, 0]).astype(np.int64)
        vi = np.rint(uv[ok, 1]).astype(np.int64)
        seen[sel] += 1
        bg_votes[sel[mask[vi, ui] < 128]] += 1

    occ_flat = np.zeros(in_cyl.size, bool)
    occ_flat[idx] = (bg_votes < min_bg_votes) & (seen >= min(min_seen_views, max(1, used)))
    occupancy = occ_flat.reshape(shape)
    hull = {
        "occupancy": occupancy,
        "origin": np.array([xs[0], ys[0], xs[0]], dtype=np.float64),
        "voxel_m": float(voxel_m),
        "views_used": used,
        "seconds": round(time.time() - t0, 2),
    }
    logger.info(
        f"Visual hull: {int(occupancy.sum()):,d} voxels of {voxel_m * 1000:.1f} mm from {used} masks "
        f"({hull['seconds']:.1f} s)."
    )
    return hull


def hull_stats(hull: Dict[str, Any]) -> Dict[str, Any]:
    """Volume, height range and maximum radius of the occupied voxels (None when empty)."""
    occ = hull["occupancy"]
    if not occ.any():
        return {"voxels": 0}
    ii, jj, kk = np.nonzero(occ)
    v = hull["voxel_m"]
    o = hull["origin"]
    x = o[0] + ii * v
    y = o[1] + jj * v
    z = o[2] + kk * v
    return {
        "voxels": int(occ.sum()),
        "voxel_mm": round(v * 1000, 2),
        "volume_cm3": round(float(occ.sum()) * v ** 3 * 1e6, 1),
        "y_min_mm": round(float(y.min()) * 1000, 1),
        "y_max_mm": round(float(y.max()) * 1000, 1),
        "radius_max_mm": round(float(np.hypot(x, z).max()) * 1000, 1),
        "views_used": hull.get("views_used"),
        "seconds": hull.get("seconds"),
    }


def proxy_bounds_from_hull(
    hull: Optional[Dict[str, Any]],
    cylinder_radius_m: float,
    cylinder_y_max_m: float,
    margin_m: float = 0.010,
) -> Dict[str, float]:
    """
    Volume for bridge_sfm.generate_proxy_3d_points: the hull box plus a margin, never larger than the masking
    cylinder. Without a usable hull: the masking cylinder itself.
    """
    stats = hull_stats(hull) if hull is not None else {"voxels": 0}
    if stats.get("voxels", 0) < 100:
        return {"y_max": float(cylinder_y_max_m), "r_max": float(cylinder_radius_m)}
    return {
        "y_max": float(min(cylinder_y_max_m, stats["y_max_mm"] / 1000 + margin_m)),
        "r_max": float(min(cylinder_radius_m, stats["radius_max_mm"] / 1000 + margin_m)),
    }


def save_hull(hull: Dict[str, Any], path: str) -> None:
    np.savez_compressed(path, occupancy=hull["occupancy"], origin=hull["origin"], voxel_m=hull["voxel_m"],
                        views_used=hull.get("views_used", 0), seconds=hull.get("seconds", 0.0))


def load_hull(path: str) -> Dict[str, Any]:
    d = np.load(path)
    return {"occupancy": d["occupancy"].astype(bool), "origin": d["origin"], "voxel_m": float(d["voxel_m"]),
            "views_used": int(d["views_used"]), "seconds": float(d["seconds"])}


def surface_points(hull: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Centres of the boundary voxels (occupied, with an empty 6-neighbour) and outward unit normals."""
    from scipy.ndimage import binary_erosion, gaussian_filter

    occ = hull["occupancy"]
    if not occ.any():
        return np.zeros((0, 3)), np.zeros((0, 3))
    padded = np.pad(occ, 1)
    boundary = (padded & ~binary_erosion(padded))[1:-1, 1:-1, 1:-1]
    # Outward normal = minus the gradient of the smoothed occupancy
    smooth = gaussian_filter(occ.astype(np.float32), sigma=1.5)
    gx, gy, gz = np.gradient(smooth)
    ii, jj, kk = np.nonzero(boundary)
    n = -np.stack([gx[ii, jj, kk], gy[ii, jj, kk], gz[ii, jj, kk]], 1).astype(np.float64)
    norm = np.linalg.norm(n, axis=1)
    keep = norm > 1e-6
    v = hull["voxel_m"]
    pts = hull["origin"] + np.stack([ii, jj, kk], 1)[keep] * v
    return pts, n[keep] / norm[keep, None]


# -----------------------------------------------------------------------------
# 2. OpenMVS dense point cloud (scene_dense.ply)
# -----------------------------------------------------------------------------

def read_dense_ply(path: str) -> Dict[str, Any]:
    """
    Reads the binary scene_dense.ply of DensifyPointCloud: header, raw body, xyz and a sample of view weights.
    Raises ValueError on any other layout (the caller then leaves the cloud untouched).
    """
    with open(path, "rb") as f:
        data = f.read()
    end = data.find(b"end_header\n")
    if end < 0:
        raise ValueError("no PLY header")
    header = data[:end].decode("ascii", errors="replace").splitlines()
    body = memoryview(data)[end + len(b"end_header\n"):]
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("not a binary little-endian PLY")
    elements = [ln for ln in header if ln.startswith("element ")]
    if len(elements) != 1 or not elements[0].startswith("element vertex "):
        raise ValueError(f"unexpected elements {elements}")
    props = [ln.strip() for ln in header if ln.startswith("property ")]
    if props != DENSE_PLY_PROPERTIES:
        raise ValueError(f"unexpected vertex properties {props}")
    n = int(elements[0].split()[2])

    # Records have a variable length (two lists of n_views entries): walk them once
    raw = np.frombuffer(body, dtype=np.uint8)
    offsets = np.empty(n, dtype=np.int64)
    off = 0
    b = bytes(body)
    for i in range(n):
        offsets[i] = off
        nv = b[off + _FIXED_BYTES]
        off += _FIXED_BYTES + 1 + 4 * nv
        if b[off] != nv:
            raise ValueError(f"point {i}: {nv} view indices but {b[off]} weights")
        off += 1 + 4 * nv
    if off != len(b):
        raise ValueError(f"body has {len(b)} bytes, records end at {off}")

    xyz = np.empty((n, 3), dtype=np.float32)
    for s in range(0, n, 500_000):
        o = offsets[s:s + 500_000]
        xyz[s:s + len(o)] = raw[o[:, None] + np.arange(12)].view(np.float32).reshape(-1, 3)

    # Views per point and the sum of their weights (OpenMVS confidence of the fused depth)
    n_views = raw[offsets + _FIXED_BYTES].astype(np.int64)
    weight_sum = np.zeros(n, dtype=np.float64)
    w_start = offsets + _FIXED_BYTES + 1 + 4 * n_views + 1
    for k in range(int(n_views.max()) if n else 0):
        has = n_views > k
        pos = w_start[has] + 4 * k
        weight_sum[has] += raw[pos[:, None] + np.arange(4)].view(np.float32).ravel()
    first_weight = weight_sum[n_views > 0] / np.maximum(n_views[n_views > 0], 1)
    return {"header": header, "body": b, "count": n, "xyz": xyz, "offsets": offsets,
            "n_views": n_views, "weight_sum": weight_sum,
            "median_weight": float(np.median(first_weight)) if len(first_weight) else 1.0}


def write_dense_ply(path: str, dense: Dict[str, Any], pts: np.ndarray, rgb: np.ndarray, normals: np.ndarray,
                    views: List[np.ndarray], weights: List[np.ndarray], keep: Optional[np.ndarray] = None) -> int:
    """
    Writes the MVS records (all, or those where `keep` is True) unchanged plus the new points (same layout),
    atomically. Returns the number of points written.
    """
    body = dense["body"]
    count = dense["count"]
    if keep is not None and not keep.all():
        rec_len = (_FIXED_BYTES + 2 + 8 * dense["n_views"]).astype(np.int64)
        body = np.frombuffer(body, dtype=np.uint8)[np.repeat(keep, rec_len)].tobytes()
        count = int(keep.sum())
    total = count + len(pts)
    header = [f"element vertex {total}" if ln.startswith("element vertex ") else ln for ln in dense["header"]]
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(("\n".join(header) + "\nend_header\n").encode("ascii"))
        f.write(body)
        for p, c, nrm, vi, vw in zip(pts, rgb, normals, views, weights):
            f.write(np.asarray(p, np.float32).tobytes())
            f.write(np.asarray(c, np.uint8).tobytes())
            f.write(np.asarray(nrm, np.float32).tobytes())
            f.write(bytes([len(vi)]) + np.asarray(vi, np.uint32).tobytes())
            f.write(bytes([len(vw)]) + np.asarray(vw, np.float32).tobytes())
    os.replace(tmp, path)
    return total


def depth_inside_hull(hull: Dict[str, Any], P: np.ndarray) -> np.ndarray:
    """Distance (m) of each point to the hull surface, measured inwards; 0 outside the hull or the grid."""
    from scipy.ndimage import distance_transform_edt

    occ = hull["occupancy"]
    edt = distance_transform_edt(occ) * hull["voxel_m"]
    ijk = np.rint((np.asarray(P, np.float64) - hull["origin"]) / hull["voxel_m"]).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < np.array(occ.shape)), axis=1)
    depth = np.zeros(len(P))
    depth[ok] = edt[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    # Centre-to-centre distance: a point in the boundary voxel is ~0.5 voxel inside
    return np.maximum(depth - 0.5 * hull["voxel_m"], 0.0)


# -----------------------------------------------------------------------------
# 3. Fusion: hull surface where MVS has nothing
# -----------------------------------------------------------------------------

def _zbuffer(cam: Dict[str, Any], P: np.ndarray, scale: int) -> np.ndarray:
    """Nearest depth per (scale x scale) pixel block of the hull surface points, seen from one undistorted view."""
    uv, z, ok = project_points(cam, P, distorted=False)
    hb, wb = cam["h"] // scale + 1, cam["w"] // scale + 1
    zb = np.full(hb * wb, np.inf)
    cell = (uv[ok, 1] // scale).astype(np.int64) * wb + (uv[ok, 0] // scale).astype(np.int64)
    np.minimum.at(zb, cell, z[ok])
    return zb.reshape(hb, wb)


def refine_dense_cloud_with_hull(
    dense_ply_path: str,
    hull: Dict[str, Any],
    cameras: List[Dict[str, Any]],
    image_paths: List[Optional[str]],
    mask_paths: List[Optional[str]],
    prune_depth_m: float = 0.005,
    prune_max_views: int = 3,
    fill: bool = True,
    support_dist_m: float = 0.003,
    min_y_m: float = 0.005,
    max_normal_y: float = 0.8,
    min_views: int = 3,
    max_bg_fraction: float = 0.5,
    max_views_per_point: int = 8,
    weight_scale: float = 0.5,
    brightness_white_threshold: float = 135.0,
    chroma_threshold: float = 20.0,
    min_facing: float = 0.3,
) -> Dict[str, Any]:
    """
    Corrects the dense cloud of DensifyPointCloud with the visual hull, before ReconstructMesh:
    1. Prune: points deeper than `prune_depth_m` inside the hull and fused from <= `prune_max_views` views are
       dropped. On uniform plastic PatchMatch converges to wrong depths inside the object (fixativ cap, 180-210 mm:
       61 % of the points >= 6 mm inside, 92 % of them from <= 3 views) and the mesh followed them (radius 8 mm
       instead of 25). A real concavity seen by enough views keeps its points.
    2. Fill (`fill`): hull surface points are appended where no point is left within `support_dist_m`, the surface
       is not near horizontal, >= `min_views` views see it, and in those views the pixel is not white background
       or outside the mask (rejects a membrane across a closed handle and white labels).
    `cameras` must be in the order of the OpenMVS scene images (load_sfm_cameras = bridge_sfm order): the
    appended view_indices index them. `image_paths` / `mask_paths`: the undistorted images and masks of the bridge.
    The original cloud is kept next to it as <name>_mvs.ply and is always the input, so a rerun never applies twice.
    Returns statistics (counts per reason).
    """
    from scipy.spatial import cKDTree

    t0 = time.time()
    ply = Path(dense_ply_path)
    original = ply.with_name(ply.stem + "_mvs.ply")
    source = original if original.is_file() else ply
    dense = read_dense_ply(str(source))
    xyz = dense["xyz"].astype(np.float64)
    stats: Dict[str, Any] = {"mvs_points": dense["count"], "added": 0}

    # 1. Prune wrong depths inside the object
    depth = depth_inside_hull(hull, xyz)
    prune = (depth > prune_depth_m) & (dense["n_views"] <= prune_max_views) if prune_depth_m > 0 else \
        np.zeros(dense["count"], bool)
    keep = ~prune
    stats["pruned_inside"] = int(prune.sum())
    stats["deep_inside_kept"] = int(((depth > prune_depth_m) & keep).sum())
    if prune.any():
        y = xyz[prune, 1] * 1000
        stats["pruned_y_mm_p10_p50_p90"] = [round(float(q), 1) for q in np.percentile(y, [10, 50, 90])]

    # 2. Fill where nothing is left
    P = np.zeros((0, 3))
    N = np.zeros((0, 3))
    views: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    rgb = np.zeros((0, 3), np.uint8)
    surf, normals = surface_points(hull) if fill else (np.zeros((0, 3)), np.zeros((0, 3)))
    stats["hull_surface_points"] = int(len(surf))
    if len(surf):
        tree = cKDTree(xyz[keep])
        dist, _ = tree.query(surf, k=1, distance_upper_bound=support_dist_m * 1.01)
        unsupported = ~np.isfinite(dist) | (dist > support_dist_m)
        flat_ok = np.abs(normals[:, 1]) <= max_normal_y
        low_ok = surf[:, 1] >= min_y_m
        cand = np.flatnonzero(unsupported & flat_ok & low_ok)
        stats.update(supported=int((~unsupported).sum()), rejected_horizontal=int((unsupported & ~flat_ok).sum()),
                     rejected_base=int((unsupported & flat_ok & ~low_ok).sum()), candidates=int(len(cand)))
        if len(cand):
            P, N, views, weights, rgb, reasons = _accept_hull_points(
                surf, normals, cand, hull["voxel_m"], cameras, image_paths, mask_paths, min_views, max_bg_fraction,
                max_views_per_point, float(dense["median_weight"]) * weight_scale, brightness_white_threshold,
                chroma_threshold, min_facing)
            stats.update(reasons)

    if prune.any() or len(P):
        if source == ply:
            os.replace(ply, original)
        stats["written_points"] = write_dense_ply(str(ply), dense, P.astype(np.float32), rgb, N.astype(np.float32),
                                                  views, weights, keep=keep)
    elif source == original:
        # A previous run had changed the cloud; this one changes nothing: put the plain MVS cloud back
        os.replace(original, ply)

    stats["added"] = int(len(P))
    if len(P):
        stats["added_y_mm"] = [round(float(P[:, 1].min()) * 1000, 1), round(float(P[:, 1].max()) * 1000, 1)]
    stats["seconds"] = round(time.time() - t0, 2)
    logger.info(
        f"Visual hull refinement of {ply.name}: {stats['pruned_inside']:,d} points pruned (> "
        f"{prune_depth_m * 1000:.0f} mm inside the hull, <= {prune_max_views} views), {stats['added']:,d} hull points "
        f"added ({stats.get('candidates', 0):,d} unsupported candidates, {stats.get('rejected_background', 0):,d} on "
        f"background); {stats['seconds']:.1f} s."
    )
    return stats


def _accept_hull_points(surf, normals, cand, voxel_m, cameras, image_paths, mask_paths, min_views, max_bg_fraction,
                        max_views_per_point, weight, brightness_white_threshold, chroma_threshold, min_facing):
    """Visibility (facing + hull z-buffer) and photometric test of the candidate hull points; see the caller."""
    P, N = surf[cand], normals[cand]
    n_views = len(cameras)
    facing = np.zeros((len(P), n_views), np.float32)       # cos(normal, direction to camera) when visible, else 0
    bg = np.zeros((len(P), n_views), bool)
    colour = np.zeros((len(P), 3), np.float64)
    colour_n = np.zeros(len(P), np.float64)
    scale = 4
    tol = 2.5 * voxel_m
    for vi, cam in enumerate(cameras):
        img_path = image_paths[vi] if vi < len(image_paths) else None
        if not img_path or not os.path.isfile(img_path):
            continue
        to_cam = cam["C"] - P
        dist_c = np.linalg.norm(to_cam, axis=1)
        cos = np.einsum("ij,ij->i", N, to_cam) / np.maximum(dist_c, 1e-9)
        uv, z, ok = project_points(cam, P, distorted=False)
        ok &= cos > min_facing
        if not np.any(ok):
            continue
        zb = _zbuffer(cam, surf, scale)
        cell_v = (uv[ok, 1] // scale).astype(np.int64)
        cell_u = (uv[ok, 0] // scale).astype(np.int64)
        visible = z[ok] <= zb[cell_v, cell_u] + tol
        sel = np.flatnonzero(ok)[visible]
        if len(sel) == 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        ui = np.clip(np.rint(uv[sel, 0]).astype(np.int64), 0, img.shape[1] - 1)
        vv = np.clip(np.rint(uv[sel, 1]).astype(np.int64), 0, img.shape[0] - 1)
        px = img[vv, ui].astype(np.float64)                    # BGR
        bright = px.mean(axis=1)
        chroma = px.max(axis=1) - px.min(axis=1)
        is_bg = (bright > brightness_white_threshold) & (chroma < chroma_threshold)
        mpath = mask_paths[vi] if vi < len(mask_paths) else None
        if mpath and os.path.isfile(mpath):
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape == img.shape[:2]:
                is_bg |= m[vv, ui] < 128
        facing[sel, vi] = cos[sel]
        bg[sel, vi] = is_bg
        obj = sel[~is_bg]
        colour[obj] += px[~is_bg][:, ::-1]
        colour_n[obj] += 1

    n_vis = (facing > 0).sum(axis=1)
    bg_frac = bg.sum(axis=1) / np.maximum(n_vis, 1)
    few_views = n_vis < min_views
    background = ~few_views & (bg_frac > max_bg_fraction)
    acc = np.flatnonzero(~few_views & ~background)
    views, weights = [], []
    for i in acc:
        order = np.argsort(-facing[i])
        order = order[facing[i, order] > 0]
        obj_views = order[~bg[i, order]]
        order = (obj_views if len(obj_views) >= 2 else order)[:max_views_per_point]
        views.append(order.astype(np.uint32))
        weights.append(np.full(len(order), weight, np.float32))
    rgb = np.where(colour_n[acc, None] > 0, colour[acc] / np.maximum(colour_n[acc, None], 1), 128).astype(np.uint8)
    reasons = {"rejected_few_views": int(few_views.sum()), "rejected_background": int(background.sum()),
               "view_weight": round(weight, 4)}
    return P[acc], N[acc], views, weights, rgb, reasons


def restore_dense_cloud(dense_ply_path: str) -> bool:
    """Puts the plain MVS cloud back (used when ReconstructMesh fails on the fused one)."""
    ply = Path(dense_ply_path)
    original = ply.with_name(ply.stem + "_mvs.ply")
    if original.is_file():
        os.replace(original, ply)
        return True
    return False


# -----------------------------------------------------------------------------
# 4. File naming of the bridge (undistorted/<viewId>_<name>, masks/<viewId>_<stem>.mask.png)
# -----------------------------------------------------------------------------

def bridge_image_paths(cameras: List[Dict[str, Any]], undistorted_dir: str) -> Tuple[List[str], List[str]]:
    """Undistorted image and mask of each camera, as written by bridge_sfm."""
    images, masks = [], []
    for c in cameras:
        name = f"{c['view_id']}_{Path(c['path']).name}"
        images.append(str(Path(undistorted_dir) / name))
        masks.append(str(Path(undistorted_dir) / f"{Path(name).stem}.mask.png"))
    return images, masks


def source_mask_paths(cameras: List[Dict[str, Any]], masks_dir: str) -> List[Optional[str]]:
    """Masks of the original (distorted) photos written by mask_processor: <viewId>_<stem>.mask.png."""
    out = []
    for c in cameras:
        p = Path(masks_dir) / f"{c['view_id']}_{Path(c['path']).stem}.mask.png"
        out.append(str(p) if p.is_file() else None)
    return out
