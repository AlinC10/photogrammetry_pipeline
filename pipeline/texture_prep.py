"""
texture_prep.py — Builds the images texrecon textures from, after the mesh has been cleaned.

For every view:
  1. Undistort the ORIGINAL photo with the same K -> K_new mapping as bridge_sfm (so the .cam
     files written for texrecon stay valid).
  2. Per-camera colour normalisation: a per-channel gain measured on the white paper of the
     turntable, which both cameras see in every photo. The rig only copies exposure from
     camera 0 to camera 1 (copy_colour_gains is off by default), so each camera keeps its
     own AWB and the charts of the two cameras show up as colour patches.
  3. Mask = silhouette of the cleaned mesh rendered with the locked pose (mesh_masks.py),
     dilated by a small margin. Light-grey / white parts of the object are kept, unlike the
     photometric mask. White background that a few pixels of pose error pull inside the
     silhouette is trimmed, but only in a narrow band connected to the outside, so no hole
     can appear inside the object.
  4. Background = exactly RGB(0, 0, 0), object pixels never 0, written as lossless PNG.
     texrecon only invalidates pure-black pixels connected to the image corners
     (TextureView::generate_validity_mask); JPEG ringing around the silhouette used to break
     that flood fill, so the black halo was sampled as texture.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .mesh_masks import adjust_mask_margin, load_sfm_cameras, render_silhouette

logger = logging.getLogger("TurntableStudio.TexturePrep")


def texrecon_validity_mask(image_bgr: np.ndarray, erode: bool = True) -> np.ndarray:
    """
    Re-implementation of mvs-texturing TextureView::generate_validity_mask (+ erode_validity_mask,
    used with the GMI data term): pixels whose R+G+B == 0 and that are 4-connected to one of the
    four image corners are invalid. Returns a boolean map of VALID pixels.
    """
    zero = image_bgr.reshape(image_bgr.shape[0], image_bgr.shape[1], -1).astype(np.int32).sum(axis=2) == 0
    h, w = zero.shape
    # label 0 = non-black pixels; black components get labels >= 1
    _, labels = cv2.connectedComponents(zero.astype(np.uint8), connectivity=4)
    corner_labels = {int(labels[0, 0]), int(labels[h - 1, 0]), int(labels[0, w - 1]), int(labels[h - 1, w - 1])} - {0}
    invalid = np.isin(labels, sorted(corner_labels))
    if erode:
        invalid = cv2.dilate(invalid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        invalid[0, :] = invalid[-1, :] = invalid[:, 0] = invalid[:, -1] = True
    return ~invalid


def trim_background_band(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    band_px: int,
    brightness_threshold: float,
    chroma_threshold: float,
) -> np.ndarray:
    """
    Removes from `mask` the white/neutral background pixels (booth wall, turntable paper) that lie
    within `band_px` of the mask border AND are connected to the outside through such pixels.
    Interior light parts of the object are never touched, so the result has no new holes.
    """
    if band_px <= 0:
        return mask
    inside = mask > 0
    f = image_bgr.astype(np.int16)
    brightness = f.sum(axis=2) / 3.0
    chroma = f.max(axis=2) - f.min(axis=2)
    white = (brightness > brightness_threshold) & (chroma < chroma_threshold)
    dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    candidate = (white & inside & (dist <= band_px)) | ~inside
    n, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
    outside_labels = np.unique(labels[~inside])
    trim = np.isin(labels, outside_labels) & inside
    out = mask.copy()
    out[trim] = 0
    return out


def disc_paper_points(
    tags: List[Dict[str, Any]],
    r_min_m: float = 0.045,
    r_max_m: float = 0.125,
    tag_clearance_m: float = 0.030,
    count: int = 6000,
    seed: int = 3,
) -> np.ndarray:
    """Random points on the white paper of the turntable (y = 0), away from the AprilTags."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(r_min_m ** 2, r_max_m ** 2, count))
    th = rng.uniform(0.0, 2.0 * np.pi, count)
    P = np.stack([r * np.cos(th), np.zeros_like(r), r * np.sin(th)], axis=1)
    keep = np.ones(len(P), dtype=bool)
    for t in tags:
        keep &= np.linalg.norm(P - np.asarray(t["center"], dtype=np.float64), axis=1) > tag_clearance_m
    return P[keep]


def sample_paper_colour(
    image_bgr: np.ndarray, cam: Dict[str, Any], paper: np.ndarray, object_hull: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Median BGR of the turntable paper in one ORIGINAL (distorted) photo, object area excluded."""
    R, C = cam["R"], cam["C"]
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ C
    Xc = (paper - C) @ R.T
    pts = paper[Xc[:, 2] > 0.05]
    if len(pts) == 0:
        return None
    uv, _ = cv2.projectPoints(pts, rvec, tvec, cam["K"], cam["dist"])
    uv = uv.reshape(-1, 2)
    h, w = image_bgr.shape[:2]
    ok = (uv[:, 0] >= 2) & (uv[:, 0] < w - 3) & (uv[:, 1] >= 2) & (uv[:, 1] < h - 3)
    if object_hull is not None and len(object_hull) >= 3:
        hull = object_hull.astype(np.float32).reshape(-1, 1, 2)
        ok &= np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) < -15.0 for x, y in uv])
    uv = uv[ok]
    if len(uv) < 50:
        return None
    x = np.round(uv[:, 0]).astype(int)
    y = np.round(uv[:, 1]).astype(int)
    col = image_bgr[y, x].astype(np.float64)
    bright = col.mean(axis=1)
    good = (bright > 60) & (col.max(axis=1) < 250)
    if good.sum() < 50:
        return None
    return np.median(col[good], axis=0)


def compute_camera_gains(
    paper_by_camera: Dict[str, List[np.ndarray]], max_deviation: float = 0.25
) -> Dict[str, np.ndarray]:
    """
    Per-camera BGR gains mapping each camera's paper colour onto the same neutral grey
    (the mean paper brightness over all cameras): fixes white balance and exposure differences.
    """
    medians = {cid: np.median(np.stack(v), axis=0) for cid, v in paper_by_camera.items() if v}
    if not medians:
        return {}
    level = float(np.mean([m.mean() for m in medians.values()]))
    gains = {}
    for cid, m in medians.items():
        g = level / np.maximum(m, 1.0)
        gains[cid] = np.clip(g, 1.0 - max_deviation, 1.0 + max_deviation)
    return gains


def _object_hull(vertices: np.ndarray, cam: Dict[str, Any], step: int = 50) -> Optional[np.ndarray]:
    """Convex hull of the mesh projected into the distorted photo (for excluding paper behind the object)."""
    sub = vertices[::step]
    R, C = cam["R"], cam["C"]
    rvec, _ = cv2.Rodrigues(R)
    Xc = (sub - C) @ R.T
    sub = sub[Xc[:, 2] > 0.05]
    if len(sub) < 3:
        return None
    uv, _ = cv2.projectPoints(sub, rvec, -R @ C, cam["K"], cam["dist"])
    return cv2.convexHull(uv.reshape(-1, 2).astype(np.float32)).reshape(-1, 2)


def prepare_texturing_images(
    sfm_json_path: str,
    mesh_path: str,
    output_dir: str,
    margin_px: int = 2,
    trim_band_px: int = 25,
    brightness_threshold: float = 135.0,
    chroma_threshold: float = 20.0,
    colour_normalize: bool = True,
    max_gain_deviation: float = 0.25,
    tag_layout_path: Optional[str] = None,
    workers: int = 4,
) -> Dict[str, Any]:
    """
    Writes `<viewId>_<stem>.png` (texturing image, background exactly 0) and
    `<viewId>_<stem>.mask.png` for every registered view into `output_dir`.
    Returns a summary with the per-camera gains, the image list and per-view mask statistics.
    """
    import trimesh

    from .mask_processor import load_tag_layout

    out = Path(output_dir).resolve()
    if out.exists():
        for old in list(out.glob("*.png")) + list(out.glob("*.jpg")):
            old.unlink()
    out.mkdir(parents=True, exist_ok=True)

    cams = load_sfm_cameras(sfm_json_path)
    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    # Undistortion maps, one per intrinsic (identical to cv2.undistort used by bridge_sfm)
    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cam in cams:
        if cam["intrinsic_id"] not in maps:
            maps[cam["intrinsic_id"]] = cv2.initUndistortRectifyMap(
                cam["K"], cam["dist"], None, cam["K_new"], (cam["w"], cam["h"]), cv2.CV_32FC1)

    # 1. Per-camera colour gains from the turntable paper
    gains: Dict[str, np.ndarray] = {}
    if colour_normalize:
        paper = disc_paper_points(load_tag_layout(tag_layout_path))
        by_cam: Dict[str, List[np.ndarray]] = {}

        def measure(cam):
            img = cv2.imread(cam["path"], cv2.IMREAD_COLOR)
            if img is None:
                return cam["intrinsic_id"], None
            return cam["intrinsic_id"], sample_paper_colour(img, cam, paper, _object_hull(V, cam))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for cid, med in ex.map(measure, cams):
                if med is not None:
                    by_cam.setdefault(cid, []).append(med)
        if len(by_cam) >= 2:
            gains = compute_camera_gains(by_cam, max_deviation=max_gain_deviation)
            for cid in sorted(gains):
                m = np.median(np.stack(by_cam[cid]), axis=0)
                logger.info(
                    f"Camera {cid}: paper BGR {m.round(1).tolist()} from {len(by_cam[cid])} photos "
                    f"-> gains B/G/R {gains[cid].round(3).tolist()}"
                )
        else:
            logger.info("Colour normalisation skipped (paper visible for fewer than two cameras).")

    # 2. Texturing images
    def process(cam):
        img = cv2.imread(cam["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read {cam['path']}")
        mx, my = maps[cam["intrinsic_id"]]
        und = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        g = gains.get(cam["intrinsic_id"])
        if g is not None:
            # per-channel gain, saturated to uint8
            und = cv2.transform(und, np.diag(g).astype(np.float32))

        sil = render_silhouette(V, F, cam["R"], cam["C"], cam["K_new"], cam["w"], cam["h"])
        mask = adjust_mask_margin(sil, margin_px)
        mask = trim_background_band(und, mask, trim_band_px, brightness_threshold, chroma_threshold)

        fg = mask > 0
        und[~fg] = 0
        zero_fg = fg & (und.reshape(-1, 3).max(axis=1).reshape(fg.shape) == 0)
        und[zero_fg] = 1     # only the background may be exactly black

        stem = f"{cam['view_id']}_{Path(cam['path']).stem}"
        img_path = out / f"{stem}.png"
        cv2.imwrite(str(img_path), und, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(out / f"{stem}.mask.png"), mask)
        return {
            "view_id": cam["view_id"], "image": str(img_path),
            "silhouette_px": int((sil > 0).sum()), "mask_px": int(fg.sum()),
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        views = list(ex.map(process, cams))
    logger.info(f"Wrote {len(views)} lossless texturing images with mesh masks to {out}")

    summary = {
        "images_dir": str(out),
        "views": views,
        "margin_px": margin_px,
        "trim_band_px": trim_band_px,
        "camera_gains_bgr": {cid: g.round(4).tolist() for cid, g in gains.items()},
    }
    with open(out / "texturing_images.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
