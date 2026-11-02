"""
mask_fusion.py — Fused foreground masks (masking.method "fused"): the AI model on a crop around the object, checked
against a background model of the scan.

Experiment of 01.11.2026 on 70 views of 7 scans (plans/CUNOSTINTE_SI_MASURATORI.md §5.6): mean IoU against the
reprojected final mesh 0.32 (AI on the whole photo) -> 0.83, no view missed (15 / 70 before), the turntable disc
mostly removed from the masks (5-18 % of the mask instead of 35-70 %), ~2x faster than the AI on the whole photo.

Per view, all of it at 1/4 resolution in the frame of the original (distorted) photo:
1. Background model:
   - "cabin" pixels (the ray misses the disc): difference to the per-(camera, level) median of the scan, only where
     that median is clearly the booth (bright, colourless). Where the object always stands the median IS the object,
     so there the AI decides; this also holds for a white object, whose median looks like the booth: pixels the
     model marks as object in most shots of the (camera, level) group are left to it as well;
   - "disc" pixels (the ray lands on the turntable top): difference to the disc rendered from the tag layout (white
     paper + the tag16h5 markers at their layout positions x tag_scale), with a +-3 px tolerance for registration
     errors and the shadow rule (same hue, darker = still the disc).
2. AI (BiRefNet_lite) on a square crop around the object (rough box from the background model and the photometric
   test, +30 %): the object gets ~2-5x more pixels at the model input, the booth rails and most of the disc leave it.
3. Fusion: AI "object" accepted where the background model sees a difference nearby (or cannot judge), strong
   background differences accepted on their own, masking cylinder and tag blackout as before, small specks dropped.
The per-view result is upsampled to the photo size; the tag blackout is applied again at full resolution.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mask_processor import (ai_foreground_probability, project_cylinder_to_2d, project_tags_to_2d,
                             tag_may_be_occluded, write_mask_files)

logger = logging.getLogger("TurntableStudio.MaskFusion")

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is a runtime dependency
    cv2 = None

SCALE = 4                      # working resolution: 1/4 of the photo (1014 x 760 for the IMX477)
DISC_TEX_RES_M = 0.00025       # rendered disc: 0.25 mm per texel
DISC_TEX_HALF_M = 0.135        # texture covers x, z in [-135, 135] mm


# -----------------------------------------------------------------------------
# 1. Geometry: disc rendering and camera rays
# -----------------------------------------------------------------------------

def disc_texture(tags: Sequence[Dict[str, Any]], tag_scale: float, disc_radius_m: float,
                 res_m: float = DISC_TEX_RES_M, half_m: float = DISC_TEX_HALF_M) -> np.ndarray:
    """Top view of the turntable (uint8, row = z, column = x): white disc with the tag16h5 markers of the layout."""
    n = int(round(2 * half_m / res_m))
    tex = np.zeros((n, n), np.uint8)
    cv2.circle(tex, (n // 2, n // 2), int(disc_radius_m / res_m), 255, -1)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    for t in tags:
        corners = t.get("corners") or []
        if len(corners) != 4:
            continue
        marker = cv2.aruco.generateImageMarker(dic, int(t["id"]), 120, borderBits=1)
        src = np.float32([[0, 0], [119, 0], [119, 119], [0, 119]])
        dst = np.float32([[(c[0] * tag_scale + half_m) / res_m, (c[2] * tag_scale + half_m) / res_m] for c in corners])
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(marker, H, (n, n), flags=cv2.INTER_LINEAR, borderValue=255)
        inside = cv2.warpPerspective(np.full_like(marker, 255), H, (n, n), flags=cv2.INTER_NEAREST)
        tex[inside > 0] = warped[inside > 0]
    return tex


def pixel_rays(K: np.ndarray, dist: np.ndarray, wq: int, hq: int, scale: int = SCALE) -> np.ndarray:
    """Normalized undistorted camera coordinates (hq, wq, 2) of the centres of the 1/scale pixels."""
    gx, gy = np.meshgrid((np.arange(wq) + 0.5) * scale - 0.5, (np.arange(hq) + 0.5) * scale - 0.5)
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist).reshape(hq, wq, 2).astype(np.float32)


def disc_zone_and_render(rays: np.ndarray, R: np.ndarray, C: np.ndarray, tex: np.ndarray, disc_radius_m: float,
                         res_m: float = DISC_TEX_RES_M, half_m: float = DISC_TEX_HALF_M,
                         near_margin_m: float = 0.012) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per pixel: does the ray land on the disc top (2 mm inside the rim), near it (`near_margin_m` past the rim: rim,
    edge shadow) and what the disc looks like there (float32 0-255, 0 off the disc).
    """
    hq, wq = rays.shape[:2]
    d_cam = np.concatenate([rays, np.ones((hq, wq, 1), np.float32)], axis=2).reshape(-1, 3).astype(np.float64)
    d = d_cam @ np.asarray(R, np.float64)                  # world directions (R is world -> camera)
    C = np.asarray(C, np.float64).ravel()
    down = d[:, 1] < -1e-9
    t = np.where(down, -C[1] / np.where(down, d[:, 1], -1.0), -1.0)
    x = C[0] + t * d[:, 0]
    z = C[2] + t * d[:, 2]
    r = np.hypot(x, z)
    hits = down & (t > 0)
    zone = (hits & (r <= disc_radius_m - 0.002)).reshape(hq, wq)
    near = (hits & (r <= disc_radius_m + near_margin_m)).reshape(hq, wq)
    mx = np.where(hits, (x + half_m) / res_m, -10.0).reshape(hq, wq).astype(np.float32)
    my = np.where(hits, (z + half_m) / res_m, -10.0).reshape(hq, wq).astype(np.float32)
    D = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
    return zone, near, D


# -----------------------------------------------------------------------------
# 2. Background model and fusion (pure numpy / OpenCV, unit tested)
# -----------------------------------------------------------------------------

def background_model(I: np.ndarray, M: np.ndarray, disc_zone: np.ndarray, D: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    I: the view, M: the median of its (camera, level) group (both BGR or RGB uint8, 1/4 resolution), D: rendered disc.
    Returns (object, strong object, unreliable) boolean maps. The thresholds are those of the 01.11 experiment.
    """
    I = I.astype(np.float32)
    M = M.astype(np.float32)
    gray = I.mean(2)
    chroma = I.max(2) - I.min(2)
    white = disc_zone & (D > 200)
    ref = float(np.percentile(gray[white], 90)) if white.sum() > 100 else 200.0
    r = gray / max(ref, 1.0)
    # On the disc: object = coloured, much darker than the paper / tag there, or much brighter than a black tag.
    # The pixel must leave the range of the darkest and brightest disc value within +-3 px (registration errors);
    # grey at 0.3-1.2 of the disc brightness is the disc or its shadow
    k7 = np.ones((7, 7), np.uint8)
    d_min = 0.33 + 0.67 * cv2.erode(D, k7) / 255.0
    d_max = 0.33 + 0.67 * cv2.dilate(D, k7) / 255.0
    obj_disc = (chroma > 45) | (r < 0.30 * d_min) | (r > 1.2 * d_max + 0.05)
    # Booth: difference to the median where the median is clearly the booth
    med_ok = (M.mean(2) > 110) & ((M.max(2) - M.min(2)) < 25)
    diff = cv2.GaussianBlur(np.abs(I - M).max(2), (5, 5), 0)
    obj_cab = med_ok & (diff > 28)
    obj = np.where(disc_zone, obj_disc, obj_cab)
    strong = np.where(disc_zone, obj_disc & ((chroma > 45) | (r < 0.2)), med_ok & (diff > 55))
    unreliable = (~disc_zone) & (~med_ok)
    return obj, strong, unreliable


def drop_small_components(mask: np.ndarray, min_area: int, rel_to_largest: float = 0.0) -> np.ndarray:
    """Connected components (8-neighbourhood) smaller than min_area (or than rel_to_largest x the largest) removed."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= max(int(min_area), int(rel_to_largest * areas.max()))
    return keep[lab]


def fuse_masks(ai: np.ndarray, obj: np.ndarray, strong: np.ndarray, unreliable: np.ndarray, roi: np.ndarray,
               disc_zone: Optional[np.ndarray] = None, disc_tolerance_px: int = int(os.environ.get("TT_DISC_TOL", "3"))
               ) -> np.ndarray:
    """
    Fusion F of the 01.11 experiment (all boolean, same size): the model's "object" is accepted within 3 px of a
    background difference (edges the background model misses), on the disc within `disc_tolerance_px`.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near_obj = cv2.dilate((obj & roi).astype(np.uint8), k) > 0
    if disc_zone is not None and disc_tolerance_px < 3:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * disc_tolerance_px + 1, 2 * disc_tolerance_px + 1))
        near_obj = np.where(disc_zone, cv2.dilate((obj & roi).astype(np.uint8), kd) > 0, near_obj)
    fused = roi & ((ai & (near_obj | unreliable)) | (strong & roi))
    fused = drop_small_components(fused, max(30, int(0.002 * roi.sum())))
    return cv2.morphologyEx(fused.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0


def keep_touching(mask: np.ndarray, region: np.ndarray) -> np.ndarray:
    """
    Components of `mask` that touch `region` (the object stands on the disc, so its silhouette reaches the pixels
    over the disc; static booth parts such as the grey rails do not). The whole mask when none touches it.
    """
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    hit = np.unique(lab[region & mask])
    hit = hit[hit > 0]
    if len(hit) == 0:
        return mask.astype(bool)
    keep = np.zeros(n, bool)
    keep[hit] = True
    return keep[lab]


def crop_box(rough: np.ndarray, w: int, h: int, scale: int = SCALE, grow: float = 1.3, pad_px: int = 64
             ) -> Tuple[int, int, int, int]:
    """Square box (x0, y0, x1, y1, full-resolution pixels) around the rough object mask (1/scale), +30 % and a pad."""
    ys, xs = np.nonzero(rough)
    if len(xs) < 20:
        return 0, 0, w, h
    x0, x1 = (xs.min() + 0.0) * scale, (xs.max() + 1.0) * scale
    y0, y1 = (ys.min() + 0.0) * scale, (ys.max() + 1.0) * scale
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * grow + pad_px
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
            int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))


def _poly_mask(polys: Sequence[Sequence[Tuple[float, float]]], w: int, h: int, scale: float = 1.0) -> np.ndarray:
    out = np.zeros((h, w), np.uint8)
    for p in polys:
        if len(p) >= 3:
            cv2.fillPoly(out, [np.int32(np.round(np.asarray(p, np.float64) / scale))], 255)
    return out > 0


# -----------------------------------------------------------------------------
# 3. Batch
# -----------------------------------------------------------------------------

def read_quarter(path: str) -> Optional[np.ndarray]:
    """BGR image at 1/4 resolution (the JPEG decoder scales the DCT: ~4x faster than a full decode)."""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    return img


def median_image(imgs: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Per-pixel median (uint8) of the decoded images of one group (those of the first one's size)."""
    stack = [im for im in imgs if im is not None]
    if not stack:
        return None
    shape = stack[0].shape
    arr = np.stack([s for s in stack if s.shape == shape])
    med = np.empty(shape, np.uint8)
    for r0 in range(0, shape[0], 128):                 # row chunks: np.median works in float64
        med[r0:r0 + 128] = np.median(arr[:, r0:r0 + 128], axis=0).astype(np.uint8)
    return med


def group_medians(paths: Sequence[str], groups: Sequence[Any], out_dir: Path, threads: int = 4,
                  cache_dir: Optional[str] = None) -> Dict[Any, str]:
    """
    Median image (1/4 resolution) of every group of views, saved as .npy; {group: path}. `cache_dir`
    (stream_cache): a median already computed from exactly the same photos is used as it is.
    """
    from . import stream_cache

    out_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(set(groups), key=str)
    members = {g: [paths[i] for i in range(len(paths)) if groups[i] == g] for g in order}
    res: Dict[Any, str] = {}
    todo = []
    for g in order:
        hit = stream_cache.median_path(cache_dir, members[g]) if cache_dir else None
        if hit is not None:
            res[g] = str(hit)
        else:
            todo.append(g)
    if todo:
        need = sorted({p for g in todo for p in members[g]})
        with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:     # decoding releases the GIL
            decoded = dict(zip(need, ex.map(read_quarter, need)))
        for k, g in enumerate(todo):
            med = median_image([decoded[p] for p in members[g]])
            if med is None:
                continue
            pth = out_dir / f"median_{order.index(g)}.npy"
            np.save(str(pth), med)
            res[g] = str(pth)
    return res


_RAYS: Dict[Tuple[str, int, int], np.ndarray] = {}


def _ai_on_crop(task: Dict[str, Any], cam: Dict[str, Any], I: np.ndarray, full_img: List[Optional[np.ndarray]],
                box: Tuple[int, int, int, int], roi: np.ndarray, zone: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    The model on one crop (full-resolution box), its mask at 1/4 resolution in the frame of the photo, and whether
    its part off the disc touches an inner side of the crop (the object goes on beyond it; the disc itself, which
    the model keeps, always leaves the crop and does not count).
    """
    hq, wq = I.shape[:2]
    x0, y0, x1, y1 = box
    q0, q1, p0, p1 = x0 // SCALE, -(-x1 // SCALE), y0 // SCALE, -(-y1 // SCALE)
    # Keyed on the photo, not the view id (that depends on how many photos the scan has: the streaming pre-work
    # runs on one level before the others exist)
    cache = (Path(task["ai_cache_dir"]) / f"{Path(cam['path']).parent.name}_{Path(cam['path']).stem}_{x0}_{y0}_{x1}_{y1}.npy"
             if task.get("ai_cache_dir") else None)
    if cache is not None and cache.is_file():
        task["_ai_hits"] = task.get("_ai_hits", 0) + 1
    if min(q1 - q0, p1 - p0) >= task["ai_resolution"]:
        crop = cv2.cvtColor(np.ascontiguousarray(I[p0:p1, q0:q1]), cv2.COLOR_BGR2RGB)
    else:
        if full_img[0] is None:
            full_img[0] = cv2.imread(cam["path"], cv2.IMREAD_COLOR)
        crop = cv2.cvtColor(np.ascontiguousarray(full_img[0][y0:y1, x0:x1]), cv2.COLOR_BGR2RGB)
    prob = ai_foreground_probability(crop, task["ai_model"], task["ai_resolution"], task["ai_threads"],
                                     cache_path=cache)
    prob_q = cv2.resize(prob, (q1 - q0, p1 - p0), interpolation=cv2.INTER_AREA)
    full_q = np.zeros((hq, wq), np.float32)
    full_q[p0:p1, q0:q1] = prob_q[:hq - p0, :wq - q0]
    ai = drop_small_components((full_q >= 0.5) & roi, 20, 0.2)
    off = ai & ~zone
    sides = []
    pe, qe = min(p1, hq) - 1, min(q1, wq) - 1
    if p0 > 0:
        sides.append(off[p0, q0:qe + 1])
    if p1 < hq:
        sides.append(off[pe, q0:qe + 1])
    if q0 > 0:
        sides.append(off[p0:pe + 1, q0])
    if q1 < wq:
        sides.append(off[p0:pe + 1, qe])
    touches = any(int(sd.sum()) >= 3 for sd in sides)
    return ai, touches


def _phase1(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Worker: background model, crop box and AI on the crop for one view; the 1/4 resolution maps are saved to
    task["npz"] for the fusion. None when the photo is missing.
    """
    cam = task["cam"]
    t0 = time.time()
    I = read_quarter(cam["path"])
    if I is None:
        return None
    hq, wq = I.shape[:2]
    w, h = int(cam["w"]), int(cam["h"])
    key = (cam["intrinsic_id"], wq, hq)
    if key not in _RAYS:
        _RAYS[key] = pixel_rays(cam["K"], cam["dist"], wq, hq)
    rays = _RAYS[key]
    tex = np.load(task["disc_tex"])
    zone, near, D = disc_zone_and_render(rays, cam["R"], cam["C"], tex, task["disc_radius_m"])
    M = np.load(task["median"]) if task.get("median") else I
    if M.shape != I.shape:
        M = cv2.resize(M, (wq, hq), interpolation=cv2.INTER_AREA)
    obj, strong, unreliable = background_model(I, M, zone, D)

    # Masking cylinder and the tags the object cannot cover (as mask_processor)
    hull_pts = project_cylinder_to_2d(cam["R"], cam["C"], cam["K"], task["cylinder_radius_m"],
                                      task["cylinder_y_min_m"], task["cylinder_y_max_m"])
    roi = _poly_mask([hull_pts], wq, hq, SCALE) if len(hull_pts) >= 3 else np.ones((hq, wq), bool)
    tags = [t for t in task["tags"] if not tag_may_be_occluded(
        t, cam["C"], task["cylinder_radius_m"], task["cylinder_y_min_m"], task["cylinder_y_max_m"])]
    tag_polys = project_tags_to_2d(tags, cam["R"], cam["C"], cam["K"], task["tag_expansion_factor"]) if tags else []
    tagq = _poly_mask(tag_polys, wq, hq, SCALE)
    roi &= ~tagq

    # Rough object for the crop box: background model + photometric test, inside the cylinder
    If = I.astype(np.float32)
    photometric = ~((If.mean(2) > task["brightness_white_threshold"])
                    & ((If.max(2) - If.min(2)) < task["chroma_threshold"]))
    # Seeded by the background difference (the static grey rails of the booth are not in it), grown by the
    # photometric test. Across the dark band around the disc rim the growth can reach the rails (phone lying near
    # the rim: a crop of the whole photo, model missed the phone), so it is used only when its box stays within
    # 2.5x the area of the box grown without that band (a long object reaching over the rim: remote control)
    seed = drop_small_components(obj & roi, 20, 0.2)
    seed_d = cv2.dilate(seed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 if seed.any() else zone
    rough_s = keep_touching(drop_small_components((obj | (photometric & ~(near & ~zone))) & roi, 20, 0.2), seed_d)
    rough_l = keep_touching(drop_small_components((obj | photometric) & roi, 20, 0.2), seed_d)
    box_s, box_l = crop_box(rough_s, w, h), crop_box(rough_l, w, h)
    area = lambda b: max(1, (b[2] - b[0]) * (b[3] - b[1]))
    rough, box = (rough_l, box_l) if area(box_l) <= 2.5 * area(box_s) else (rough_s, box_s)
    t1 = time.time()

    ai = np.zeros((hq, wq), bool)
    ai_note = None
    retried = 0
    if task.get("ai_model"):
        try:
            full_img: List[Optional[np.ndarray]] = [None]
            ai, touches = _ai_on_crop(task, cam, I, full_img, box, roi, zone)
            # The object leaves the crop (a tall or long object whose middle the median hides: the seed of the box
            # saw only part of it): again on a larger box around the model's mask, until it fits (at most 3 times)
            while touches and retried < 3:
                grown = crop_box(ai | rough, w, h, grow=1.6)
                if grown == box:
                    break
                box = grown
                ai, touches = _ai_on_crop(task, cam, I, full_img, box, roi, zone)
                retried += 1
        except Exception as e:  # one bad view must not stop the scan
            ai_note = f"ai error: {str(e)[:120]}"
    # Pixels whose ray lands on (or next to) the black print of a tag: a dark object there looks like the tag
    tag_ink = zone & (cv2.erode(D, np.ones((7, 7), np.uint8)) < 128)
    np.savez_compressed(task["npz"], ai=ai, obj=obj, strong=strong, unreliable=unreliable, roi=roi, near=near,
                        zone=zone, rough=rough, tag_ink=tag_ink)
    return {"index": task["index"], "box": box, "ai_note": ai_note, "retried": retried,
            "ai_cache_hits": task.get("_ai_hits", 0),
            "t_model_s": round(time.time() - t1, 3),
            "t_s": round(time.time() - t0, 3)}


def _phase2(task: Dict[str, Any], freq: Optional[np.ndarray], out_dir: str, min_ai_recall: float
            ) -> Tuple[str, str, str, str]:
    """Fusion of one view, upsampled to the photo and written; returns (file name, view id, mask path, method)."""
    cam = task["cam"]
    d = np.load(task["npz"])
    ai, obj, strong, unreliable, roi, near = (d[k] for k in ("ai", "obj", "strong", "unreliable", "roi", "near"))
    method = "fused"
    if freq is not None and freq.shape == ai.shape:
        # Where the model sees the object in most shots of this camera and level, the median is the object (a
        # white one looks like the booth): the model decides there, as where the median is not the booth
        unreliable = unreliable | ((freq >= 0.5) & ~near)
    fused = fuse_masks(ai, obj, strong, unreliable, roi, d["zone"])
    if "tag_ink" in d.files and fused.any():
        # A dark object in front of a black tag differs from neither the disc rendering nor the paper: the fusion
        # bit a notch out of it (lighter 153311). The model's pixels on tag ink inside the closed mask go back in.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        closed = cv2.morphologyEx(fused.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        fused |= ai & d["tag_ink"] & closed
    cabin_ai = ai & ~near
    if cabin_ai.sum() > 200 and (fused & cabin_ai).sum() < min_ai_recall * cabin_ai.sum():
        # The background model removes most of what the model sees above the disc: trust the model there (a
        # larger mask cannot cut the visual hull; a smaller one can)
        fused |= cabin_ai
        method = "fused (ai kept above the disc)"
    if fused.sum() < 50:
        fused = ai if ai.sum() >= 50 else d["rough"]
        method = "ai crop (fusion empty)" if ai.sum() >= 50 else "photometric (fusion empty)"
    w, h = int(cam["w"]), int(cam["h"])
    full = cv2.resize(fused.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.where(full > 127, 255, 0).astype(np.uint8)
    if task["tag_polys"]:
        mask[_poly_mask(task["tag_polys"], w, h)] = 0
    dst = write_mask_files(mask, Path(cam["path"]), str(cam["view_id"]), Path(out_dir))
    return Path(cam["path"]).name, str(cam["view_id"]), dst, method


def batch_fused_masks(
    cameras: List[Dict[str, Any]],
    groups: Sequence[Any],
    output_masks_dir: str,
    work_dir: str,
    tags: Sequence[Dict[str, Any]],
    tag_scale: float = 0.98,
    disc_radius_m: float = 0.127,
    cylinder_radius_m: float = 0.080,
    cylinder_y_min_m: float = 0.001,
    cylinder_y_max_m: float = 0.350,
    chroma_threshold: float = 20.0,
    brightness_white_threshold: float = 135.0,
    tag_blackout: bool = True,
    tag_expansion_factor: float = 1.25,
    workers: int = 1,
    ai_model: Optional[str] = None,
    ai_resolution: int = 512,
    ai_threads: int = 2,
    ai_cache_dir: Optional[str] = None,
    min_ai_recall: float = 0.5,
    stats: Optional[Dict[str, Any]] = None,
    median_cache_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Fused masks for `cameras` (mesh_masks.load_sfm_cameras entries); `groups[i]` = (camera, level) of view i (the
    background median is taken per group). Without `ai_model` the background model alone decides (with the
    photometric test where it cannot judge). Writes <viewId>_<stem>.mask.png like batch_generate_masks and
    returns the same {image name: mask path} mapping; `stats` receives methods and timings.
    `median_cache_dir` (stream_cache): background medians computed while the photos arrived.
    """
    t0 = time.time()
    out_dir = Path(output_masks_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    medians = group_medians([c["path"] for c in cameras], list(groups), work, threads=max(1, workers),
                            cache_dir=median_cache_dir)
    t_med = time.time() - t0
    tex_path = work / "disc_texture.npy"
    np.save(str(tex_path), disc_texture(tags, tag_scale, disc_radius_m))
    blackout_tags = list(tags) if tag_blackout else []
    tasks = []
    for i, (cam, g) in enumerate(zip(cameras, groups)):
        task = {
            "cam": cam, "index": i, "npz": str(work / f"view_{i}.npz"), "median": medians.get(g),
            "disc_tex": str(tex_path), "disc_radius_m": disc_radius_m, "tags": blackout_tags,
            "cylinder_radius_m": cylinder_radius_m, "cylinder_y_min_m": cylinder_y_min_m,
            "cylinder_y_max_m": cylinder_y_max_m, "tag_expansion_factor": tag_expansion_factor,
            "chroma_threshold": chroma_threshold, "brightness_white_threshold": brightness_white_threshold,
            "ai_model": ai_model, "ai_resolution": ai_resolution, "ai_threads": ai_threads,
            "ai_cache_dir": ai_cache_dir,
        }
        vis = [t for t in blackout_tags if not tag_may_be_occluded(
            t, cam["C"], cylinder_radius_m, cylinder_y_min_m, cylinder_y_max_m)]
        task["tag_polys"] = project_tags_to_2d(vis, cam["R"], cam["C"], cam["K"], tag_expansion_factor) if vis else []
        tasks.append(task)

    workers = max(1, min(int(workers or 1), len(tasks) or 1))
    if workers == 1:
        results = [_phase1(t) for t in tasks]
    else:
        # "spawn": the pipeline has a telemetry thread running, and fork() of a multi-threaded process can deadlock
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(_phase1, tasks))
    t_p1 = time.time() - t0 - t_med

    # Share of the shots of each group in which the model sees the object, per pixel
    freq: Dict[Any, np.ndarray] = {}
    if ai_model:
        for g in set(groups):
            arrs = [np.load(tasks[i]["npz"])["ai"] for i in range(len(tasks))
                    if groups[i] == g and results[i] is not None]
            arrs = [a for a in arrs if a.shape == arrs[0].shape] if arrs else []
            if len(arrs) >= 4:
                freq[g] = np.mean(np.stack(arrs), axis=0)

    mask_map: Dict[str, str] = {}
    methods: Dict[str, int] = {}
    done = [(t, results[i]) for i, t in enumerate(tasks) if results[i] is not None]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        outs = list(ex.map(lambda tr: _phase2(tr[0], freq.get(groups[tr[0]["index"]]), str(out_dir), min_ai_recall),
                           done))
    for (task, r), (name, view_id, dst, method) in zip(done, outs):
        if r.get("ai_note"):
            method = f"{method} ({r['ai_note']})"
        mask_map[name] = dst
        mask_map[f"{view_id}_{name}"] = dst
        methods[method] = methods.get(method, 0) + 1
    for t in tasks:
        try:
            os.remove(t["npz"])
        except OSError:
            pass
    model_t = [r["t_model_s"] for r in results if r]
    retries = sum(1 for r in results if r and r.get("retried"))
    model_runs = sum(1 + int(r.get("retried") or 0) for r in results if r)
    if stats is not None:
        stats.update(methods=methods, seconds={"medians": round(t_med, 1), "views": round(t_p1, 1),
                                               "fusion": round(time.time() - t0 - t_med - t_p1, 1)},
                     model_s_per_view=round(float(np.mean(model_t)), 3) if model_t else None,
                     missing=len(tasks) - len(done), crops_grown=retries,
                     model_runs=model_runs, ai_cache_hits=sum(int(r.get("ai_cache_hits") or 0) for r in results if r))
    logger.info(f"Fused masks: {len(done)}/{len(tasks)} views in {time.time() - t0:.1f} s "
                f"(medians {t_med:.1f} s, {workers} worker(s); "
                f"{', '.join(f'{k}: {v}' for k, v in sorted(methods.items()))}).")
    return mask_map
