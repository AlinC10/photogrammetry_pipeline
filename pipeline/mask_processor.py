"""
mask_processor.py — Automated background matting, silhouette masking, and turntable disc elimination.
Combines 3D cylinder visual hull projection with photometric foreground segmentation.
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("TurntableStudio.MaskProcessor")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    from PIL import Image, ImageDraw, ImageFilter
    HAS_PIL = True
except ImportError:
    Image = None
    ImageDraw = None
    ImageFilter = None
    HAS_PIL = False


# -----------------------------------------------------------------------------
# 1. 2D Convex Hull & 3D Cylinder Projection
# -----------------------------------------------------------------------------

def convex_hull_2d(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Computes 2D convex hull of points via Andrew's monotone chain algorithm (O(N log N)).
    """
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def project_cylinder_to_2d(
    R: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    radius_m: float = 0.080,
    y_min_m: float = 0.003,
    y_max_m: float = 0.350,
    n_angles: int = 64,
) -> List[Tuple[float, float]]:
    """
    Projects the 3D bounding cylinder of the turntable scanning volume into the 2D camera view.
    Points at y <= y_min_m (e.g. 0.003m) are strictly excluded, eliminating the turntable disc (y=0.0).
    Returns the ordered 2D convex hull polygon boundary points.
    """
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Sample bottom, middle, and top circular rings of the cylinder
    bot = np.stack([radius_m * cos_t, np.full_like(thetas, y_min_m), radius_m * sin_t], axis=1)
    mid = np.stack([radius_m * cos_t, np.full_like(thetas, (y_min_m + y_max_m) / 2.0), radius_m * sin_t], axis=1)
    top = np.stack([radius_m * cos_t, np.full_like(thetas, y_max_m), radius_m * sin_t], axis=1)

    pts_3d = np.vstack([bot, mid, top])

    # World-to-camera projection: P_cam = R @ (P_world - C)
    C_vec = C.flatten()
    P_cam = (pts_3d - C_vec) @ R.T

    # Filter points in front of the camera
    valid = P_cam[:, 2] > 0.05
    if not np.any(valid):
        return []

    pts_valid = P_cam[valid]
    u = K[0, 0] * (pts_valid[:, 0] / pts_valid[:, 2]) + K[0, 2]
    v = K[1, 1] * (pts_valid[:, 1] / pts_valid[:, 2]) + K[1, 2]

    pts_2d = [(float(px), float(py)) for px, py in zip(u, v)]
    return convex_hull_2d(pts_2d)


def load_tag_layout(layout_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Loads AprilTag disc layout specification (e.g. platan_260mm_layout.json).
    Returns list of tag dictionaries with 'id', 'center', and 'corners'.
    """
    import json
    if layout_path is None:
        layout_path = Path(__file__).resolve().parent / "platan_260mm_layout.json"
    l_path = Path(layout_path).resolve()
    if not l_path.is_file():
        logger.warning(f"Tag layout file not found: {l_path}")
        return []
    try:
        with open(l_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("tags", [])
    except Exception as e:
        logger.warning(f"Failed to load tag layout from {l_path}: {e}")
        return []


def project_tags_to_2d(
    tags: List[Dict[str, Any]],
    R: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    expansion_factor: float = 1.25,
) -> List[List[Tuple[float, float]]]:
    """
    Projects 3D AprilTag coordinates into 2D camera coordinates.
    Each tag polygon is expanded by `expansion_factor` around its 3D center
    to guarantee suppression of the tag's black border and quiet zone.
    Returns list of 2D quad polygons [(u0, v0), (u1, v1), (u2, v2), (u3, v3)].
    """
    tag_polys: List[List[Tuple[float, float]]] = []
    C_vec = C.flatten()

    for tag in tags:
        corners = np.array(tag.get("corners", []), dtype=np.float64)
        if len(corners) < 3:
            continue
        center = np.array(tag.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)

        # 3D radial expansion around center to cover black border
        if expansion_factor != 1.0:
            corners_exp = center + (corners - center) * expansion_factor
        else:
            corners_exp = corners

        # World-to-camera projection: P_cam = (P_world - C) @ R.T
        P_cam = (corners_exp - C_vec) @ R.T

        # Require points to be in front of camera
        if not np.all(P_cam[:, 2] > 0.05):
            continue

        u = K[0, 0] * (P_cam[:, 0] / P_cam[:, 2]) + K[0, 2]
        v = K[1, 1] * (P_cam[:, 1] / P_cam[:, 2]) + K[1, 2]

        poly = [(float(px), float(py)) for px, py in zip(u, v)]
        tag_polys.append(poly)

    return tag_polys


# -----------------------------------------------------------------------------
# 2. Hybrid Mask Generator
# -----------------------------------------------------------------------------

def generate_hybrid_mask(
    image: Union[np.ndarray, "Image.Image", str, Path],
    R: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    cylinder_radius_m: float = 0.035,
    cylinder_y_min_m: float = 0.001,
    cylinder_y_max_m: float = 0.245,
    chroma_threshold: float = 20.0,
    brightness_white_threshold: float = 135.0,
    safety_dilation_px: int = 1,
    fill_holes: bool = True,
    tag_polygons: Optional[List[List[Tuple[float, float]]]] = None,
    tag_blackout: bool = True,
) -> np.ndarray:
    """
    Generates a high-precision binary silhouette mask (0 = background, 255 = foreground).
    Guarantees that ONLY the scanned physical object is visible:
    1. 3D Bounding Cylinder projection (cuts away booth walls, aluminum rails, and turntable disc at y <= 0.001m).
    2. Explicit geometric AprilTag blackout (forces all 12 AprilTag regions to 0/background).
    3. Photometric segmentation (separates colorful plastic / dark rubber from bright white booth & white disc).
    4. Connected component filtering (isolates largest component / tool, drops isolated tag fragments).
    5. Morphological safety boundary.
    """
    # 1. Load image as RGB NumPy array
    if isinstance(image, (str, Path)):
        if HAS_PIL:
            with Image.open(str(image)) as p_img:
                rgb_arr = np.array(p_img.convert("RGB"), dtype=np.float32)
        elif HAS_CV2:
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise FileNotFoundError(f"Failed to load image: {image}")
            rgb_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        else:
            raise RuntimeError("PIL or cv2 required for image loading.")
    elif HAS_PIL and isinstance(image, Image.Image):
        rgb_arr = np.array(image.convert("RGB"), dtype=np.float32)
    elif isinstance(image, np.ndarray):
        rgb_arr = image.astype(np.float32)
        if rgb_arr.ndim == 2:
            rgb_arr = np.repeat(rgb_arr[:, :, None], 3, axis=2)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    h, w = rgb_arr.shape[:2]

    # 2. Geometric 3D Cylinder Visual Hull ROI
    roi_mask = np.ones((h, w), dtype=bool)
    if R is not None and C is not None and K is not None:
        hull_pts = project_cylinder_to_2d(
            R=R, C=C, K=K,
            radius_m=cylinder_radius_m,
            y_min_m=cylinder_y_min_m,
            y_max_m=cylinder_y_max_m,
        )
        if len(hull_pts) >= 3:
            if HAS_PIL:
                roi_pil = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(roi_pil)
                draw.polygon(hull_pts, fill=255)
                roi_mask = np.array(roi_pil) > 128
            elif HAS_CV2:
                pts_cv = np.array(hull_pts, dtype=np.int32).reshape((-1, 1, 2))
                roi_cv = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(roi_cv, [pts_cv], 255)
                roi_mask = roi_cv > 128
    else:
        # Fallback ROI: center 70% width, top 75% height (cuts off turntable disc & side frame)
        roi_mask = np.zeros((h, w), dtype=bool)
        x_min, x_max = int(w * 0.15), int(w * 0.85)
        y_min, y_max = int(h * 0.05), int(h * 0.72)
        roi_mask[y_min:y_max, x_min:x_max] = True

    # 3. Geometric AprilTag Blackout Mask
    tag_mask = np.zeros((h, w), dtype=bool)
    if tag_blackout and tag_polygons:
        if HAS_PIL:
            t_pil = Image.new("L", (w, h), 0)
            tdraw = ImageDraw.Draw(t_pil)
            for tpoly in tag_polygons:
                if len(tpoly) >= 3:
                    tdraw.polygon(tpoly, fill=255)
            tag_mask = np.array(t_pil) > 128
        elif HAS_CV2:
            t_cv = np.zeros((h, w), dtype=np.uint8)
            for tpoly in tag_polygons:
                if len(tpoly) >= 3:
                    pts_t = np.array(tpoly, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(t_cv, [pts_t], 255)
            tag_mask = t_cv > 128

    # 4. Photometric Color & Brightness Segmentation
    r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    chroma = max_c - min_c
    brightness = (r + g + b) / 3.0

    # Flat white photo-booth wall & white turntable disc:
    is_white_bg = (brightness > brightness_white_threshold) & (chroma < chroma_threshold)

    # Foreground object: inside cylinder ROI, not white background, and strictly NOT on any AprilTag
    is_fg = (~is_white_bg) & roi_mask & (~tag_mask)

    # 5. Morphological Cleaning & Connected Component Filtering
    scale_w = min(800, w)
    scale_h = int(h * (scale_w / w))

    if HAS_PIL:
        fg_pil = Image.fromarray((is_fg * 255).astype(np.uint8)).resize((scale_w, scale_h), Image.Resampling.NEAREST)
        if fill_holes:
            closed_pil = fg_pil.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(5))
        else:
            closed_pil = fg_pil

        sub_arr = np.array(closed_pil) > 128
    elif HAS_CV2:
        sub_fg = cv2.resize((is_fg * 255).astype(np.uint8), (scale_w, scale_h), interpolation=cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed_cv = cv2.morphologyEx(sub_fg, cv2.MORPH_CLOSE, kernel) if fill_holes else sub_fg
        sub_arr = closed_cv > 128
    else:
        sub_arr = is_fg

    # Connected Component Filtering (keep main physical object, eliminate isolated tags/reflections)
    sh, sw = sub_arr.shape
    visited = np.zeros_like(sub_arr, dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    for y in range(0, sh, 2):
        for x in range(0, sw, 2):
            if sub_arr[y, x] and not visited[y, x]:
                comp: List[Tuple[int, int]] = []
                queue = [(y, x)]
                visited[y, x] = True
                while queue:
                    cy, cx = queue.pop()
                    comp.append((cy, cx))
                    for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < sh and 0 <= nx < sw and sub_arr[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                if len(comp) > 20:
                    components.append(comp)

    clean_sub = np.zeros_like(sub_arr, dtype=np.uint8)
    if components:
        # Keep components that are >= 20% of the largest component (isolates the tool)
        largest_size = max(len(c) for c in components)
        min_keep_size = max(50, int(largest_size * 0.20))
        for comp in components:
            if len(comp) >= min_keep_size:
                for cy, cx in comp:
                    clean_sub[max(0, cy - 1):min(sh, cy + 2), max(0, cx - 1):min(sw, cx + 2)] = 255
    else:
        clean_sub = (sub_arr.astype(np.uint8)) * 255

    # 6. Safety Dilation / Erosion & Upsample to Native Resolution
    if HAS_PIL:
        clean_pil = Image.fromarray(clean_sub)
        if safety_dilation_px > 0:
            clean_pil = clean_pil.filter(ImageFilter.MaxFilter(int(safety_dilation_px * 2 + 1)))
        final_mask_pil = clean_pil.resize((w, h), Image.Resampling.BILINEAR)
        final_mask = (np.array(final_mask_pil) > 128).astype(np.uint8) * 255
    elif HAS_CV2:
        if safety_dilation_px > 0:
            d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safety_dilation_px * 2 + 1, safety_dilation_px * 2 + 1))
            clean_sub = cv2.dilate(clean_sub, d_kernel)
        final_mask = cv2.resize(clean_sub, (w, h), interpolation=cv2.INTER_LINEAR)
        final_mask = (final_mask > 128).astype(np.uint8) * 255
    else:
        final_mask = (is_fg.astype(np.uint8)) * 255

    # 7. Strictly enforce AprilTag blackout on final mask (overrides dilation/upsampling)
    if np.any(tag_mask):
        final_mask[tag_mask] = 0

    return final_mask


# -----------------------------------------------------------------------------
# 3. Batch Generation & Application
# -----------------------------------------------------------------------------

def apply_mask_to_image(image_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    """
    Sets background pixels (mask < 128) to pure black RGB(0, 0, 0).
    Guarantees that TextureMesh and texrecon never sample white walls or AprilTags.
    """
    out_arr = image_arr.copy()
    if mask_arr.ndim == 2:
        m = mask_arr >= 128
        if out_arr.ndim == 3:
            out_arr[~m] = 0
        else:
            out_arr[~m] = 0
    else:
        m = mask_arr[..., 0] >= 128
        out_arr[~m] = 0
    return out_arr


def _mask_one_view(task: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """
    Worker: one view -> mask files. Module level so ProcessPoolExecutor can pickle it.
    Returns (source file name, view id, mask path) or None when the image is missing.
    """
    v = task["view"]
    src_path = Path(v["path"])
    if not src_path.is_file():
        return None

    R, C, K = task["R"], task["C"], task["K"]
    tags = task["tags"]
    tag_polygons = None
    if task["tag_blackout"] and tags and R is not None and C is not None and K is not None:
        tag_polygons = project_tags_to_2d(
            tags=tags, R=R, C=C, K=K,
            expansion_factor=task["tag_expansion_factor"],
        )

    mask = generate_hybrid_mask(
        image=src_path,
        R=R, C=C, K=K,
        cylinder_radius_m=task["cylinder_radius_m"],
        cylinder_y_min_m=task["cylinder_y_min_m"],
        cylinder_y_max_m=task["cylinder_y_max_m"],
        chroma_threshold=task["chroma_threshold"],
        brightness_white_threshold=task["brightness_white_threshold"],
        safety_dilation_px=task["safety_dilation_px"],
        tag_polygons=tag_polygons,
        tag_blackout=task["tag_blackout"],
    )

    out_dir = Path(task["out_dir"])
    view_id = str(v.get("viewId", task["index"]))
    stem_name = src_path.stem

    # Canonical OpenMVS naming: {viewId}_{stem}.mask.png (e.g. 1000_f_1000.mask.png matching scene.mvs views).
    # No flat f_1000.mask.png alias: in rig mode both cameras have an f_1000 and one overwrote the other.
    dst_mask_path = out_dir / f"{view_id}_{stem_name}.mask.png"
    targets = [dst_mask_path]
    # In rig mode keep the per-camera copy rig/0 or 0/ (unique per camera folder)
    if src_path.parent.name in {"0", "1", "rig0", "rig1"}:
        rig_sub = out_dir / src_path.parent.name
        rig_sub.mkdir(parents=True, exist_ok=True)
        targets += [rig_sub / f"{stem_name}.mask.png", rig_sub / f"{src_path.name}.mask.png"]

    if HAS_PIL:
        p_mask = Image.fromarray(mask)
        for t in targets:
            p_mask.save(str(t), format="PNG")
    elif HAS_CV2:
        for t in targets:
            cv2.imwrite(str(t), mask)

    return src_path.name, view_id, str(dst_mask_path)


def batch_generate_masks(
    views: List[Dict[str, Any]],
    poses_dict: Dict[str, Dict[str, Any]],
    intrinsics_dict: Dict[str, Dict[str, Any]],
    output_masks_dir: Union[str, Path],
    cylinder_radius_m: float = 0.035,
    cylinder_y_min_m: float = 0.001,
    cylinder_y_max_m: float = 0.245,
    chroma_threshold: float = 20.0,
    brightness_white_threshold: float = 135.0,
    safety_dilation_px: int = 1,
    tag_blackout: bool = True,
    tag_expansion_factor: float = 1.25,
    tag_layout_path: Optional[Union[str, Path]] = None,
    workers: int = 1,
) -> Dict[str, str]:
    """
    Generates automated binary masks for all views in the dataset, `workers` images at a time
    (separate processes; the result is identical to the serial run).
    Saves `<viewId>_<stem>.mask.png` in output_masks_dir.
    Returns mapping: {image_filename: mask_file_path}.
    """
    out_dir = Path(output_masks_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tags = load_tag_layout(tag_layout_path) if tag_blackout else []
    if tag_blackout and tags:
        logger.info(f"Loaded {len(tags)} AprilTags for disc blackout suppression (expansion={tag_expansion_factor:.2f}).")

    tasks = []
    for idx, v in enumerate(views):
        pose_info = poses_dict.get(str(v.get("poseId", v.get("viewId", ""))))
        intr_info = intrinsics_dict.get(str(v.get("intrinsicId", "0")))
        tasks.append({
            "view": v, "index": idx, "out_dir": str(out_dir), "tags": tags,
            "R": pose_info.get("R") if pose_info else None,
            "C": pose_info.get("C") if pose_info else None,
            "K": intr_info.get("K") if intr_info else None,
            "cylinder_radius_m": cylinder_radius_m,
            "cylinder_y_min_m": cylinder_y_min_m,
            "cylinder_y_max_m": cylinder_y_max_m,
            "chroma_threshold": chroma_threshold,
            "brightness_white_threshold": brightness_white_threshold,
            "safety_dilation_px": safety_dilation_px,
            "tag_blackout": tag_blackout,
            "tag_expansion_factor": tag_expansion_factor,
        })

    mask_map: Dict[str, str] = {}
    total = len(tasks)
    workers = max(1, min(int(workers or 1), total or 1))
    logger.info(f"Starting automated background matting for {total} images ({workers} worker(s))...")

    def collect(done: int, result) -> None:
        if result is not None:
            name, view_id, dst = result
            mask_map[name] = dst
            mask_map[f"{view_id}_{name}"] = dst
        if done % 10 == 0 or done == total:
            logger.info(f"  Processed {done}/{total} masks ({done / total * 100:.0f}%)")

    if workers == 1:
        for i, task in enumerate(tasks, 1):
            collect(i, _mask_one_view(task))
    else:
        # "spawn": the pipeline has a telemetry thread running, and fork() of a multi-threaded process can deadlock
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            for i, result in enumerate(ex.map(_mask_one_view, tasks), 1):
                collect(i, result)

    logger.info(f"Completed automated masking: {len(mask_map)} masks saved in {out_dir}")
    return mask_map
