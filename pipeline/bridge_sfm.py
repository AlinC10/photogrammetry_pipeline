"""
bridge_sfm.py — Converts AliceVision cameras_known.sfm to undistorted images,
undistorted masks, and COLMAP sparse models for OpenMVS InterfaceCOLMAP.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger("TurntableStudio.BridgeSFM")


def build_camera_matrix_and_distortion(
    intr: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Constructs the 3x3 camera calibration matrix K and distortion coefficients
    from an AliceVision intrinsic record.
    """
    w = int(intr["width"])
    h = int(intr["height"])
    sensor_w = float(intr.get("sensorWidth") or 6.287)
    focal_mm = float(intr.get("focalLength") or 6.0)
    pixel_ratio = float(intr.get("pixelRatio") or 1.0)

    # Calculate focal length in pixels
    fx = focal_mm * (w / sensor_w)
    fy = fx * pixel_ratio

    raw_pp = intr.get("principalPoint", [0.0, 0.0])
    raw_cx = float(raw_pp[0])
    raw_cy = float(raw_pp[1])

    # AliceVision stores principal point as offset from image center (e.g. [-1.5, +2.3]).
    # If raw_cx is smaller than w/4, it is definitely a center-relative offset.
    if abs(raw_cx) < (w / 4.0):
        cx = (w / 2.0) + raw_cx
    else:
        cx = raw_cx

    if abs(raw_cy) < (h / 4.0):
        cy = (h / 2.0) + raw_cy
    else:
        cy = raw_cy

    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # AliceVision radial/tangential distortion: [k1, k2, k3] or [k1, k2, p1, p2, k3]
    d_params = intr.get("distortionParams", [])
    dist = np.zeros(5, dtype=np.float64)
    if len(d_params) == 1:
        dist[0] = float(d_params[0])
    elif len(d_params) == 2:
        dist[0] = float(d_params[0])
        dist[1] = float(d_params[1])
    elif len(d_params) == 3:
        dist[0] = float(d_params[0])
        dist[1] = float(d_params[1])
        dist[4] = float(d_params[2])  # k3 in OpenCV is index 4
    elif len(d_params) >= 5:
        dist[0] = float(d_params[0])
        dist[1] = float(d_params[1])
        dist[2] = float(d_params[2])  # p1
        dist[3] = float(d_params[3])  # p2
        dist[4] = float(d_params[4])  # k3

    return K, dist, w, h


def undistort_single_image(
    src_path: str,
    dst_path: str,
    K: np.ndarray,
    dist: np.ndarray,
    K_new: np.ndarray,
    is_mask: bool = False,
) -> None:
    """
    Undistorts an image or mask using OpenCV.
    For masks, uses nearest-neighbor interpolation to preserve sharp binary edges.
    """
    if is_mask:
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Could not load mask: {src_path}")
            return
        undist = cv2.undistort(img, K, dist, None, K_new)
        # Binarize mask cleanly
        _, undist = cv2.threshold(undist, 127, 255, cv2.THRESH_BINARY)
    else:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Could not load image: {src_path}")
            return
        undist = cv2.undistort(img, K, dist, None, K_new)

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst_path, undist)


def generate_proxy_3d_points(
    n_points: int = 1500,
    y_min: float = 0.002,
    y_max: float = 0.220,
    r_min: float = 0.010,
    r_max: float = 0.085,
) -> np.ndarray:
    """
    Generates synthetic 3D proxy points bounded inside a cylinder around the
    turntable center. These guarantee OpenMVS InterfaceCOLMAP has valid 3D points
    and rich covisibility tracks without needing unconstrained feature matching.
    """
    np.random.seed(42)
    # 1. Surface spiral around object boundary
    n_surf = int(n_points * 0.6)
    theta_surf = np.linspace(0, 20 * math.pi, n_surf)
    y_surf = np.linspace(y_min, y_max, n_surf)
    r_surf = np.random.uniform(r_min, r_max, n_surf)
    x_surf = r_surf * np.cos(theta_surf)
    z_surf = r_surf * np.sin(theta_surf)
    surf_pts = np.column_stack([x_surf, y_surf, z_surf])

    # 2. Internal core volume
    n_core = int(n_points * 0.25)
    r_core = np.random.uniform(0.002, r_min, n_core)
    theta_core = np.random.uniform(0, 2 * math.pi, n_core)
    y_core = np.random.uniform(y_min, y_max, n_core)
    x_core = r_core * np.cos(theta_core)
    z_core = r_core * np.sin(theta_core)
    core_pts = np.column_stack([x_core, y_core, z_core])

    # 3. Base plate discs (near y=0)
    n_base = n_points - n_surf - n_core
    r_base = np.random.uniform(0.005, 0.110, n_base)
    theta_base = np.random.uniform(0, 2 * math.pi, n_base)
    y_base = np.full(n_base, y_min)
    x_base = r_base * np.cos(theta_base)
    z_base = r_base * np.sin(theta_base)
    base_pts = np.column_stack([x_base, y_base, z_base])

    points = np.vstack([surf_pts, core_pts, base_pts])
    return points


def convert_sfm_to_colmap(
    sfm_json_path: str,
    output_colmap_dir: str,
    undistorted_images_dir: str,
    masks_src_dir: Optional[str] = None,
    masks_dst_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Reads AliceVision cameras_known.sfm, undistorts images and masks,
    converts rotation and translation to COLMAP format, projects proxy 3D points
    to establish bidirectional tracks, and outputs cameras.txt, images.txt, points3D.txt.
    """
    colmap_path = Path(output_colmap_dir).resolve()
    colmap_path.mkdir(parents=True, exist_ok=True)
    undist_img_path = Path(undistorted_images_dir).resolve()
    undist_img_path.mkdir(parents=True, exist_ok=True)

    if masks_dst_dir:
        Path(masks_dst_dir).resolve().mkdir(parents=True, exist_ok=True)

    with open(sfm_json_path, "r", encoding="utf-8") as f:
        sfm = json.load(f)

    # 1. Process Intrinsics and compute K_new
    intr_data = {}
    colmap_cameras = []

    for intr in sfm["intrinsics"]:
        intr_id = str(intr["intrinsicId"])
        K, dist, w, h = build_camera_matrix_and_distortion(intr)

        # Optimal new camera matrix with alpha=0.0 (no black borders)
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)

        intr_data[intr_id] = {
            "K_orig": K,
            "dist": dist,
            "K_new": K_new,
            "w": w,
            "h": h,
            "colmap_id": int(intr_id) + 1,
        }

        fx_new = float(K_new[0, 0])
        fy_new = float(K_new[1, 1])
        cx_new = float(K_new[0, 2])
        cy_new = float(K_new[1, 2])

        # COLMAP PINHOLE: CAMERA_ID, MODEL, WIDTH, HEIGHT, fx, fy, cx, cy
        colmap_cameras.append(
            f"{intr_data[intr_id]['colmap_id']} PINHOLE {w} {h} {fx_new:.6f} {fy_new:.6f} {cx_new:.6f} {cy_new:.6f}"
        )

    # Prepare COLMAP directories (both root and sparse/ / sparse/0/ for all OpenMVS versions)
    sparse_dirs = [
        colmap_path,
        colmap_path / "sparse",
        colmap_path / "sparse" / "0",
    ]
    for sd in sparse_dirs:
        sd.mkdir(parents=True, exist_ok=True)

    # Write cameras.txt to all candidate locations
    cameras_content = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "\n".join(colmap_cameras) + "\n"
    )
    for sd in sparse_dirs:
        with open(sd / "cameras.txt", "w", encoding="utf-8") as f:
            f.write(cameras_content)

    # 2. Build View and Pose mappings
    poses_by_id = {}
    for p in sfm.get("poses", []):
        pose_id = str(p["poseId"])
        # In AliceVision, rotation is stored Fortran order (Column-Major)
        R = np.array(p["pose"]["transform"]["rotation"], dtype=np.float64).reshape((3, 3), order="F")
        C = np.array(p["pose"]["transform"]["center"], dtype=np.float64)
        poses_by_id[pose_id] = {"R": R, "C": C}

    # 3. Generate 1,500 proxy 3D points spanning turntable scanning volume
    proxy_points = generate_proxy_3d_points(n_points=1500)

    # 4. Undistort images and collect view geometry
    views = sfm.get("views", [])
    registered_views = []

    for v in views:
        pose_id = str(v["poseId"])
        if pose_id not in poses_by_id:
            continue

        intr_id = str(v["intrinsicId"])
        c_info = intr_data[intr_id]
        K_new = c_info["K_new"]
        w = c_info["w"]
        h = c_info["h"]

        pose = poses_by_id[pose_id]
        R = pose["R"]
        C = pose["C"]

        # COLMAP Translation: T = -R @ C
        T = -R @ C

        # SciPy quaternion -> [qx, qy, qz, qw]
        q_scipy = Rotation.from_matrix(R).as_quat()
        # COLMAP quaternion -> [qw, qx, qy, qz]
        qw = float(q_scipy[3])
        qx = float(q_scipy[0])
        qy = float(q_scipy[1])
        qz = float(q_scipy[2])

        # Undistort original image
        src_img_path = Path(v["path"])
        dst_img_name = f"{v['viewId']}_{src_img_path.name}"
        dst_img_path = undist_img_path / dst_img_name

        undistort_single_image(
            str(src_img_path), str(dst_img_path),
            c_info["K_orig"], c_info["dist"], K_new, is_mask=False
        )

        # Undistort mask if available
        mask_candidates = [
            src_img_path.parent / f"{src_img_path.stem}.mask.png",
            src_img_path.parent / f"{src_img_path.stem}.png",
        ]
        if masks_src_dir:
            m_src = Path(masks_src_dir)
            mask_candidates.extend([
                m_src / f"{src_img_path.stem}.mask.png",
                m_src / f"{src_img_path.stem}.png",
                m_src / f"{src_img_path.name}",
                m_src / src_img_path.parent.name / f"{src_img_path.stem}.mask.png",
                m_src / src_img_path.parent.name / f"{src_img_path.stem}.png",
                m_src / "rig" / src_img_path.parent.name / f"{src_img_path.stem}.mask.png",
                m_src / "rig" / src_img_path.parent.name / f"{src_img_path.stem}.png",
            ])

        if masks_dst_dir:
            for m_cand in mask_candidates:
                if m_cand.is_file():
                    # OpenMVS expects masks formatted as: <image_name>.mask.png
                    dst_mask_name = f"{dst_img_name}.mask.png"
                    dst_mask_path = Path(masks_dst_dir) / dst_mask_name
                    undistort_single_image(
                        str(m_cand), str(dst_mask_path),
                        c_info["K_orig"], c_info["dist"], K_new, is_mask=True
                    )
                    break

        colmap_image_id = len(registered_views) + 1
        registered_views.append({
            "colmap_image_id": colmap_image_id,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
            "T": T, "R": R, "C": C,
            "colmap_id": c_info["colmap_id"],
            "dst_img_name": dst_img_name,
            "K_new": K_new,
            "w": w, "h": h,
        })

    # Pass 1: Project points into all views to calculate visibility
    pt_visibility = {pt_idx + 1: [] for pt_idx in range(len(proxy_points))}
    for rv_idx, rv in enumerate(registered_views):
        R = rv["R"]
        C = rv["C"]
        K_new = rv["K_new"]
        w = rv["w"]
        h = rv["h"]

        for pt_idx, P_world in enumerate(proxy_points):
            pt_id = pt_idx + 1
            P_cam = R @ (P_world - C)
            if P_cam[2] > 0.05:
                u = K_new[0, 0] * (P_cam[0] / P_cam[2]) + K_new[0, 2]
                v = K_new[1, 1] * (P_cam[1] / P_cam[2]) + K_new[1, 2]
                if 0.0 <= u < float(w) and 0.0 <= v < float(h):
                    pt_visibility[pt_id].append((rv_idx, u, v))

    # Keep only points observed by at least 2 cameras (or 1 if only 1 view is registered in test)
    min_obs_required = 2 if len(registered_views) >= 2 else 1
    valid_pt_ids = {pt_id for pt_id, obs_list in pt_visibility.items() if len(obs_list) >= min_obs_required}

    # Pass 2: Build final observations and tracks
    view_obs_triplets = {rv_idx: [] for rv_idx in range(len(registered_views))}
    point_tracks = {pt_id: [] for pt_id in valid_pt_ids}

    for pt_id in valid_pt_ids:
        for obs in pt_visibility[pt_id]:
            rv_idx, u, v = obs
            colmap_image_id = registered_views[rv_idx]["colmap_image_id"]
            point_2d_idx = len(view_obs_triplets[rv_idx])
            view_obs_triplets[rv_idx].append(f"{u:.3f} {v:.3f} {pt_id}")
            point_tracks[pt_id].append((colmap_image_id, point_2d_idx))

    # Build COLMAP images.txt lines
    colmap_images_lines = []
    for rv_idx, rv in enumerate(registered_views):
        line1 = (
            f"{rv['colmap_image_id']} {rv['qw']:.8f} {rv['qx']:.8f} {rv['qy']:.8f} {rv['qz']:.8f} "
            f"{rv['T'][0]:.8f} {rv['T'][1]:.8f} {rv['T'][2]:.8f} {rv['colmap_id']} {rv['dst_img_name']}"
        )
        line2 = " ".join(view_obs_triplets[rv_idx])
        colmap_images_lines.append(line1)
        colmap_images_lines.append(line2)

    # Write images.txt to all candidate locations
    images_content = (
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "\n".join(colmap_images_lines) + "\n"
    )
    for sd in sparse_dirs:
        with open(sd / "images.txt", "w", encoding="utf-8") as f:
            f.write(images_content)

    # Write points3D.txt to all candidate locations
    points3d_lines = [
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    ]
    for pt_idx, P_world in enumerate(proxy_points):
        pt_id = pt_idx + 1
        if pt_id not in valid_pt_ids:
            continue
        tracks = point_tracks[pt_id]
        track_str = " ".join(f"{img_id} {p2d_idx}" for img_id, p2d_idx in tracks)
        points3d_lines.append(
            f"{pt_id} {P_world[0]:.6f} {P_world[1]:.6f} {P_world[2]:.6f} 128 128 128 0.5 {track_str}\n"
        )
    points3d_content = "".join(points3d_lines)
    for sd in sparse_dirs:
        with open(sd / "points3D.txt", "w", encoding="utf-8") as f:
            f.write(points3d_content)

    registered_views_count = len(registered_views)
    logger.info(
        f"COLMAP Export complete: {registered_views_count} cameras registered, "
        f"{len(valid_pt_ids)} valid proxy 3D points with multi-view covisibility tracks."
    )

    return {
        "colmap_dir": str(colmap_path),
        "undistorted_images_dir": str(undist_img_path),
        "registered_views": registered_views_count,
        "proxy_points_count": len(proxy_points),
    }
