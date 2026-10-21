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

    pp = intr.get("principalPoint", [w / 2.0, h / 2.0])
    cx = float(pp[0])
    cy = float(pp[1])

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
    n_points: int = 300,
    y_min: float = 0.005,
    y_max: float = 0.20,
    r_min: float = 0.015,
    r_max: float = 0.065,
) -> np.ndarray:
    """
    Generates synthetic 3D proxy points bounded inside a cylinder around the
    turntable center. These guarantee OpenMVS InterfaceCOLMAP has valid 3D points
    and covisibility tracks without needing unconstrained feature matching.
    """
    np.random.seed(42)
    # Golden spiral distribution on cylinder
    theta = np.linspace(0, 10 * math.pi, n_points)
    y = np.linspace(y_min, y_max, n_points)
    r = np.random.uniform(r_min, r_max, n_points)

    x = r * np.cos(theta)
    z = r * np.sin(theta)

    points = np.column_stack([x, y, z])
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

    # Write cameras.txt
    with open(colmap_path / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for line in colmap_cameras:
            f.write(line + "\n")

    # 2. Build View and Pose mappings
    poses_by_id = {}
    for p in sfm.get("poses", []):
        pose_id = str(p["poseId"])
        # In AliceVision, rotation is stored Fortran order (Column-Major)
        R = np.array(p["pose"]["transform"]["rotation"], dtype=np.float64).reshape((3, 3), order="F")
        C = np.array(p["pose"]["transform"]["center"], dtype=np.float64)
        poses_by_id[pose_id] = {"R": R, "C": C}

    # 3. Generate 300 proxy 3D points
    proxy_points = generate_proxy_3d_points(n_points=300)
    point_tracks = {i + 1: [] for i in range(len(proxy_points))}

    # 4. Undistort images and prepare COLMAP image records
    views = sfm.get("views", [])
    colmap_images_lines = []
    registered_views_count = 0

    for idx, v in enumerate(views):
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

        # Project proxy 3D points to compute 2D observations
        colmap_image_id = registered_views_count + 1
        obs_triplets = []

        point_2d_idx = 0
        for pt_idx, P_world in enumerate(proxy_points):
            pt_id = pt_idx + 1
            # Camera coordinate: P_cam = R @ (P_world - C)
            P_cam = R @ (P_world - C)

            # Check if point is in front of camera
            if P_cam[2] > 0.05:
                # Pinhole projection: (u, v) = K_new @ P_cam / P_cam[2]
                u = K_new[0, 0] * (P_cam[0] / P_cam[2]) + K_new[0, 2]
                v = K_new[1, 1] * (P_cam[1] / P_cam[2]) + K_new[1, 2]

                # Bounding box check
                if 0.0 <= u < float(w) and 0.0 <= v < float(h):
                    obs_triplets.append(f"{u:.3f} {v:.3f} {pt_id}")
                    point_tracks[pt_id].append((colmap_image_id, point_2d_idx))
                    point_2d_idx += 1

        # Format COLMAP images.txt entry (2 lines per image)
        # Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        line1 = f"{colmap_image_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {T[0]:.8f} {T[1]:.8f} {T[2]:.8f} {c_info['colmap_id']} {dst_img_name}"
        # Line 2: POINTS2D[] as (X, Y, POINT3D_ID)
        line2 = " ".join(obs_triplets)

        colmap_images_lines.append(line1)
        colmap_images_lines.append(line2)
        registered_views_count += 1

    # Write images.txt
    with open(colmap_path / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for line in colmap_images_lines:
            f.write(line + "\n")

    # Write points3D.txt
    with open(colmap_path / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pt_idx, P_world in enumerate(proxy_points):
            pt_id = pt_idx + 1
            tracks = point_tracks[pt_id]
            track_str = " ".join(f"{img_id} {p2d_idx}" for img_id, p2d_idx in tracks)
            f.write(
                f"{pt_id} {P_world[0]:.6f} {P_world[1]:.6f} {P_world[2]:.6f} 128 128 128 0.5 {track_str}\n"
            )

    logger.info(
        f"COLMAP Export complete: {registered_views_count} cameras registered, "
        f"{len(proxy_points)} proxy 3D points with covisibility tracks."
    )

    return {
        "colmap_dir": str(colmap_path),
        "undistorted_images_dir": str(undist_img_path),
        "registered_views": registered_views_count,
        "proxy_points_count": len(proxy_points),
    }
