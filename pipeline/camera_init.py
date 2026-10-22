"""
camera_init.py — Image scanner, metadata extractor, cameraInit.sfm generator,
and turntable_poses caller.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import ExifTags, Image

from .config import (
    DEFAULT_FOCAL_LENGTH_MM,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_SENSOR_WIDTH_MM,
    TurntableConfig,
)

logger = logging.getLogger("TurntableStudio.CameraInit")


def extract_image_metadata(
    img_path: str, default_focal_mm: Optional[float] = None
) -> Dict[str, Any]:
    """
    Extracts image dimensions and EXIF metadata (focal length, sensor size).
    Falls back to Sony IMX477 defaults if EXIF is absent or incomplete.
    """
    width = DEFAULT_IMAGE_WIDTH
    height = DEFAULT_IMAGE_HEIGHT
    focal_length_mm = default_focal_mm if default_focal_mm is not None else DEFAULT_FOCAL_LENGTH_MM
    sensor_width_mm = DEFAULT_SENSOR_WIDTH_MM
    make = ""
    model = ""

    try:
        with Image.open(img_path) as img:
            width, height = img.size
            exif = img.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == "Make":
                        make = str(value).strip()
                    elif tag_name == "Model":
                        model = str(value).strip()
                    elif tag_name == "FocalLength":
                        try:
                            # Rational to float
                            focal_length_mm = float(value)
                        except Exception:
                            pass
    except Exception as e:
        logger.warning(f"Could not read EXIF from {img_path}: {e}")

    # Check for known camera models
    if "imx477" in model.lower() or "raspberry" in make.lower():
        sensor_width_mm = 6.287

    return {
        "width": width,
        "height": height,
        "focalLength": round(focal_length_mm, 4),
        "sensorWidth": round(sensor_width_mm, 4),
        "make": make,
        "model": model,
    }


def scan_input_directory(
    input_dir: str,
    rig_mode: bool = False,
    default_focal_mm: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], str]:
    """
    Discovers images in input_dir, automatically detecting rig structure
    (rig/0 and rig/1) or flat files.

    Returns:
        views: List of view dicts for AliceVision SFM
        intrinsics: Dict of intrinsicId -> intrinsic metadata
        name_regex: Regex used to parse level and index from filenames
    """
    p = Path(input_dir).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {p}")

    valid_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

    # Check if rig mode is present on disk
    rig_0 = p / "rig" / "0"
    rig_1 = p / "rig" / "1"
    is_rig = rig_mode or (rig_0.is_dir() and rig_1.is_dir())

    views = []
    intrinsics = {}
    name_regex = r"(?P<level>\d)[_]?(?P<index>\d{1,4})$"

    if is_rig:
        logger.info(f"Scanning RIG layout: {rig_0} and {rig_1}")
        cam_dirs = [("0", rig_0), ("1", rig_1)]
        view_counter = 1000

        for cam_id, cdir in cam_dirs:
            if not cdir.is_dir():
                continue
            img_files = sorted([
                f for f in cdir.iterdir()
                if f.suffix.lower() in valid_exts and not f.name.endswith(".mask.png")
            ])
            logger.info(f"Found {len(img_files)} images for Camera {cam_id}")

            for f in img_files:
                meta = extract_image_metadata(str(f), default_focal_mm=default_focal_mm)

                # Register intrinsic for camera if not present
                if cam_id not in intrinsics:
                    intrinsics[cam_id] = {
                        "intrinsicId": cam_id,
                        "width": meta["width"],
                        "height": meta["height"],
                        "sensorWidth": meta["sensorWidth"],
                        "sensorHeight": round(meta["sensorWidth"] * (meta["height"] / meta["width"]), 4),
                        "focalLength": meta["focalLength"],
                        "initialFocalLength": meta["focalLength"],
                        "pixelRatio": 1.0,
                    }

                stem = f.stem
                # Clean stem for regex matching
                # E.g. if stem is "0001", regex matches level 0, index 001
                # If stem has non-digits, extract trailing numbers
                digits = re.findall(r"\d+", stem)
                if digits:
                    num_val = int(digits[-1])
                    # level 0, index num_val
                    level = 0
                    index = num_val
                else:
                    level = 0
                    index = view_counter - 1000

                views.append({
                    "viewId": str(view_counter),
                    "poseId": str(view_counter),
                    "intrinsicId": str(cam_id),
                    "path": str(f.resolve()),
                    "width": meta["width"],
                    "height": meta["height"],
                    "level": level,
                    "index": index,
                })
                view_counter += 1

        name_regex = r"(?P<level>\d)[_]?(?P<index>\d{1,4})$"

    else:
        # Flat directory scanning
        logger.info(f"Scanning FLAT layout in {p}")
        img_files = sorted([
            f for f in p.iterdir()
            if f.suffix.lower() in valid_exts and not f.name.endswith(".mask.png")
        ])
        if not img_files:
            raise FileNotFoundError(f"No valid images found in {p}")

        logger.info(f"Found {len(img_files)} images in input directory")
        view_counter = 1000

        for f in img_files:
            meta = extract_image_metadata(str(f), default_focal_mm=default_focal_mm)
            cam_id = "0"

            if cam_id not in intrinsics:
                intrinsics[cam_id] = {
                    "intrinsicId": cam_id,
                    "width": meta["width"],
                    "height": meta["height"],
                    "sensorWidth": meta["sensorWidth"],
                    "sensorHeight": round(meta["sensorWidth"] * (meta["height"] / meta["width"]), 4),
                    "focalLength": meta["focalLength"],
                    "initialFocalLength": meta["focalLength"],
                    "pixelRatio": 1.0,
                }

            stem = f.stem
            digits = re.findall(r"\d+", stem)
            if digits:
                raw_num = digits[-1]
                if len(raw_num) >= 4:
                    level = int(raw_num[0])
                    index = int(raw_num[1:])
                else:
                    level = 0
                    index = int(raw_num)
            else:
                level = 0
                index = view_counter - 1000

            views.append({
                "viewId": str(view_counter),
                "poseId": str(view_counter),
                "intrinsicId": cam_id,
                "path": str(f.resolve()),
                "width": meta["width"],
                "height": meta["height"],
                "level": level,
                "index": index,
            })
            view_counter += 1

    return views, intrinsics, name_regex


def generate_camera_init_sfm(
    views: List[Dict[str, Any]],
    intrinsics: Dict[str, Dict[str, Any]],
    out_sfm_path: str,
) -> str:
    """
    Serializes views and intrinsics into the AliceVision CameraInit SFM JSON format.
    """
    out_p = Path(out_sfm_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    sfm_views = []
    for v in views:
        sfm_views.append({
            "viewId": v["viewId"],
            "poseId": v["poseId"],
            "intrinsicId": v["intrinsicId"],
            "path": v["path"],
            "width": v["width"],
            "height": v["height"],
        })

    sfm_data = {
        "version": ["1", "0", "0"],
        "featuresFolders": [],
        "matchesFolders": [],
        "views": sfm_views,
        "intrinsics": list(intrinsics.values()),
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(sfm_data, f, indent=2)

    logger.info(f"Generated CameraInit SFM: {out_p} ({len(sfm_views)} views, {len(intrinsics)} cameras)")
    return str(out_p)


def run_turntable_poses(
    camera_init_sfm: str,
    output_known_sfm: str,
    config: TurntableConfig,
    report_path: Optional[str] = None,
    calib_out_path: Optional[str] = None,
    custom_python_exe: Optional[str] = None,
) -> str:
    """
    Calls the external turntable_poses.py script to calibrate camera intrinsics,
    estimate turntable axis, and rigidly lock all camera poses.

    IMPORTANT: turntable_poses.py is executed as an external process and is NEVER modified.
    """
    script_path = Path(config.turntable_poses_script).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(
            f"turntable_poses.py not found at: {script_path}. "
            "Please check config.turntable.turntable_poses_script."
        )

    out_sfm = Path(output_known_sfm).resolve()
    out_sfm.parent.mkdir(parents=True, exist_ok=True)

    py_exe = custom_python_exe or sys.executable

    cmd = [
        py_exe,
        str(script_path),
        "--input", str(Path(camera_init_sfm).resolve()),
        "--output", str(out_sfm),
        "--stepDeg", str(config.step_deg),
        "--tagScale", str(config.tag_scale),
        "--nameRegex", config.name_regex,
    ]

    if config.tag_layout:
        cmd.extend(["--layout", str(Path(config.tag_layout).resolve())])

    if calib_out_path:
        cmd.extend(["--calibOut", str(Path(calib_out_path).resolve())])

    if report_path:
        cmd.extend(["--report", str(Path(report_path).resolve())])

    logger.info(f"Executing TurntablePoses: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error(f"turntable_poses.py failed (exit {result.returncode}):\n{result.stderr}")
        raise RuntimeError(f"TurntablePoses failed:\n{result.stderr}")

    if not out_sfm.is_file():
        raise FileNotFoundError(f"TurntablePoses succeeded but output file not found: {out_sfm}")

    logger.info(f"TurntablePoses completed successfully -> {out_sfm}")
    return str(out_sfm)
