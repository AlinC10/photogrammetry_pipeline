"""
main.py — Master orchestrator for Turntable Studio.
Integrates TurntablePoses, OpenMVS, Watertight Mesh Cleaning, Multi-LOD Export,
and Full Hardware/Memory Telemetry.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"

# Console: INFO. The root logger lets DEBUG through so run_log.txt can keep the OpenMVS output
# (logged at DEBUG by openmvs_runner); the console handler filters it back to INFO.
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
logging.getLogger().setLevel(logging.DEBUG)
for _h in logging.getLogger().handlers:
    _h.setLevel(logging.INFO)
logger = logging.getLogger("TurntableStudio")


class _RunLogFilter(logging.Filter):
    """run_log.txt: everything from INFO up, plus DEBUG from our own loggers (OpenMVS output), no library noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.INFO or record.name.startswith("TurntableStudio")


def attach_run_log(out_path: Path) -> logging.Handler:
    """UTF-8 log file written by the pipeline itself, whatever the shell (PowerShell tee wrote UTF-16)."""
    handler = logging.FileHandler(out_path / "run_log.txt", mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    handler.addFilter(_RunLogFilter())
    logging.getLogger().addHandler(handler)
    return handler


def detach_run_log(handler: logging.Handler) -> None:
    handler.flush()
    handler.close()
    logging.getLogger().removeHandler(handler)


from pipeline.texrecon_runner import TexReconRunner, find_texrecon_binary
from pipeline.bridge_sfm import convert_sfm_to_colmap, undistort_rgb_images
from pipeline.mask_processor import ai_model_available, batch_generate_masks, load_tag_layout
from pipeline.mask_fusion import batch_fused_masks
from pipeline.camera_init import (
    generate_camera_init_sfm,
    run_turntable_poses,
    scan_input_directory,
)
from pipeline.config import (
    MaskingConfig,
    TexReconConfig,
    TexturingImagesConfig,
    MeshCleaningConfig,
    OpenMVSConfig,
    PipelineConfig,
    RigRefineConfig,
    TurntableConfig,
    VisualHullConfig,
)
from pipeline.exporter import (
    collect_log_files,
    decimate_mesh,
    export_multi_lod_bundle,
    export_print_assets,
    EarlyExports,
    OutputZip,
)
from pipeline.parallel import LoggingProcessPool
from pipeline.mesh_cleaner import clean_and_repair_mesh, face_target
from pipeline.object_info import write_object_info
from pipeline.mesh_masks import load_sfm_cameras
from pipeline.rig_refine import refine_rig_poses, view_groups
from pipeline import visual_hull
from pipeline.openmvs_runner import OpenMVSRunner
from pipeline import run_metrics
from pipeline import stream_cache
from pipeline.resources import InsufficientRamError, check_ram, describe_resources, resolve_workers
from pipeline.telemetry import SystemTelemetry
from pipeline.texrecon_runner import TEXRECON_DATA_TERMS

DENSIFY_MASK_MODES = ("blackout", "mask_path", "none")
MASK_METHODS = ("fused", "ai", "photometric")
# Job options with a server default the clients (tools/cloud_scan.py, the Pi) do not need to know or forward:
# a new image changes the behaviour of every scan without touching the scripts that upload the photos
SERVER_DEFAULT_OPTIONS = frozenset({"rig_refine", "mask_method", "ai_view_stride"})
from pipeline.texture_prep import prepare_texturing_images


def turntable_config(job: Dict[str, Any]) -> TurntableConfig:
    """Turntable / pose options of a job (also used by the streaming pre-work, pipeline/stream_prework.py)."""
    layout = job.get("tag_layout") or TurntableConfig.tag_layout
    tag_scale_val = job.get("tag_scale")
    if tag_scale_val is None:
        tag_scale = 0.98 if (layout and "260" in str(layout)) else 0.99
    else:
        tag_scale = float(tag_scale_val)
    return TurntableConfig(
        tag_layout=layout,
        step_deg=job.get("step_deg", 18.0),
        tag_scale=tag_scale,
        default_focal_mm=job.get("focal_length_mm", 3.9),
        lift_mm=float(job.get("lift_mm", TurntableConfig.lift_mm)),
        estimate_lift=None if job.get("estimate_lift") in (None, "auto") else bool(job["estimate_lift"]),
        tag_refine=bool(job.get("tag_refine", True)),
        intrinsics_in=job.get("intrinsics_in"),
    )


def masking_config(job: Dict[str, Any]) -> MaskingConfig:
    """Masking options of a job (also used by the streaming pre-work)."""
    layout = job.get("tag_layout") or TurntableConfig.tag_layout
    is_260 = bool(layout and "260" in str(layout))
    cyl_r = job.get("cylinder_radius_m")
    if cyl_r is None:
        cyl_r = 0.080 if is_260 else 0.138
    disc_r = job.get("disc_radius_m")
    if disc_r is None:
        disc_r = 0.127 if is_260 else 0.170
    wide_cyl_r = 0.095 if is_260 else 0.138
    return MaskingConfig(
        densify_masks=job.get("densify_masks", "blackout"),
        method=job.get("mask_method", "fused"),
        ai_view_stride=int(job.get("ai_view_stride", 2)),
        cylinder_radius_m=float(cyl_r),
        wide_cylinder_radius_m=float(wide_cyl_r),
        disc_radius_m=float(disc_r),
    )


def ai_mask_workers(m: MaskingConfig, workers: int) -> int:
    """Processes for the AI masks: each holds the model (~3 GB peak) and uses ai_threads threads."""
    res = describe_resources()
    by_cpu = max(1, int(res["cpus"]) // max(1, m.ai_threads))
    by_ram = max(1, int((float(res["ram_free_gb"]) - 4.0) // m.ai_ram_per_worker_gb))
    return max(1, min(workers, by_cpu, by_ram))


def run_pipeline(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for pipeline execution, compatible with RunPod/Cloudflare job specifications.

    Example job dictionary:
    {
        "input_dir": "./input_images",
        "output_dir": "./output",
        "rig_mode": True,
        "resolution_level": 1,
        "step_deg": 18.0,
        "mock_openmvs": False
    }
    """
    input_dir = job.get("input_dir", "./input_images")
    output_dir = job.get("output_dir", "./output")
    rig_mode = job.get("rig_mode", False)
    mock_openmvs = job.get("mock_openmvs", False)

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    run_log_handler = attach_run_log(out_path)

    # Large intermediate files (undistorted images, depth maps) go to a fast work directory when one is
    # given: a Windows folder bind-mounted through Docker Desktop/WSL is several times slower.
    work_dir = job.get("work_dir") or os.environ.get("TURNTABLE_WORK_DIR") or None
    work_path = Path(work_dir).resolve() if work_dir else out_path
    work_path.mkdir(parents=True, exist_ok=True)

    # Initialize Telemetry
    telemetry = SystemTelemetry(output_dir=str(out_path))
    telemetry.on_step_done = job.get("on_step_done")      # runpod_job.py: progress.json for the scanner
    telemetry.start_sampling(interval_sec=0.2)
    if job.get("stream_info"):
        stream_meta = dict(job["stream_info"])
        if job.get("capture_done_at") and job.get("started_at") and "time_until_capture_done_s" not in stream_meta:
            stream_meta["time_until_capture_done_s"] = round(float(job["capture_done_at"]) - float(job["started_at"]), 2)
        telemetry.set_info(stream=stream_meta)

    resources = describe_resources()
    workers = resolve_workers(job.get("workers"))
    telemetry.set_info(resources=resources, workers=workers, work_dir=str(work_path))
    telemetry.watch(str(out_path), str(work_path))
    # Features for predicting RAM / time / disk of future scans (tools/collect_stats.py)
    disk = {"work": run_metrics.disk_benchmark(str(work_path))}
    if work_path != out_path:
        disk["output"] = run_metrics.disk_benchmark(str(out_path))
    telemetry.set_info(
        version=run_metrics.code_version(),
        hardware={"cpu": run_metrics.cpu_info(), "gpus": run_metrics.gpu_info(), "host": run_metrics.host_info()},
        disk=disk,
        settings={k: job.get(k) for k in (
            "rig_mode", "resolution_level", "texturing_engine", "data_term", "estimate_lift", "lift_mm",
            "tag_refine", "tag_scale", "intrinsics_in", "target_low_poly_faces", "target_high_poly_faces",
            "high_poly_ratio", "low_poly_ratio", "low_poly_max_faces", "title", "export_scale", "glb_scale", "keep_depth_maps",
            "mesh_texture_masks", "colour_normalize", "resume", "mock_openmvs", "ram_margin", "visual_hull",
            "visual_hull_refine", "visual_hull_voxel_mm", "densify_masks", "glb_draco", "rig_refine", "mask_method", "ai_view_stride")},
    )

    logger.info("=" * 70)
    logger.info("          STARTING TURNTABLE STUDIO 3D SCANNING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Input Directory  : {input_dir}")
    logger.info(f"Output Directory : {output_dir}")
    logger.info(f"Rig Mode Enabled : {rig_mode}")
    logger.info(f"Work Directory   : {work_path}")
    logger.info(
        f"Resources        : {resources['cpus']} CPUs (host {resources['host_cpus']}), "
        f"{resources['ram_total_gb']} GB RAM (host {resources['host_ram_gb']} GB, "
        f"cgroup limit {resources['ram_cgroup_limit_gb']}), {workers} worker(s)"
    )

    texturing_engine = job.get("texturing_engine", "texrecon")
    texrecon_bin = job.get("texrecon_bin") or job.get("texrecon_bin_dir")

    # If texrecon is requested, verify availability or fall back to OpenMVS TextureMesh
    if texturing_engine == "texrecon":
        detected_tr = find_texrecon_binary(custom_dir=texrecon_bin)
        if not detected_tr and not job.get("mock_openmvs"):
            logger.warning("texrecon executable not detected on system. Defaulting to OpenMVS texturing engine.")
            texturing_engine = "openmvs"
        elif detected_tr:
            logger.info(f"Using mvs-texturing (texrecon): {detected_tr}")

    if job.get("mask_method", "fused") not in MASK_METHODS:
        raise ValueError(f"mask_method must be one of {MASK_METHODS}, got '{job.get('mask_method')}'")
    densify_masks = job.get("densify_masks", "blackout")
    if densify_masks not in DENSIFY_MASK_MODES:
        raise ValueError(f"densify_masks must be one of {DENSIFY_MASK_MODES}, got '{densify_masks}'")

    data_term = job.get("data_term", "gmi")
    if data_term not in TEXRECON_DATA_TERMS:
        raise ValueError(f"data_term must be one of {TEXRECON_DATA_TERMS} (texrecon rejects anything else), got '{data_term}'")

    layout = job.get("tag_layout") or TurntableConfig.tag_layout

    # Build Configuration
    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        rig_mode=rig_mode,
        texturing_engine=texturing_engine,
        texrecon=TexReconConfig(
            bin_dir=texrecon_bin,
            data_term=data_term,
            smoothness_term=job.get("smoothness_term", "potts"),
            outlier_removal=job.get("outlier_removal", "gauss_damping"),
            num_threads=job.get("num_threads") or resources["cpus"],
            skip_global_seam_leveling=not bool(job.get("texrecon_global_seam_leveling", 0)),
            keep_unseen_faces=bool(job.get("keep_unseen_faces", True)),
            write_intermediate_results=bool(job.get("write_intermediate_results", False)),
        ),
        texturing_images=TexturingImagesConfig(
            enabled=bool(job.get("mesh_texture_masks", True)),
            margin_px=int(job.get("texture_mask_margin_px", 2)),
            trim_band_px=int(job.get("texture_mask_trim_band_px", 25)),
            colour_normalize=bool(job.get("colour_normalize", True)),
            workers=workers,
        ),
        openmvs=OpenMVSConfig(
            bin_dir=job.get("openmvs_bin_dir"),
            resolution_level=job.get("resolution_level", 1),
            smooth_mesh=job.get("smooth_mesh", 2),
            min_point_distance=float(job.get("min_point_distance", 0.5)),
            min_resolution=job.get("min_resolution", 640),
            texture_resolution=job.get("texture_resolution", 4096),
            empty_color=job.get("empty_color", 0),
            cost_smoothness_ratio=job.get("cost_smoothness_ratio", 1.0),
            outlier_threshold=job.get("outlier_threshold", 0.0),
            global_seam_leveling=job.get("global_seam_leveling", 0),
            local_seam_leveling=job.get("local_seam_leveling", 1),
            refine_mesh_enabled=bool(job.get("refine_mesh", False)),
            refine_resolution_level=int(job.get("refine_resolution_level", 1)),
            max_threads=resources["cpus"],
            keep_depth_maps=bool(job.get("keep_depth_maps", False)),
        ),
        turntable=turntable_config(job),
        cleaning=MeshCleaningConfig(
            slice_y_m=job.get("slice_y_m", 0.003),
            skirt_bottom_y_m=job.get("skirt_bottom_y_m", 0.000),
            max_radius_m=job.get("max_radius_m", 0.12 if (layout and "260" in str(layout)) else 0.160),
            rim_radius_m=job.get("rim_radius_m", 0.115 if (layout and "260" in str(layout)) else 0.155),
            high_poly_ratio=float(job.get("high_poly_ratio", 0.75)),
            low_poly_ratio=float(job.get("low_poly_ratio", 0.2)),
            low_poly_max_faces=int(job.get("low_poly_max_faces", 50000)),
            target_high_poly_faces=int(job.get("target_high_poly_faces") or 0),
            target_low_poly_faces=int(job.get("target_low_poly_faces") or 0),
        ),
        masking=masking_config(job),
        rig_refine=RigRefineConfig(enabled=bool(job.get("rig_refine", True))),
        visual_hull=VisualHullConfig(
            enabled=bool(job.get("visual_hull", True)),
            refine_cloud=bool(job.get("visual_hull_refine", True)),
            voxel_m=float(job.get("visual_hull_voxel_mm", 1.5)) / 1000.0 if job.get("visual_hull_voxel_mm") is not None else 0.0015,
        ),
        export_scale=job.get("export_scale", 1000.0),
        glb_scale=job.get("glb_scale", 100.0),
        scale_stl_to_mm=job.get("scale_stl_to_mm", True),
        glb_draco=bool(job.get("glb_draco", True)),
        stream_cache_dir=job.get("stream_cache") or None,
    )

    bg: Optional[LoggingProcessPool] = None
    debug_parts: Optional[threading.Thread] = None
    early_exports: Optional[EarlyExports] = None
    out_zip: Dict[str, OutputZip] = {}
    if job.get("debug_lite_parts"):
        Path(job["debug_lite_parts"]).unlink(missing_ok=True)       # never the parts of an earlier run
    try:
        # ----------------------------------------------------------------------
        # STEP 1: Scan Images and Generate CameraInit SFM
        # ----------------------------------------------------------------------
        with telemetry.step("1. Camera Discovery & Init"):
            views, intrinsics, name_regex = scan_input_directory(
                input_dir=config.input_dir,
                rig_mode=config.rig_mode,
                default_focal_mm=job.get("focal_length_mm", 3.9),
            )
            config.turntable.name_regex = name_regex

            camera_init_sfm = str(out_path / "cameraInit.sfm")
            generate_camera_init_sfm(views, intrinsics, camera_init_sfm)
        inputs = run_metrics.input_stats(camera_init_sfm, name_regex)
        telemetry.set_info(inputs=inputs)
        n_levels = int((inputs or {}).get("n_levels") or 1)
        n_images = int((inputs or {}).get("n_images") or len(views))
        if job.get("visual_hull_voxel_mm") is None and config.visual_hull.enabled:
            config.visual_hull.voxel_m = 0.0025 if n_images >= 80 else 0.0015
            logger.info(f"Visual hull voxel size: {config.visual_hull.voxel_m * 1000:.1f} mm "
                        f"({'adaptive for ' + str(n_images) + ' views' if n_images >= 80 else 'standard'}).")
        if config.turntable.estimate_lift is None:
            config.turntable.estimate_lift = n_levels >= 2
            logger.info(f"Lift per level: {'estimated from the photos' if n_levels >= 2 else 'not used'} "
                        f"({n_levels} level(s)).")

        # ----------------------------------------------------------------------
        # STEP 2: Calibrate and Lock Poses via TurntablePoses
        # ----------------------------------------------------------------------
        cameras_known_sfm = str(out_path / "cameras_known.sfm")
        resume_poses = bool(job.get("resume", False)) and Path(cameras_known_sfm).is_file()
        with telemetry.step("2. Turntable Poses (Rigid Lock)"):
            if not mock_openmvs and resume_poses:
                # The resumed mesh was reconstructed with these exact poses: recomputing them with
                # other options (e.g. tag_refine) would misalign texture and geometry.
                logger.info(f"[RESUME] Reusing locked poses from {cameras_known_sfm}.")
            elif not mock_openmvs:
                run_turntable_poses(
                    camera_init_sfm=camera_init_sfm,
                    output_known_sfm=cameras_known_sfm,
                    config=config.turntable,
                    report_path=str(out_path / "turntable_report.txt"),
                    calib_out_path=str(out_path / "turntable_calib.json"),
                )
            else:
                logger.info("[MOCK] Simulating TurntablePoses output.")
                # Write minimal mock cameras_known.sfm
                with open(camera_init_sfm, "r") as f:
                    mock_data = json.load(f)
                mock_data["poses"] = []
                for v in mock_data["views"]:
                    mock_data["poses"].append({
                        "poseId": v["poseId"],
                        "pose": {
                            "transform": {
                                "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                                "center": [0.0, 0.05, 0.35],
                            }
                        }
                    })
                with open(cameras_known_sfm, "w") as f:
                    json.dump(mock_data, f)

        if not mock_openmvs and (out_path / "turntable_report.txt").is_file():
            pose_q = run_metrics.pose_quality(str(out_path / "turntable_report.txt"))
            telemetry.set_info(pose_quality=pose_q)
            _check_tag_corrections(pose_q, config, telemetry)
            _check_lift(pose_q, config.turntable, n_levels, telemetry)

        # The undistorted photos need only the intrinsics: beside the rig refinement and the masks (B1)
        early_undistort = None if mock_openmvs else _start_early_undistort(
            cameras_known_sfm, str(work_path / "undistorted"), workers)

        # ----------------------------------------------------------------------
        # STEP 2.1: Rigid per-(camera, level) correction from sparse object features (rig_refine.py)
        # ----------------------------------------------------------------------
        if config.rig_refine.enabled and not mock_openmvs and not resume_poses:
            _refine_rig(cameras_known_sfm, config, out_path, workers, telemetry)

        # ----------------------------------------------------------------------
        # STEP 2.5: Automated Hybrid Background Matting & Turntable Elimination
        # ----------------------------------------------------------------------
        masks_out_dir = str(out_path / "masks")
        mask_hull: Optional[Dict[str, Any]] = None
        # Streaming: the AI outputs of the crops computed while the photos arrived (same scan) are kept
        ai_cache_dir = stream_cache.ai_dir(config.stream_cache_dir) or str(work_path / "ai_masks_cache")
        masks_src: Optional[str] = config.masks_dir or str(Path(input_dir) / "masks")
        if not Path(masks_src).is_dir():
            masks_src = None

        if job.get("before_masks"):
            # Streaming: the AI of the levels the pod pre-computed finishes first (RAM: one set of model workers)
            job["before_masks"]()
        if config.use_masks and config.masking.enabled and config.masking.auto_generate:
            with telemetry.step("2.5. Automated Masking & Turntable Elimination"):
                # Masks of an earlier scan with more photos would otherwise be picked up by name
                stale = list(Path(masks_out_dir).rglob("*.mask.png")) if Path(masks_out_dir).is_dir() else []
                for old_mask in stale:
                    old_mask.unlink()
                if stale:
                    logger.info(f"Removed {len(stale)} mask file(s) of a previous run from {masks_out_dir}.")
                if not mock_openmvs:
                    # Model outputs of this scan only: the cache serves the second masking (wider cylinder). The
                    # streaming cache belongs to this scan already (keyed on photo + crop box): kept
                    if not config.stream_cache_dir:
                        shutil.rmtree(ai_cache_dir, ignore_errors=True)
                    n_masks, mask_hull = _generate_masks(cameras_known_sfm, masks_out_dir, config, workers, telemetry,
                                                         ai_cache_dir)
                    masks_src = masks_out_dir
                    logger.info(f"Generated {n_masks} foreground masks in {masks_out_dir}.")
                else:
                    Path(masks_out_dir).mkdir(parents=True, exist_ok=True)
                    masks_src = masks_out_dir

        # ----------------------------------------------------------------------
        # STEP 2.6: Visual hull from the masks (proxy-point bounds; dense-cloud correction in 4.2b)
        # ----------------------------------------------------------------------
        hull = None
        hull_path = work_path / "visual_hull.npz"
        proxy_bounds = None
        if config.visual_hull.enabled and not mock_openmvs and masks_src and Path(masks_src).is_dir():
            hull = _carve_visual_hull(cameras_known_sfm, masks_src, config, str(hull_path), telemetry, mask_hull)
            if hull is not None and masks_src == masks_out_dir and _hull_touches_cylinder(hull, config):
                # The object reaches the masking cylinder (off centre, or wider than it): masks and hull are cut.
                # Once, with the cylinder out to the disc rim.
                old_r = config.masking.cylinder_radius_m
                config.masking.cylinder_radius_m = config.masking.wide_cylinder_radius_m
                logger.warning(f"The object reaches the {old_r * 1000:.0f} mm masking cylinder (off centre or too "
                               f"wide?): masks and visual hull again with {config.masking.cylinder_radius_m * 1000:.0f} mm.")
                telemetry.set_info(masking_cylinder_widened={"from_mm": old_r * 1000,
                                                             "to_mm": config.masking.cylinder_radius_m * 1000})
                with telemetry.step("2.5b Masks Again (Wide Cylinder)"):
                    for old_mask in Path(masks_out_dir).rglob("*.mask.png"):
                        old_mask.unlink()
                    _, mask_hull = _generate_masks(cameras_known_sfm, masks_out_dir, config, workers, telemetry,
                                                   ai_cache_dir)
                hull = _carve_visual_hull(cameras_known_sfm, masks_src, config, str(hull_path), telemetry, mask_hull)
            if masks_src == masks_out_dir and config.masking.excluded_views:
                repaired = _repair_excluded_masks(cameras_known_sfm, masks_out_dir, hull, config)
                telemetry.set_info(masks_left_out_of_hull={"views": list(config.masking.excluded_views),
                                                           "masks_with_hull_silhouette": repaired})
            _check_hull_cone_only(hull, config, telemetry)
            proxy_bounds = visual_hull.proxy_bounds_from_hull(
                hull, config.masking.cylinder_radius_m, config.masking.cylinder_y_max_m,
                config.visual_hull.proxy_margin_m,
            )

        # ----------------------------------------------------------------------
        # STEP 3: Convert SFM to COLMAP & Undistort Images/Masks
        # ----------------------------------------------------------------------
        colmap_dir = str(work_path / "colmap")
        undistorted_dir = str(work_path / "undistorted")

        images_ready = _join_early_undistort(early_undistort, telemetry)
        with telemetry.step("3. Bridge SFM & Image Undistortion"):
            bridge = convert_sfm_to_colmap(
                sfm_json_path=cameras_known_sfm,
                output_colmap_dir=colmap_dir,
                undistorted_images_dir=undistorted_dir,
                masks_src_dir=masks_src if masks_src and Path(masks_src).is_dir() else None,
                masks_dst_dir=masks_out_dir if masks_src and Path(masks_src).is_dir() else None,
                workers=workers,
                proxy_bounds=proxy_bounds,
                blackout_background=config.masking.densify_masks == "blackout",
                images_ready=images_ready,
            )
        telemetry.set_info(bridge={"masks_applied": bridge.get("masks_applied"), "proxy_bounds": proxy_bounds,
                                   "densify_masks": config.masking.densify_masks})
        # Metrics for stats.json, the object card, the low-poly decimation and the print exports run in worker
        # processes beside the main steps (results collected before the export / the final stats.json)
        bg = LoggingProcessPool(max(1, min(3, int(resources["cpus"]) - 1)))
        bg.submit("object_images", run_metrics.object_image_stats, undistorted_dir)

        # ----------------------------------------------------------------------
        # STEP 4: OpenMVS Dense Reconstruction
        # ----------------------------------------------------------------------
        openmvs_dir = work_path / "openmvs"
        openmvs_dir.mkdir(parents=True, exist_ok=True)
        runner = OpenMVSRunner(config=config.openmvs, work_dir=str(openmvs_dir))
        raw_reconstructed_mesh = str(openmvs_dir / "scene_mesh.ply")
        cleaned_obj_path = str(out_path / "model_cleaned_watertight.obj")
        textured_obj_path = str(openmvs_dir / "scene_textured.obj")
        textured_low_obj_path: Optional[str] = None

        resume = bool(job.get("resume", False))

        if not mock_openmvs:
            skip_densify = False
            skip_reconstruct = False

            if resume:
                existing_meshes = [
                    openmvs_dir / "scene_mesh.ply",
                    openmvs_dir / "scene_mesh.mvs",
                    openmvs_dir / "scene_dense_mesh.ply",
                ]
                if any(m.is_file() for m in existing_meshes):
                    logger.info("[RESUME] Found existing reconstructed surface mesh, skipping DensifyPointCloud & ReconstructMesh.")
                    skip_densify = True
                    skip_reconstruct = True
                elif (openmvs_dir / "scene_dense.mvs").is_file():
                    logger.info("[RESUME] Found existing scene_dense.mvs point cloud, skipping DensifyPointCloud.")
                    skip_densify = True

            if not skip_densify:
                # Fail in seconds instead of an OOM kill after 20 minutes
                ram_check = _check_densify_ram(cameras_known_sfm, config, job)
                telemetry.set_info(ram_check=ram_check)

                with telemetry.step("4.1 OpenMVS InterfaceCOLMAP"):
                    runner.run_interface_colmap(colmap_dir, undistorted_dir)

                with telemetry.step("4.2 OpenMVS DensifyPointCloud"):
                    # densify_masks "blackout" (default): DensifyPointCloud IS masked, through the background
                    # pixels set to 0 in undistorted/ by bridge_sfm. "mask_path": unchanged images, and OpenMVS
                    # reads <image stem>.mask.png from undistorted/ (A/B test, plans/arhiva/implemented/AUDIT_RECONSTRUCTIE_GENERALA.md
                    # G7). The disc left in either case is sliced at Y = 3 mm by mesh_cleaner.py.
                    effective_densify_masks = (
                        undistorted_dir
                        if config.masking.enabled and config.masking.densify_masks == "mask_path" and masks_src
                        else None
                    )
                    runner.run_densify(masks_dir=effective_densify_masks)
                # A fresh cloud: the plain-cloud copy of a previous run (work dirs persist, e.g. the Docker
                # volume) must not be taken for the original of this one by the 4.2b refinement
                stale_original = openmvs_dir / "scene_dense_mvs.ply"
                if stale_original.is_file():
                    stale_original.unlink()

            if not skip_reconstruct:
                dense_ply = openmvs_dir / "scene_dense.ply"
                refined = False
                if config.visual_hull.enabled and config.visual_hull.refine_cloud and dense_ply.is_file():
                    if hull is None and hull_path.is_file():
                        hull = visual_hull.load_hull(str(hull_path))      # resumed run
                    if hull is not None:
                        with telemetry.step("4.2b Visual Hull Cloud Refinement"):
                            debug_dir = work_path / "debug_lite"
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            refined = _refine_dense_cloud(str(dense_ply), hull, cameras_known_sfm, undistorted_dir,
                                                          config, telemetry, str(debug_dir))

                with telemetry.step("4.3 OpenMVS ReconstructMesh"):
                    try:
                        runner.run_reconstruct_mesh()
                    except Exception as rm_err:
                        if not (refined and visual_hull.restore_dense_cloud(str(dense_ply))):
                            raise
                        logger.warning(f"ReconstructMesh failed on the refined cloud ({rm_err}); "
                                       f"retrying on the plain DensifyPointCloud output.")
                        telemetry.set_info(visual_hull_refine_reverted=str(rm_err)[:300])
                        runner.run_reconstruct_mesh()

            # Identify raw reconstructed surface geometry (.ply or .obj)
            candidate_raw_meshes = [
                openmvs_dir / "scene_mesh.ply",
                openmvs_dir / "scene_mesh.obj",
                openmvs_dir / "scene_dense_mesh.ply",
                openmvs_dir / "scene_dense_mesh.obj",
            ]
            raw_reconstructed_mesh = None
            for cand in candidate_raw_meshes:
                if cand.is_file():
                    raw_reconstructed_mesh = str(cand)
                    break
            if not raw_reconstructed_mesh:
                any_meshes = list(openmvs_dir.glob("*.ply")) + list(openmvs_dir.glob("*.obj"))
                if any_meshes:
                    raw_reconstructed_mesh = str(any_meshes[0])
                else:
                    raw_reconstructed_mesh = str(openmvs_dir / "scene_mesh.ply")

            # ----------------------------------------------------------------------
            # STEP 4.5: OpenMVS RefineMesh (Photo-consistency surface regularizer)
            # ----------------------------------------------------------------------
            if config.openmvs.refine_mesh_enabled and raw_reconstructed_mesh and Path(raw_reconstructed_mesh).is_file():
                with telemetry.step("4.5 OpenMVS RefineMesh"):
                    try:
                        refined_ply = runner.run_refine_mesh(
                            input_mvs="scene_dense.mvs",
                            mesh_file=raw_reconstructed_mesh,
                            output_mvs="scene_mesh_refined.mvs",
                        )
                        if refined_ply and Path(refined_ply).is_file() and Path(refined_ply).stat().st_size > 0:
                            logger.info(f"Using refined mesh geometry: {refined_ply}")
                            raw_reconstructed_mesh = refined_ply
                    except Exception as rf_err:
                        logger.warning(f"RefineMesh failed ({rf_err}); continuing with raw reconstructed mesh.")

            bg.submit("openmvs", run_metrics.openmvs_log_stats, str(openmvs_dir))
            bg.submit("raw_mesh", run_metrics.mesh_quality, raw_reconstructed_mesh, "raw")
            if job.get("debug_lite_parts"):
                # Large parts of the debug archive (raw mesh, masks, hull) while the mesh is cleaned and textured
                debug_parts = _start_debug_parts(str(out_path), str(work_path), job["debug_lite_parts"])

            # ----------------------------------------------------------------------
            # STEP 5: Watertight Base Slicing & Mesh Cleaning (BEFORE Texturing!)
            # ----------------------------------------------------------------------
            with telemetry.step("5. Watertight Base Slicing & Pedestal Sealing"):
                if hull is None and hull_path.is_file():
                    hull = visual_hull.load_hull(str(hull_path))      # resumed run
                hook = _hull_patch_hook(hull, config)
                clean_res = clean_and_repair_mesh(
                    input_obj_path=raw_reconstructed_mesh,
                    output_obj_path=cleaned_obj_path,
                    config=config.cleaning,
                    patch_hook=hook,
                    binary_copies=True,       # PLY for texrecon, arrays for the texturing images (B5)
                )
            _report_cleaning(clean_res, hook, telemetry)

            _resolve_low_poly_target(config, clean_res, telemetry)
            # Dedicated low-poly geometry for branching texturing (eliminates white untextured low-poly!), the print
            # exports of the high poly, its metrics and object.json: in the workers, while the texturing images are made
            cleaned_low_path = out_path / "cleaned_mesh_low.obj"
            if cleaned_low_path.is_file():
                cleaned_low_path.unlink()
            bg.submit("decimate_low", decimate_mesh, cleaned_obj_path, str(cleaned_low_path),
                      config.cleaning.target_low_poly_faces)
            print_tmp = work_path / "print_high"
            print_tmp.mkdir(parents=True, exist_ok=True)
            bg.submit("print_high", export_print_assets, cleaned_obj_path, str(print_tmp / "high_model.stl"),
                      str(print_tmp / "high_model.3mf"), float(config.export_scale))
            bg.submit("cleaned_mesh", run_metrics.mesh_quality, cleaned_obj_path, "cleaned")
            bg.submit("object_info", write_object_info, cleaned_obj_path, str(out_path), job.get("title"),
                      job.get("description"), _object_extra(job, clean_res, config))

            # Determine appropriate MVS project file for TextureMesh
            mvs_project = "scene_mesh.mvs"
            if not (openmvs_dir / "scene_mesh.mvs").is_file():
                if (openmvs_dir / "scene_dense.mvs").is_file():
                    mvs_project = "scene_dense.mvs"
                else:
                    mvs_project = "scene.mvs"

            # ----------------------------------------------------------------------
            # STEP 5.5: Texturing images from the cleaned mesh (texrecon only)
            #   lossless PNG, background exactly 0, mesh-silhouette masks, per-camera colour gains
            # ----------------------------------------------------------------------
            texturing_images_dir = undistorted_dir
            if config.texturing_engine == "texrecon" and config.texturing_images.enabled:
                ti = config.texturing_images
                try:
                    with telemetry.step("5.5 Texturing Images (Mesh Masks, PNG, Colour Gains)"):
                        prep = prepare_texturing_images(
                            sfm_json_path=cameras_known_sfm,
                            mesh_path=cleaned_obj_path,
                            output_dir=str(out_path / "texturing_images"),
                            margin_px=ti.margin_px,
                            trim_band_px=ti.trim_band_px,
                            brightness_threshold=ti.brightness_white_threshold,
                            chroma_threshold=ti.chroma_threshold,
                            colour_normalize=ti.colour_normalize,
                            max_gain_deviation=ti.max_gain_deviation,
                            tag_layout_path=config.turntable.tag_layout,
                            workers=ti.workers,
                            mesh_arrays=clean_res.get("arrays_path"),
                        )
                        texturing_images_dir = prep["images_dir"]
                except Exception as prep_err:
                    logger.warning(
                        f"Texturing image preparation failed ({prep_err}). "
                        f"texrecon will use the bridge images in {undistorted_dir}."
                    )

            try:
                bg.result("decimate_low", raise_errors=True)
            except Exception as dec_err:
                logger.warning(f"Pre-texturing low-poly decimation failed ({dec_err}). Will fall back to standard decimation.")
            if not cleaned_low_path.is_file():
                cleaned_low_path = None

            # ----------------------------------------------------------------------
            # STEP 4.4: Texture Mesh on Watertight Sealed Geometry
            # ----------------------------------------------------------------------
            if config.texturing_engine == "texrecon":
                try:
                    early_exports = EarlyExports(str(out_path), cleaned_obj_path, config, bg)
                    with telemetry.step("4.4 mvs-texturing (texrecon)"):
                        textured_obj_path, textured_low_obj_path = _run_texrecon(
                            config, cameras_known_sfm, cleaned_obj_path, cleaned_low_path, texturing_images_dir,
                            out_path, job.get("mock_openmvs", False), telemetry, early_exports,
                            clean_res.get("ply_path"))
                except Exception as tr_err:
                    logger.warning(
                        f"mvs-texturing failed ({tr_err}). Falling back to OpenMVS texturing."
                    )
                    with telemetry.step("4.4 OpenMVS TextureMesh (Fallback)"):
                        effective_mask_dir = masks_out_dir if masks_src and Path(masks_out_dir).is_dir() else None
                        _, textured_obj_path = runner.run_texture_mesh(
                            input_mvs=mvs_project,
                            mesh_file=cleaned_obj_path,
                            output_mvs="scene_texture.mvs",
                            masks_dir=effective_mask_dir,
                        )
                        if cleaned_low_path and cleaned_low_path.is_file():
                            try:
                                _, textured_low_obj_path = runner.run_texture_mesh(
                                    input_mvs=mvs_project,
                                    mesh_file=str(cleaned_low_path),
                                    output_mvs="scene_texture_low.mvs",
                                    masks_dir=effective_mask_dir,
                                )
                            except Exception as omvs_low_err:
                                logger.warning(f"OpenMVS fallback low-poly texturing failed ({omvs_low_err}).")
            else:
                with telemetry.step("4.4 OpenMVS TextureMesh (Watertight)"):
                    effective_mask_dir = masks_out_dir if masks_src and Path(masks_out_dir).is_dir() else None
                    _, textured_obj_path = runner.run_texture_mesh(
                        input_mvs=mvs_project,
                        mesh_file=cleaned_obj_path,
                        output_mvs="scene_texture.mvs",
                        masks_dir=effective_mask_dir,
                    )
                    if cleaned_low_path and cleaned_low_path.is_file():
                        try:
                            logger.info("Running branching texturing on low-poly mesh (OpenMVS)...")
                            _, textured_low_obj_path = runner.run_texture_mesh(
                                input_mvs=mvs_project,
                                mesh_file=str(cleaned_low_path),
                                output_mvs="scene_texture_low.mvs",
                                masks_dir=effective_mask_dir,
                            )
                        except Exception as omvs_low_err:
                            logger.warning(f"OpenMVS low-poly texturing failed ({omvs_low_err}).")

        else:
            logger.info("[MOCK] Creating mock OpenMVS geometry.")
            import trimesh
            R_mock = trimesh.transformations.rotation_matrix(1.5707963, [1, 0, 0])
            cylinder = trimesh.creation.cylinder(radius=0.03, height=0.10, transform=R_mock)
            cylinder.apply_translation([0.0, 0.03, 0.0])
            cylinder.export(raw_reconstructed_mesh)

            with telemetry.step("5. Watertight Base Slicing & Pedestal Sealing"):
                clean_res = clean_and_repair_mesh(
                    input_obj_path=raw_reconstructed_mesh,
                    output_obj_path=cleaned_obj_path,
                    config=config.cleaning,
                )
            _report_cleaning(clean_res, None, telemetry)
            _resolve_low_poly_target(config, clean_res, telemetry)
            write_object_info(cleaned_obj_path, str(out_path), job.get("title"), job.get("description"),
                              _object_extra(job, clean_res, config))

            cleaned_low_path = out_path / "cleaned_mesh_low.obj"
            try:
                decimate_mesh(
                    input_mesh_path=cleaned_obj_path,
                    output_mesh_path=str(cleaned_low_path),
                    target_faces=config.cleaning.target_low_poly_faces,
                )
            except Exception as dec_err:
                logger.warning(f"Mock decimation failed: {dec_err}")
                cleaned_low_path = None

            if config.texturing_engine == "texrecon":
                tr_runner = TexReconRunner(
                    config=config.texrecon,
                    work_dir=str(out_path / "texrecon"),
                )
                textured_obj_path, _ = tr_runner.run_texturing(
                    sfm_data_path=cameras_known_sfm,
                    input_mesh_path=cleaned_obj_path,
                    images_folder=undistorted_dir,
                    out_prefix=str(out_path / "texrecon" / "model_textured"),
                    mock_mode=True,
                )
                if cleaned_low_path and cleaned_low_path.is_file():
                    textured_low_obj_path, _ = tr_runner.run_texturing(
                        sfm_data_path=cameras_known_sfm,
                        input_mesh_path=str(cleaned_low_path),
                        images_folder=undistorted_dir,
                        out_prefix=str(out_path / "texrecon" / "model_textured_low"),
                        mock_mode=True,
                    )
            else:
                textured_obj_path = cleaned_obj_path
                textured_low_obj_path = str(cleaned_low_path) if cleaned_low_path and cleaned_low_path.is_file() else None

        # ----------------------------------------------------------------------
        # STEP 6: Multi-LOD Asset Packaging & Cloud Archive (output.zip)
        # ----------------------------------------------------------------------
        # Save preliminary stats.json so it is included in output.zip
        stats_file = telemetry.save_json()

        with telemetry.step("6. Multi-LOD Export & Compression"):
            print_high = bg.result("print_high") if bg is not None and bg.has("print_high") else None
            bundle = export_multi_lod_bundle(
                cleaned_mesh_path=cleaned_obj_path,
                textured_obj_path=textured_obj_path,
                output_dir=str(out_path),
                telemetry=telemetry,
                config=config,
                stats_json_path=stats_file,
                textured_low_obj_path=textured_low_obj_path,
                make_zip=False,                  # built once, at the very end, with the final stats.json and logs
                pool=bg,
                print_high=print_high,
                # the deliverables go into output.zip while their metrics are computed (B4)
                on_files_ready=lambda: _start_output_zip(out_zip, str(out_path), config.zip_name),
                early=early_exports,
            )
        if bg is not None:
            done = bg.results()
            telemetry.set_info(**{k: done.get(k) for k in ("object_images", "openmvs", "raw_mesh", "cleaned_mesh")
                                  if k in done})
            bg.shutdown()
            bg = None

        logger.info("=" * 70)
        logger.info("       PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

        # Stop telemetry sampling and finalize report
        telemetry.stop_sampling()
        run_log_handler.flush()
        telemetry.set_info(log_counters=run_metrics.log_counters(str(out_path / "run_log.txt")))
        if config.stream_cache_dir or job.get("stream_info"):
            stream_meta = dict(job.get("stream_info") or {})
            now = time.time()
            if job.get("capture_done_at"):
                t_after = round(now - float(job["capture_done_at"]), 2)
                stream_meta["time_after_capture_done_s"] = t_after
                stream_meta["wait_after_capture_s"] = t_after
            if job.get("started_at"):
                stream_meta["total_time_s"] = round(now - float(job["started_at"]), 2)
            if config.stream_cache_dir:
                stream_meta["cache_hits"] = stream_cache.stats()
            telemetry.set_info(stream=stream_meta)
        telemetry.set_outcome("ok")
        stats_file = telemetry.save_json()
        telemetry.print_table()

        # output.zip closed last, so it carries the final stats.json and the complete logs
        run_log_handler.flush()
        if not out_zip:
            _start_output_zip(out_zip, str(out_path), config.zip_name, run_async=False)
        zip_path = out_zip.pop("zip").finish(
            stats_file,
            collect_log_files(str(out_path), extra_dirs=[str(openmvs_dir)] if work_path != out_path else None),
        )
        bundle["zip_path"] = str(zip_path)
        bundle["zip_size_mb"] = round(zip_path.stat().st_size / (1024 * 1024), 2)
        telemetry.set_info(output_zip_mb=bundle["zip_size_mb"])
        stats_file = telemetry.save_json()
        _join_debug_parts(debug_parts)
        detach_run_log(run_log_handler)

        return {
            "status": "SUCCESS",
            "stats_json": stats_file,
            "bundle": bundle,
            "telemetry": telemetry.summary(),
            "warnings": telemetry.warnings,
        }

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        if bg is not None:
            bg.shutdown(cancel=True)
        if out_zip:                       # no half-written output.zip next to a failed run
            z = out_zip.pop("zip")
            try:
                z.finish()
            except Exception:
                pass
            z.path.unlink(missing_ok=True)
        _join_debug_parts(debug_parts)
        telemetry.stop_sampling()
        telemetry.set_outcome("error", error_type=type(e).__name__, error=str(e)[:500],
                              failed_step=telemetry.current_step)
        run_log_handler.flush()
        telemetry.set_info(log_counters=run_metrics.log_counters(str(out_path / "run_log.txt")))
        telemetry.save_json()
        telemetry.print_table()
        detach_run_log(run_log_handler)
        raise


def _start_early_undistort(cameras_known_sfm: str, undistorted_dir: str, workers: int) -> Dict[str, Any]:
    """bridge_sfm.undistort_rgb_images in a thread (B1); the sfm is read now, before rig_refine rewrites it."""
    state: Dict[str, Any] = {}
    try:
        with open(cameras_known_sfm, "r", encoding="utf-8") as f:
            sfm = json.load(f)
    except Exception as e:
        logger.warning(f"Early undistortion not started ({e}).")
        return state

    def run() -> None:
        t0 = time.time()
        try:
            state["images"] = undistort_rgb_images(sfm, undistorted_dir, max(1, workers // 2))
        except Exception as e:
            state["error"] = str(e)[:200]
        state["seconds"] = round(time.time() - t0, 2)

    state["thread"] = threading.Thread(target=run, name="undistort-rgb", daemon=True)
    state["thread"].start()
    return state


def _join_early_undistort(state: Optional[Dict[str, Any]], telemetry: SystemTelemetry) -> bool:
    """True when the undistorted photos are all written (otherwise step 3 writes them as before)."""
    if not state or "thread" not in state:
        return False
    t0 = time.time()
    state["thread"].join()
    info = {k: state.get(k) for k in ("images", "seconds", "error") if state.get(k) is not None}
    info["waited_s"] = round(time.time() - t0, 2)
    telemetry.set_info(undistort_early=info)
    if state.get("error"):
        logger.warning(f"Early undistortion failed ({state['error']}): done in step 3.")
        return False
    return True


def _start_output_zip(holder: Dict[str, OutputZip], out_dir: str, zip_name: str, run_async: bool = True) -> None:
    """output.zip opened and, in a thread, filled with the deliverables already written (OutputZip)."""
    z = OutputZip(out_dir, zip_name)
    holder["zip"] = z
    if run_async:
        z.add_deliverables_async()


def _start_debug_parts(out_dir: str, work_dir: str, parts_path: str) -> threading.Thread:
    """debug_lite.build_debug_parts in a thread (B2); a failure only means runpod_job builds them at the end."""
    from pipeline.debug_lite import build_debug_parts

    def run() -> None:
        try:
            info = build_debug_parts(out_dir, work_dir, parts_path)
            logger.info(f"Debug archive parts ready in the background: {info['parts']} part(s), {info['seconds']} s.")
        except Exception as e:
            logger.warning(f"Debug archive parts not built during the run ({e}).")
            Path(parts_path).unlink(missing_ok=True)

    t = threading.Thread(target=run, name="debug-parts", daemon=True)
    t.start()
    return t


def _join_debug_parts(t: Optional[threading.Thread]) -> None:
    if t is not None and t.is_alive():
        t.join(timeout=300)


def _run_texrecon(config: PipelineConfig, cameras_known_sfm: str, cleaned_obj_path: str,
                  cleaned_low_path: Optional[Path], images_dir: str, out_path: Path, mock: bool,
                  telemetry: SystemTelemetry, early: Optional[EarlyExports] = None,
                  cleaned_ply: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    `cleaned_ply`: the cleaned mesh already converted to PLY (B5).
    `early` (EarlyExports): the export of each model starts as soon as its texturing is done (B3).
    Step 4.4: texrecon on the high- and the low-poly mesh. The .cam scene is written once and both run at the same
    time (the low poly in texrecon/low/: texrecon refuses a "tmp" folder it did not make) when the RAM allows; the
    low-poly run takes ~20-45 s that no longer add up. A failed high-poly run raises (TextureMesh fallback); a failed
    low-poly run only warns. Returns (high OBJ, low OBJ or None).
    """
    tr_runner = TexReconRunner(config=config.texrecon, work_dir=str(out_path / "texrecon"))
    tr_runner.prepare_scene(cameras_known_sfm, images_dir)

    def started(kind: str, path: str) -> str:
        if early is not None:
            try:
                (early.start_high if kind == "high" else early.start_low)(path)
            except Exception as e:      # the export at the end does it then
                logger.warning(f"Early {kind}-poly export not started ({e}).")
        return path

    def high() -> str:
        return started("high", tr_runner.run_texturing(
            sfm_data_path=cameras_known_sfm, input_mesh_path=cleaned_obj_path, images_folder=images_dir,
            out_prefix=str(out_path / "texrecon" / "model_textured"), mock_mode=mock, scene_ready=True,
            mesh_ply=cleaned_ply)[0])

    def low() -> str:
        logger.info("Running branching texturing on low-poly mesh (texrecon)...")
        return started("low", tr_runner.run_texturing(
            sfm_data_path=cameras_known_sfm, input_mesh_path=str(cleaned_low_path), images_folder=images_dir,
            out_prefix=str(out_path / "texrecon" / "low" / "model_textured_low"), mock_mode=mock,
            scene_ready=True)[0])

    if not (cleaned_low_path and Path(cleaned_low_path).is_file()):
        return high(), None
    free_gb = float(describe_resources().get("ram_free_gb") or 0.0)
    parallel = config.texrecon.parallel_low and free_gb >= config.texrecon.parallel_min_free_gb
    telemetry.set_info(texrecon_parallel={"parallel": parallel, "free_ram_gb": round(free_gb, 1)})
    if not parallel:
        high_obj = high()
        try:
            return high_obj, low()
        except Exception as tr_low_err:
            logger.warning(f"texrecon low-poly texturing failed ({tr_low_err}).")
            return high_obj, None
    with ThreadPoolExecutor(max_workers=2) as ex:       # texrecon is a subprocess: threads are enough
        f_high, f_low = ex.submit(high), ex.submit(low)
        try:
            low_obj = f_low.result()
        except Exception as tr_low_err:
            logger.warning(f"texrecon low-poly texturing failed ({tr_low_err}).")
            low_obj = None
        return f_high.result(), low_obj


def _check_lift(pose_q: Optional[Dict[str, Any]], turntable: TurntableConfig, n_levels: int,
                telemetry: SystemTelemetry) -> None:
    """
    Warns when the lift estimated from the photos is far from the commanded one (steps/mm of the Z axis, lead
    screw, or a tag sheet printed at another scale: see tag_scale) or tilted from the turntable axis.
    """
    if not turntable.estimate_lift or n_levels < 2 or not pose_q or pose_q.get("lift_mm") is None:
        return
    est, cmd = float(pose_q["lift_mm"]), float(turntable.lift_mm)
    angle = float(pose_q.get("lift_angle_deg") or 0.0)
    rel = (est - cmd) / cmd if cmd else 0.0
    telemetry.set_info(lift_check={"commanded_mm": cmd, "estimated_mm": est, "difference_pct": round(100 * rel, 1),
                                   "angle_deg": angle, "tag_scale": turntable.tag_scale})
    if abs(rel) > 0.10:
        logger.warning(f"Estimated lift {est:.1f} mm/level differs by {100 * rel:+.0f} % from the commanded "
                       f"{cmd:g} mm: check the Z axis steps/mm or the tag sheet scale (tag_scale {turntable.tag_scale}).")
    if angle > 2.0:
        logger.warning(f"Lift direction is {angle:.1f} deg off the turntable axis: is the Z axis or the disc tilted?")


def _check_tag_corrections(pose_q: Optional[Dict[str, Any]], config: PipelineConfig,
                           telemetry: SystemTelemetry) -> None:
    """
    tagRefine fits the height of every tag; a glued sheet is flat to ~1.5 mm (same corrections scan after scan).
    Several mm means the object covers or stands on that tag (orange bottle: -6.8 mm, axis tilted 2.1 deg). Only a
    warning: computing the poses again without tagRefine made the object features agree less, not more (3.09 ->
    3.63 px Sampson on that scan), and the rig refinement corrects what it can afterwards.
    """
    worst = (pose_q or {}).get("tag_correction_max_mm")
    if not config.turntable.tag_refine or worst is None or worst <= config.turntable.max_tag_correction_mm:
        return
    logger.warning(f"tagRefine moved a tag by {worst:.1f} mm (the sheet itself is flat to ~1.5 mm): the object "
                   f"probably covers or stands on it. Keep the object inside the tag ring for reliable poses.")
    telemetry.set_info(tag_correction_warning={"worst_correction_mm": worst,
                                               "limit_mm": config.turntable.max_tag_correction_mm})


def _hull_touches_cylinder(hull: Dict[str, Any], config: PipelineConfig, min_voxels: int = 20) -> bool:
    """
    True when the carved hull reaches the masking cylinder and a wider one is still available. Only voxels the
    silhouettes decided count (not the ambiguous cone over the disc, whose rim used to trigger it: lighter 153808,
    26 s for nothing), and at least `min_voxels` of them within 2 voxels of the wall (not a few stray ones).
    """
    if config.masking.wide_cylinder_radius_m <= config.masking.cylinder_radius_m:
        return False
    occ = visual_hull.decided_occupancy(hull)
    if not occ.any():
        return False
    v = hull["voxel_m"]
    ii, _, kk = np.nonzero(occ)
    r = np.hypot(hull["origin"][0] + ii * v, hull["origin"][2] + kk * v)
    return int((r >= config.masking.cylinder_radius_m - 2 * v).sum()) >= min_voxels


def _check_hull_cone_only(hull: Optional[Dict[str, Any]], config: PipelineConfig,
                          telemetry: SystemTelemetry) -> None:
    """Warning when the silhouettes decided nothing (the hull is only the cone over the disc of the 'ai' masks)."""
    if hull is None or hull.get("ambiguous") is None:
        return
    st = visual_hull.hull_stats(hull)
    if st.get("voxels", 0) < 100 and st.get("voxels_total", 0) >= 100:
        telemetry.add_warning(
            "hull_cone_only", "The visual hull is only the cone over the disc the masks cannot judge: the working "
            "volume is the whole masking cylinder and the object may be small or cut; check the model.")


def _generate_masks(cameras_known_sfm: str, masks_out_dir: str, config: PipelineConfig, workers: int,
                    telemetry: Optional[SystemTelemetry] = None,
                    ai_cache_dir: Optional[str] = None) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Step 2.5: one foreground mask per photo (masking cylinder + foreground + tag blackout).
    AI foreground (masking.method "ai", model available): the model runs on 1 view in ai_view_stride, the visual hull is
    carved from those masks and the other views get its silhouette. Returns (masks written, that hull or None).
    """
    m = config.masking
    with open(cameras_known_sfm, "r", encoding="utf-8") as f:
        sfm_data_mask = json.load(f)

    m_views = sfm_data_mask.get("views", [])
    m_intrinsics = {}
    for intr in sfm_data_mask.get("intrinsics", []):
        w_val = float(intr.get("width", 4056))
        h_val = float(intr.get("height", 3040))
        sensor_w = float(intr.get("sensorWidth", 6.287) or 6.287)
        focal_mm = float(intr.get("focalLength", 3.9) or 3.9)
        fx = focal_mm * (w_val / sensor_w)
        fy = fx * float(intr.get("pixelRatio", 1.0) or 1.0)
        pp = intr.get("principalPoint", [0.0, 0.0])
        cx = (w_val / 2.0) + float(pp[0]) if abs(float(pp[0])) < (w_val / 4.0) else float(pp[0])
        cy = (h_val / 2.0) + float(pp[1]) if abs(float(pp[1])) < (h_val / 4.0) else float(pp[1])
        m_intrinsics[str(intr["intrinsicId"])] = {
            "K": np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        }

    m_poses = {}
    for p in sfm_data_mask.get("poses", []):
        m_poses[str(p["poseId"])] = {
            "R": np.array(p["pose"]["transform"]["rotation"], dtype=np.float64).reshape((3, 3), order="F"),
            "C": np.array(p["pose"]["transform"]["center"], dtype=np.float64),
        }

    use_ai = m.method in ("ai", "fused") and ai_model_available(m.ai_model_path)
    if m.method in ("ai", "fused") and not use_ai:
        logger.warning(f"AI mask model not available ({m.ai_model_path} or onnxruntime missing): photometric masks.")
    ai_workers = ai_mask_workers(m, workers) if use_ai else workers

    def run(views: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
        stats: Dict[str, Any] = {}
        out = batch_generate_masks(
            views=views,
            poses_dict=m_poses,
            intrinsics_dict=m_intrinsics,
            output_masks_dir=masks_out_dir,
            cylinder_radius_m=m.cylinder_radius_m,
            cylinder_y_min_m=m.cylinder_y_min_m,
            cylinder_y_max_m=m.cylinder_y_max_m,
            chroma_threshold=m.chroma_threshold,
            brightness_white_threshold=m.brightness_white_threshold,
            safety_dilation_px=m.safety_dilation_px,
            tag_blackout=m.tag_blackout,
            tag_expansion_factor=m.tag_expansion_factor,
            workers=ai_workers if use_ai else workers,
            ai_model=m.ai_model_path if use_ai else None,
            ai_resolution=m.ai_resolution,
            ai_threads=m.ai_threads,
            ai_min_area_ratio=m.ai_min_area_ratio,
            stats=stats,
            ai_cache_dir=ai_cache_dir,
        )
        return out, stats

    fused = use_ai and m.method == "fused"
    m.ai_used = use_ai and not fused          # the "ai" masks keep the turntable; the fused ones leave it out
    m.excluded_views = []
    info: Dict[str, Any] = {"method": m.method if use_ai else "photometric",
                            "workers": ai_workers if use_ai else workers}
    hull = None
    stride = max(1, int(m.ai_view_stride)) if use_ai and not fused else 1
    if fused:
        cams = load_sfm_cameras(cameras_known_sfm)
        cam_i, lvl, _ = view_groups(cams, config.turntable.name_regex)
        stats: Dict[str, Any] = {}
        fusion_dir = Path(ai_cache_dir or masks_out_dir).parent / "mask_fusion"
        batch_fused_masks(
            cams, list(zip(cam_i.tolist(), lvl.tolist())), masks_out_dir, str(fusion_dir),
            load_tag_layout(config.turntable.tag_layout), tag_scale=config.turntable.tag_scale,
            disc_radius_m=m.disc_radius_m, cylinder_radius_m=m.cylinder_radius_m,
            cylinder_y_min_m=m.cylinder_y_min_m, cylinder_y_max_m=m.cylinder_y_max_m,
            chroma_threshold=m.chroma_threshold, brightness_white_threshold=m.brightness_white_threshold,
            tag_blackout=m.tag_blackout, tag_expansion_factor=m.tag_expansion_factor, workers=ai_workers,
            ai_model=m.ai_model_path, ai_resolution=m.ai_resolution, ai_threads=m.ai_threads,
            ai_cache_dir=ai_cache_dir, stats=stats, median_cache_dir=config.stream_cache_dir)
        info.update(views_model=len(cams), methods=stats.get("methods"), seconds=stats.get("seconds"),
                    model_s_per_view=stats.get("model_s_per_view"), model_runs=stats.get("model_runs"),
                    ai_cache_hits=stats.get("ai_cache_hits"))
        check = _check_masks_against_hull(cams, masks_out_dir, config, telemetry)
        info["hull_check"] = check
    elif stride == 1:
        written, stats = run(m_views)
        info.update(views_model=len(m_views), methods=stats.get("methods"))
    else:
        cams = load_sfm_cameras(cameras_known_sfm)
        cam_i, lvl, shot = view_groups(cams, config.turntable.name_regex)
        model_ids = {c["view_id"] for c, a, b, k in zip(cams, cam_i, lvl, shot) if (a + b + k) % stride == 0}
        written, stats = run([v for v in m_views if str(v["viewId"]) in model_ids])
        info.update(views_model=len(model_ids), methods=stats.get("methods"))
        sub = [c for c in cams if c["view_id"] in model_ids]
        vh = config.visual_hull
        hull = visual_hull.carve_visual_hull(
            sub, visual_hull.source_mask_paths(sub, masks_out_dir), radius_m=m.cylinder_radius_m,
            y_min_m=config.cleaning.slice_y_m, y_max_m=m.cylinder_y_max_m, voxel_m=vh.voxel_m,
            min_bg_votes=vh.min_bg_votes, distorted=True, disc_radius_m=m.disc_radius_m)
        rest = [c for c in cams if c["view_id"] not in model_ids]
        if visual_hull.hull_stats(hull).get("voxels_total", 0) >= 100:
            def write_silhouette(c: Dict[str, Any]) -> None:
                dst = Path(masks_out_dir) / f"{c['view_id']}_{Path(c['path']).stem}.mask.png"
                cv2.imwrite(str(dst), visual_hull.hull_silhouette(hull, c, distorted=True))

            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:   # numpy / OpenCV release the GIL
                list(ex.map(write_silhouette, rest))
            info["views_hull_silhouette"] = len(rest)
            written = dict(written, **{c["view_id"]: "hull" for c in rest})
            logger.info(f"AI masks on {len(model_ids)} views; the other {len(rest)} views get the silhouette of the "
                        f"visual hull carved from them.")
        else:
            logger.warning("Visual hull from the AI masks is empty: AI masks for the remaining views too.")
            hull = None
            more, stats2 = run([v for v in m_views if str(v["viewId"]) not in model_ids])
            written = dict(written, **more)
            info.update(views_model=len(m_views), methods_rest=stats2.get("methods"))
    if telemetry is not None:
        telemetry.set_info(masks=info)
    return len(m_views), hull


def _check_masks_against_hull(cams: List[Dict[str, Any]], masks_dir: str, config: PipelineConfig,
                              telemetry: Optional[SystemTelemetry] = None) -> Dict[str, Any]:
    """
    Every mask against a loose hull of all of them (3 mm voxels, a voxel is cut only by >= 3 views, so 1-2 wrong
    masks cannot cut it): a view whose mask covers less than masking.min_hull_agreement of that hull's silhouette
    misses part of the object the other views see. Such views go to masking.excluded_views (left out of the visual
    hull; their masks get its silhouette added in _repair_excluded_masks). Returns the per-view numbers.
    """
    m = config.masking
    try:
        paths = visual_hull.source_mask_paths(cams, masks_dir)
        loose = visual_hull.carve_visual_hull(
            cams, paths, radius_m=m.cylinder_radius_m, y_min_m=config.cleaning.slice_y_m,
            y_max_m=m.cylinder_y_max_m, voxel_m=0.003, min_bg_votes=3, distorted=True)
        agree = visual_hull.mask_agreement(loose, cams, paths)
    except Exception as e:
        logger.warning(f"Mask check against the hull failed ({e}); every mask is used.")
        return {"error": str(e)[:200]}
    bad = [(c["view_id"], a) for c, a in zip(cams, agree) if a is not None and a < m.min_hull_agreement]
    known = [a for a in agree if a is not None]
    res: Dict[str, Any] = {"agreement_median": round(float(np.median(known)), 3) if known else None,
                           "agreement_min": round(float(min(known)), 3) if known else None,
                           "views_below": len(bad)}
    if bad and len(bad) <= m.max_excluded_share * len(cams):
        m.excluded_views = [vid for vid, _ in bad]
        res["excluded"] = {vid: a for vid, a in bad}
        logger.warning(f"{len(bad)} mask(s) miss part of the object the other views see (agreement "
                       f"{', '.join(f'{a:.2f}' for _, a in bad)}): left out of the visual hull.")
    elif bad:
        res["not_excluded"] = "too many views disagree"
        if telemetry is not None:
            telemetry.add_warning("masks_disagree", f"{len(bad)} of {len(cams)} masks disagree with the others: "
                                  f"the object may be cut or padded; check the model.", views=len(bad))
    return res


def _repair_excluded_masks(cameras_known_sfm: str, masks_dir: str, hull: Optional[Dict[str, Any]],
                           config: PipelineConfig) -> int:
    """Masks of the views left out of the hull: mask OR hull silhouette, so Densify does not black out the object."""
    excluded = set(config.masking.excluded_views or [])
    if not excluded or hull is None:
        return 0
    n = 0
    for c in load_sfm_cameras(cameras_known_sfm):
        if c["view_id"] not in excluded:
            continue
        dst = Path(masks_dir) / f"{c['view_id']}_{Path(c['path']).stem}.mask.png"
        old = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE)
        if old is None:
            continue
        sil = visual_hull.hull_silhouette(hull, c, distorted=True)
        cv2.imwrite(str(dst), np.maximum(old, sil))
        for extra in (dst.parent / Path(c["path"]).parent.name / f"{Path(c['path']).stem}.mask.png",
                      dst.parent / Path(c["path"]).parent.name / f"{Path(c['path']).name}.mask.png"):
            if extra.is_file():
                cv2.imwrite(str(extra), np.maximum(old, sil))
        n += 1
    return n


def _refine_rig(cameras_known_sfm: str, config: PipelineConfig, out_path: Path, workers: int,
                telemetry: SystemTelemetry) -> None:
    """Step 2.1: rewrites cameras_known.sfm when the correction is accepted (tag-only poses kept next to it)."""
    try:
        with telemetry.step("2.1 Rig Refinement (Sparse Features)"):
            res = refine_rig_poses(
                cameras_known_sfm, config.turntable.name_regex, config.rig_refine, workers=workers,
                report_path=str(out_path / "rig_refine.json"),
                backup_sfm=str(out_path / "cameras_known_tags.sfm"),
                feature_cache=config.stream_cache_dir,
            )
        telemetry.set_info(rig_refine={k: res.get(k) for k in (
            "accepted", "reason", "holdout_before", "holdout_after", "max_rotation_deg", "max_translation_mm",
            "observations", "observations_per_image_median", "corrections", "timings_s")})
    except Exception as e:
        logger.warning(f"Rig refinement failed ({e}); continuing with the tag poses.")
        telemetry.set_info(rig_refine={"accepted": False, "reason": f"error: {str(e)[:300]}"})


def _resolve_low_poly_target(config: PipelineConfig, clean_res: Dict[str, Any], telemetry: SystemTelemetry) -> None:
    """
    Low-poly face count: min(low_poly_ratio x sealed faces, low_poly_max_faces) unless a fixed target was given. Written back into
    config.cleaning.target_low_poly_faces: the exporter's fallback decimations read the same number.
    """
    c = config.cleaning
    sealed = int(clean_res.get("sealed_faces") or clean_res.get("face_count") or 0)
    c.target_low_poly_faces = face_target(sealed, c.low_poly_ratio, c.target_low_poly_faces, minimum=500,
                                          maximum=c.low_poly_max_faces)
    telemetry.set_info(decimation={"sealed_faces": sealed, "high_faces": clean_res.get("face_count"),
                                   "low_target_faces": c.target_low_poly_faces,
                                   "high_poly_ratio": c.high_poly_ratio, "low_poly_ratio": c.low_poly_ratio,
                                   "low_poly_max_faces": c.low_poly_max_faces})
    logger.info(f"Face counts: sealed {sealed:,d} -> high {clean_res.get('face_count') or 0:,d}, "
                f"low target {c.target_low_poly_faces:,d}.")


def _hull_patch_hook(hull: Optional[Dict[str, Any]], config: PipelineConfig):
    """
    patch_hook of clean_and_repair_mesh: the inner vertices of each hole-filling patch that sag less than
    visual_hull.snap_max_dist_m inside the visual hull are moved onto it (2.11). None without a decided hull.
    """
    vh = config.visual_hull
    if hull is None or not vh.snap_patches or visual_hull.hull_stats(hull).get("voxels", 0) < 100:
        return None
    import pymeshlab

    sdf = visual_hull.hull_signed_distance(hull)
    stats = {"vertices_snapped": 0, "patch_vertices": 0}

    def hook(ms) -> None:
        m = ms.current_mesh()
        sel = m.face_selection_array()
        F = m.face_matrix()
        inner = np.setdiff1d(np.unique(F[sel]), np.unique(F[~sel]))   # patch vertices not on its border
        stats["patch_vertices"] += int(len(inner))
        if not len(inner):
            return
        V = m.vertex_matrix().copy()                                   # millimetres inside the cleaner
        moved, n = visual_hull.snap_to_hull(V[inner] / 1000.0, hull, vh.snap_max_dist_m, sdf)
        if n == 0:
            return
        V[inner] = moved * 1000.0
        old = ms.current_mesh_id()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F))
        new = ms.current_mesh_id()
        ms.set_current_mesh(old)
        ms.delete_current_mesh()
        ms.set_current_mesh(new)
        stats["vertices_snapped"] += n

    hook.stats = stats
    return hook


def _report_cleaning(clean_res: Dict[str, Any], hook, telemetry: SystemTelemetry) -> None:
    """Numbers of the cleaning in stats.json, its warnings in status.json."""
    info = {k: clean_res.get(k) for k in ("components", "base_band", "base", "holes", "patched_area_cm2",
                                           "is_watertight", "extent_mm")}
    if hook is not None:
        info["patches_on_hull"] = getattr(hook, "stats", None)
    telemetry.set_info(cleaning=info)
    for w in clean_res.get("warnings") or []:
        telemetry.add_warning(**w)


def _object_extra(job: Dict[str, Any], clean_res: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
    """Fields of object.json besides the measurements."""
    return {
        "job_id": job.get("job_id"),
        "watertight": clean_res.get("is_watertight"),
        # Surface added by the hole filling (unseen underside, cavities): an estimate, not a measurement
        "patched_area_cm2": clean_res.get("patched_area_cm2"),
        "faces_high": clean_res.get("face_count"),
        "faces_low_target": config.cleaning.target_low_poly_faces,
        "tag_scale": job.get("tag_scale", TurntableConfig.tag_scale),
    }


def _carve_visual_hull(cameras_known_sfm: str, masks_dir: str, config: PipelineConfig, hull_path: str,
                       telemetry: SystemTelemetry, hull: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Step 2.6: hull from the masks of the original photos; None (with a warning) when it cannot be built.
    `hull`: already carved by the AI masking step (from the views the model ran on): only checked and saved.
    """
    try:
        with telemetry.step("2.6 Visual Hull"):
            if hull is None:
                cams = load_sfm_cameras(cameras_known_sfm)
                mask_paths = visual_hull.source_mask_paths(cams, masks_dir)
                excluded = set(config.masking.excluded_views or [])
                found = sum(1 for m in mask_paths if m)
                mask_paths = [None if c["view_id"] in excluded else p for c, p in zip(cams, mask_paths)]
                if found < max(3, int(0.8 * len(cams))):
                    logger.warning(f"Visual hull skipped: masks found for only {found}/{len(cams)} views in {masks_dir}.")
                    return None
                vh = config.visual_hull
                hull = visual_hull.carve_visual_hull(
                    cams, mask_paths,
                    radius_m=config.masking.cylinder_radius_m,
                    y_min_m=config.cleaning.slice_y_m,
                    y_max_m=config.masking.cylinder_y_max_m,
                    voxel_m=vh.voxel_m,
                    min_bg_votes=vh.min_bg_votes,
                    distorted=True,
                    disc_radius_m=config.masking.disc_radius_m if config.masking.ai_used else None,
                )
            stats = visual_hull.hull_stats(hull)
            telemetry.set_info(visual_hull=stats)
            if stats.get("voxels_total", stats.get("voxels", 0)) < 100:
                logger.warning("Visual hull is empty (masks or poses wrong?); proxy points use the masking cylinder.")
                return None
            visual_hull.save_hull(hull, hull_path)
            if stats.get("voxels", 0) < 100:
                logger.info("Visual hull: the object lies inside the cone over the disc the silhouettes cannot judge; "
                            "hull used for the proxy volume only.")
            else:
                amb = (f", {stats['voxels_ambiguous']:,d} voxels over the disc left undecided"
                       if stats.get("voxels_ambiguous") else "")
                logger.info(
                    f"Visual hull: {stats['volume_cm3']} cm3, height {stats['y_min_mm']}-{stats['y_max_mm']} mm, "
                    f"max radius {stats['radius_max_mm']} mm{amb}."
                )
            return hull
    except Exception as e:
        logger.warning(f"Visual hull failed ({e}); continuing without it.")
        return None


def _refine_dense_cloud(dense_ply: str, hull: Dict[str, Any], cameras_known_sfm: str, undistorted_dir: str,
                        config: PipelineConfig, telemetry: SystemTelemetry, debug_dir: Optional[str] = None) -> bool:
    """Step 4.2b: prune wrong depths inside the hull and fill bare surfaces; True when the cloud was changed."""
    vh = config.visual_hull
    try:
        cams = load_sfm_cameras(cameras_known_sfm)
        images, masks = visual_hull.bridge_image_paths(cams, undistorted_dir)
        stats = visual_hull.refine_dense_cloud_with_hull(
            dense_ply, hull, cams, images, masks,
            prune_depth_m=vh.prune_depth_m,
            prune_max_views=vh.prune_max_views,
            fill=vh.fill,
            support_dist_m=vh.support_dist_m,
            max_normal_y=vh.max_normal_y,
            min_y_m=config.cleaning.slice_y_m + 0.002,
            brightness_white_threshold=config.masking.brightness_white_threshold,
            chroma_threshold=config.masking.chroma_threshold,
            debug_npz=str(Path(debug_dir) / "refine_points.npz") if debug_dir else None,
        )
        telemetry.set_info(visual_hull_refine=stats)
        return bool(stats.get("pruned_inside") or stats.get("added"))
    except Exception as e:
        logger.warning(f"Visual hull refinement failed ({e}); ReconstructMesh uses the plain dense cloud.")
        visual_hull.restore_dense_cloud(dense_ply)
        return False


def _check_densify_ram(cameras_known_sfm: str, config: PipelineConfig, job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RAM estimate for DensifyPointCloud against the container RAM (resources.check_ram).
    Full resolution that does not fit (or barely fits) drops to level 1; anything else that does not fit
    raises InsufficientRamError before OpenMVS starts.
    """
    with open(cameras_known_sfm, "r", encoding="utf-8") as f:
        sfm = json.load(f)
    n_images = len(sfm.get("views", []))
    intr = (sfm.get("intrinsics") or [{}])[0]
    width, height = int(intr.get("width", 4056)), int(intr.get("height", 3040))
    margin = float(job.get("ram_margin", 1.0))
    level = int(config.openmvs.resolution_level)
    try:
        result = check_ram(n_images, width, height, level, margin=margin)
    except InsufficientRamError as err:
        if level != 0:
            raise
        logger.warning(f"{err} Dropping DensifyPointCloud from full resolution to --resolution-level 1.")
        config.openmvs.resolution_level = 1
        result = check_ram(n_images, width, height, 1, margin=margin)
        result["downgraded_from_level"] = 0
    if result["status"] == "tight" and config.openmvs.resolution_level == 0:
        logger.warning("RAM is tight at full resolution: dropping DensifyPointCloud to --resolution-level 1.")
        config.openmvs.resolution_level = 1
        result = check_ram(n_images, width, height, 1, margin=margin)
        result["downgraded_from_level"] = 0
    log = logger.warning if result["status"] == "tight" else logger.info
    log(
        f"RAM check ({result['status']}): DensifyPointCloud needs ~{result['required_gb']} GB for "
        f"{n_images} images at level {result['resolution_level']}, container has {result['available_gb']} GB."
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Turntable Studio - Studio Grade 3D Photogrammetry")
    parser.add_argument("--input_dir", "--input", "-i", default="./input_images", help="Path to input images directory")
    parser.add_argument("--output_dir", "--output", "-o", default="./output", help="Path to output deliverables directory")
    parser.add_argument("--rig_mode", action="store_true", help="Enable rig mode (input_images/rig/0 and rig/1)")
    parser.add_argument("--openmvs_bin_dir", "--bin_dir", default=None, help="Custom path to OpenMVS binaries directory (e.g. C:\\OpenMVS\\vc17\\x64\\Release)")
    parser.add_argument("--resolution_level", type=int, default=1, help="OpenMVS densify downscale (0=full 4K, 1=2x 2K, 2=4x 1K)")
    parser.add_argument("--focal_length_mm", "--focal_mm", type=float, default=3.9, help="Camera lens focal length in mm (default: 3.9 for Sony IMX477 M12 lens)")
    parser.add_argument("--step_deg", type=float, default=18.0, help="Turntable step angle in degrees (default: 18.0 for 20 steps)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing OpenMVS dense cloud or mesh if available")
    parser.add_argument("--cost_smoothness_ratio", type=float, default=1.0, help="OpenMVS texturing smoothness ratio (1.0 = large continuous charts)")
    parser.add_argument("--outlier_threshold", type=float, default=0.05, help="OpenMVS texturing outlier rejection threshold (default: 0.05 for plastics)")
    parser.add_argument("--empty_color", type=int, default=0, help="OpenMVS empty/unobserved face color in decimal RGB (0=black, 16744231=orange)")
    parser.add_argument("--global_seam_leveling", type=int, default=0, help="OpenMVS global Poisson seam leveling (0=disabled to avoid dark bleeding, 1=enabled)")
    parser.add_argument("--local_seam_leveling", type=int, default=1, help="OpenMVS local seam edge blending (1=enabled, 0=disabled)")
    parser.add_argument("--min_point_distance", type=float, default=0.5, help="ReconstructMesh --min-point-distance in pixels (default 0.5; OpenMVS default 2.5: fewer points, faster Delaunay + graph-cut)")
    parser.add_argument("--high_poly_ratio", type=float, default=0.75, help="High-poly keeps this share of the faces of the sealed mesh (default 0.75)")
    parser.add_argument("--low_poly_ratio", type=float, default=0.2, help="Low-poly keeps this share of the faces of the sealed mesh (default 0.2) ...")
    parser.add_argument("--low_poly_max_faces", type=int, default=50000, help="... but at most this many faces (default 50,000; 0 = no cap)")
    parser.add_argument("--target_high_poly_faces", type=int, default=0, help="Fixed high-poly face count instead of --high_poly_ratio (0 = use the ratio)")
    parser.add_argument("--target_low_poly_faces", type=int, default=0, help="Fixed low-poly face count instead of --low_poly_ratio (0 = use the ratio)")
    parser.add_argument("--no_draco", action="store_true", help="GLB without Draco geometry compression (for viewers without a Draco decoder)")
    parser.add_argument("--title", default=None, help="Object title written to object.json")
    parser.add_argument("--description", default=None, help="Object description written to object.json")
    parser.add_argument("--export_scale", type=float, default=1000.0, help="OBJ / STL / 3MF units per metre (default 1000 = millimetres, what slicers and CAD read)")
    parser.add_argument("--glb_scale", type=float, default=100.0, help="GLB units per metre (glTF standard: 1 = metres; default 100 = centimetres, as before)")
    parser.add_argument("--scale_to_mm", action="store_true", help="Export OBJ / STL / 3MF in millimetres (export_scale=1000, now the default)")
    parser.add_argument("--texturing_engine", choices=["texrecon", "openmvs"], default="texrecon", help="Texturing engine: 'texrecon' (mvs-texturing, default) or 'openmvs' (TextureMesh)")
    parser.add_argument("--texrecon_bin", default=None, help="Custom path to texrecon executable or directory (e.g. C:\\texrecon\\bin or /usr/local/bin/texrecon)")
    parser.add_argument("--data_term", choices=list(TEXRECON_DATA_TERMS), default="gmi", help="texrecon data term: 'gmi' (default, gradient magnitude) or 'area' (less sensitive to reflections and mask borders)")
    parser.add_argument("--smoothness_term", default="potts", help="texrecon smoothness term (default: 'potts')")
    parser.add_argument("--outlier_removal", choices=["gauss_damping", "none", "gauss_clamping"], default="gauss_damping", help="texrecon outlier removal (default: 'gauss_damping')")
    parser.add_argument("--texrecon_global_seam_leveling", type=int, default=0, help="texrecon global seam leveling (0=skip, default; 1=enable, test once masks/PNG inputs are clean)")
    parser.add_argument("--no_keep_unseen_faces", action="store_true", help="Let texrecon drop faces no camera sees (leaves holes in the textured OBJ/GLB)")
    parser.add_argument("--write_intermediate_results", action="store_true", help="texrecon writes <prefix>_labeling.vec / _data_costs.spt (which view textured each face)")
    parser.add_argument("--no_mesh_texture_masks", action="store_true", help="Texture from the bridge JPEGs + photometric masks instead of PNGs masked by the cleaned mesh")
    parser.add_argument("--texture_mask_margin_px", type=int, default=2, help="Dilation (px) of the mesh silhouette used as texturing mask (default: 2)")
    parser.add_argument("--no_colour_normalize", action="store_true", help="Disable per-camera colour gains measured on the turntable paper")
    parser.add_argument("--lift_mm", type=float, default=TurntableConfig.lift_mm, help="Carriage lift per level the scanner commands, mm (default: 80); used as is with --no_estimate_lift, otherwise only to warn when the estimate differs by > 10 %%")
    lift = parser.add_mutually_exclusive_group()
    lift.add_argument("--estimate_lift", dest="estimate_lift", action="store_const", const=True, default=None, help="Always estimate the lift per level from the photos (default: automatic, when there are >= 2 levels)")
    lift.add_argument("--no_estimate_lift", dest="estimate_lift", action="store_const", const=False, help="Trust --lift_mm instead of estimating the lift")
    parser.add_argument("--no_tag_refine", action="store_true", help="Disable turntable_poses --tagRefine (per-tag paper unevenness)")
    parser.add_argument("--intrinsics_in", default=None, help="Chessboard intrinsics JSON from calibrate_intrinsics.py, passed to turntable_poses --intrinsicsIn")
    parser.add_argument("--tag_scale", type=float, default=None, help="Print scale of the tag sheet: measured / designed distance (e.g. 0.99 for 345mm disc, 0.98 for 260mm disc; default: automatic by layout)")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers for masks, undistortion and texturing images (0 = automatic from CPUs and free RAM; env TURNTABLE_WORKERS)")
    parser.add_argument("--work_dir", default=None, help="Fast directory for undistorted images and OpenMVS files (default: output_dir; env TURNTABLE_WORK_DIR)")
    parser.add_argument("--keep_depth_maps", action="store_true", help="Keep OpenMVS depth*.dmap after fusion (~74 MB per image at level 1)")
    parser.add_argument("--ram_margin", type=float, default=1.0, help="Safety factor on the DensifyPointCloud RAM estimate before starting (e.g. 1.3)")
    parser.add_argument("--mask_method", choices=list(MASK_METHODS), default="fused", help="Foreground of the masks: 'fused' (default: BiRefNet_lite on a crop, checked against the scan background and the rendered disc, every view), 'ai' (BiRefNet_lite on the whole photo, keeps the disc) or 'photometric' (white = booth); the first two fall back to photometric without the model")
    parser.add_argument("--ai_view_stride", type=int, default=2, help="mask_method 'ai' only: AI masks on 1 view in N; the others get the visual hull silhouette (1 = every view). 'fused' masks every view")
    parser.add_argument("--no_rig_refine", action="store_true", help="Keep the tag poses as they are (no per-camera, per-level correction from sparse object features)")
    parser.add_argument("--no_visual_hull", action="store_true", help="Do not carve the visual hull (proxy points use the fixed volume, no cloud refinement)")
    parser.add_argument("--no_visual_hull_refine", action="store_true", help="Keep the hull for the proxy volume, but feed ReconstructMesh the plain DensifyPointCloud output")
    parser.add_argument("--visual_hull_voxel_mm", type=float, default=None, help="Voxel size of the visual hull in mm (default: adaptive, 1.5 mm for <=40 views, 2.5 mm for >=80 views)")
    parser.add_argument("--densify_masks", choices=list(DENSIFY_MASK_MODES), default="blackout", help="How DensifyPointCloud sees the masks: 'blackout' (background pixels set to 0, default), 'mask_path' (unchanged images + --mask-path), 'none'")
    parser.add_argument("--refine_mesh", action="store_true", default=False, help="Enable OpenMVS RefineMesh surface refinement before cleaning (disabled by default for speed)")
    parser.add_argument("--no_refine_mesh", action="store_true", help="Legacy flag: RefineMesh is now disabled by default")
    parser.add_argument("--tag_layout", default=None, help="Custom tag layout JSON (e.g. pipeline/platan_345mm_layout.json)")
    parser.add_argument("--cylinder_radius_m", type=float, default=None, help="Object cylinder radius in metres (default 0.08 for 260mm disc, 0.13 for 345mm disc)")
    parser.add_argument("--disc_radius_m", type=float, default=None, help="Turntable disc radius in metres (default 0.127 for 260mm disc, 0.170 for 345mm disc)")
    parser.add_argument("--mock_openmvs", action="store_true", help="Simulate OpenMVS for fast testing")
    return parser.parse_args()


def main():
    args = parse_args()
    export_scale = 1000.0 if args.scale_to_mm else args.export_scale
    job = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "rig_mode": args.rig_mode,
        "texturing_engine": args.texturing_engine,
        "texrecon_bin": args.texrecon_bin,
        "data_term": args.data_term,
        "smoothness_term": args.smoothness_term,
        "outlier_removal": args.outlier_removal,
        "texrecon_global_seam_leveling": args.texrecon_global_seam_leveling,
        "keep_unseen_faces": not args.no_keep_unseen_faces,
        "write_intermediate_results": args.write_intermediate_results,
        "mesh_texture_masks": not args.no_mesh_texture_masks,
        "texture_mask_margin_px": args.texture_mask_margin_px,
        "colour_normalize": not args.no_colour_normalize,
        "lift_mm": args.lift_mm,
        "estimate_lift": args.estimate_lift,
        "tag_refine": not args.no_tag_refine,
        "intrinsics_in": args.intrinsics_in,
        "tag_scale": args.tag_scale,
        "tag_layout": args.tag_layout,
        "cylinder_radius_m": args.cylinder_radius_m,
        "disc_radius_m": args.disc_radius_m,
        "refine_mesh": bool(args.refine_mesh and not args.no_refine_mesh),
        "workers": args.workers,
        "work_dir": args.work_dir,
        "keep_depth_maps": args.keep_depth_maps,
        "ram_margin": args.ram_margin,
        "openmvs_bin_dir": args.openmvs_bin_dir,
        "focal_length_mm": args.focal_length_mm,
        "resolution_level": args.resolution_level,
        "step_deg": args.step_deg,
        "resume": args.resume,
        "cost_smoothness_ratio": args.cost_smoothness_ratio,
        "outlier_threshold": args.outlier_threshold,
        "empty_color": args.empty_color,
        "global_seam_leveling": args.global_seam_leveling,
        "local_seam_leveling": args.local_seam_leveling,
        "target_low_poly_faces": args.target_low_poly_faces,
        "target_high_poly_faces": args.target_high_poly_faces,
        "high_poly_ratio": args.high_poly_ratio,
        "low_poly_ratio": args.low_poly_ratio,
        "low_poly_max_faces": args.low_poly_max_faces,
        "glb_draco": not args.no_draco,
        "title": args.title,
        "description": args.description,
        "min_point_distance": args.min_point_distance,
        "rig_refine": not args.no_rig_refine,
        "mask_method": args.mask_method,
        "ai_view_stride": args.ai_view_stride,
        "visual_hull": not args.no_visual_hull,
        "visual_hull_refine": not args.no_visual_hull_refine,
        "visual_hull_voxel_mm": args.visual_hull_voxel_mm,
        "densify_masks": args.densify_masks,
        "export_scale": export_scale,
        "glb_scale": args.glb_scale,
        "scale_stl_to_mm": True,
        "mock_openmvs": args.mock_openmvs,
    }
    run_pipeline(job)


if __name__ == "__main__":
    main()
