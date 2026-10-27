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
from pathlib import Path
from typing import Any, Dict, Optional

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
from pipeline.bridge_sfm import convert_sfm_to_colmap
from pipeline.mask_processor import batch_generate_masks
from pipeline.camera_init import (
    generate_camera_init_sfm,
    run_turntable_poses,
    scan_input_directory,
)
from pipeline.config import (
    TexReconConfig,
    TexturingImagesConfig,
    MeshCleaningConfig,
    OpenMVSConfig,
    PipelineConfig,
    TurntableConfig,
)
from pipeline.exporter import collect_log_files, decimate_mesh, export_multi_lod_bundle, package_output_zip
from pipeline.mesh_cleaner import clean_and_repair_mesh
from pipeline.openmvs_runner import OpenMVSRunner
from pipeline import run_metrics
from pipeline.resources import InsufficientRamError, check_ram, describe_resources, resolve_workers
from pipeline.telemetry import SystemTelemetry
from pipeline.texrecon_runner import TEXRECON_DATA_TERMS
from pipeline.texture_prep import prepare_texturing_images


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
    telemetry.start_sampling(interval_sec=0.2)

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
            "tag_refine", "tag_scale", "intrinsics_in", "target_low_poly_faces", "export_scale", "keep_depth_maps",
            "mesh_texture_masks", "colour_normalize", "resume", "mock_openmvs", "ram_margin")},
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

    data_term = job.get("data_term", "gmi")
    if data_term not in TEXRECON_DATA_TERMS:
        raise ValueError(f"data_term must be one of {TEXRECON_DATA_TERMS} (texrecon rejects anything else), got '{data_term}'")

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
            min_resolution=job.get("min_resolution", 640),
            texture_resolution=job.get("texture_resolution", 4096),
            empty_color=job.get("empty_color", 0),
            cost_smoothness_ratio=job.get("cost_smoothness_ratio", 1.0),
            outlier_threshold=job.get("outlier_threshold", 0.0),
            global_seam_leveling=job.get("global_seam_leveling", 0),
            local_seam_leveling=job.get("local_seam_leveling", 1),
            max_threads=resources["cpus"],
            keep_depth_maps=bool(job.get("keep_depth_maps", False)),
        ),
        turntable=TurntableConfig(
            step_deg=job.get("step_deg", 18.0),
            tag_scale=job.get("tag_scale", 1.0),
            default_focal_mm=job.get("focal_length_mm", 3.9),
            lift_mm=float(job.get("lift_mm", 100.0)),
            estimate_lift=bool(job.get("estimate_lift", False)),
            tag_refine=bool(job.get("tag_refine", True)),
            intrinsics_in=job.get("intrinsics_in"),
        ),
        cleaning=MeshCleaningConfig(
            slice_y_m=job.get("slice_y_m", 0.003),
            skirt_bottom_y_m=job.get("skirt_bottom_y_m", 0.000),
            max_radius_m=job.get("max_radius_m", 0.12),
            target_low_poly_faces=job.get("target_low_poly_faces", 50000),
        ),
        export_scale=job.get("export_scale", 100.0),
        scale_stl_to_mm=job.get("scale_stl_to_mm", True),
    )

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
        telemetry.set_info(inputs=run_metrics.input_stats(camera_init_sfm, name_regex))

        # ----------------------------------------------------------------------
        # STEP 2: Calibrate and Lock Poses via TurntablePoses
        # ----------------------------------------------------------------------
        cameras_known_sfm = str(out_path / "cameras_known.sfm")
        with telemetry.step("2. Turntable Poses (Rigid Lock)"):
            if not mock_openmvs and job.get("resume", False) and Path(cameras_known_sfm).is_file():
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
            telemetry.set_info(pose_quality=run_metrics.pose_quality(str(out_path / "turntable_report.txt")))

        # ----------------------------------------------------------------------
        # STEP 2.5: Automated Hybrid Background Matting & Turntable Elimination
        # ----------------------------------------------------------------------
        masks_out_dir = str(out_path / "masks")
        masks_src: Optional[str] = config.masks_dir or str(Path(input_dir) / "masks")
        if not Path(masks_src).is_dir():
            masks_src = None

        if config.use_masks and config.masking.enabled and config.masking.auto_generate:
            with telemetry.step("2.5. Automated Masking & Turntable Elimination"):
                if not mock_openmvs:
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

                    auto_masks = batch_generate_masks(
                        views=m_views,
                        poses_dict=m_poses,
                        intrinsics_dict=m_intrinsics,
                        output_masks_dir=masks_out_dir,
                        cylinder_radius_m=config.masking.cylinder_radius_m,
                        cylinder_y_min_m=config.masking.cylinder_y_min_m,
                        cylinder_y_max_m=config.masking.cylinder_y_max_m,
                        chroma_threshold=config.masking.chroma_threshold,
                        brightness_white_threshold=config.masking.brightness_white_threshold,
                        safety_dilation_px=config.masking.safety_dilation_px,
                        tag_blackout=config.masking.tag_blackout,
                        tag_expansion_factor=config.masking.tag_expansion_factor,
                        workers=workers,
                    )
                    masks_src = masks_out_dir
                    logger.info(f"Generated {len(auto_masks)} foreground masks in {masks_out_dir}.")
                else:
                    Path(masks_out_dir).mkdir(parents=True, exist_ok=True)
                    masks_src = masks_out_dir

        # ----------------------------------------------------------------------
        # STEP 3: Convert SFM to COLMAP & Undistort Images/Masks
        # ----------------------------------------------------------------------
        colmap_dir = str(work_path / "colmap")
        undistorted_dir = str(work_path / "undistorted")

        with telemetry.step("3. Bridge SFM & Image Undistortion"):
            convert_sfm_to_colmap(
                sfm_json_path=cameras_known_sfm,
                output_colmap_dir=colmap_dir,
                undistorted_images_dir=undistorted_dir,
                masks_src_dir=masks_src if masks_src and Path(masks_src).is_dir() else None,
                masks_dst_dir=masks_out_dir if masks_src and Path(masks_src).is_dir() else None,
                workers=workers,
            )
        telemetry.set_info(object_images=run_metrics.object_image_stats(undistorted_dir))

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
                    # Architectural principle: DensifyPointCloud runs cleanly on the full scene
                    # without mask constraints (preventing OpenMVS fail-fast aborts and edge depth voids).
                    # Turntable disc is cleanly sliced at Y = 3.0mm by mesh_cleaner.py.
                    # Masks are strictly used for Texturing (texrecon / TextureMesh).
                    effective_densify_masks = (
                        masks_out_dir
                        if config.masking.enabled and config.masking.use_for_densify and Path(masks_out_dir).is_dir()
                        else None
                    )
                    runner.run_densify(masks_dir=effective_densify_masks)

            if not skip_reconstruct:
                with telemetry.step("4.3 OpenMVS ReconstructMesh"):
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

            telemetry.set_info(
                openmvs=run_metrics.openmvs_log_stats(str(openmvs_dir)),
                raw_mesh=run_metrics.mesh_quality(raw_reconstructed_mesh, "raw"),
            )

            # ----------------------------------------------------------------------
            # STEP 5: Watertight Base Slicing & Mesh Cleaning (BEFORE Texturing!)
            # ----------------------------------------------------------------------
            with telemetry.step("5. Watertight Base Slicing & Pedestal Sealing"):
                clean_res = clean_and_repair_mesh(
                    input_obj_path=raw_reconstructed_mesh,
                    output_obj_path=cleaned_obj_path,
                    config=config.cleaning,
                )

            telemetry.set_info(cleaned_mesh=run_metrics.mesh_quality(cleaned_obj_path, "cleaned"))

            # Generate dedicated low-poly geometry for branching texturing (eliminates white untextured low-poly!)
            cleaned_low_path = out_path / "cleaned_mesh_low.obj"
            try:
                decimate_mesh(
                    input_mesh_path=cleaned_obj_path,
                    output_mesh_path=str(cleaned_low_path),
                    target_faces=config.cleaning.target_low_poly_faces,
                )
            except Exception as dec_err:
                logger.warning(f"Pre-texturing low-poly decimation failed ({dec_err}). Will fall back to standard decimation.")
                cleaned_low_path = None

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
                        )
                        texturing_images_dir = prep["images_dir"]
                except Exception as prep_err:
                    logger.warning(
                        f"Texturing image preparation failed ({prep_err}). "
                        f"texrecon will use the bridge images in {undistorted_dir}."
                    )

            # ----------------------------------------------------------------------
            # STEP 4.4: Texture Mesh on Watertight Sealed Geometry
            # ----------------------------------------------------------------------
            if config.texturing_engine == "texrecon":
                try:
                    with telemetry.step("4.4 mvs-texturing (texrecon)"):
                        tr_runner = TexReconRunner(
                            config=config.texrecon,
                            work_dir=str(out_path / "texrecon"),
                        )
                        textured_obj_path, _ = tr_runner.run_texturing(
                            sfm_data_path=cameras_known_sfm,
                            input_mesh_path=cleaned_obj_path,
                            images_folder=texturing_images_dir,
                            out_prefix=str(out_path / "texrecon" / "model_textured"),
                            mock_mode=job.get("mock_openmvs", False),
                        )
                        if cleaned_low_path and cleaned_low_path.is_file():
                            try:
                                logger.info("Running branching texturing on low-poly mesh (texrecon)...")
                                textured_low_obj_path, _ = tr_runner.run_texturing(
                                    sfm_data_path=cameras_known_sfm,
                                    input_mesh_path=str(cleaned_low_path),
                                    images_folder=texturing_images_dir,
                                    out_prefix=str(out_path / "texrecon" / "model_textured_low"),
                                    mock_mode=job.get("mock_openmvs", False),
                                )
                            except Exception as tr_low_err:
                                logger.warning(f"texrecon low-poly texturing failed ({tr_low_err}).")
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
            bundle = export_multi_lod_bundle(
                cleaned_mesh_path=cleaned_obj_path,
                textured_obj_path=textured_obj_path,
                output_dir=str(out_path),
                telemetry=telemetry,
                config=config,
                stats_json_path=stats_file,
                textured_low_obj_path=textured_low_obj_path,
            )

        logger.info("=" * 70)
        logger.info("       PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

        # Stop telemetry sampling and finalize report
        telemetry.stop_sampling()
        run_log_handler.flush()
        telemetry.set_info(log_counters=run_metrics.log_counters(str(out_path / "run_log.txt")),
                           output_zip_mb=bundle.get("zip_size_mb"))
        telemetry.set_outcome("ok")
        stats_file = telemetry.save_json()
        telemetry.print_table()

        # Repack output.zip last, so it carries the final stats.json and the complete logs
        run_log_handler.flush()
        bundle["zip_path"] = str(package_output_zip(
            str(out_path), config.zip_name, stats_file,
            collect_log_files(str(out_path), extra_dirs=[str(openmvs_dir)] if work_path != out_path else None),
        ))
        detach_run_log(run_log_handler)

        return {
            "status": "SUCCESS",
            "stats_json": stats_file,
            "bundle": bundle,
            "telemetry": telemetry.summary(),
        }

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        telemetry.stop_sampling()
        telemetry.set_outcome("error", error_type=type(e).__name__, error=str(e)[:500],
                              failed_step=telemetry.current_step)
        run_log_handler.flush()
        telemetry.set_info(log_counters=run_metrics.log_counters(str(out_path / "run_log.txt")))
        telemetry.save_json()
        telemetry.print_table()
        detach_run_log(run_log_handler)
        raise


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
    parser.add_argument("--target_low_poly_faces", type=int, default=50000, help="Target face count for low-poly mesh (default: 50,000)")
    parser.add_argument("--export_scale", type=float, default=100.0, help="Scale factor applied to exported assets (default: 100.0 for real-world scale: 2.2mm -> 22cm)")
    parser.add_argument("--scale_to_mm", action="store_true", help="Shortcut to export OBJ in millimeters (sets export_scale=1000.0)")
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
    parser.add_argument("--lift_mm", type=float, default=100.0, help="Carriage lift per level in mm passed to turntable_poses --liftMm (default: 100, scanner z_axis.distance_mm)")
    parser.add_argument("--estimate_lift", action="store_true", help="Let turntable_poses estimate the real lift per level")
    parser.add_argument("--no_tag_refine", action="store_true", help="Disable turntable_poses --tagRefine (per-tag paper unevenness)")
    parser.add_argument("--intrinsics_in", default=None, help="Chessboard intrinsics JSON from calibrate_intrinsics.py, passed to turntable_poses --intrinsicsIn")
    parser.add_argument("--tag_scale", type=float, default=1.0, help="Print scale of the tag sheet: measured / designed distance (e.g. 0-6 centres: measured mm / 200), passed to turntable_poses --tagScale")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers for masks, undistortion and texturing images (0 = automatic from CPUs and free RAM; env TURNTABLE_WORKERS)")
    parser.add_argument("--work_dir", default=None, help="Fast directory for undistorted images and OpenMVS files (default: output_dir; env TURNTABLE_WORK_DIR)")
    parser.add_argument("--keep_depth_maps", action="store_true", help="Keep OpenMVS depth*.dmap after fusion (~74 MB per image at level 1)")
    parser.add_argument("--ram_margin", type=float, default=1.0, help="Safety factor on the DensifyPointCloud RAM estimate before starting (e.g. 1.3)")
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
        "export_scale": export_scale,
        "scale_stl_to_mm": True,
        "mock_openmvs": args.mock_openmvs,
    }
    run_pipeline(job)


if __name__ == "__main__":
    main()
