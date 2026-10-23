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
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TurntableStudio")

from pipeline.alicevision_runner import AliceVisionRunner, find_alicevision_binary
from pipeline.texrecon_runner import TexReconRunner, find_texrecon_binary
from pipeline.bridge_sfm import convert_sfm_to_colmap
from pipeline.camera_init import (
    generate_camera_init_sfm,
    run_turntable_poses,
    scan_input_directory,
)
from pipeline.config import (
    AliceVisionConfig,
    TexReconConfig,
    MeshCleaningConfig,
    OpenMVSConfig,
    PipelineConfig,
    TurntableConfig,
)
from pipeline.exporter import decimate_mesh, export_multi_lod_bundle
from pipeline.mesh_cleaner import clean_and_repair_mesh
from pipeline.openmvs_runner import OpenMVSRunner
from pipeline.telemetry import SystemTelemetry


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

    # Initialize Telemetry
    telemetry = SystemTelemetry(output_dir=str(out_path))
    telemetry.start_sampling(interval_sec=0.2)

    logger.info("=" * 70)
    logger.info("          STARTING TURNTABLE STUDIO 3D SCANNING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Input Directory  : {input_dir}")
    logger.info(f"Output Directory : {output_dir}")
    logger.info(f"Rig Mode Enabled : {rig_mode}")

    texturing_engine = job.get("texturing_engine", "texrecon")
    texrecon_bin = job.get("texrecon_bin") or job.get("texrecon_bin_dir")
    alicevision_bin = job.get("alicevision_bin") or job.get("alicevision_bin_dir")

    # If texrecon is requested, verify availability or fall back gracefully
    if texturing_engine == "texrecon":
        detected_tr = find_texrecon_binary(custom_dir=texrecon_bin)
        if not detected_tr and not job.get("mock_openmvs"):
            logger.warning(
                "texrecon executable not detected on system. "
                "Checking for AliceVision as fallback..."
            )
            detected_av = find_alicevision_binary("aliceVision_texturing", custom_dir=alicevision_bin)
            if detected_av:
                logger.info(f"Using AliceVision Texturing as fallback: {detected_av}")
                texturing_engine = "alicevision"
            else:
                logger.warning("Defaulting to OpenMVS texturing engine.")
                texturing_engine = "openmvs"
        elif detected_tr:
            logger.info(f"Using mvs-texturing (texrecon): {detected_tr}")

    # If AliceVision is requested but binary is not found anywhere, warn and fall back to openmvs
    elif texturing_engine == "alicevision":
        detected_av = find_alicevision_binary("aliceVision_texturing", custom_dir=alicevision_bin)
        if not detected_av and not job.get("mock_openmvs"):
            logger.warning(
                "aliceVision_texturing executable not detected on system. "
                "Defaulting to OpenMVS texturing engine."
            )
            texturing_engine = "openmvs"
        elif detected_av:
            logger.info(f"Using AliceVision Texturing: {detected_av}")

    # Build Configuration
    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        rig_mode=rig_mode,
        texturing_engine=texturing_engine,
        texrecon=TexReconConfig(
            bin_dir=texrecon_bin,
            data_term=job.get("data_term", "gmi"),
            smoothness_term=job.get("smoothness_term", "potts"),
            outlier_removal=job.get("outlier_removal", "gauss_damping"),
            num_threads=job.get("num_threads"),
        ),
        alicevision=AliceVisionConfig(
            bin_dir=alicevision_bin,
            texture_side=job.get("texture_resolution", 4096),
            unwrap_method=job.get("unwrap_method", "Basic"),
            use_udim=job.get("use_udim", False),
            fill_holes=job.get("fill_holes", True),
            correct_ev=job.get("correct_ev", True),
            color_mapping_type=job.get("color_mapping_type", "png"),
            visibility_remapping_method=job.get("visibility_remapping_method", "MeshItself"),
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
        ),
        turntable=TurntableConfig(
            step_deg=job.get("step_deg", 18.0),
            tag_scale=job.get("tag_scale", 1.0),
            default_focal_mm=job.get("focal_length_mm", 3.9),
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

        # ----------------------------------------------------------------------
        # STEP 2: Calibrate and Lock Poses via TurntablePoses
        # ----------------------------------------------------------------------
        cameras_known_sfm = str(out_path / "cameras_known.sfm")
        with telemetry.step("2. Turntable Poses (Rigid Lock)"):
            if not mock_openmvs:
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

        # ----------------------------------------------------------------------
        # STEP 3: Convert SFM to COLMAP & Undistort Images/Masks
        # ----------------------------------------------------------------------
        colmap_dir = str(out_path / "colmap")
        undistorted_dir = str(out_path / "undistorted")
        masks_out_dir = str(out_path / "masks")

        with telemetry.step("3. Bridge SFM & Image Undistortion"):
            masks_src = config.masks_dir or str(Path(input_dir) / "masks")
            bridge_res = convert_sfm_to_colmap(
                sfm_json_path=cameras_known_sfm,
                output_colmap_dir=colmap_dir,
                undistorted_images_dir=undistorted_dir,
                masks_src_dir=masks_src if Path(masks_src).is_dir() else None,
                masks_dst_dir=masks_out_dir if Path(masks_src).is_dir() else None,
            )

        # ----------------------------------------------------------------------
        # STEP 4: OpenMVS Dense Reconstruction
        # ----------------------------------------------------------------------
        openmvs_dir = out_path / "openmvs"
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
                with telemetry.step("4.1 OpenMVS InterfaceCOLMAP"):
                    runner.run_interface_colmap(colmap_dir, undistorted_dir)

                with telemetry.step("4.2 OpenMVS DensifyPointCloud"):
                    runner.run_densify(masks_dir=masks_out_dir if Path(masks_out_dir).is_dir() else None)

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

            # ----------------------------------------------------------------------
            # STEP 5: Watertight Base Slicing & Mesh Cleaning (BEFORE Texturing!)
            # ----------------------------------------------------------------------
            with telemetry.step("5. Watertight Base Slicing & Pedestal Sealing"):
                clean_res = clean_and_repair_mesh(
                    input_obj_path=raw_reconstructed_mesh,
                    output_obj_path=cleaned_obj_path,
                    config=config.cleaning,
                )

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
                            images_folder=undistorted_dir,
                            out_prefix=str(out_path / "texrecon" / "model_textured"),
                            mock_mode=job.get("mock_openmvs", False),
                        )
                        if cleaned_low_path and cleaned_low_path.is_file():
                            try:
                                logger.info("Running branching texturing on low-poly mesh (texrecon)...")
                                textured_low_obj_path, _ = tr_runner.run_texturing(
                                    sfm_data_path=cameras_known_sfm,
                                    input_mesh_path=str(cleaned_low_path),
                                    images_folder=undistorted_dir,
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
                        _, textured_obj_path = runner.run_texture_mesh(
                            input_mvs=mvs_project,
                            mesh_file=cleaned_obj_path,
                            output_mvs="scene_texture.mvs",
                        )
                        if cleaned_low_path and cleaned_low_path.is_file():
                            try:
                                _, textured_low_obj_path = runner.run_texture_mesh(
                                    input_mvs=mvs_project,
                                    mesh_file=str(cleaned_low_path),
                                    output_mvs="scene_texture_low.mvs",
                                )
                            except Exception as omvs_low_err:
                                logger.warning(f"OpenMVS fallback low-poly texturing failed ({omvs_low_err}).")
            elif config.texturing_engine == "alicevision":
                try:
                    with telemetry.step("4.4 AliceVision Texturing (Multi-Band & LSCM)"):
                        av_runner = AliceVisionRunner(
                            config=config.alicevision,
                            work_dir=str(out_path / "alicevision"),
                        )
                        textured_obj_path, _ = av_runner.run_texturing(
                            sfm_data_path=cameras_known_sfm,
                            input_mesh_path=cleaned_obj_path,
                            images_folder=undistorted_dir,
                            output_mesh_name="texturedMesh.obj",
                        )
                        if cleaned_low_path and cleaned_low_path.is_file():
                            try:
                                logger.info("Running branching texturing on low-poly mesh (AliceVision)...")
                                av_low_runner = AliceVisionRunner(
                                    config=config.alicevision,
                                    work_dir=str(out_path / "alicevision" / "low"),
                                )
                                textured_low_obj_path, _ = av_low_runner.run_texturing(
                                    sfm_data_path=cameras_known_sfm,
                                    input_mesh_path=str(cleaned_low_path),
                                    images_folder=undistorted_dir,
                                    output_mesh_name="texturedMesh.obj",
                                )
                            except Exception as av_low_err:
                                logger.warning(f"AliceVision low-poly texturing failed ({av_low_err}).")
                except Exception as av_err:
                    logger.warning(
                        f"AliceVision texturing failed ({av_err}). Falling back to OpenMVS texturing."
                    )
                    with telemetry.step("4.4 OpenMVS TextureMesh (Fallback)"):
                        _, textured_obj_path = runner.run_texture_mesh(
                            input_mvs=mvs_project,
                            mesh_file=cleaned_obj_path,
                            output_mvs="scene_texture.mvs",
                        )
                        if cleaned_low_path and cleaned_low_path.is_file():
                            try:
                                _, textured_low_obj_path = runner.run_texture_mesh(
                                    input_mvs=mvs_project,
                                    mesh_file=str(cleaned_low_path),
                                    output_mvs="scene_texture_low.mvs",
                                )
                            except Exception as omvs_low_err:
                                logger.warning(f"OpenMVS fallback low-poly texturing failed ({omvs_low_err}).")
            else:
                with telemetry.step("4.4 OpenMVS TextureMesh (Watertight)"):
                    _, textured_obj_path = runner.run_texture_mesh(
                        input_mvs=mvs_project,
                        mesh_file=cleaned_obj_path,
                        output_mvs="scene_texture.mvs",
                    )
                    if cleaned_low_path and cleaned_low_path.is_file():
                        try:
                            logger.info("Running branching texturing on low-poly mesh (OpenMVS)...")
                            _, textured_low_obj_path = runner.run_texture_mesh(
                                input_mvs=mvs_project,
                                mesh_file=str(cleaned_low_path),
                                output_mvs="scene_texture_low.mvs",
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
        stats_file = telemetry.save_json()
        telemetry.print_table()

        return {
            "status": "SUCCESS",
            "stats_json": stats_file,
            "bundle": bundle,
            "telemetry": telemetry.summary(),
        }

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        telemetry.stop_sampling()
        telemetry.save_json()
        telemetry.print_table()
        raise


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
    parser.add_argument("--texturing_engine", choices=["texrecon", "alicevision", "openmvs"], default="texrecon", help="Texturing engine: 'texrecon' (mvs-texturing, default), 'alicevision', or 'openmvs'")
    parser.add_argument("--texrecon_bin", default=None, help="Custom path to texrecon executable or directory (e.g. C:\\texrecon\\bin or /usr/local/bin/texrecon)")
    parser.add_argument("--data_term", choices=["gmi", "color"], default="gmi", help="texrecon data term: 'gmi' (default) or 'color'")
    parser.add_argument("--smoothness_term", default="potts", help="texrecon smoothness term (default: 'potts')")
    parser.add_argument("--outlier_removal", choices=["gauss_damping", "none", "gauss_clamping"], default="gauss_damping", help="texrecon outlier removal (default: 'gauss_damping')")
    parser.add_argument("--unwrap_method", choices=["Basic", "LSCM", "ABF"], default="Basic", help="AliceVision UV unwrapping method: 'Basic' (fast & robust on high-poly), 'LSCM', or 'ABF'")
    parser.add_argument("--visibility_remapping_method", choices=["MeshItself", "PullPush", "Pull", "Push"], default="MeshItself", help="AliceVision visibility remapping: 'MeshItself' (direct mesh raycasting) or 'PullPush'")
    parser.add_argument("--alicevision_bin", default=None, help="Custom path to AliceVision bin directory or aliceVision_texturing executable (e.g. C:\\Meshroom\\aliceVision\\bin)")
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
        "unwrap_method": args.unwrap_method,
        "visibility_remapping_method": args.visibility_remapping_method,
        "alicevision_bin": args.alicevision_bin,
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
