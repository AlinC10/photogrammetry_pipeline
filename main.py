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

from pipeline.bridge_sfm import convert_sfm_to_colmap
from pipeline.camera_init import (
    generate_camera_init_sfm,
    run_turntable_poses,
    scan_input_directory,
)
from pipeline.config import (
    MeshCleaningConfig,
    OpenMVSConfig,
    PipelineConfig,
    TurntableConfig,
)
from pipeline.exporter import export_multi_lod_bundle
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

    # Build Configuration
    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        rig_mode=rig_mode,
        openmvs=OpenMVSConfig(
            bin_dir=job.get("openmvs_bin_dir"),
            resolution_level=job.get("resolution_level", 1),
            smooth_mesh=job.get("smooth_mesh", 2),
            min_resolution=job.get("min_resolution", 640),
            texture_resolution=job.get("texture_resolution", 4096),
        ),
        turntable=TurntableConfig(
            step_deg=job.get("step_deg", 18.0),
            tag_scale=job.get("tag_scale", 1.0),
        ),
        cleaning=MeshCleaningConfig(
            slice_y_m=job.get("slice_y_m", 0.003),
            skirt_bottom_y_m=job.get("skirt_bottom_y_m", 0.000),
            max_radius_m=job.get("max_radius_m", 0.12),
        ),
    )

    try:
        # ----------------------------------------------------------------------
        # STEP 1: Scan Images and Generate CameraInit SFM
        # ----------------------------------------------------------------------
        with telemetry.step("1. Camera Discovery & Init"):
            views, intrinsics, name_regex = scan_input_directory(
                input_dir=config.input_dir,
                rig_mode=config.rig_mode,
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

        if not mock_openmvs:
            with telemetry.step("4.1 OpenMVS InterfaceCOLMAP"):
                runner.run_interface_colmap(colmap_dir, undistorted_dir)

            with telemetry.step("4.2 OpenMVS DensifyPointCloud"):
                runner.run_densify(masks_dir=masks_out_dir if Path(masks_out_dir).is_dir() else None)

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

            # ----------------------------------------------------------------------
            # STEP 4.4: OpenMVS TextureMesh on Watertight Sealed Geometry
            # ----------------------------------------------------------------------
            with telemetry.step("4.4 OpenMVS TextureMesh (Watertight)"):
                _, textured_obj_path = runner.run_texture_mesh(
                    input_mvs="scene_mesh.mvs",
                    mesh_file=cleaned_obj_path,
                )

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
            textured_obj_path = cleaned_obj_path

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
    parser.add_argument("--input_dir", "-i", default="./input_images", help="Path to input images directory")
    parser.add_argument("--output_dir", "-o", default="./output", help="Path to output deliverables directory")
    parser.add_argument("--rig_mode", action="store_true", help="Enable rig mode (input_images/rig/0 and rig/1)")
    parser.add_argument("--openmvs_bin_dir", "--bin_dir", default=None, help="Custom path to OpenMVS binaries directory (e.g. C:\\OpenMVS or C:\\openMVS\\bin)")
    parser.add_argument("--resolution_level", type=int, default=1, help="OpenMVS densify downscale (0=full, 1=2x, 2=4x)")
    parser.add_argument("--step_deg", type=float, default=18.0, help="Turntable step angle in degrees")
    parser.add_argument("--mock_openmvs", action="store_true", help="Simulate OpenMVS for fast testing")
    return parser.parse_args()


def main():
    args = parse_args()
    job = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "rig_mode": args.rig_mode,
        "openmvs_bin_dir": args.openmvs_bin_dir,
        "resolution_level": args.resolution_level,
        "step_deg": args.step_deg,
        "mock_openmvs": args.mock_openmvs,
    }
    run_pipeline(job)


if __name__ == "__main__":
    main()
