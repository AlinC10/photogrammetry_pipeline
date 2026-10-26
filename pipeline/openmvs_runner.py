"""
openmvs_runner.py — Cross-platform wrapper executing OpenMVS CLI binaries:
InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, RefineMesh, TextureMesh.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import OpenMVSConfig

logger = logging.getLogger("TurntableStudio.OpenMVS")


class OpenMVSError(RuntimeError):
    """Raised when an OpenMVS executable returns a non-zero exit code."""
    pass


class OpenMVSRunner:
    """Executes the OpenMVS reconstruction pipeline."""

    def __init__(self, config: OpenMVSConfig, work_dir: str):
        self.config = config
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        missing = self.config.validate()
        if missing:
            logger.warning(
                f"Missing OpenMVS binaries: {', '.join(missing)}. "
                "Ensure OpenMVS is installed or OPENMVS_BIN_DIR is set."
            )

    def _execute_cmd(self, cmd: List[str], step_name: str) -> None:
        """Runs an external CLI command with real-time log streaming."""
        logger.info(f"Executing [{step_name}]: {' '.join(cmd)}")
        t0 = time.time()

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        output_lines = []
        if proc.stdout:
            for line in iter(proc.stdout.readline, ""):
                line_str = line.strip()
                if line_str:
                    logger.debug(f"[{step_name}] {line_str}")
                    output_lines.append(line_str)
            proc.stdout.close()

        ret = proc.wait()
        elapsed = time.time() - t0

        if ret != 0:
            tail = "\n".join(output_lines[-25:])
            err_msg = (
                f"OpenMVS {step_name} failed with return code {ret} in {elapsed:.1f}s!\n"
                f"Last output lines:\n{tail}"
            )
            logger.error(err_msg)
            raise OpenMVSError(err_msg)

        logger.info(f"Completed [{step_name}] in {elapsed:.2f}s.")

    def run_interface_colmap(
        self,
        colmap_dir: str,
        undistorted_images_dir: str,
        output_mvs: str = "scene.mvs",
    ) -> str:
        """Imports COLMAP sparse model into OpenMVS scene.mvs."""
        bin_path = self.config.interface_colmap
        if not bin_path:
            raise FileNotFoundError("InterfaceCOLMAP binary not found.")

        out_file = self.work_dir / output_mvs
        cmd = [
            bin_path,
            "-i", str(Path(colmap_dir).resolve()),
            "-o", str(out_file),
            "--image-folder", str(Path(undistorted_images_dir).resolve()),
            "-v", "2",
        ]
        self._execute_cmd(cmd, "InterfaceCOLMAP")
        if not out_file.is_file():
            raise FileNotFoundError(f"InterfaceCOLMAP output missing: {out_file}")
        return str(out_file)

    def run_densify(
        self,
        input_mvs: str = "scene.mvs",
        output_mvs: str = "scene_dense.mvs",
        masks_dir: Optional[str] = None,
    ) -> str:
        """Runs PatchMatch MVS point cloud densification."""
        bin_path = self.config.densify_point_cloud
        if not bin_path:
            raise FileNotFoundError("DensifyPointCloud binary not found.")

        in_file = self.work_dir / input_mvs
        out_file = self.work_dir / output_mvs

        cmd = [
            bin_path,
            str(in_file),
            "-o", str(out_file),
            "--resolution-level", str(self.config.resolution_level),
            "--min-resolution", str(self.config.min_resolution),
            "--max-resolution", str(self.config.max_resolution),
            "--number-views", str(self.config.number_views),
            "--number-views-fuse", "2",
            "-v", "2",
        ]
        if masks_dir and Path(masks_dir).is_dir():
            cmd.extend([
                "--mask-path", str(Path(masks_dir).resolve()),
                "--ignore-mask-label", "0",
            ])
        self._execute_cmd(cmd, "DensifyPointCloud")
        if not out_file.is_file():
            raise FileNotFoundError(f"DensifyPointCloud output missing: {out_file}")
        return str(out_file)

    def run_reconstruct_mesh(
        self,
        input_mvs: str = "scene_dense.mvs",
        output_mvs: str = "scene_mesh.mvs",
    ) -> str:
        """Reconstructs Delaunay/Graph-Cut surface mesh from dense point cloud."""
        bin_path = self.config.reconstruct_mesh
        if not bin_path:
            raise FileNotFoundError("ReconstructMesh binary not found.")

        in_file = self.work_dir / input_mvs
        out_file = self.work_dir / output_mvs

        cmd = [
            bin_path,
            str(in_file),
            "-o", str(out_file),
            "--smooth", str(self.config.smooth_mesh),
            "--min-point-distance", str(self.config.min_point_distance),
            "-v", "2",
        ]
        self._execute_cmd(cmd, "ReconstructMesh")

        # OpenMVS may save as .mvs or .ply directly depending on version
        if out_file.is_file():
            return str(out_file)

        ply_candidate = self.work_dir / f"{out_file.stem}.ply"
        if ply_candidate.is_file():
            logger.info(f"ReconstructMesh generated PLY surface: {ply_candidate}")
            return str(ply_candidate)

        for alt in [self.work_dir / "scene_mesh.ply", self.work_dir / "scene_dense_mesh.ply"]:
            if alt.is_file():
                return str(alt)

        raise FileNotFoundError(f"ReconstructMesh output missing: {out_file} or {ply_candidate}")

    def run_refine_mesh(
        self,
        input_mvs: str = "scene_mesh.mvs",
        output_mvs: str = "scene_mesh_refined.mvs",
    ) -> str:
        """Optionally refines mesh geometry using photo-consistency gradient."""
        bin_path = self.config.refine_mesh
        if not bin_path:
            logger.info("RefineMesh binary not found, skipping.")
            return str(self.work_dir / input_mvs)

        in_file = self.work_dir / input_mvs
        if not in_file.is_file():
            in_file = self.work_dir / "scene_dense.mvs"

        out_file = self.work_dir / output_mvs

        cmd = [
            bin_path,
            str(in_file),
            "-o", str(out_file),
            "--resolution-level", "1",
            "--max-face-area", "64",
            "-v", "2",
        ]
        self._execute_cmd(cmd, "RefineMesh")
        return str(out_file) if out_file.is_file() else str(in_file)

    def run_texture_mesh(
        self,
        input_mvs: str = "scene_mesh.mvs",
        output_mvs: str = "scene_textured.mvs",
        mesh_file: Optional[str] = None,
        masks_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Generates global UV atlas and textures mesh from original camera views.
        If mesh_file is provided, textures that specific sealed/cleaned mesh geometry.
        Returns paths to (output_mvs, output_obj).
        """
        bin_path = self.config.texture_mesh
        if not bin_path:
            raise FileNotFoundError("TextureMesh binary not found.")

        # Fallback to scene_dense.mvs or scene.mvs if scene_mesh.mvs does not exist
        in_file = self.work_dir / input_mvs
        if not in_file.is_file():
            for cand_mvs in ["scene_dense.mvs", "scene.mvs"]:
                cand_path = self.work_dir / cand_mvs
                if cand_path.is_file():
                    in_file = cand_path
                    break

        out_file = self.work_dir / output_mvs
        expected_obj = self.work_dir / f"{out_file.stem}.obj"

        cmd = [
            bin_path,
            str(in_file),
            "-o", str(out_file),
            "--export-type", "obj",
            "--resolution-level", "0",
            "--empty-color", str(self.config.empty_color),
            "--cost-smoothness-ratio", str(self.config.cost_smoothness_ratio),
            "--outlier-threshold", str(self.config.outlier_threshold),
            "--global-seam-leveling", str(self.config.global_seam_leveling),
            "--local-seam-leveling", str(self.config.local_seam_leveling),
            "--patch-packing-heuristic", str(getattr(self.config, "patch_packing_heuristic", 3)),
            "--close-holes", str(getattr(self.config, "close_holes", 1)),
            "-v", "2",
        ]
        if mesh_file and Path(mesh_file).is_file():
            cmd.extend(["-m", str(Path(mesh_file).resolve())])
        if masks_dir and Path(masks_dir).is_dir():
            cmd.extend([
                "--mask-path", str(Path(masks_dir).resolve()),
                "--ignore-mask-label", "0",
            ])

        self._execute_cmd(cmd, "TextureMesh")

        if not expected_obj.is_file():
            # Check if obj was exported with another name
            candidate_objs = list(self.work_dir.glob("*.obj"))
            if candidate_objs:
                expected_obj = candidate_objs[0]
            else:
                raise FileNotFoundError(f"TextureMesh OBJ output missing: {expected_obj}")

        return str(out_file), str(expected_obj)
