"""
texrecon_runner.py — High-precision multi-view texturing using mvs-texturing (texrecon).
Developed by TU Darmstadt (Waechter et al., ECCV 2014 "Let There Be Color!").
Features:
  - Global color consistency (MRF Graph-Cut optimization with GMI data term).
  - Poisson global seam leveling across camera chart boundaries.
  - Image-space local seam blending.
  - Native support for watertight CAD meshes without LSCM matrix singularities.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .config import TexReconConfig

logger = logging.getLogger("TurntableStudio.TexRecon")


class TexReconError(RuntimeError):
    """Raised when a texrecon process fails."""
    pass


def find_texrecon_binary(custom_dir: Optional[str] = None) -> Optional[str]:
    """
    Locates the texrecon executable across:
    1. Custom path / CLI argument (--texrecon_bin)
    2. Environment variables: TEXRECON_BIN, TEXRECON_BIN_DIR, MVS_TEXTURING_BIN
    3. System PATH (which texrecon)
    4. Standard Linux installation paths (/usr/local/bin, /usr/bin, /opt/mvs-texturing/build/apps/texrecon)
    5. Standard Windows installation paths (C:\\texrecon, D:\\texrecon, Visual Studio build dirs)
    """
    bin_name = "texrecon.exe" if sys.platform == "win32" else "texrecon"

    # 1. Direct path or custom directory
    if custom_dir:
        cd = Path(custom_dir)
        for cand in [cd / bin_name, cd / "bin" / bin_name, cd]:
            if cand.is_file():
                return str(cand.resolve())

    # 2. Environment variables
    for env_var in ["TEXRECON_BIN", "TEXRECON_BIN_DIR", "MVS_TEXTURING_ROOT", "MVS_TEXTURING_BIN"]:
        if env_var in os.environ:
            ep = Path(os.environ[env_var])
            for cand in [ep / bin_name, ep / "build" / "apps" / "texrecon" / bin_name, ep / "bin" / bin_name, ep]:
                if cand.is_file():
                    return str(cand.resolve())

    # 3. System PATH
    which_path = shutil.which("texrecon") or shutil.which(bin_name)
    if which_path:
        return str(Path(which_path).resolve())

    # 4. Standard Linux paths
    linux_candidates = [
        Path("/usr/local/bin") / bin_name,
        Path("/usr/bin") / bin_name,
        Path("/opt/mvs-texturing/build/apps/texrecon") / bin_name,
        Path("/opt/mvs-texturing/bin") / bin_name,
        Path.home() / "mvs-texturing" / "build" / "apps" / "texrecon" / bin_name,
        Path.home() / "mvs-texturing" / "bin" / bin_name,
    ]
    for cand in linux_candidates:
        if cand.is_file():
            return str(cand.resolve())

    # 5. Standard Windows paths
    if sys.platform == "win32":
        win_candidates = [
            Path("C:/texrecon") / bin_name,
            Path("C:/texrecon/bin") / bin_name,
            Path("C:/mvs-texturing/build/apps/texrecon/Release") / bin_name,
            Path("C:/mvs-texturing/build/apps/texrecon") / bin_name,
            Path("D:/texrecon") / bin_name,
            Path("D:/texrecon/bin") / bin_name,
            Path("D:/mvs-texturing/build/apps/texrecon/Release") / bin_name,
            Path("D:/mvs-texturing/build/apps/texrecon") / bin_name,
        ]
        for cand in win_candidates:
            if cand.is_file():
                return str(cand.resolve())

        # Glob search on drives C:\ and D:\
        for drive in ["C:\\", "D:\\"]:
            d_path = Path(drive)
            if not d_path.is_dir():
                continue
            try:
                for match_dir in list(d_path.glob("*[Tt]exrecon*")) + list(d_path.glob("*mvs-texturing*")):
                    if match_dir.is_dir():
                        for sub in [
                            match_dir / bin_name,
                            match_dir / "bin" / bin_name,
                            match_dir / "build" / "apps" / "texrecon" / "Release" / bin_name,
                            match_dir / "build" / "apps" / "texrecon" / bin_name,
                        ]:
                            if sub.is_file():
                                return str(sub.resolve())
            except Exception:
                pass

    return None


def export_sfm_to_cam_scene(
    sfm_data_path: str,
    undistorted_images_dir: str,
    output_scene_dir: str,
) -> List[Path]:
    """
    Exports AliceVision cameras_known.sfm data into an MVE / texrecon SCENE_FOLDER
    containing undistorted images and individual .cam files.

    Mathematical Invariants (Rule 3 & Rule 4 in AGENTS.md):
      - In cameras_known.sfm, rotation R is Column-Major (order='F').
      - World-to-Camera translation: T = -R @ C.
      - .cam format:
          Line 1: tx ty tz R00 R01 R02 R10 R11 R12 R20 R21 R22
          Line 2: f d0 d1 paspect ppx ppy
          where f = fx / max(w, h), paspect = fy / fx, ppx = cx / w, ppy = cy / h.
          For undistorted images, distortion coefficients d0=0, d1=0.
    """
    sfm_p = Path(sfm_data_path).resolve()
    undist_dir = Path(undistorted_images_dir).resolve()
    scene_dir = Path(output_scene_dir).resolve()
    scene_dir.mkdir(parents=True, exist_ok=True)

    with open(sfm_p, "r", encoding="utf-8") as f:
        sfm = json.load(f)

    # 1. Parse intrinsics
    intr_data = {}
    for intr in sfm.get("intrinsics", []):
        intr_id = str(intr["intrinsicId"])
        w = int(intr["width"])
        h = int(intr["height"])
        sensor_w = float(intr.get("sensorWidth") or 6.287)
        focal_mm = float(intr.get("focalLength") or 3.9)
        pixel_ratio = float(intr.get("pixelRatio") or 1.0)

        fx = focal_mm * (w / sensor_w)
        fy = fx * pixel_ratio

        raw_pp = intr.get("principalPoint", [0.0, 0.0])
        raw_cx = float(raw_pp[0])
        raw_cy = float(raw_pp[1])

        # AliceVision relative principal point offset conversion (Rule 4)
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

        d_params = intr.get("distortionParams", [])
        dist = np.zeros(5, dtype=np.float64)
        if len(d_params) >= 1: dist[0] = float(d_params[0])
        if len(d_params) >= 2: dist[1] = float(d_params[1])
        if len(d_params) >= 3: dist[4] = float(d_params[2])

        # Undistorted new camera matrix (alpha=0.0)
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)

        intr_data[intr_id] = {
            "w": w,
            "h": h,
            "K_new": K_new,
            "fx": float(K_new[0, 0]),
            "fy": float(K_new[1, 1]),
            "cx": float(K_new[0, 2]),
            "cy": float(K_new[1, 2]),
        }

    # 2. Parse poses
    poses_by_id = {}
    for p in sfm.get("poses", []):
        pose_id = str(p["poseId"])
        # Column-Major order from AliceVision
        R = np.array(p["pose"]["transform"]["rotation"], dtype=np.float64).reshape((3, 3), order="F")
        C = np.array(p["pose"]["transform"]["center"], dtype=np.float64)
        poses_by_id[pose_id] = {"R": R, "C": C}

    # 3. For each view, write image and .cam file
    written_cam_files: List[Path] = []
    nvm_camera_records: List[str] = []

    for v in sfm.get("views", []):
        pose_id = str(v["poseId"])
        if pose_id not in poses_by_id:
            continue

        intr_id = str(v["intrinsicId"])
        c_info = intr_data[intr_id]
        pose = poses_by_id[pose_id]

        R = pose["R"]
        C = pose["C"]
        T = -R @ C  # World to Camera translation

        orig_path = Path(v["path"])
        fname = orig_path.name
        view_id = str(v.get("viewId", ""))

        # Locate undistorted image in undistorted_images_dir
        # Bridge SFM names undistorted files as: {viewId}_{filename}
        candidates = [
            undist_dir / f"{view_id}_{fname}",
            undist_dir / f"{view_id}_{orig_path.stem}.png",
            undist_dir / f"{view_id}_{orig_path.stem}.jpg",
            undist_dir / fname,
            undist_dir / f"{orig_path.stem}.png",
            undist_dir / f"{orig_path.stem}.jpg",
        ]
        undist_img = None
        for cand in candidates:
            if cand.is_file():
                undist_img = cand
                break

        if undist_img is None:
            logger.warning(
                f"Undistorted image not found for view {view_id} ({fname}) in {undist_dir}. "
                f"Falling back to original path: {orig_path}"
            )
            undist_img = orig_path

        # Destination image inside scene_dir
        dst_img = scene_dir / undist_img.name
        if not dst_img.is_file():
            try:
                # Attempt relative symlink
                if hasattr(os, "symlink") and sys.platform != "win32":
                    os.symlink(str(undist_img), str(dst_img))
                else:
                    shutil.copy2(str(undist_img), str(dst_img))
            except Exception:
                shutil.copy2(str(undist_img), str(dst_img))

        # Camera parameters
        w = c_info["w"]
        h = c_info["h"]
        max_dim = max(w, h)
        f_norm = c_info["fx"] / max_dim
        paspect = c_info["fy"] / c_info["fx"]
        ppx = c_info["cx"] / w
        ppy = c_info["cy"] / h

        # Rotation row-major 9 elements
        r_flat = R.flatten()

        cam_file = scene_dir / f"{dst_img.stem}.cam"
        with open(cam_file, "w", encoding="utf-8") as cf:
            # Line 1: Extrinsics
            cf.write(
                f"{T[0]:.9f} {T[1]:.9f} {T[2]:.9f} "
                f"{r_flat[0]:.9f} {r_flat[1]:.9f} {r_flat[2]:.9f} "
                f"{r_flat[3]:.9f} {r_flat[4]:.9f} {r_flat[5]:.9f} "
                f"{r_flat[6]:.9f} {r_flat[7]:.9f} {r_flat[8]:.9f}\n"
            )
            # Line 2: Intrinsics (f d0 d1 paspect ppx ppy)
            cf.write(f"{f_norm:.9f} 0.0 0.0 {paspect:.9f} {ppx:.9f} {ppy:.9f}\n")

        written_cam_files.append(cam_file)

        # Collect record for VisualSfM NVM format as well
        # COLMAP/SciPy quaternion -> [qw, qx, qy, qz]
        q_scipy = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
        qw, qx, qy, qz = q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]
        f_px = c_info["fx"]
        nvm_camera_records.append(
            f"{dst_img.name} {f_px:.4f} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {C[0]:.8f} {C[1]:.8f} {C[2]:.8f} 0 0"
        )

    # 4. Also write scene.nvm in scene_dir as alternative/universal bundle file
    nvm_path = scene_dir / "scene.nvm"
    with open(nvm_path, "w", encoding="utf-8") as nf:
        nf.write("NVM_V3\n\n")
        nf.write(f"{len(nvm_camera_records)}\n")
        for rec in nvm_camera_records:
            nf.write(f"{rec}\n")
        nf.write("\n0\n")  # 0 3D points required for pure texturing

    logger.info(
        f"Exported {len(written_cam_files)} .cam parameters and scene.nvm to {scene_dir}"
    )
    return written_cam_files


class TexReconRunner:
    """Manages execution of mvs-texturing (texrecon)."""

    def __init__(self, config: TexReconConfig, work_dir: str):
        self.config = config
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Locate binary if not explicitly provided
        if not self.config.texrecon_bin:
            self.config.texrecon_bin = find_texrecon_binary(self.config.bin_dir)

    def run_texturing(
        self,
        sfm_data_path: str,
        input_mesh_path: str,
        images_folder: str,
        out_prefix: Optional[str] = None,
        mock_mode: bool = False,
    ) -> Tuple[str, str]:
        """
        Executes texrecon on the input mesh and cameras.

        Arguments:
            sfm_data_path: Path to cameras_known.sfm
            input_mesh_path: Path to input OBJ or PLY mesh
            images_folder: Path to undistorted images directory
            out_prefix: Output path prefix for generated textured OBJ/MTL
            mock_mode: If True, simulates execution for testing without texrecon binary

        Returns:
            Tuple of (textured_obj_path, textured_mtl_path)
        """
        in_mesh_p = Path(input_mesh_path).resolve()
        if not in_mesh_p.is_file():
            raise FileNotFoundError(f"Input mesh not found: {in_mesh_p}")

        # 1. Prepare scene folder with .cam files and images
        scene_dir = self.work_dir / "scene"
        export_sfm_to_cam_scene(
            sfm_data_path=sfm_data_path,
            undistorted_images_dir=images_folder,
            output_scene_dir=str(scene_dir),
        )

        # 2. texrecon expects input mesh in .ply format. Convert if OBJ.
        if in_mesh_p.suffix.lower() == ".ply":
            mesh_ply_p = in_mesh_p
        else:
            mesh_ply_p = self.work_dir / f"{in_mesh_p.stem}.ply"
            logger.info(f"Converting input mesh {in_mesh_p.name} to PLY format for texrecon: {mesh_ply_p.name}")
            mesh = trimesh.load(str(in_mesh_p), process=False)
            mesh.export(str(mesh_ply_p))

        # 3. Setup output prefix
        if out_prefix:
            out_pref = Path(out_prefix).resolve()
        else:
            out_pref = self.work_dir / "model_textured"
        out_pref.parent.mkdir(parents=True, exist_ok=True)

        # Ensure any leftover "tmp" directory from a prior run is removed to prevent texrecon abort
        tmp_dir = out_pref.parent / "tmp"
        if tmp_dir.is_dir():
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

        final_obj = Path(f"{out_pref}.obj")
        final_mtl = Path(f"{out_pref}.mtl")

        # Handle mock mode or missing binary when mock is allowed
        if mock_mode or not self.config.texrecon_bin:
            if not self.config.texrecon_bin and not mock_mode:
                raise FileNotFoundError(
                    "texrecon executable not found! Please specify --texrecon_bin "
                    "or ensure mvs-texturing is installed."
                )
            logger.info("[Mock Mode] Generating simulated texrecon textured OBJ output...")
            return self._generate_mock_output(in_mesh_p, final_obj, final_mtl)

        # 4. Build command line arguments
        cmd = [
            str(self.config.texrecon_bin),
            f"--data_term={self.config.data_term}",
            f"--smoothness_term={self.config.smoothness_term}",
            f"--outlier_removal={self.config.outlier_removal}",
            f"--tone_mapping={self.config.tone_mapping}",
        ]

        if self.config.skip_global_seam_leveling:
            cmd.append("--skip_global_seam_leveling")
        if self.config.skip_local_seam_leveling:
            cmd.append("--skip_local_seam_leveling")
        if self.config.num_threads:
            cmd.append(f"--num_threads={self.config.num_threads}")
        if self.config.write_timings:
            cmd.append("--write_timings")
        if self.config.write_intermediate_results:
            cmd.append("--write_intermediate_results")

        # Positional arguments: IN_SCENE IN_MESH OUT_PREFIX
        cmd.extend([
            str(scene_dir),
            str(mesh_ply_p),
            str(out_pref),
        ])

        logger.info(f"Executing texrecon: {' '.join(cmd)}")
        t0 = time.time()
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self.work_dir),
        )
        elapsed = time.time() - t0
        logger.info(f"texrecon finished in {elapsed:.2f}s with return code {res.returncode}")

        if res.returncode != 0:
            logger.error(f"texrecon stdout/stderr:\n{res.stdout}")
            raise TexReconError(f"texrecon process failed with return code {res.returncode}")

        if not final_obj.is_file():
            # Check if texrecon outputted with slightly different suffix
            cands = list(out_pref.parent.glob(f"{out_pref.name}*.obj"))
            if cands:
                final_obj = cands[0]
            else:
                raise FileNotFoundError(f"Expected texrecon output OBJ not found: {final_obj}")

        logger.info(f"Successfully generated textured OBJ with texrecon: {final_obj.name}")
        return str(final_obj), str(final_mtl)

    def _generate_mock_output(
        self,
        in_mesh_p: Path,
        final_obj: Path,
        final_mtl: Path,
    ) -> Tuple[str, str]:
        """Creates dummy textured OBJ + MTL + PNG texture for test environments."""
        from PIL import Image

        mesh = trimesh.load(str(in_mesh_p), process=False)
        tex_img_path = final_obj.parent / f"{final_obj.stem}_material_0_map_Kd.png"

        # Create dummy texture map
        img = Image.new("RGB", (512, 512), color=(140, 140, 140))
        img.save(str(tex_img_path))

        # Write MTL
        with open(final_mtl, "w", encoding="utf-8") as f:
            f.write(
                f"# Mock Material for {final_obj.name}\n"
                f"newmtl material_0\n"
                f"Ka 1.0 1.0 1.0\n"
                f"Kd 1.0 1.0 1.0\n"
                f"map_Kd {tex_img_path.name}\n"
            )

        # Write OBJ with UV coordinates
        with open(final_obj, "w", encoding="utf-8") as f:
            f.write(f"mtllib {final_mtl.name}\n")
            f.write(f"usemtl material_0\n")
            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            # UV coordinates
            f.write("vt 0.0 0.0\nvt 1.0 0.0\nvt 0.5 1.0\n")
            for face in mesh.faces:
                f.write(f"f {face[0]+1}/1 {face[1]+1}/2 {face[2]+1}/3\n")

        return str(final_obj), str(final_mtl)
