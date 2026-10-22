"""
alicevision_runner.py — High-fidelity Multi-Band texturing using AliceVision Texturing.
Uses LSCM UV unwrapping, frequency decomposition (multi-band blending),
exposure compensation, and Poisson hole-filling (--fillHoles 1).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AliceVisionConfig

logger = logging.getLogger("TurntableStudio.AliceVision")


class AliceVisionError(RuntimeError):
    """Raised when an AliceVision process fails."""
    pass


def find_alicevision_binary(bin_name: str = "aliceVision_texturing", custom_dir: Optional[str] = None) -> Optional[str]:
    """
    Locates an AliceVision binary (e.g. aliceVision_texturing or aliceVision_texturing.exe)
    by checking:
    1. Custom directory argument / CLI override.
    2. Environment variable ALICEVISION_BIN_DIR or ALICEVISION_ROOT.
    3. System PATH.
    4. Sibling/parent Meshroom directories (Linux and Windows).
    5. Common Windows install locations (C:\\, D:\\, Program Files).
    """
    exe_name = f"{bin_name}.exe" if sys.platform == "win32" else bin_name

    # 1. Custom directory
    if custom_dir:
        cd = Path(custom_dir)
        for cand in [cd / exe_name, cd / "bin" / exe_name, cd]:
            if cand.is_file():
                return str(cand.resolve())

    # 2. Environment variables
    for env_var in ["ALICEVISION_BIN_DIR", "ALICEVISION_ROOT", "MESHROOM_ROOT"]:
        if env_var in os.environ:
            ep = Path(os.environ[env_var])
            for cand in [ep / exe_name, ep / "bin" / exe_name, ep / "aliceVision" / "bin" / exe_name]:
                if cand.is_file():
                    return str(cand.resolve())

    # 3. System PATH
    which_path = shutil.which(bin_name) or shutil.which(exe_name)
    if which_path:
        return str(Path(which_path).resolve())

    # 4. Known local project directories (e.g. Meshroom-2025.1.0 in workspace)
    project_root = Path(__file__).resolve().parent.parent.parent
    for mdir in [
        project_root / "Meshroom-2025.1.0" / "aliceVision" / "bin",
        project_root.parent / "Meshroom-2025.1.0" / "aliceVision" / "bin",
    ]:
        cand = mdir / exe_name
        if cand.is_file():
            return str(cand.resolve())

    # 5. Windows drive scanning (C:\ and D:\ for Meshroom folders)
    if sys.platform == "win32":
        for drive in ["C:\\", "D:\\"]:
            d_path = Path(drive)
            if not d_path.is_dir():
                continue
            try:
                for match_dir in list(d_path.glob("*[Mm]eshroom*")) + list(d_path.glob("*[Aa]lice[Vv]ision*")):
                    if match_dir.is_dir():
                        for sub in [
                            match_dir / "aliceVision" / "bin" / exe_name,
                            match_dir / "bin" / exe_name,
                            match_dir / exe_name,
                        ]:
                            if sub.is_file():
                                return str(sub.resolve())
            except Exception:
                pass

    return None


class AliceVisionRunner:
    """Manages execution of AliceVision Texturing."""

    def __init__(self, config: AliceVisionConfig, work_dir: str):
        self.config = config
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Validate or locate binary
        if not self.config.texturing_bin:
            self.config.texturing_bin = find_alicevision_binary(
                "aliceVision_texturing", self.config.bin_dir
            )

        if not self.config.texturing_bin:
            raise FileNotFoundError(
                "aliceVision_texturing executable not found! "
                "Please specify --alicevision_bin or set ALICEVISION_ROOT."
            )

        # Determine root directory for OCIO config and libraries
        bin_p = Path(self.config.texturing_bin).resolve()
        if bin_p.parent.name == "bin":
            self.root_dir = bin_p.parent.parent
        else:
            self.root_dir = bin_p.parent

    def _prepare_environment(self) -> Dict[str, str]:
        """Prepares environment variables (ALICEVISION_ROOT, LD_LIBRARY_PATH, PATH)."""
        env = os.environ.copy()
        root_str = str(self.root_dir.resolve())
        env["ALICEVISION_ROOT"] = root_str

        lib_dir = self.root_dir / "lib"
        bin_dir = self.root_dir / "bin"

        if sys.platform != "win32":
            # Linux: prepend to LD_LIBRARY_PATH
            curr_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{curr_ld}" if lib_dir.is_dir() else curr_ld
        else:
            # Windows: prepend to PATH
            curr_path = env.get("PATH", "")
            add_paths = []
            if bin_dir.is_dir(): add_paths.append(str(bin_dir))
            if lib_dir.is_dir(): add_paths.append(str(lib_dir))
            if add_paths:
                env["PATH"] = ";".join(add_paths) + ";" + curr_path

        return env

    def prepare_sfm_data(
        self,
        input_sfm_path: str,
        input_mesh_path: str,
        undistorted_dir: Optional[str] = None,
    ) -> str:
        """
        Enriches cameras_known.sfm for AliceVision texturing requirements:
        1. Ensures all required intrinsic fields (serialNumber, pxFocalLength, etc.) are present.
        2. Points views to undistorted images if available (setting distortionType to 'none').
        3. Populates the 'structure' field with 3D reference landmarks (derived from mesh vertices)
           and 2D observations so AliceVision can remap visibility without throwing
           'Cannot remap visibilities as there is no reference points'.
        """
        import json
        import cv2
        import numpy as np
        import trimesh

        with open(input_sfm_path, "r", encoding="utf-8") as f:
            sfm_data = json.load(f)

        # 1. Enrich intrinsics
        intrinsics_dict = {}
        for intr in sfm_data.get("intrinsics", []):
            iid = str(intr.get("intrinsicId", "0"))
            w = float(intr.get("width", 4056))
            h = float(intr.get("height", 3040))
            sw = float(intr.get("sensorWidth", 6.287))
            f_mm = float(intr.get("focalLength", 6.0))
            px_f = f_mm * (w / sw)

            intr["serialNumber"] = intr.get("serialNumber") or f"camera_{iid}"
            intr["pxFocalLength"] = str(round(px_f, 4))
            intr["pxInitialFocalLength"] = str(round(px_f, 4))
            intr["sensorHeight"] = str(round(sw * (h / w), 4))
            intr["pixelRatio"] = "1"
            intr["type"] = intr.get("type", "pinhole")
            intr["initializationMode"] = intr.get("initializationMode", "calibrated")
            intr["distortionInitializationMode"] = intr.get("distortionInitializationMode", "calibrated")
            intr["locked"] = "true"
            intr["undistortionType"] = "none"
            intr["undistortionParams"] = ""
            intr["undistortionOffset"] = ["0", "0"]

            if undistorted_dir and Path(undistorted_dir).is_dir():
                intr["distortionType"] = "none"
                intr["distortionParams"] = []

            intrinsics_dict[iid] = intr

        # 2. Update view image paths if undistorted directory is provided
        if undistorted_dir and Path(undistorted_dir).is_dir():
            undist_path = Path(undistorted_dir)
            for v in sfm_data.get("views", []):
                orig_file = Path(v["path"]).name
                cand = undist_path / orig_file
                if cand.is_file():
                    v["path"] = str(cand.resolve())

        # 3. Populate structure landmarks if empty
        structure = sfm_data.get("structure", [])
        if len(structure) < 20 and Path(input_mesh_path).is_file():
            logger.info("Populating AliceVision SFM reference landmarks from mesh geometry...")
            mesh = trimesh.load(input_mesh_path, process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)

            # Subsample up to 2,000 vertices for fast reference mapping
            if len(vertices) > 2000:
                indices = np.linspace(0, len(vertices) - 1, 2000, dtype=int)
                sample_pts = vertices[indices]
            else:
                sample_pts = vertices
                indices = np.arange(len(vertices))

            # Build pose transforms
            pose_map = {}
            for p in sfm_data.get("poses", []):
                pid = str(p["poseId"])
                t = p["pose"]["transform"]
                R = np.array([float(x) for x in t["rotation"]], dtype=np.float64).reshape((3, 3), order="F")
                C = np.array([float(x) for x in t["center"]], dtype=np.float64)
                pose_map[pid] = (R, C)

            new_structure = []
            for pt_idx, (orig_idx, pt) in enumerate(zip(indices, sample_pts)):
                obs = []
                for v in sfm_data.get("views", []):
                    vid = str(v["viewId"])
                    pid = str(v.get("poseId", vid))
                    if pid not in pose_map:
                        continue
                    R, C = pose_map[pid]
                    # Transform to camera space
                    p_cam = R @ (pt - C)
                    if p_cam[2] <= 0.02:  # Behind or too close to lens
                        continue

                    intr = intrinsics_dict.get(str(v["intrinsicId"]))
                    if not intr:
                        continue

                    w = float(intr["width"])
                    h = float(intr["height"])
                    fx = float(intr["pxFocalLength"])
                    fy = fx

                    # Principal point
                    pp = intr.get("principalPoint", ["0.0", "0.0"])
                    cx = float(pp[0]) + w / 2.0
                    cy = float(pp[1]) + h / 2.0

                    u = fx * (p_cam[0] / p_cam[2]) + cx
                    v_px = fy * (p_cam[1] / p_cam[2]) + cy

                    # Margin check (5 px from edge)
                    if 5.0 <= u <= (w - 5.0) and 5.0 <= v_px <= (h - 5.0):
                        obs.append({
                            "observationId": vid,
                            "featureId": str(orig_idx),
                            "x": [str(round(u, 2)), str(round(v_px, 2))],
                            "scale": "1.0",
                        })

                if len(obs) >= 2:
                    new_structure.append({
                        "landmarkId": str(pt_idx + 1),
                        "descType": "sift",
                        "color": ["128", "128", "128"],
                        "X": [str(round(x, 6)) for x in pt],
                        "observations": obs,
                    })

            sfm_data["structure"] = new_structure
            logger.info(f"Generated {len(new_structure)} reference landmarks for AliceVision visibility remapping.")

        prepared_sfm_path = self.work_dir / "cameras_alicevision.sfm"
        with open(prepared_sfm_path, "w", encoding="utf-8") as f:
            json.dump(sfm_data, f, indent=2)

        return str(prepared_sfm_path.resolve())

    def run_texturing(
        self,
        sfm_data_path: str,
        input_mesh_path: str,
        images_folder: Optional[str] = None,
        output_mesh_name: str = "texturedMesh.obj",
    ) -> Tuple[str, str]:
        """
        Executes aliceVision_texturing on the sealed watertight geometry.
        Returns paths to (output_obj_path, output_mtl_path).
        """
        bin_path = self.config.texturing_bin
        logger.info(f"Running AliceVision Texturing: {bin_path}")

        # Prepare and enrich SFM for AliceVision 3.0 requirements
        prepared_sfm = self.prepare_sfm_data(
            input_sfm_path=sfm_data_path,
            input_mesh_path=input_mesh_path,
            undistorted_dir=images_folder,
        )

        cmd = [
            bin_path,
            "-i", str(Path(prepared_sfm).resolve()),
            "--inputMesh", str(Path(input_mesh_path).resolve()),
            "-o", str(self.work_dir),
            "--textureSide", str(self.config.texture_side),
            "--outputMeshFileType", "obj",
            "--colorMappingFileType", self.config.color_mapping_type,
            "--unwrapMethod", self.config.unwrap_method,
            "--useUDIM", "1" if self.config.use_udim else "0",
            "--fillHoles", "1" if self.config.fill_holes else "0",
            "--correctEV", "1" if self.config.correct_ev else "0",
            "--padding", str(self.config.padding),
            "--multiBandDownscale", str(self.config.multi_band_downscale),
            "-v", "info",
        ]

        env = self._prepare_environment()
        start_time = time.time()
        logger.info(f"Executing: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        if proc.stdout:
            for line in proc.stdout:
                line_str = line.rstrip()
                output_lines.append(line_str)
                if any(k in line_str.lower() for k in ["error", "warning", "progress", "%", "completed"]):
                    logger.info(f"[AliceVision] {line_str}")
                else:
                    logger.debug(f"[AliceVision] {line_str}")

        ret = proc.wait()
        elapsed = time.time() - start_time

        if ret != 0:
            tail = "\n".join(output_lines[-25:])
            err_msg = f"AliceVision Texturing failed with code {ret} in {elapsed:.1f}s!\nLast output lines:\n{tail}"
            logger.error(err_msg)
            raise AliceVisionError(err_msg)

        logger.info(f"Completed AliceVision Texturing in {elapsed:.2f}s.")

        # Find output OBJ and MTL
        out_obj = self.work_dir / output_mesh_name
        if not out_obj.is_file():
            # Search for any OBJ generated
            objs = list(self.work_dir.glob("*.obj"))
            if objs:
                out_obj = objs[0]
            else:
                raise FileNotFoundError(f"AliceVision texturing OBJ missing in {self.work_dir}")

        out_mtl = out_obj.parent / f"{out_obj.stem}.mtl"
        if not out_mtl.is_file():
            mtls = list(self.work_dir.glob("*.mtl"))
            if mtls:
                out_mtl = mtls[0]

        return str(out_obj), str(out_mtl)
