"""
alicevision_runner.py — High-fidelity Multi-Band texturing using AliceVision Texturing.
Uses Basic/LSCM UV unwrapping, frequency decomposition (multi-band blending),
direct mesh raycasting (MeshItself visibility), exposure compensation,
and Poisson hole-filling (--fillHoles 1).
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

    def prepare_sfm_and_images(
        self,
        input_sfm_path: str,
        input_mesh_path: str,
        undistorted_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Prepares cameras_known.sfm and standardized undistorted images for AliceVision 3.0:
        1. Copies/symlinks undistorted images into a dedicated images folder named <viewId>.<ext>.
        2. Configures intrinsics with distortionType="none".
        3. Returns (prepared_sfm_path, images_folder_path).
        """
        with open(input_sfm_path, "r", encoding="utf-8") as f:
            sfm_data = json.load(f)

        images_folder = self.work_dir / "images"
        images_folder.mkdir(parents=True, exist_ok=True)

        undist_p = Path(undistorted_dir).resolve() if undistorted_dir else None

        # 1. Standardize undistorted images (<viewId>.<ext>)
        for v in sfm_data.get("views", []):
            vid = str(v["viewId"])
            orig_path = Path(v["path"])
            orig_file = orig_path.name

            found_img: Optional[Path] = None
            if undist_p and undist_p.is_dir():
                candidates = [
                    undist_p / f"{vid}_{orig_file}",
                    undist_p / f"{vid}.png",
                    undist_p / f"{vid}.jpg",
                    undist_p / f"{vid}.exr",
                    undist_p / orig_file,
                ]
                for cand in candidates:
                    if cand.is_file():
                        found_img = cand
                        break

            if not found_img and orig_path.is_file():
                found_img = orig_path

            if found_img:
                ext = found_img.suffix.lower()
                dest_img = images_folder / f"{vid}{ext}"
                if not dest_img.is_file():
                    try:
                        # Try relative symlink first on Linux
                        if sys.platform != "win32":
                            dest_img.symlink_to(found_img)
                        else:
                            shutil.copy2(found_img, dest_img)
                    except Exception:
                        shutil.copy2(found_img, dest_img)

                v["path"] = str(dest_img.resolve())

        # 2. Enrich intrinsics for AliceVision requirements
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
            intr["distortionType"] = "none"
            intr["distortionParams"] = []
            intr["undistortionType"] = "none"
            intr["undistortionParams"] = ""
            intr["undistortionOffset"] = ["0", "0"]
            intrinsics_dict[iid] = intr

        # 3. If structure is requested or fallback needed, compute backface-culled landmarks
        structure = sfm_data.get("structure", [])
        if len(structure) == 0 and Path(input_mesh_path).is_file():
            try:
                import numpy as np
                import trimesh
                mesh = trimesh.load(input_mesh_path, process=False)
                vertices = np.asarray(mesh.vertices, dtype=np.float64)
                normals = np.asarray(mesh.vertex_normals, dtype=np.float64) if len(mesh.vertex_normals) == len(vertices) else None

                # Subsample 1,000 vertices for reference
                sample_count = min(1000, len(vertices))
                indices = np.linspace(0, len(vertices) - 1, sample_count, dtype=int)
                sample_pts = vertices[indices]

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
                    pt_norm = normals[orig_idx] if normals is not None else None

                    for v in sfm_data.get("views", []):
                        vid = str(v["viewId"])
                        pid = str(v.get("poseId", vid))
                        if pid not in pose_map:
                            continue
                        R, C = pose_map[pid]

                        # Backface culling: Vector from point to camera must align with surface normal
                        cam_dir = C - pt
                        dist = np.linalg.norm(cam_dir)
                        if dist > 1e-6 and pt_norm is not None:
                            cos_angle = np.dot(pt_norm, cam_dir / dist)
                            if cos_angle < 0.15:  # Facing > 81 degrees away -> backfacing
                                continue

                        p_cam = R @ (pt - C)
                        if p_cam[2] <= 0.02:
                            continue

                        intr = intrinsics_dict.get(str(v["intrinsicId"]))
                        if not intr:
                            continue

                        w = float(intr["width"])
                        h = float(intr["height"])
                        fx = float(intr["pxFocalLength"])
                        pp = intr.get("principalPoint", ["0.0", "0.0"])
                        cx = float(pp[0]) + w / 2.0
                        cy = float(pp[1]) + h / 2.0

                        u = fx * (p_cam[0] / p_cam[2]) + cx
                        v_px = fx * (p_cam[1] / p_cam[2]) + cy

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
            except Exception as e:
                logger.debug(f"Could not build landmark structure: {e}")

        prepared_sfm_path = self.work_dir / "cameras_alicevision.sfm"
        with open(prepared_sfm_path, "w", encoding="utf-8") as f:
            json.dump(sfm_data, f, indent=2)

        return str(prepared_sfm_path.resolve()), str(images_folder.resolve())

    def prepare_sfm_data(
        self,
        input_sfm_path: str,
        input_mesh_path: str,
        undistorted_dir: Optional[str] = None,
    ) -> str:
        """Compatibility wrapper returning prepared SFM path."""
        sfm_p, _ = self.prepare_sfm_and_images(input_sfm_path, input_mesh_path, undistorted_dir)
        return sfm_p

    def prepare_mesh_with_material(self, mesh_path: str) -> str:
        """
        Ensures the input OBJ has an associated .mtl and usemtl tag.
        AliceVision's loadWithAtlas partitions textures by material groups.
        """
        in_p = Path(mesh_path).resolve()
        if in_p.suffix.lower() != ".obj":
            return str(in_p)

        mtl_file = in_p.parent / f"{in_p.stem}.mtl"
        has_usemtl = False
        has_mtllib = False

        try:
            with open(in_p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("mtllib "):
                        has_mtllib = True
                    elif line.startswith("usemtl "):
                        has_usemtl = True
                        break
        except Exception:
            pass

        if has_mtllib and has_usemtl:
            return str(in_p)

        # Create basic material file
        if not mtl_file.is_file():
            with open(mtl_file, "w", encoding="utf-8") as mf:
                mf.write("newmtl material_0\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nillum 1\n")

        # Create a mesh copy with material headers
        mesh_with_mtl = self.work_dir / f"{in_p.stem}_with_mtl.obj"
        with open(in_p, "r", encoding="utf-8", errors="ignore") as src, open(mesh_with_mtl, "w", encoding="utf-8") as dst:
            dst.write(f"mtllib {mtl_file.name}\nusemtl material_0\n")
            for line in src:
                if not line.startswith("mtllib ") and not line.startswith("usemtl "):
                    dst.write(line)

        return str(mesh_with_mtl.resolve())

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

        # 1. Prepare SFM and standardize undistorted images folder
        prepared_sfm, av_images_folder = self.prepare_sfm_and_images(
            input_sfm_path=sfm_data_path,
            input_mesh_path=input_mesh_path,
            undistorted_dir=images_folder,
        )

        # 2. Ensure input mesh has material definition
        prepared_mesh = self.prepare_mesh_with_material(input_mesh_path)

        cmd = [
            bin_path,
            "-i", str(Path(prepared_sfm).resolve()),
            "--inputMesh", str(Path(prepared_mesh).resolve()),
            "-o", str(self.work_dir),
            "--imagesFolder", str(av_images_folder),
            "--textureSide", str(self.config.texture_side),
            "--downscale", "1",
            "--outputMeshFileType", "obj",
            "--colorMappingFileType", self.config.color_mapping_type,
            "--unwrapMethod", self.config.unwrap_method,
            "--useUDIM", "1" if self.config.use_udim else "0",
            "--fillHoles", "1" if self.config.fill_holes else "0",
            "--correctEV", "1" if self.config.correct_ev else "0",
            "--padding", str(self.config.padding),
            "--multiBandDownscale", str(self.config.multi_band_downscale),
            "--multiBandNbContrib", "1", "5", "10", "0",
            "--useScore", "1",
            "--bestScoreThreshold", "0.1",
            "--angleHardThreshold", "90.0",
            "--visibilityRemappingMethod", getattr(self.config, "visibility_remapping_method", "MeshItself"),
            "--subdivisionTargetRatio", "0.8",
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
