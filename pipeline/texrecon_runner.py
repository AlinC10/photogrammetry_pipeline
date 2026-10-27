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


# The only data terms texrecon accepts (tex/settings.h: choice_strings<DataTerm>)
TEXRECON_DATA_TERMS = ("gmi", "area")


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


def _link_or_copy(src: Path, dst: Path) -> None:
    """Relative symlink (survives moving the whole output folder), copy as fallback."""
    try:
        if hasattr(os, "symlink") and sys.platform != "win32":
            os.symlink(os.path.relpath(str(src), str(dst.parent)), str(dst))
            return
    except OSError:
        pass
    shutil.copy2(str(src), str(dst))


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
          where f = fx / max(w, h), paspect = fy / fx, ppx = (cx + 0.5) / w, ppy = (cy + 0.5) / h.
          MVE puts pixel centres at +0.5 (TextureView::get_pixel_coords subtracts 0.5), while
          OpenCV/AliceVision centres are integers, hence the half-pixel shift.
          For undistorted images, distortion coefficients d0=0, d1=0.

    The scene folder is rebuilt from scratch on every call: texrecon loads EVERY .cam file it
    finds, so leftovers from a previous run in the same output folder become extra (wrong) views.
    Each view is written as `<viewId>_<stem>` so the two rig cameras (same file names in
    rig/0 and rig/1) can never overwrite each other's .cam file.
    """
    sfm_p = Path(sfm_data_path).resolve()
    undist_dir = Path(undistorted_images_dir).resolve()
    scene_dir = Path(output_scene_dir).resolve()
    if scene_dir.exists():
        shutil.rmtree(str(scene_dir))
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

    # A bare file name is only trusted if no other view has the same one (rig/0 and rig/1 do).
    name_counts: Dict[str, int] = {}
    for v in sfm.get("views", []):
        n = Path(v["path"]).name
        name_counts[n] = name_counts.get(n, 0) + 1

    missing: List[str] = []
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

        # Locate the texturing image (undistorted with K_new) for this view.
        # Bridge SFM / texture_prep name them: {viewId}_{stem}.png or {viewId}_{filename}
        candidates = [
            undist_dir / f"{view_id}_{orig_path.stem}.png",
            undist_dir / f"{view_id}_{fname}",
            undist_dir / f"{view_id}_{orig_path.stem}.jpg",
        ]
        if name_counts.get(fname, 0) == 1:
            candidates += [
                undist_dir / fname,
                undist_dir / f"{orig_path.stem}.png",
                undist_dir / f"{orig_path.stem}.jpg",
            ]
        undist_img = next((c for c in candidates if c.is_file()), None)

        if undist_img is None:
            # Never fall back to the original photo: it is distorted and unmasked, and the
            # .cam written below describes the undistorted camera K_new.
            missing.append(f"{view_id} ({fname})")
            continue

        # Destination image inside scene_dir, unique per view
        dst_img = scene_dir / f"{view_id}_{orig_path.stem}{undist_img.suffix.lower()}"
        _link_or_copy(undist_img, dst_img)

        # Camera parameters
        w = c_info["w"]
        h = c_info["h"]
        max_dim = max(w, h)
        f_norm = c_info["fx"] / max_dim
        paspect = c_info["fy"] / c_info["fx"]
        ppx = (c_info["cx"] + 0.5) / w
        ppy = (c_info["cy"] + 0.5) / h

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

    if missing:
        raise TexReconError(
            f"Texturing images not found in {undist_dir} for {len(missing)} view(s): "
            f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
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


def remove_texrecon_outputs(out_prefix: Path) -> int:
    """
    Deletes every file texrecon writes for `out_prefix` (OBJ, MTL, atlases, intermediate results).
    Patterns are anchored on the exact prefix, so "model_textured" never touches "model_textured_low_*".
    """
    out_prefix = Path(out_prefix)
    name = out_prefix.name
    patterns = [
        f"{name}.obj", f"{name}.mtl", f"{name}.conf",
        f"{name}_material[0-9]*_map_Kd.*",
        f"{name}_labeling.vec", f"{name}_data_costs.spt", f"{name}_timings.csv",
        f"{name}_view_selection*",
    ]
    removed = 0
    for pat in patterns:
        for f in out_prefix.parent.glob(pat):
            if f.is_file() or f.is_symlink():
                f.unlink()
                removed += 1
    return removed


def _read_obj_faces(obj_path: Path):
    """Minimal OBJ reader for texrecon output: (vt [N,2], face v [F,3], face vt [F,3], face material, names, mtllib)."""
    vt: List[Tuple[float, float]] = []
    fv: List[Tuple[int, int, int]] = []
    ft: List[Tuple[int, int, int]] = []
    fm: List[int] = []
    names: List[str] = []
    index: Dict[str, int] = {}
    cur = -1
    mtllib = None
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("vt "):
                p = line.split()
                vt.append((float(p[1]), float(p[2])))
            elif line.startswith("f "):
                p = [c.split("/") for c in line.split()[1:4]]
                if len(p) == 3 and all(len(c) > 1 and c[1] for c in p):
                    fv.append((int(p[0][0]), int(p[1][0]), int(p[2][0])))
                    ft.append((int(p[0][1]), int(p[1][1]), int(p[2][1])))
                    fm.append(cur)
            elif line.startswith("usemtl "):
                n = line.split(maxsplit=1)[1].strip()
                if n not in index:
                    index[n] = len(names)
                    names.append(n)
                cur = index[n]
            elif line.startswith("mtllib "):
                mtllib = line.split(maxsplit=1)[1].strip()
    return (np.array(vt, dtype=np.float64).reshape(-1, 2), np.array(fv, dtype=np.int64).reshape(-1, 3) - 1,
            np.array(ft, dtype=np.int64).reshape(-1, 3) - 1, np.array(fm, dtype=np.int64), names, mtllib)


def fill_unseen_face_texels(
    obj_path: str,
    min_faces: int = 3,
    fallback_rgb: Tuple[int, int, int] = (25, 25, 25),
    max_samples: int = 20000,
) -> Optional[Tuple[int, int, int]]:
    """
    With --keep_unseen_faces texrecon maps every face no view could see onto one 3x3 black patch
    (generate_texture_patches.cpp) inside one atlas, surrounded by `atlas_size >> 7` px of padding.
    All those faces therefore share the same three texture coordinates. This repaints that patch
    with the median colour of the textured faces touching the unseen ones, so undersides and the
    flat base read as the surrounding plastic instead of black holes.
    Returns the RGB colour used, or None if the OBJ has no unseen-face patch.
    """
    obj_p = Path(obj_path)
    vt, fv, ft, fm, names, mtllib = _read_obj_faces(obj_p)
    if len(ft) == 0:
        return None
    _, inverse, counts = np.unique(np.sort(ft, axis=1), axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    best = int(np.argmax(counts))
    if counts[best] < min_faces:
        return None
    unseen = inverse == best

    maps: Dict[str, str] = {}
    if mtllib and (obj_p.parent / mtllib).is_file():
        cur = None
        for line in (obj_p.parent / mtllib).read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("newmtl "):
                cur = line.split(maxsplit=1)[1].strip()
            elif line.startswith("map_Kd") and cur:
                maps[cur] = line.split(maxsplit=1)[1].strip()

    def atlas_path(mat_idx: int) -> Optional[Path]:
        if mat_idx < 0 or mat_idx >= len(names) or names[mat_idx] not in maps:
            return None
        p = obj_p.parent / maps[names[mat_idx]]
        return p if p.is_file() else None

    # Median colour of textured faces sharing a vertex with an unseen face
    seen_v = np.zeros(int(fv.max()) + 1, dtype=bool)
    seen_v[fv[unseen].ravel()] = True
    neigh = np.nonzero(~unseen & seen_v[fv].any(axis=1))[0]
    if len(neigh) == 0:
        neigh = np.nonzero(~unseen)[0]
    if len(neigh) > max_samples:
        neigh = np.random.default_rng(0).choice(neigh, max_samples, replace=False)
    samples = []
    for mat_idx in np.unique(fm[neigh]):
        ap = atlas_path(int(mat_idx))
        if ap is None:
            continue
        img = cv2.imread(str(ap), cv2.IMREAD_COLOR)
        if img is None:
            continue
        sel = neigh[fm[neigh] == mat_idx]
        uv = vt[ft[sel]].mean(axis=1)
        x = np.clip((uv[:, 0] * img.shape[1]).astype(int), 0, img.shape[1] - 1)
        y = np.clip(((1.0 - uv[:, 1]) * img.shape[0]).astype(int), 0, img.shape[0] - 1)
        samples.append(img[y, x])
    if samples:
        bgr = np.median(np.concatenate(samples), axis=0).round().astype(int)
        colour = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    else:
        colour = tuple(int(c) for c in fallback_rgb)

    mat_idx = int(fm[unseen][0])
    ap = atlas_path(mat_idx)
    if ap is None:
        return None
    img = cv2.imread(str(ap), cv2.IMREAD_COLOR)
    size = img.shape[1]
    padding = size >> 7
    tri = vt[ft[unseen][0]]
    # patch-relative texcoords are (2,1), (1,1), (1,2): the 3x3 patch starts one texel before the minimum
    x0 = int(round(tri[:, 0].min() * size)) - 1 - padding
    y0 = int(round((1.0 - tri[:, 1].max()) * img.shape[0])) - 1 - padding
    x1, y1 = x0 + 3 + 2 * padding, y0 + 3 + 2 * padding
    img[max(0, y0):min(img.shape[0], y1), max(0, x0):min(size, x1)] = (colour[2], colour[1], colour[0])
    cv2.imwrite(str(ap), img)
    logger.info(f"Recoloured {int(unseen.sum()):,d} unseen faces in {ap.name} (patch at x={x0}, y={y0}).")
    return colour


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
        if self.config.data_term not in TEXRECON_DATA_TERMS:
            raise ValueError(
                f"texrecon --data_term must be one of {TEXRECON_DATA_TERMS}, got '{self.config.data_term}'"
            )

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

        # Remove outputs of a previous run with the same prefix: a run that produces fewer atlases
        # would otherwise leave old *_materialNNNN_map_Kd.png files next to the new ones.
        removed = remove_texrecon_outputs(out_pref)
        if removed:
            logger.info(f"Removed {removed} stale texrecon output file(s) for prefix {out_pref.name}")

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
        if self.config.keep_unseen_faces:
            cmd.append("--keep_unseen_faces")
        if self.config.num_threads:
            cmd.append(f"--num_threads={self.config.num_threads}")
        if self.config.write_timings:
            cmd.append("--write_timings")
        # texrecon writes <prefix>_labeling.vec / _data_costs.spt by default; the only switch is the opt-out
        if not self.config.write_intermediate_results:
            cmd.append("--no_intermediate_results")

        # Positional arguments: IN_SCENE IN_MESH OUT_PREFIX
        cmd.extend([
            str(scene_dir),
            str(mesh_ply_p),
            str(out_pref),
        ])

        logger.info(f"Executing texrecon: {' '.join(cmd)}")
        env = os.environ.copy()
        if self.config.num_threads:
            env["OMP_NUM_THREADS"] = str(self.config.num_threads)
        t0 = time.time()
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self.work_dir),
            env=env,
        )
        elapsed = time.time() - t0
        logger.info(f"texrecon finished in {elapsed:.2f}s with return code {res.returncode}")
        # Full texrecon output next to its results (packed into output.zip under logs/)
        try:
            Path(f"{out_pref}.log").write_text(res.stdout or "", encoding="utf-8")
        except OSError as log_err:
            logger.warning(f"Could not write the texrecon log: {log_err}")

        if res.returncode != 0:
            logger.error(f"texrecon stdout/stderr:\n{res.stdout}")
            raise TexReconError(f"texrecon process failed with return code {res.returncode}")

        if not final_obj.is_file():
            # No glob fallback: "<prefix>*.obj" would also match e.g. model_textured_low.obj
            raise FileNotFoundError(f"Expected texrecon output OBJ not found: {final_obj}")

        if self.config.keep_unseen_faces:
            try:
                colour = fill_unseen_face_texels(final_obj)
                if colour is not None:
                    logger.info(f"Unseen faces kept and coloured with neighbour median RGB {colour}")
            except Exception as fill_err:
                logger.warning(f"Could not recolour unseen faces ({fill_err}); they stay black.")

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
