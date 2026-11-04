"""
openmvs_runner.py — Cross-platform wrapper executing OpenMVS CLI binaries:
InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh, RefineMesh, TextureMesh.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import OpenMVSConfig

logger = logging.getLogger("TurntableStudio.OpenMVS")


class OpenMVSError(RuntimeError):
    """Raised when an OpenMVS executable returns a non-zero exit code."""
    pass


_PLY_TYPES = {"int8": "i1", "char": "i1", "uint8": "u1", "uchar": "u1", "int16": "<i2", "short": "<i2",
              "uint16": "<u2", "ushort": "<u2", "int32": "<i4", "int": "<i4", "uint32": "<u4", "uint": "<u4",
              "float32": "<f4", "float": "<f4", "float64": "<f8", "double": "<f8"}


def textured_ply_to_obj(ply_path: str, obj_path: str) -> str:
    """
    Converts the textured PLY TextureMesh writes (vertex x/y/z; face vertex_indices + texcoord, texnumber with
    several atlases; "comment TextureFile <png>" per atlas) into OBJ + MTL next to it, referencing the same PNGs.
    TextureMesh v2.3.0 crashes (SIGSEGV) while writing OBJ after texturing, so it exports PLY and this converts.
    The texcoords are already in the OBJ convention (v up): written unchanged.
    """
    ply_p, obj_p = Path(ply_path), Path(obj_path)
    data = ply_p.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii", errors="replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise OpenMVSError(f"{ply_p.name}: expected a binary little-endian PLY")
    textures = [ln.split(maxsplit=2)[2] for ln in header if ln.startswith("comment TextureFile ")]

    elements, current = [], None
    for ln in header:
        parts = ln.split()
        if parts[:1] == ["element"]:
            current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(current)
        elif parts[:1] == ["property"] and current is not None:
            current["props"].append(parts[1:])
    offset = end
    arrays = {}
    for el in elements:
        if el["name"] == "vertex":
            fields = [(pr[-1], _PLY_TYPES[pr[0]]) for pr in el["props"]]
            arr = np.frombuffer(data, np.dtype(fields), el["count"], offset)
        elif el["name"] == "face":
            # Triangles with 3 indices and 6 texcoords: fixed-size records (checked below)
            fields = []
            for pr in el["props"]:
                if pr[0] == "list":
                    n = 3 if pr[-1] == "vertex_indices" else 6
                    fields += [(pr[-1] + "_n", _PLY_TYPES[pr[1]]), (pr[-1], _PLY_TYPES[pr[2]], n)]
                else:
                    fields.append((pr[-1], _PLY_TYPES[pr[0]]))
            arr = np.frombuffer(data, np.dtype(fields), el["count"], offset)
            if not (np.all(arr["vertex_indices_n"] == 3) and np.all(arr["texcoord_n"] == 6)):
                raise OpenMVSError(f"{ply_p.name}: only triangles with 3 texcoords are supported")
        else:
            raise OpenMVSError(f"{ply_p.name}: unexpected element {el['name']}")
        arrays[el["name"]] = arr
        offset += arr.nbytes
    if "face" not in arrays or "texcoord" not in arrays["face"].dtype.names:
        raise OpenMVSError(f"{ply_p.name}: no texture coordinates")

    v = arrays["vertex"]
    f = arrays["face"]
    faces = f["vertex_indices"].astype(np.int64) + 1
    uv = f["texcoord"].reshape(-1, 2)
    tex_index = f["texnumber"].astype(np.int64) if "texnumber" in f.dtype.names else np.zeros(len(f), np.int64)
    corner = np.arange(1, 3 * len(f) + 1).reshape(-1, 3)

    mtl_p = obj_p.with_suffix(".mtl")
    with open(mtl_p, "w", encoding="utf-8") as m:
        for i, tex in enumerate(textures or [""]):
            m.write(f"newmtl material_{i}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\n")
            if tex:
                m.write(f"map_Kd {tex}\n")
            m.write("\n")
    with open(obj_p, "w", encoding="utf-8") as o:
        o.write(f"mtllib {mtl_p.name}\n")
        np.savetxt(o, np.stack([v["x"], v["y"], v["z"]], 1), fmt="v %.6f %.6f %.6f")
        np.savetxt(o, uv, fmt="vt %.6f %.6f")
        for i in range(max(1, len(textures))):
            sel = tex_index == i
            if not sel.any():
                continue
            o.write(f"usemtl material_{i}\n")
            rows = np.stack([faces[sel, 0], corner[sel, 0], faces[sel, 1], corner[sel, 1],
                             faces[sel, 2], corner[sel, 2]], 1)
            np.savetxt(o, rows, fmt="f %d/%d %d/%d %d/%d")
    logger.info(f"Converted {ply_p.name} -> {obj_p.name} ({len(f):,d} faces, {len(textures)} texture(s))")
    return str(obj_p)


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
        if self.config.max_threads and self.config.max_threads > 0:
            cmd = list(cmd) + ["--max-threads", str(self.config.max_threads)]
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
        if not self.config.keep_depth_maps:
            self.remove_depth_maps()
        return str(out_file)

    def remove_depth_maps(self) -> int:
        """Deletes depth*.dmap (only needed during fusion; ~9 GB for 120 images at level 1)."""
        removed, freed = 0, 0
        for dmap in self.work_dir.glob("depth*.dmap"):
            try:
                freed += dmap.stat().st_size
                dmap.unlink()
                removed += 1
            except OSError as e:
                logger.warning(f"Could not delete {dmap.name}: {e}")
        if removed:
            logger.info(f"Deleted {removed} depth maps ({freed / 1024 ** 3:.2f} GB) after fusion.")
        return removed

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
        input_mvs: str = "scene_dense.mvs",
        mesh_file: Optional[str] = None,
        output_mvs: str = "scene_mesh_refined.mvs",
    ) -> str:
        """
        Refines the reconstructed surface with photo-consistency (OpenMVS RefineMesh): removes the
        small dents PatchMatch leaves on dark / glossy plastic without blurring sharp edges.
        Returns the refined mesh path (.ply), or mesh_file unchanged if RefineMesh is unavailable.
        """
        bin_path = self.config.refine_mesh
        if not bin_path:
            logger.info("RefineMesh binary not found, skipping.")
            return str(mesh_file) if mesh_file else str(self.work_dir / input_mvs)

        in_file = self.work_dir / input_mvs
        if not in_file.is_file():
            in_file = self.work_dir / "scene_dense.mvs"

        out_file = self.work_dir / output_mvs
        out_ply = self.work_dir / f"{out_file.stem}.ply"

        # Remove stale refined outputs from previous runs
        if out_ply.is_file():
            out_ply.unlink()
        if out_file.is_file():
            out_file.unlink()

        cmd = [
            bin_path,
            str(in_file),
            "-o", str(out_file),
            "--resolution-level", str(getattr(self.config, "refine_resolution_level", 2)),
            "--max-face-area", str(getattr(self.config, "refine_max_face_area", 256)),
            "--scales", str(getattr(self.config, "refine_scales", 1)),
            "-v", "2",
        ]
        # ReconstructMesh (v2.3) leaves the surface in scene_mesh.ply next to a mesh-less .mvs
        if mesh_file and Path(mesh_file).is_file():
            cmd.extend(["--mesh-file", str(Path(mesh_file).resolve())])

        self._execute_cmd(cmd, "RefineMesh")
        if out_ply.is_file():
            return str(out_ply)
        if out_file.is_file():
            return str(out_file)
        return str(mesh_file) if mesh_file else str(in_file)

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
        expected_ply = self.work_dir / f"{out_file.stem}.ply"
        for stale in (expected_obj, expected_ply):
            if stale.is_file():
                stale.unlink()

        # --export-type ply: TextureMesh v2.3.0 finishes texturing, then dies with SIGSEGV (-11) while writing OBJ
        # (reproduced with the v4 image on the fixativ, any mesh); the PLY export works and is converted below.
        cmd = [
            bin_path,
            str(in_file),
            "-o", str(out_file),
            "--export-type", "ply",
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
        # TextureMesh in OpenMVS v2.3.0 (Dockerfile) has no --mask-path option ("unrecognised option")
        # and aborts; the masks are already burned into the undistorted images (black background)
        # by bridge_sfm, so masks_dir is accepted for API compatibility but not passed.

        self._execute_cmd(cmd, "TextureMesh")

        if expected_ply.is_file():
            textured_ply_to_obj(str(expected_ply), str(expected_obj))
        if not expected_obj.is_file():
            # Check if obj was exported with another name
            candidate_objs = list(self.work_dir.glob("*.obj"))
            if candidate_objs:
                expected_obj = candidate_objs[0]
            else:
                raise FileNotFoundError(f"TextureMesh OBJ output missing: {expected_obj}")

        return str(out_file), str(expected_obj)
