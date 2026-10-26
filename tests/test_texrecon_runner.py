"""
test_texrecon_runner.py — Unit tests for mvs-texturing (texrecon) integration.
Verifies:
  1. Binary locator (find_texrecon_binary) across custom dirs and env vars.
  2. Mathematical conversion of cameras_known.sfm to .cam scene format (T = -R @ C, normalized f, ppx, ppy).
  3. Mesh conversion from OBJ to PLY required by texrecon.
  4. End-to-end execution flow with mock mode generating valid textured OBJ + MTL + PNG.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure turntable_studio root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh
from PIL import Image

from pipeline.config import TexReconConfig
from pipeline.texrecon_runner import (
    TexReconError,
    TexReconRunner,
    export_sfm_to_cam_scene,
    fill_unseen_face_texels,
    find_texrecon_binary,
    remove_texrecon_outputs,
)


class TestTexReconRunner(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.work_p = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_find_texrecon_binary(self):
        """Verifies binary discovery via custom dir and env var."""
        bin_name = "texrecon.exe" if sys.platform == "win32" else "texrecon"
        fake_bin_dir = self.work_p / "fake_bin"
        fake_bin_dir.mkdir(parents=True, exist_ok=True)
        fake_bin = fake_bin_dir / bin_name
        fake_bin.touch()

        # 1. Direct custom dir
        found = find_texrecon_binary(custom_dir=str(fake_bin_dir))
        self.assertIsNotNone(found)
        self.assertEqual(Path(found).resolve(), fake_bin.resolve())

        # 2. Environment variable
        old_env = os.environ.get("TEXRECON_BIN")
        try:
            os.environ["TEXRECON_BIN"] = str(fake_bin)
            found_env = find_texrecon_binary()
            self.assertIsNotNone(found_env)
            self.assertEqual(Path(found_env).resolve(), fake_bin.resolve())
        finally:
            if old_env is not None:
                os.environ["TEXRECON_BIN"] = old_env
            else:
                os.environ.pop("TEXRECON_BIN", None)

    def test_export_sfm_to_cam_scene(self):
        """Verifies mathematical conversion of AliceVision SFM to .cam files."""
        sfm_file = self.work_p / "cameras_known.sfm"
        img_dir = self.work_p / "undistorted_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        scene_dir = self.work_p / "scene"

        # Create dummy image
        img_file = img_dir / "frame_00000.png"
        img = Image.new("RGB", (4000, 3000), color=(200, 100, 50))
        img.save(str(img_file))

        # Camera center and rotation
        C_world = np.array([0.15, -0.05, 0.40], dtype=np.float64)
        # 90 deg rotation around X
        R_orig = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)

        # AliceVision rotation is stored Fortran order (Column-Major)
        R_fortran_flat = [float(x) for x in R_orig.flatten(order="F")]

        sfm_data = {
            "version": ["1", "0", "0"],
            "featuresFolders": [],
            "matchesFolders": [],
            "views": [
                {
                    "viewId": "100",
                    "poseId": "100",
                    "intrinsicId": "1",
                    "resectionId": "100",
                    "path": str(img_file),
                    "width": "4000",
                    "height": "3000",
                }
            ],
            "intrinsics": [
                {
                    "intrinsicId": "1",
                    "width": "4000",
                    "height": "3000",
                    "sensorWidth": "6.0",
                    "sensorHeight": "4.5",
                    "focalLength": "6.0",
                    "principalPoint": ["0.0", "0.0"],
                    "distortionParams": ["0.0", "0.0"],
                }
            ],
            "poses": [
                {
                    "poseId": "100",
                    "pose": {
                        "transform": {
                            "rotation": R_fortran_flat,
                            "center": [float(c) for c in C_world],
                        }
                    },
                }
            ],
        }

        with open(sfm_file, "w", encoding="utf-8") as f:
            json.dump(sfm_data, f)

        # Run conversion
        cam_files = export_sfm_to_cam_scene(
            sfm_data_path=str(sfm_file),
            undistorted_images_dir=str(img_dir),
            output_scene_dir=str(scene_dir),
        )

        self.assertEqual(len(cam_files), 1)
        cam_path = cam_files[0]
        self.assertTrue(cam_path.is_file())

        # Verify .cam file contents
        with open(cam_path, "r", encoding="utf-8") as cf:
            lines = [l.strip() for l in cf if l.strip()]

        self.assertEqual(len(lines), 2)
        ext_parts = [float(x) for x in lines[0].split()]
        self.assertEqual(len(ext_parts), 12)  # tx, ty, tz, R00..R22

        # Check translation T = -R @ C
        T_expected = -R_orig @ C_world
        np.testing.assert_allclose(ext_parts[:3], T_expected, atol=1e-5)

        # Check rotation matrix components (row-major)
        R_read = np.array(ext_parts[3:]).reshape((3, 3))
        np.testing.assert_allclose(R_read, R_orig, atol=1e-5)

        # Check Intrinsics line: f d0 d1 paspect ppx ppy
        intr_parts = [float(x) for x in lines[1].split()]
        self.assertEqual(len(intr_parts), 6)
        # f = fx / max(w, h) = 4000 / 4000 = 1.0
        self.assertAlmostEqual(intr_parts[0], 1.0, places=4)
        self.assertEqual(intr_parts[1], 0.0)  # d0
        self.assertEqual(intr_parts[2], 0.0)  # d1
        self.assertAlmostEqual(intr_parts[3], 1.0, places=4)  # paspect
        self.assertAlmostEqual(intr_parts[4], 0.5, places=3)  # ppx
        self.assertAlmostEqual(intr_parts[5], 0.5, places=3)  # ppy

        # Verify scene.nvm was also generated
        nvm_file = scene_dir / "scene.nvm"
        self.assertTrue(nvm_file.is_file())

    def test_texrecon_runner_mock_flow(self):
        """Verifies end-to-end execution with mock mode."""
        sfm_file = self.work_p / "cameras.sfm"
        img_dir = self.work_p / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_file = img_dir / "frame_001.png"
        Image.new("RGB", (100, 100), color=(100, 150, 200)).save(str(img_file))

        sfm_data = {
            "version": ["1", "0", "0"],
            "featuresFolders": [],
            "matchesFolders": [],
            "views": [{"viewId": "1", "poseId": "1", "intrinsicId": "1", "path": str(img_file), "width": "100", "height": "100"}],
            "intrinsics": [{"intrinsicId": "1", "width": "100", "height": "100", "sensorWidth": "6.0", "focalLength": "6.0", "principalPoint": ["0.0", "0.0"]}],
            "poses": [{"poseId": "1", "pose": {"transform": {"rotation": [1,0,0, 0,1,0, 0,0,1], "center": [0,0,1]}}}],
        }
        with open(sfm_file, "w", encoding="utf-8") as f:
            json.dump(sfm_data, f)

        # Create input OBJ mesh
        obj_file = self.work_p / "input_model.obj"
        box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        box.export(str(obj_file))

        cfg = TexReconConfig(
            data_term="gmi",
            smoothness_term="potts",
            outlier_removal="gauss_damping",
        )
        runner = TexReconRunner(config=cfg, work_dir=str(self.work_p / "tex_out"))

        out_obj, out_mtl = runner.run_texturing(
            sfm_data_path=str(sfm_file),
            input_mesh_path=str(obj_file),
            images_folder=str(img_dir),
            mock_mode=True,
        )

        self.assertTrue(Path(out_obj).is_file())
        self.assertTrue(Path(out_mtl).is_file())

        # Verify input OBJ was properly converted to PLY for texrecon
        self.assertTrue((self.work_p / "tex_out" / "input_model.ply").is_file())

        # Verify OBJ has texture coordinates and face references
        with open(out_obj, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("vt ", content)
            self.assertIn("usemtl material_0", content)
            self.assertIn("mtllib ", content)

    def test_export_sfm_with_prefixed_undistorted_image(self):
        """Verifies lookup matches bridge_sfm naming: {viewId}_{filename}."""
        sfm_file = self.work_p / "sfm_prefixed.json"
        img_dir = self.work_p / "undist_prefixed"
        img_dir.mkdir(parents=True, exist_ok=True)
        scene_dir = self.work_p / "scene_prefixed"

        # Image is saved with viewId prefix as generated by bridge_sfm: 100_frame.jpg
        orig_img_path = Path("/raw/input/frame.jpg")
        actual_undist_img = img_dir / "100_frame.jpg"
        Image.new("RGB", (100, 100), color=(10, 20, 30)).save(str(actual_undist_img))

        sfm_data = {
            "version": ["1", "0", "0"],
            "views": [{
                "viewId": "100",
                "poseId": "100",
                "intrinsicId": "1",
                "path": str(orig_img_path),
                "width": "100",
                "height": "100",
            }],
            "intrinsics": [{
                "intrinsicId": "1",
                "width": "100",
                "height": "100",
                "sensorWidth": "6.0",
                "focalLength": "6.0",
                "principalPoint": ["0.0", "0.0"],
            }],
            "poses": [{
                "poseId": "100",
                "pose": {
                    "transform": {
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "center": [0.0, 0.0, 0.5],
                    }
                },
            }],
        }
        with open(sfm_file, "w", encoding="utf-8") as f:
            json.dump(sfm_data, f)

        cam_files = export_sfm_to_cam_scene(
            sfm_data_path=str(sfm_file),
            undistorted_images_dir=str(img_dir),
            output_scene_dir=str(scene_dir),
        )
        self.assertEqual(len(cam_files), 1)
        # Verify the copied image in scene_dir is the undistorted one (100_frame.jpg), not raw fallback
        copied_images = list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png"))
        self.assertEqual(len(copied_images), 1)
        self.assertEqual(copied_images[0].name, "100_frame.jpg")

    def _rig_sfm(self, img_dir: Path) -> Path:
        """Two rig cameras whose photos share the file name f_1000.jpg (rig/0 and rig/1)."""
        views, poses = [], []
        for cam, (vid, cx) in enumerate((("1000", 0.30), ("1020", -0.30))):
            views.append({"viewId": vid, "poseId": vid, "intrinsicId": "0",
                          "path": f"/workspace/input_images/rig/{cam}/f_1000.jpg", "width": "100", "height": "100"})
            poses.append({"poseId": vid, "pose": {"transform": {
                "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "center": [cx, 0.0, -0.5]}}})
            Image.new("RGB", (100, 100), color=(10 + cam, 20, 30)).save(str(img_dir / f"{vid}_f_1000.png"))
        sfm = {"version": ["1", "0", "0"], "views": views, "poses": poses,
               "intrinsics": [{"intrinsicId": "0", "width": "100", "height": "100", "sensorWidth": "6.0",
                               "focalLength": "6.0", "principalPoint": ["0.0", "0.0"]}]}
        sfm_file = self.work_p / "rig.sfm"
        sfm_file.write_text(json.dumps(sfm))
        return sfm_file

    def test_scene_folder_is_rebuilt_without_stale_views(self):
        """Leftover .cam files from an older run were loaded by texrecon as extra (wrong) views."""
        img_dir = self.work_p / "texturing_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        sfm_file = self._rig_sfm(img_dir)
        scene_dir = self.work_p / "scene"
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "f_1000.cam").write_text("0 0 0 1 0 0 0 1 0 0 0 1\n1 0 0 1 0.5 0.5\n")
        (scene_dir / "f_1000.jpg").write_bytes(b"")

        cam_files = export_sfm_to_cam_scene(str(sfm_file), str(img_dir), str(scene_dir))

        names = sorted(p.name for p in scene_dir.iterdir())
        self.assertEqual(names, ["1000_f_1000.cam", "1000_f_1000.png", "1020_f_1000.cam", "1020_f_1000.png", "scene.nvm"])
        self.assertEqual(len(cam_files), 2)
        # Same file name in both cameras: each .cam keeps its own pose (T = -R C)
        t0 = [float(x) for x in (scene_dir / "1000_f_1000.cam").read_text().split()[:3]]
        t1 = [float(x) for x in (scene_dir / "1020_f_1000.cam").read_text().split()[:3]]
        self.assertAlmostEqual(t0[0], -0.30, places=6)
        self.assertAlmostEqual(t1[0], 0.30, places=6)
        # ...and its own image
        self.assertEqual(Image.open(scene_dir / "1020_f_1000.png").getpixel((0, 0))[0], 11)

    def test_missing_texturing_image_raises_instead_of_using_raw_photo(self):
        img_dir = self.work_p / "texturing_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        sfm_file = self._rig_sfm(img_dir)
        (img_dir / "1020_f_1000.png").unlink()
        with self.assertRaises(TexReconError):
            export_sfm_to_cam_scene(str(sfm_file), str(img_dir), str(self.work_p / "scene"))

    def test_remove_outputs_is_anchored_on_prefix(self):
        out = self.work_p / "texrecon"
        out.mkdir(parents=True, exist_ok=True)
        for name in ("model_textured.obj", "model_textured.mtl", "model_textured_material0024_map_Kd.png",
                     "model_textured_labeling.vec", "model_textured_low.obj", "model_textured_low_material0000_map_Kd.png"):
            (out / name).write_bytes(b"x")
        removed = remove_texrecon_outputs(out / "model_textured")
        self.assertEqual(removed, 4)
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         ["model_textured_low.obj", "model_textured_low_material0000_map_Kd.png"])

    def test_fill_unseen_face_texels(self):
        """--keep_unseen_faces maps all unseen faces to one black 3x3 patch: it is repainted with the neighbours' colour."""
        out = self.work_p / "unseen"
        out.mkdir(parents=True, exist_ok=True)
        size = 256
        padding = size >> 7
        atlas = np.zeros((size, size, 3), np.uint8)
        atlas[:128, :] = (40, 160, 60)                       # textured faces, RGB
        Image.fromarray(atlas).save(str(out / "m_material0000_map_Kd.png"))
        (out / "m.mtl").write_text("newmtl material0000\nKd 1 1 1\nmap_Kd m_material0000_map_Kd.png\n")
        ox, oy = 200 + padding, 200 + padding                 # unseen patch rect origin + padding
        rel = [(2.0, 1.0), (1.0, 1.0), (1.0, 2.0)]
        vts = [((ox + x) / size, 1.0 - (oy + y) / size) for x, y in rel]
        vts += [(0.2, 0.9), (0.5, 0.9), (0.2, 0.7)]          # a textured face in the green half
        lines = ["mtllib m.mtl\n"]
        lines += [f"v {i} {i % 2} 0\n" for i in range(6)]
        lines += [f"vt {u:.6f} {v:.6f}\n" for u, v in vts]
        lines += ["usemtl material0000\n", "f 1/4 2/5 3/6\n", "f 2/1 3/2 4/3\n", "f 4/1 5/2 6/3\n", "f 5/1 6/2 1/3\n"]
        (out / "m.obj").write_text("".join(lines))

        colour = fill_unseen_face_texels(str(out / "m.obj"))
        self.assertEqual(colour, (40, 160, 60))
        res = np.array(Image.open(out / "m_material0000_map_Kd.png"))
        self.assertEqual(tuple(res[oy + 1, ox + 1]), (40, 160, 60))
        self.assertEqual(tuple(res[oy - padding, ox - padding]), (40, 160, 60))
        self.assertEqual(tuple(res[140, 20]), (0, 0, 0))     # rest of the atlas untouched


if __name__ == "__main__":
    unittest.main()
