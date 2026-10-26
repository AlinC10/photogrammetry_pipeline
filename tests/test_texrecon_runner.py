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
    TexReconRunner,
    export_sfm_to_cam_scene,
    find_texrecon_binary,
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


if __name__ == "__main__":
    unittest.main()
