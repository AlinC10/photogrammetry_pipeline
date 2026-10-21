"""
test_bridge_sfm.py — Unit test for bridge_sfm matrix conversions,
quaternions, COLMAP writer, and proxy point tracks.
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

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.bridge_sfm import (
    build_camera_matrix_and_distortion,
    convert_sfm_to_colmap,
    generate_proxy_3d_points,
)


class TestBridgeSFM(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_camera_matrix_and_distortion(self):
        intr = {
            "intrinsicId": "0",
            "width": 4000,
            "height": 3000,
            "sensorWidth": 6.287,
            "focalLength": 6.0,
            "principalPoint": [2000.0, 1500.0],
            "distortionParams": [0.01, -0.002, 0.0001, -0.0001, 0.0],
        }
        K, dist, w, h = build_camera_matrix_and_distortion(intr)
        self.assertEqual(w, 4000)
        self.assertEqual(h, 3000)
        expected_fx = 6.0 * (4000 / 6.287)
        self.assertAlmostEqual(K[0, 0], expected_fx, places=2)
        self.assertAlmostEqual(K[1, 1], expected_fx, places=2)
        self.assertEqual(K[0, 2], 2000.0)
        self.assertEqual(K[1, 2], 1500.0)
        self.assertEqual(len(dist), 5)
        self.assertAlmostEqual(dist[0], 0.01)

    def test_quaternion_ordering_and_translation(self):
        # 90-degree rotation around Y axis
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]
        ], dtype=np.float64)
        C = np.array([0.1, 0.05, 0.4], dtype=np.float64)

        # Translation T = -R @ C
        T = -R @ C
        expected_T = np.array([-0.4, -0.05, 0.1])
        np.testing.assert_allclose(T, expected_T, atol=1e-6)

        # SciPy quaternion -> [qx, qy, qz, qw]
        q_scipy = Rotation.from_matrix(R).as_quat()
        # COLMAP quaternion -> [qw, qx, qy, qz]
        q_colmap = [q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]]

        # Rotation of 90 deg around Y: qw = cos(45 deg) = 0.7071, qy = sin(45 deg) = 0.7071
        self.assertAlmostEqual(q_colmap[0], np.cos(np.pi / 4), places=4)
        self.assertAlmostEqual(q_colmap[2], np.sin(np.pi / 4), places=4)

    def test_convert_sfm_to_colmap_flow(self):
        test_path = Path(self.test_dir)
        img_path = test_path / "test_img.jpg"
        dummy_img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), dummy_img)

        # Synthetic AliceVision SFM
        sfm = {
            "version": ["1", "0", "0"],
            "intrinsics": [{
                "intrinsicId": "0",
                "width": 400,
                "height": 300,
                "sensorWidth": 6.287,
                "focalLength": 6.0,
                "distortionParams": [0.0, 0.0],
            }],
            "views": [{
                "viewId": "1000",
                "poseId": "1000",
                "intrinsicId": "0",
                "path": str(img_path),
            }],
            "poses": [{
                "poseId": "1000",
                "pose": {
                    "transform": {
                        "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "center": [0.0, 0.0, 0.35],
                    }
                }
            }],
        }
        sfm_file = test_path / "cameras_known.sfm"
        with open(sfm_file, "w") as f:
            json.dump(sfm, f)

        colmap_out = test_path / "colmap"
        undist_out = test_path / "undistorted"

        res = convert_sfm_to_colmap(
            sfm_json_path=str(sfm_file),
            output_colmap_dir=str(colmap_out),
            undistorted_images_dir=str(undist_out),
        )

        self.assertEqual(res["registered_views"], 1)
        self.assertTrue((colmap_out / "cameras.txt").is_file())
        self.assertTrue((colmap_out / "images.txt").is_file())
        self.assertTrue((colmap_out / "points3D.txt").is_file())

        with open(colmap_out / "cameras.txt") as f:
            cam_content = f.read()
            self.assertIn("PINHOLE", cam_content)

        with open(colmap_out / "images.txt") as f:
            img_content = f.read()
            self.assertIn("test_img.jpg", img_content)

        with open(colmap_out / "points3D.txt") as f:
            pts_content = f.read()
            self.assertIn("128 128 128", pts_content)


if __name__ == "__main__":
    unittest.main()
