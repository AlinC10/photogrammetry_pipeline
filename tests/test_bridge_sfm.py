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

    def test_camera_matrix_with_offset_principal_point(self):
        intr = {
            "intrinsicId": "1",
            "width": 4056,
            "height": 3040,
            "sensorWidth": 6.287,
            "focalLength": 6.0,
            "principalPoint": ["1.25", "-2.50"],  # AliceVision offset from (w/2, h/2)
            "distortionParams": [0.0],
        }
        K, dist, w, h = build_camera_matrix_and_distortion(intr)
        self.assertEqual(w, 4056)
        self.assertEqual(h, 3040)
        self.assertAlmostEqual(K[0, 2], 2028.0 + 1.25, places=2)
        self.assertAlmostEqual(K[1, 2], 1520.0 - 2.50, places=2)

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
                        "center": [0.0, 0.0, -0.35],
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
        self.assertTrue((colmap_out / "sparse" / "cameras.txt").is_file())
        self.assertTrue((colmap_out / "sparse" / "images.txt").is_file())
        self.assertTrue((colmap_out / "sparse" / "points3D.txt").is_file())

        with open(colmap_out / "cameras.txt") as f:
            cam_content = f.read()
            self.assertIn("PINHOLE", cam_content)

        with open(colmap_out / "images.txt") as f:
            img_content = f.read()
            self.assertIn("test_img.jpg", img_content)

        with open(colmap_out / "points3D.txt") as f:
            pts_content = f.read()
            self.assertIn("128 128 128", pts_content)

    def test_multiview_turntable_tracks(self):
        test_path = Path(self.test_dir) / "multiview"
        test_path.mkdir(parents=True, exist_ok=True)

        # Create dummy image
        img_file = test_path / "cam_img.jpg"
        cv2.imwrite(str(img_file), np.zeros((3040, 4056, 3), dtype=np.uint8))

        # Generate 4 turntable camera poses around the origin looking inward
        sfm = {
            "version": ["1", "0", "0"],
            "intrinsics": [{
                "intrinsicId": "0",
                "width": 4056,
                "height": 3040,
                "sensorWidth": 6.287,
                "focalLength": 6.0,
                "principalPoint": ["0.5", "-1.2"],
                "distortionParams": [0.0, 0.0],
            }],
            "views": [],
            "poses": [],
        }

        dist = 0.35
        for i in range(4):
            angle = i * (np.pi / 2.0)
            C = np.array([dist * np.sin(angle), 0.08, dist * np.cos(angle)])
            # Camera orientation looking towards origin in OpenCV coords (X right, Y down, Z forward)
            forward = -C / np.linalg.norm(C)
            down = np.array([0.0, 1.0, 0.0])
            right = np.cross(down, forward)
            right /= np.linalg.norm(right)
            down_cam = np.cross(forward, right)
            R = np.vstack([right, down_cam, forward])

            pose_id = str(100 + i)
            view_id = str(200 + i)

            sfm["views"].append({
                "viewId": view_id,
                "poseId": pose_id,
                "intrinsicId": "0",
                "path": str(img_file),
            })
            sfm["poses"].append({
                "poseId": pose_id,
                "pose": {
                    "transform": {
                        "rotation": R.flatten(order="F").tolist(),
                        "center": C.tolist(),
                    }
                }
            })

        sfm_file = test_path / "cameras_known.sfm"
        with open(sfm_file, "w") as f:
            json.dump(sfm, f)

        colmap_out = test_path / "colmap"
        undist_out = test_path / "undistorted"

        res = convert_sfm_to_colmap(
            sfm_json_path=str(sfm_file),
            output_colmap_dir=str(colmap_out),
            undistorted_images_dir=str(undist_out),
            workers=4,  # undistortion in a thread pool: every view still gets its own image
        )

        self.assertEqual(res["registered_views"], 4)
        self.assertEqual(sorted(p.name for p in undist_out.glob("*.jpg")),
                         [f"{200 + i}_cam_img.jpg" for i in range(4)])
        # Verify valid proxy points with multi-view coverage
        with open(colmap_out / "points3D.txt") as f:
            lines = [l for l in f if not l.startswith("#")]
            self.assertGreater(len(lines), 100)
            # Verify tracks in points3D have at least 2 camera views
            for line in lines:
                parts = line.strip().split()
                # Line format: id, x, y, z, r, g, b, err, (img_id, pt2d_idx)...
                track_elements = parts[8:]
                self.assertGreaterEqual(len(track_elements), 4)  # at least 2 pairs (img_id, pt2d_idx)


    def _one_view_sfm(self, test_path, img_path):
        sfm = {
            "version": ["1", "0", "0"],
            "intrinsics": [{"intrinsicId": "0", "width": 400, "height": 300, "sensorWidth": 6.287,
                            "focalLength": 6.0, "distortionParams": [0.0, 0.0]}],
            "views": [{"viewId": "1000", "poseId": "1000", "intrinsicId": "0", "path": str(img_path)}],
            "poses": [{"poseId": "1000", "pose": {"transform": {
                "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "center": [0.0, 0.0, -0.35]}}}],
        }
        sfm_file = test_path / "cameras_known.sfm"
        sfm_file.write_text(json.dumps(sfm))
        return sfm_file

    def test_generated_masks_are_found_and_applied(self):
        """
        mask_processor writes <viewId>_<stem>.mask.png: the bridge must find it (a return inside the candidate
        loop used to give up after the first candidate) and black out the background, or leave the image unchanged.
        """
        test_path = Path(self.test_dir)
        img_path = test_path / "rig" / "0" / "f_1000.jpg"
        img_path.parent.mkdir(parents=True)
        cv2.imwrite(str(img_path), np.full((300, 400, 3), 120, dtype=np.uint8))
        masks = test_path / "masks"
        masks.mkdir()
        mask = np.zeros((300, 400), dtype=np.uint8)
        mask[:, 200:] = 255
        cv2.imwrite(str(masks / "1000_f_1000.mask.png"), mask)
        sfm_file = self._one_view_sfm(test_path, img_path)

        for blackout in (True, False):
            und = test_path / f"undistorted_{blackout}"
            res = convert_sfm_to_colmap(str(sfm_file), str(test_path / "colmap"), str(und), masks_src_dir=str(masks),
                                        masks_dst_dir=str(masks), blackout_background=blackout)
            self.assertEqual(res["masks_applied"], 1)
            self.assertTrue((und / "1000_f_1000.mask.png").is_file())
            out = cv2.imread(str(und / "1000_f_1000.jpg"))
            self.assertGreater(out[150, 300].mean(), 100)
            if blackout:
                self.assertLess(out[150, 50].mean(), 5)
            else:
                self.assertGreater(out[150, 50].mean(), 100)

    def test_proxy_points_follow_the_given_bounds(self):
        """The proxy volume (DensifyPointCloud ROI and depth range) comes from proxy_bounds, not 220 mm / 85 mm."""
        pts = generate_proxy_3d_points(n_points=1500, y_max=0.30, r_max=0.05, r_min=0.01)
        above_base = pts[pts[:, 1] > 0.002 + 1e-9]
        self.assertAlmostEqual(above_base[:, 1].max(), 0.30, places=6)
        self.assertLessEqual(np.hypot(above_base[:, 0], above_base[:, 2]).max(), 0.05 + 1e-9)

        test_path = Path(self.test_dir)
        img_path = test_path / "img.jpg"
        cv2.imwrite(str(img_path), np.zeros((300, 400, 3), dtype=np.uint8))
        sfm_file = self._one_view_sfm(test_path, img_path)
        convert_sfm_to_colmap(str(sfm_file), str(test_path / "colmap"), str(test_path / "und"),
                              proxy_bounds={"y_max": 0.12, "r_max": 0.04})
        rows = [ln.split() for ln in (test_path / "colmap" / "points3D.txt").read_text().splitlines()
                if ln and not ln.startswith("#")]
        xyz = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows])
        body = xyz[xyz[:, 1] > 0.002 + 1e-9]
        self.assertLessEqual(body[:, 1].max(), 0.12 + 1e-6)
        self.assertLessEqual(np.hypot(body[:, 0], body[:, 2]).max(), 0.04 + 1e-6)


if __name__ == "__main__":
    unittest.main()
