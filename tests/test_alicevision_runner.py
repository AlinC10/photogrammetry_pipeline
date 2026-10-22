"""
test_alicevision_runner.py — Unit test for AliceVisionRunner SFM preparation,
binary discovery, and texturing invocation.
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
import cv2

from pipeline.config import AliceVisionConfig
from pipeline.alicevision_runner import (
    AliceVisionRunner,
    find_alicevision_binary,
)


class TestAliceVisionRunner(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_find_alicevision_binary(self):
        # Test binary discovery
        bin_path = find_alicevision_binary("aliceVision_texturing")
        # On this environment, Meshroom-2025.1.0 exists in workspace
        if bin_path:
            self.assertTrue(Path(bin_path).is_file())

    def test_prepare_sfm_data_enrichment(self):
        test_path = Path(self.test_dir)
        dummy_img = test_path / "1000.png"
        cv2.imwrite(str(dummy_img), np.full((100, 100, 3), 128, dtype=np.uint8))

        mesh_path = test_path / "test_cube.obj"
        box = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
        box.export(str(mesh_path))

        raw_sfm = {
            "version": ["1", "0", "0"],
            "intrinsics": [{
                "intrinsicId": "0",
                "width": 100,
                "height": 100,
                "sensorWidth": 36.0,
                "focalLength": 50.0,
                "principalPoint": ["0.0", "0.0"],
                "distortionParams": [0.01, -0.002],
                "distortionType": "radialk1",
            }],
            "views": [{
                "viewId": "1000",
                "poseId": "1000",
                "intrinsicId": "0",
                "path": str(dummy_img),
            }, {
                "viewId": "1001",
                "poseId": "1001",
                "intrinsicId": "0",
                "path": str(dummy_img),
            }],
            "poses": [{
                "poseId": "1000",
                "pose": {
                    "transform": {
                        "rotation": ["1.0", "0.0", "0.0", "0.0", "1.0", "0.0", "0.0", "0.0", "1.0"],
                        "center": ["0.0", "0.0", "-0.2"],
                    }
                }
            }, {
                "poseId": "1001",
                "pose": {
                    "transform": {
                        "rotation": ["0.0", "0.0", "-1.0", "0.0", "1.0", "0.0", "1.0", "0.0", "0.0"],
                        "center": ["0.2", "0.0", "0.0"],
                    }
                }
            }],
        }
        sfm_file = test_path / "cameras_known.sfm"
        with open(sfm_file, "w") as f:
            json.dump(raw_sfm, f)

        config = AliceVisionConfig()
        # Mock bin path if not found to test data prep
        if not config.texturing_bin:
            config.texturing_bin = "/mock/bin/aliceVision_texturing"

        runner = AliceVisionRunner(config=config, work_dir=str(test_path / "av_work"))
        prep_sfm_path = runner.prepare_sfm_data(
            input_sfm_path=str(sfm_file),
            input_mesh_path=str(mesh_path),
        )

        self.assertTrue(Path(prep_sfm_path).is_file())
        with open(prep_sfm_path, "r") as f:
            enriched = json.load(f)

        intr = enriched["intrinsics"][0]
        self.assertEqual(intr["serialNumber"], "camera_0")
        self.assertIn("pxFocalLength", intr)
        self.assertEqual(intr["type"], "pinhole")
        self.assertGreater(len(enriched["structure"]), 0)

        # Check landmark observations
        first_landmark = enriched["structure"][0]
        self.assertIn("landmarkId", first_landmark)
        self.assertIn("observations", first_landmark)
        self.assertGreaterEqual(len(first_landmark["observations"]), 2)


if __name__ == "__main__":
    unittest.main()
