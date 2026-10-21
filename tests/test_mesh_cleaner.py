"""
test_mesh_cleaner.py — Unit test for watertight pedestal base sealing,
2D Ear-Clipping triangulation, and normal orientation.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure turntable_studio root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from pipeline.config import MeshCleaningConfig
from pipeline.mesh_cleaner import (
    clean_and_repair_mesh,
    seal_watertight_pedestal,
    triangulate_2d_polygon,
)


class TestMeshCleaner(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_2d_ear_clipping_circle(self):
        # 16-sided regular polygon in 2D
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
        triangles = triangulate_2d_polygon(coords)

        # A 16-sided polygon must produce exactly 16 - 2 = 14 triangles
        self.assertEqual(len(triangles), 14)
        self.assertEqual(triangles.shape[1], 3)

    def test_seal_watertight_pedestal(self):
        # Create cylinder with height along Y axis, crossing the slice plane
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        cylinder = trimesh.creation.cylinder(radius=0.03, height=0.10, sections=24, transform=R)
        # Shift so it rests with bottom at Y = -0.02 and top at Y = 0.08
        cylinder.apply_translation([0.0, 0.03, 0.0])

        # Slicing at slice_y = 0.003 and extruding skirt down to skirt_y = 0.000
        sealed = seal_watertight_pedestal(
            input_mesh=cylinder,
            slice_y=0.003,
            skirt_y=0.000,
            max_radius=0.10,
        )

        self.assertIsNotNone(sealed)
        self.assertGreater(len(sealed.faces), 0)

        # Check lowest Y coordinate is exactly 0.000 m
        min_y = float(sealed.vertices[:, 1].min())
        self.assertAlmostEqual(min_y, 0.000, places=3)

        # Check watertightness
        self.assertTrue(sealed.is_watertight)
        self.assertGreater(sealed.volume, 0.0)

    def test_full_clean_and_repair_flow(self):
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        cylinder = trimesh.creation.cylinder(radius=0.025, height=0.08, sections=16, transform=R)
        cylinder.apply_translation([0.0, 0.03, 0.0])

        in_obj = Path(self.test_dir) / "test_raw.obj"
        out_obj = Path(self.test_dir) / "test_cleaned.obj"
        cylinder.export(str(in_obj))

        config = MeshCleaningConfig(
            slice_y_m=0.003,
            skirt_bottom_y_m=0.000,
            max_radius_m=0.10,
        )

        res = clean_and_repair_mesh(
            input_obj_path=str(in_obj),
            output_obj_path=str(out_obj),
            config=config,
        )

        self.assertTrue(out_obj.is_file())
        self.assertGreater(res["vertex_count"], 0)
        self.assertGreater(res["face_count"], 0)
        self.assertTrue(res["is_watertight"])


if __name__ == "__main__":
    unittest.main()
