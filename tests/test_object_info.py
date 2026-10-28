"""
test_object_info.py — object.json: title/description and real-size measurements of the cleaned mesh.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from pipeline.object_info import OBJECT_INFO_FILE, measure_mesh, write_object_info


class TestObjectInfo(unittest.TestCase):

    def test_measurements_do_not_depend_on_the_turn_on_the_disc(self):
        """A 80 x 40 x 120 mm box turned 30 deg on the disc: length / width from the footprint, volume 384 cm3."""
        box = trimesh.creation.box(extents=[0.080, 0.120, 0.040])
        box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30), [0, 1, 0]))
        box.apply_translation([0.0, 0.060, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "box.obj")
            box.export(path)
            m = measure_mesh(path)
            self.assertAlmostEqual(m["dimensions_mm"]["length"], 80.0, delta=0.2)
            self.assertAlmostEqual(m["dimensions_mm"]["width"], 40.0, delta=0.2)
            self.assertAlmostEqual(m["dimensions_mm"]["height"], 120.0, delta=0.2)
            self.assertAlmostEqual(m["volume_cm3"], 384.0, delta=0.5)
            self.assertTrue(m["watertight"])

    def test_write_object_json(self):
        cyl = trimesh.creation.cylinder(radius=0.03, height=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "c.obj")
            cyl.export(path)
            out = write_object_info(path, tmp, "Fixativ Taft", "Doza de fixativ, 250 ml", {"faces_high": 123})
            self.assertEqual(Path(out).name, OBJECT_INFO_FILE)
            info = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(info["title"], "Fixativ Taft")
            self.assertEqual(info["description"], "Doza de fixativ, 250 ml")
            self.assertEqual(info["faces_high"], 123)
            self.assertAlmostEqual(info["volume_cm3"], np.pi * 3 ** 2 * 10, delta=3.0)   # polygonal cylinder
            self.assertIsNone(write_object_info(str(Path(tmp) / "missing.obj"), tmp))

            # No title: named after the scan time; no description: ""; the time is always there
            info = json.loads(Path(write_object_info(path, tmp)).read_text(encoding="utf-8"))
            self.assertTrue(info["title"].startswith("Scan 20"))
            self.assertEqual(info["description"], "")
            self.assertRegex(info["created_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")


if __name__ == "__main__":
    unittest.main()
