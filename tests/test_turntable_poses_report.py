"""
turntable_poses.write_report with photos that keep fewer than 4 tag corners after the outlier pass.

A large object (a bag) hides most tags: a photo can have 2 tags detected but only 1-3 corners left after the
corners with a big error are excluded. cv2.solvePnP needs >= 4 points and raised (-215 Assertion failed), which
stopped the whole job before cameras_known.sfm was written.
"""

import math
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import turntable_poses as tp  # noqa: E402


def _model():
    m = tp.TurntableModel(["0"], [1], 1, 18.0, 0.08)
    m.p["K_0"] = np.array([2500.0, 2028.0, 1520.0])
    m.p["D_0"] = np.zeros(3)
    m.p["R_0"] = np.array([-math.pi / 2, 0.0, 0.0])     # looks straight down (camera z = -Y)
    m.p["C_0"] = np.array([0.0, 0.40, 0.0])
    return m


def _tag(cx, cz, s=0.02):
    return np.array([[cx - s, 0, cz - s], [cx + s, 0, cz - s], [cx + s, 0, cz + s], [cx - s, 0, cz + s]], float)


class TestReportFewCorners(unittest.TestCase):
    def test_photo_with_two_tags_but_three_corners_left_does_not_crash(self):
        model = _model()
        layout = {0: _tag(-0.08, 0.0), 1: _tag(0.08, 0.0)}
        views = [
            {"viewId": "1", "intrinsicId": "0", "path": "/x/0/f_1000.jpg", "level": 1, "index": 0, "tags": {}},
            {"viewId": "2", "intrinsicId": "0", "path": "/x/0/f_1001.jpg", "level": 1, "index": 1, "tags": {}},
        ]
        for k, v in enumerate(views):
            R, C = model.pose("0", v["level"], v["index"])
            for i in layout:
                uv, _ = tp.project(R, C, model.p["K_0"], model.p["D_0"], layout[i])
                v["tags"][i] = uv
        obs = tp.build_observations(views, layout)
        # Outlier pass: the second photo keeps only 3 of its 8 corners
        keep = (obs["vi"] == 0) | (np.cumsum(obs["vi"] == 1) <= 3) & (obs["vi"] == 1)
        obs = {k: v[keep] for k, v in obs.items()}
        self.assertEqual(int((obs["vi"] == 1).sum()), 3)

        lines = []
        args = types.SimpleNamespace(stepDeg=18.0)
        intr = {"0": {"width": 4056, "height": 3040, "sensorWidth": 6.287}}
        tp.write_report(model, views, obs, intr, args, lines.append)

        row0 = next(l for l in lines if l.startswith("0/f_1000.jpg"))
        row1 = next(l for l in lines if l.startswith("0/f_1001.jpg"))
        self.assertRegex(row0, r"/ [\d.]+$")        # free-pose difference computed (8 corners)
        self.assertNotIn("/", row1.split()[-1])     # skipped (3 corners), no crash


if __name__ == "__main__":
    unittest.main()
