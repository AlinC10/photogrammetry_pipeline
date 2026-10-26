"""
test_camera_init.py — Unit tests for the turntable_poses.py command line (RULE 1: the script is
called as a subprocess and never modified, only its existing options are passed).
"""

import sys
import unittest
from pathlib import Path

# Ensure turntable_studio root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.camera_init import build_turntable_poses_cmd
from pipeline.config import TurntableConfig


class TestTurntablePosesCommand(unittest.TestCase):

    def _opt(self, cmd, name):
        return cmd[cmd.index(name) + 1]

    def test_lift_and_tag_refine_are_passed(self):
        cfg = TurntableConfig()
        cmd = build_turntable_poses_cmd(Path("turntable_poses.py"), "cameraInit.sfm", "cameras_known.sfm", cfg)
        # the script's own default is 80 mm, the scanner raises 100 mm per level
        self.assertEqual(float(self._opt(cmd, "--liftMm")), 100.0)
        self.assertEqual(self._opt(cmd, "--estimateLift"), "False")
        self.assertEqual(self._opt(cmd, "--tagRefine"), "True")
        self.assertNotIn("--intrinsicsIn", cmd)

    def test_custom_options(self):
        cfg = TurntableConfig(lift_mm=120.0, estimate_lift=True, tag_refine=False, intrinsics_in="calib.json")
        cmd = build_turntable_poses_cmd(Path("turntable_poses.py"), "a.sfm", "b.sfm", cfg,
                                        report_path="r.txt", calib_out_path="c.json")
        self.assertEqual(float(self._opt(cmd, "--liftMm")), 120.0)
        self.assertEqual(self._opt(cmd, "--estimateLift"), "True")
        self.assertEqual(self._opt(cmd, "--tagRefine"), "False")
        self.assertTrue(self._opt(cmd, "--intrinsicsIn").endswith("calib.json"))
        self.assertTrue(self._opt(cmd, "--report").endswith("r.txt"))


if __name__ == "__main__":
    unittest.main()
