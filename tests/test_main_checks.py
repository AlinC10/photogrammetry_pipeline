"""
test_main_checks.py — Automatic checks of main.py: warning when a tag correction is implausible (object on the tags),
masking cylinder widened when the visual hull reaches it.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import main
from pipeline.config import PipelineConfig

REPORT_BAD = """\
Corecția tag-urilor (denivelări ale hârtiei, față de planul discului): 0: +2.43 mm, 1: -6.79 mm, 2: +0.17 mm
Total: 2111 colțuri, RMS 2.646 px, maxim 7.913 px
"""


class FakeTelemetry:
    def __init__(self):
        self.info = {}

    def set_info(self, **kw):
        self.info.update(kw)


class TestMainChecks(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_tag_correction_over_limit_warns_and_keeps_poses(self):
        (self.out / "turntable_report.txt").write_text(REPORT_BAD, encoding="utf-8")
        config, tel = PipelineConfig(), FakeTelemetry()
        q = main.run_metrics.pose_quality(str(self.out / "turntable_report.txt"))
        with mock.patch.object(main, "run_turntable_poses") as run:
            main._check_tag_corrections(q, config, tel)
        run.assert_not_called()
        self.assertEqual(tel.info["tag_correction_warning"]["worst_correction_mm"], 6.79)

    def test_plausible_tag_corrections_no_warning(self):
        config, tel = PipelineConfig(), FakeTelemetry()
        main._check_tag_corrections({"tag_correction_max_mm": 1.49, "rms_px": 2.2}, config, tel)
        self.assertNotIn("tag_correction_warning", tel.info)

    def test_hull_touching_cylinder(self):
        config = PipelineConfig()
        v = 0.0015
        n = int(round(0.26 / v)) + 1
        occ = np.zeros((n, 20, n), bool)
        origin = np.array([-0.13, 0.003, -0.13])
        centre = int(round(0.13 / v))
        inside = lambda r: occ.__setitem__((slice(centre - int(r / v), centre + int(r / v) + 1), slice(None),
                                            slice(centre - 2, centre + 3)), True)
        inside(0.050)
        hull = {"occupancy": occ.copy(), "origin": origin, "voxel_m": v}
        self.assertFalse(main._hull_touches_cylinder(hull, config))
        inside(0.0795)                                    # reaches the 80 mm cylinder
        hull = {"occupancy": occ, "origin": origin, "voxel_m": v}
        self.assertTrue(main._hull_touches_cylinder(hull, config))
        config.masking.cylinder_radius_m = config.masking.wide_cylinder_radius_m
        self.assertFalse(main._hull_touches_cylinder(hull, config))   # already wide: only once


if __name__ == "__main__":
    unittest.main()
