"""
test_regression_tool.py — tools/regression.py: rows from the scan folders, matching by name, regressions flagged
(watertight lost, size / volume changed, new warning, slower) and improvements reported, exit code.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import regression  # noqa: E402


def write_scan(root: Path, job: str, watertight=True, height=100.0, volume=50.0, warnings=(), time_s=200.0):
    d = root / job
    d.mkdir(parents=True)
    (d / "object.json").write_text(json.dumps({
        "job_id": job, "title": job, "watertight": watertight, "volume_cm3": volume, "faces_high": 1000,
        "dimensions_mm": {"length": 50.0, "width": 40.0, "height": height}}))
    (d / "stats.json").write_text(json.dumps({"total_execution_time_sec": time_s}))
    (d / "status.json").write_text(json.dumps({"state": "ok", "warnings": [{"code": w, "message": ""}
                                                                           for w in warnings]}))


class TestRegressionTool(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.names = self.root / "names.json"
        self.names.write_text(json.dumps({"old-1": "duck", "new-1": "duck", "old-2": "box", "new-2": "box"}))

    def test_same_results_no_regression(self):
        write_scan(self.root / "a", "old-1")
        write_scan(self.root / "b", "new-1")
        base = self.root / "base.json"
        self.assertEqual(regression.main([str(self.root / "a"), "--names", str(self.names),
                                          "--save-baseline", str(base)]), 0)
        self.assertEqual(regression.main([str(self.root / "b"), "--names", str(self.names),
                                          "--baseline", str(base)]), 0)

    def test_regressions_and_improvements(self):
        ref = {"watertight": True, "height_mm": 100.0, "volume_cm3": 50.0, "warnings": [], "time_s": 200.0}
        row = {"watertight": False, "height_mm": 97.0, "volume_cm3": 45.0, "warnings": ["model_tiny"],
               "time_s": 300.0}
        d = regression.compare(row, ref)
        self.assertEqual(sum(x.startswith("REGRESSION") for x in d), 5)
        better = regression.compare({"watertight": True, "height_mm": 100.2, "volume_cm3": 50.0, "warnings": [],
                                     "time_s": 100.0},
                                    {"watertight": False, "height_mm": 100.0, "volume_cm3": None,
                                     "warnings": ["not_watertight"], "time_s": 200.0})
        self.assertFalse([x for x in better if x.startswith("REGRESSION")])
        self.assertEqual(len(better), 4)          # watertight, volume known, warning gone, faster

    def test_exit_code_on_regression(self):
        write_scan(self.root / "a", "old-2", height=60.0)
        write_scan(self.root / "b", "new-2", height=43.0)          # the lighter cut at the cone
        base = self.root / "base.json"
        regression.main([str(self.root / "a"), "--names", str(self.names), "--save-baseline", str(base)])
        self.assertEqual(regression.main([str(self.root / "b"), "--names", str(self.names),
                                          "--baseline", str(base)]), 1)


if __name__ == "__main__":
    unittest.main()
