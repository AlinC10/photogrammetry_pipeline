"""
test_telemetry.py — Unit test for RAM/VRAM telemetry and benchmark tracking.
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Ensure turntable_studio root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.telemetry import SystemTelemetry, get_process_tree_ram_mb


class TestTelemetry(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_process_ram_measurement(self):
        ram_mb = get_process_tree_ram_mb()
        self.assertGreater(ram_mb, 0.0)

    def test_telemetry_step_recording(self):
        telemetry = SystemTelemetry(output_dir=self.test_dir)
        telemetry.start_sampling(interval_sec=0.05)

        with telemetry.step("Sample Step 1"):
            time.sleep(0.1)

        with telemetry.step("Sample Step 2"):
            time.sleep(0.1)

        telemetry.stop_sampling()
        summary = telemetry.summary()

        self.assertEqual(len(summary["steps"]), 2)
        self.assertEqual(summary["steps"][0]["name"], "Sample Step 1")
        self.assertGreaterEqual(summary["steps"][0]["duration_sec"], 0.08)
        self.assertGreater(summary["global_peak_ram_mb"], 0.0)

        json_file = telemetry.save_json()
        self.assertTrue(Path(json_file).is_file())


if __name__ == "__main__":
    unittest.main()
