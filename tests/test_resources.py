"""
test_resources.py — cgroup-aware CPU/RAM limits, DensifyPointCloud RAM estimate and worker count.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import resources
from pipeline.resources import (
    InsufficientRamError,
    available_cpus,
    check_ram,
    estimate_ram_gb,
    resolve_workers,
    total_ram_gb,
    working_megapixels,
)

GB = 1024 ** 3


class FakeVM:
    def __init__(self, total_gb, available_gb):
        self.total = int(total_gb * GB)
        self.available = int(available_gb * GB)


class TestResources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        os.environ.pop(resources.WORKERS_ENV, None)
        self.addCleanup(env.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def test_cgroup_v2_cpu_quota_caps_cpus(self):
        self.write("cpu.max", "400000 100000")  # 4 CPUs
        with mock.patch("os.sched_getaffinity", return_value=set(range(64))):
            self.assertEqual(available_cpus(self.root), 4)

    def test_unlimited_cgroup_uses_affinity(self):
        self.write("cpu.max", "max 100000")
        with mock.patch("os.sched_getaffinity", return_value=set(range(12))):
            self.assertEqual(available_cpus(self.root), 12)

    def test_cgroup_v1_quota(self):
        self.write("cpu/cpu.cfs_quota_us", "200000")
        self.write("cpu/cpu.cfs_period_us", "100000")
        with mock.patch("os.sched_getaffinity", return_value=set(range(8))):
            self.assertEqual(available_cpus(self.root), 2)

    def test_container_ram_limit_wins_over_host(self):
        """psutil sees the host (256 GB); the container may use 32 GB."""
        self.write("memory.max", str(32 * GB))
        with mock.patch("psutil.virtual_memory", return_value=FakeVM(256, 200)):
            self.assertAlmostEqual(total_ram_gb(self.root), 32.0, places=3)

    def test_unlimited_memory_falls_back_to_host(self):
        self.write("memory.max", "max")
        with mock.patch("psutil.virtual_memory", return_value=FakeVM(16, 10)):
            self.assertAlmostEqual(total_ram_gb(self.root), 16.0, places=3)

    def test_estimate_matches_measured_runs(self):
        """Fitted on the multimeter (80 photos, 9.8 GB) and the hairspray (120 photos, 11.3 GB) at level 1."""
        self.assertAlmostEqual(working_megapixels(4056, 3040, 1), 3.08, places=2)
        self.assertAlmostEqual(estimate_ram_gb(80, 4056, 3040, 1), 9.8, delta=0.15)
        self.assertAlmostEqual(estimate_ram_gb(120, 4056, 3040, 1), 11.3, delta=0.15)
        # Full resolution needs much more than half resolution
        self.assertGreater(estimate_ram_gb(120, 4056, 3040, 0), 20.0)

    def test_workers_limited_by_cpus_and_ram(self):
        self.write("memory.max", str(4 * GB))
        self.write("memory.current", str(1 * GB))
        with mock.patch("os.sched_getaffinity", return_value=set(range(32))), \
                mock.patch("psutil.virtual_memory", return_value=FakeVM(64, 60)):
            # free = 3 GB inside the cgroup -> (3 - 1) / 0.5 = 4 workers, not 31
            self.assertEqual(resolve_workers(root=self.root), 4)
        with mock.patch("os.sched_getaffinity", return_value=set(range(6))), \
                mock.patch("psutil.virtual_memory", return_value=FakeVM(64, 60)):
            self.write("memory.max", "max")
            self.assertEqual(resolve_workers(root=self.root), 5)  # CPUs - 1

    def test_workers_override_and_minimum(self):
        self.assertEqual(resolve_workers(3, root=self.root), 3)
        with mock.patch.dict(os.environ, {resources.WORKERS_ENV: "7"}):
            self.assertEqual(resolve_workers(0, root=self.root), 7)
        self.write("memory.max", str(1 * GB))
        self.write("memory.current", str(1 * GB))
        with mock.patch("os.sched_getaffinity", return_value={0}):
            self.assertEqual(resolve_workers(root=self.root), 1)

    def test_check_ram(self):
        self.write("memory.max", str(16 * GB))
        with mock.patch("psutil.virtual_memory", return_value=FakeVM(64, 60)):
            self.assertEqual(check_ram(80, 4056, 3040, 1, root=self.root)["status"], "ok")
            self.assertEqual(check_ram(120, 4056, 3040, 1, margin=1.3, root=self.root)["status"], "tight")
            with self.assertRaises(InsufficientRamError) as ctx:
                check_ram(120, 4056, 3040, 0, root=self.root)
            self.assertGreater(ctx.exception.required_gb, ctx.exception.available_gb)


if __name__ == "__main__":
    unittest.main()
