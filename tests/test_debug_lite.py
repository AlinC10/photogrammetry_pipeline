"""
test_debug_lite.py — The debug archive of a scan (pipeline/debug_lite.py): raw mesh as Draco, masks at 1/4, hull,
refinement points and the small files; size budget; the pod skips the upload without debug-bucket credentials and
never uses the delivery bucket for it.
"""

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import cv2
    import trimesh
    import DracoPy  # noqa: F401
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from pipeline import debug_lite


@unittest.skipUnless(HAS_DEPS, "OpenCV, trimesh and DracoPy required")
class TestDebugLite(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.out, self.work = root / "output", root / "work"
        (self.out / "masks").mkdir(parents=True)
        (self.work / "openmvs").mkdir(parents=True)
        (self.work / "debug_lite").mkdir(parents=True)
        self.sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
        self.sphere.export(str(self.work / "openmvs" / "scene_mesh.ply"))
        for i in range(3):
            m = np.zeros((400, 600), np.uint8)
            m[100:300, 200:400] = 255
            cv2.imwrite(str(self.out / "masks" / f"{1000 + i}_f_{1000 + i}.mask.png"), m)
        (self.out / "stats.json").write_text(json.dumps({"a": 1}))
        (self.out / "cameras_known.sfm").write_text("{}")
        np.savez_compressed(self.work / "visual_hull.npz", occupancy=np.ones((3, 3, 3), bool))
        np.savez_compressed(self.work / "debug_lite" / "refine_points.npz", pruned=np.zeros((5, 3), np.float32),
                            added=np.ones((2, 3), np.float32))

    def test_archive_has_every_part(self):
        info = debug_lite.build_debug_lite(str(self.out), str(self.work), str(self.work / "d.zip"))
        names = zipfile.ZipFile(info["path"]).namelist()
        for n in ("stats.json", "cameras_known.sfm", "visual_hull.npz", "refine_points.npz", "raw_mesh.drc"):
            self.assertIn(n, names)
        self.assertEqual(sum(n.startswith("masks/") for n in names), 3)
        V, F = debug_lite.read_raw_mesh(debug_lite.unzip_member(info["path"], "raw_mesh.drc"))
        self.assertEqual(len(F), len(self.sphere.faces))
        self.assertAlmostEqual(float(np.abs(V).max()), 0.04, delta=0.001)
        mask = cv2.imdecode(np.frombuffer(debug_lite.unzip_member(info["path"], "masks/1000_f_1000.mask.png"),
                                          np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask.shape, (100, 150))
        pts = debug_lite.load_npz_member(info["path"], "refine_points.npz")
        self.assertEqual(pts["pruned"].shape, (5, 3))

    def test_prebuilt_parts_give_the_same_archive(self):
        """B2: the large parts built during the run (build_debug_parts) + the small files = the archive built at the end."""
        ref = debug_lite.build_debug_lite(str(self.out), str(self.work), str(self.work / "ref.zip"))
        parts = debug_lite.build_debug_parts(str(self.out), str(self.work), str(self.work / "parts.zip"))
        self.assertGreaterEqual(parts["parts"], 6)
        (self.out / "stats.json").write_text(json.dumps({"a": 2}))     # the final stats.json comes later
        info = debug_lite.build_debug_lite(str(self.out), str(self.work), str(self.work / "d.zip"),
                                           parts_zip=parts["path"])
        with zipfile.ZipFile(ref["path"]) as a, zipfile.ZipFile(info["path"]) as b:
            self.assertEqual(a.namelist(), b.namelist())
            for n in a.namelist():
                if n != "stats.json":
                    self.assertEqual(a.read(n), b.read(n), n)
            self.assertEqual(json.loads(b.read("stats.json")), {"a": 2})

    def test_size_budget(self):
        info = debug_lite.build_debug_lite(str(self.out), str(self.work), str(self.work / "d.zip"), max_mb=0.0005)
        self.assertTrue(info["skipped"])
        self.assertLess(os.path.getsize(info["path"]), 4096)

    def test_pod_skips_without_debug_credentials_and_never_uses_the_delivery_bucket(self):
        import runpod_job

        with mock.patch.dict(os.environ, {"R2_DEBUG_BUCKET": "", "R2_DEBUG_ACCESS_KEY_ID": ""}):
            self.assertFalse(runpod_job.upload_debug_lite("20261102-000000-abcdef")["uploaded"])
        with mock.patch.dict(os.environ, {"R2_BUCKET": "photogrammetry", "R2_DEBUG_BUCKET": "photogrammetry",
                                          "R2_DEBUG_ACCESS_KEY_ID": "x"}):
            res = runpod_job.upload_debug_lite("20261102-000000-abcdef")
            self.assertFalse(res["uploaded"])
            self.assertIn("delivery", res["reason"])

    def test_debug_credentials_only_for_a_separate_bucket(self):
        from cloud import cloudflare_r2 as r2

        with mock.patch.dict(os.environ, {"R2_BUCKET": "photogrammetry", "R2_DEBUG_BUCKET": "photogrammetry"}):
            self.assertEqual(r2.debug_credentials_env("20261102-000000-abcdef", 600), {})
        with mock.patch.dict(os.environ, {"R2_DEBUG_BUCKET": ""}):
            self.assertEqual(r2.debug_credentials_env("20261102-000000-abcdef", 600), {})
        fake = r2.TempCredentials("AK", "SK", "TOK")
        with mock.patch.dict(os.environ, {"R2_BUCKET": "photogrammetry", "R2_DEBUG_BUCKET": "photogrammetry-debug",
                                          "CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_R2_API_TOKEN": "tok",
                                          "R2_ACCESS_KEY_ID": "parent"}), \
                mock.patch.object(r2, "create_temp_credentials", return_value=fake) as create:
            env = r2.debug_credentials_env("20261102-000000-abcdef", 600)
        self.assertEqual(env["R2_DEBUG_BUCKET"], "photogrammetry-debug")
        self.assertEqual(env["R2_DEBUG_SESSION_TOKEN"], "TOK")
        args = create.call_args[0]
        self.assertEqual(args[2], "photogrammetry-debug")
        self.assertEqual(args[5], ["jobs/20261102-000000-abcdef/"])


if __name__ == "__main__":
    unittest.main()
