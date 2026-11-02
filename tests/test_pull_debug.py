"""
tools/pull_debug.py: debug data moved from its own R2 bucket to the PC, deleted only after a verified download, never
from the delivery bucket.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cloud import cloudflare_r2 as r2  # noqa: E402
from tests.test_cloud import FakeS3  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pull_debug", Path(__file__).resolve().parent.parent / "tools" / "pull_debug.py")
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)


class SizedS3(FakeS3):
    """FakeS3 whose listing also gives Size, like R2."""

    def __init__(self, short=()):
        super().__init__()
        self.short = set(short)    # keys whose download comes back truncated

    def download_file(self, bucket, key, file_name):
        super().download_file(bucket, key, file_name)
        if key in self.short:
            Path(file_name).write_bytes(self.objects[(bucket, key)][:-1])

    def get_paginator(self, _name):
        store = self.objects

        class Pager:
            def paginate(self, Bucket, Prefix=""):
                yield {"Contents": [{"Key": k, "Size": len(v)} for (b, k), v in sorted(store.items())
                                    if b == Bucket and k.startswith(Prefix)]}
        return Pager()


class TestPullDebug(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_downloads_everything_then_deletes_only_the_debug_bucket(self):
        s3 = SizedS3()
        s3.objects[("dbg", "jobs/a/debug/scene_mesh.drc")] = b"mesh"
        s3.objects[("dbg", "jobs/b/debug/masks/0.png")] = b"png!"
        s3.objects[("deliver", "jobs/a/output/output.zip")] = b"zip"
        objs = pd.list_objects(s3, "dbg")
        verified, failed = pd.pull(s3, "dbg", self.dest, objs)
        self.assertEqual(len(verified), 2)
        self.assertEqual(failed, [])
        self.assertEqual((self.dest / "jobs/a/debug/scene_mesh.drc").read_bytes(), b"mesh")
        self.assertEqual((self.dest / "jobs/b/debug/masks/0.png").read_bytes(), b"png!")
        self.assertEqual(list(s3.objects), [("deliver", "jobs/a/output/output.zip")])

    def test_size_mismatch_or_unsafe_key_stays_in_the_bucket(self):
        s3 = SizedS3(short={"jobs/a/x.ply"})
        s3.objects[("dbg", "jobs/a/x.ply")] = b"12345"
        s3.objects[("dbg", "../escape.txt")] = b"no"
        s3.objects[("dbg", "jobs/a/ok.json")] = b"{}"
        verified, failed = pd.pull(s3, "dbg", self.dest, pd.list_objects(s3, "dbg"))
        self.assertEqual(verified, ["jobs/a/ok.json"])
        self.assertEqual(sorted(failed), ["../escape.txt", "jobs/a/x.ply"])
        self.assertIn(("dbg", "jobs/a/x.ply"), s3.objects)
        self.assertIn(("dbg", "../escape.txt"), s3.objects)
        self.assertFalse((self.dest / "jobs/a/x.ply").exists())
        self.assertFalse((self.dest / "jobs/a/x.ply.part").exists())
        self.assertFalse((self.dest.parent / "escape.txt").exists())

    def test_keep_remote_and_identical_local_copy(self):
        s3 = SizedS3()
        s3.objects[("dbg", "jobs/a/m.bin")] = b"abc"
        (self.dest / "jobs/a").mkdir(parents=True)
        (self.dest / "jobs/a/m.bin").write_bytes(b"abc")
        verified, _ = pd.pull(s3, "dbg", self.dest, pd.list_objects(s3, "dbg"), delete=False)
        self.assertEqual(verified, ["jobs/a/m.bin"])
        self.assertIn(("dbg", "jobs/a/m.bin"), s3.objects)

    def test_refuses_the_delivery_bucket_or_no_bucket(self):
        with self.assertRaises(r2.R2Error):
            pd.check_bucket("photogrammetry-pipeline", "photogrammetry-pipeline")
        with self.assertRaises(r2.R2Error):
            pd.check_bucket("", "photogrammetry-pipeline")
        pd.check_bucket("photogrammetry-debug", "photogrammetry-pipeline")


if __name__ == "__main__":
    unittest.main()
