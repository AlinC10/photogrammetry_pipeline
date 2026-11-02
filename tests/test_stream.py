"""
test_stream.py — Streaming mode of the pod (plans/PLAN_STREAMING_DAG.md §2): photos downloaded as they appear in an
in-memory S3, level markers, capture_done.json, cancel.json, the scanner gone silent; the pre-work (SIFT, medians)
gives cache entries equal to what the pipeline computes; runpod_job.main() end to end with the mock pipeline.
"""

import io
import json
import os
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from cloud import cloudflare_r2 as r2
from test_cloud import FakeS3

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

BUCKET = "test-bucket"


class LockedS3(FakeS3):
    """FakeS3 written by a 'scanner' thread while the pod lists it."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()

    def upload_file(self, file_name, bucket, key):
        with self.lock:
            super().upload_file(file_name, bucket, key)

    def put_object(self, Bucket, Key, Body, **kw):
        with self.lock:
            super().put_object(Bucket=Bucket, Key=Key, Body=Body, **kw)

    def get_paginator(self, name):
        outer = self

        class Pager:
            def paginate(self, Bucket, Prefix=""):
                with outer.lock:
                    keys = [k for (b, k) in sorted(outer.objects) if b == Bucket and k.startswith(Prefix)]
                yield {"Contents": [{"Key": k} for k in keys]}
        return Pager()


def photo_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def scanner(s3, job_id, photos, gap_s=0.02, level_markers=True, done=True, extra_before=None):
    """Uploads `photos` ({relative key: bytes}) one by one, level markers, then capture_done.json."""
    def run():
        if extra_before:
            for rel, data in extra_before.items():
                s3.put_object(Bucket=BUCKET, Key=r2.input_key(job_id, rel), Body=data)
        by_level = {}
        for rel, data in photos.items():
            s3.put_object(Bucket=BUCKET, Key=r2.input_key(job_id, rel), Body=data)
            by_level.setdefault(Path(rel).stem[2], []).append(rel)
            time.sleep(gap_s)
        if level_markers:
            for L, files in by_level.items():
                s3.put_object(Bucket=BUCKET, Key=r2.input_key(job_id, f"level_done_{L}.json"),
                              Body=json.dumps({"level": int(L), "files": files}))
        if done:
            s3.put_object(Bucket=BUCKET, Key=r2.input_key(job_id, "capture_done.json"),
                          Body=json.dumps({"n_images": len(photos), "files": sorted(photos), "step_deg": 120}))
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def rig_photos(levels=(1, 2), per_level=3):
    return {f"rig/{cam}/f_{L}{k:03d}.jpg": photo_bytes(100 * L + 10 * int(cam) + k)
            for L in levels for cam in ("0", "1") for k in range(per_level)}


@unittest.skipUnless(HAS_CV2, "OpenCV required")
class TestStreamInputs(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.s3 = LockedS3()

    def test_photos_downloaded_as_they_arrive(self):
        import runpod_job

        photos = rig_photos()
        seen, levels = [], []
        t = scanner(self.s3, "j1", photos)
        got = runpod_job.stream_inputs("j1", BUCKET, self.s3, self.dir / "in",
                                       on_photo=lambda rel, p: seen.append((rel, Path(p).is_file())),
                                       on_level=lambda L, files: levels.append((L, len(files))),
                                       poll_s=0.01, idle_s=10)
        t.join()
        self.assertEqual(got["photos"], len(photos))
        self.assertEqual(sorted(r for r, _ in seen), sorted(photos))
        self.assertTrue(all(ok for _, ok in seen))                # the file exists when on_photo is called
        for rel, data in photos.items():
            self.assertEqual((self.dir / "in" / rel).read_bytes(), data)
        self.assertFalse(list((self.dir / "in").rglob("*.part")))
        # the markers are not downloaded as inputs
        self.assertFalse(list((self.dir / "in").glob("*.json")))
        self.assertIn(sorted(levels)[0], [(1, 6)])
        self.assertEqual(got["manifest"]["n_images"], len(photos))

    def test_cancel(self):
        import runpod_job

        self.s3.put_object(Bucket=BUCKET, Key=r2.input_key("j2", "cancel.json"), Body="{}")
        with self.assertRaises(runpod_job.StreamCancelled):
            runpod_job.stream_inputs("j2", BUCKET, self.s3, self.dir / "in", poll_s=0.01, idle_s=10)

    def test_scanner_gone_silent(self):
        import runpod_job

        t = scanner(self.s3, "j3", rig_photos(levels=(1,)), done=False)
        t0 = time.time()
        with self.assertRaises(runpod_job.StreamInputError) as cm:
            runpod_job.stream_inputs("j3", BUCKET, self.s3, self.dir / "in", poll_s=0.01, idle_s=0.4)
        t.join()
        self.assertIn("abandoned", str(cm.exception))
        self.assertLess(time.time() - t0, 5)

    def test_manifest_photo_missing_in_r2(self):
        import runpod_job

        self.s3.put_object(Bucket=BUCKET, Key=r2.input_key("j4", "rig/0/f_1000.jpg"), Body=photo_bytes(1))
        self.s3.put_object(Bucket=BUCKET, Key=r2.input_key("j4", "capture_done.json"),
                           Body=json.dumps({"files": ["rig/0/f_1000.jpg", "rig/0/f_1001.jpg"]}))
        with self.assertRaises(runpod_job.StreamInputError):
            runpod_job.stream_inputs("j4", BUCKET, self.s3, self.dir / "in", poll_s=0.01, idle_s=10)

    def test_photo_outside_the_manifest_is_dropped(self):
        """A photo the scanner did not verify (left from an earlier attempt) does not reach the pipeline."""
        import runpod_job

        photos = rig_photos(levels=(1,))
        t = scanner(self.s3, "j5", photos, extra_before={"rig/0/f_1999.jpg": photo_bytes(9)})
        got = runpod_job.stream_inputs("j5", BUCKET, self.s3, self.dir / "in", poll_s=0.01, idle_s=10)
        t.join()
        self.assertEqual(got["photos"], len(photos))
        self.assertFalse((self.dir / "in" / "rig" / "0" / "f_1999.jpg").exists())


@unittest.skipUnless(HAS_CV2, "OpenCV required")
class TestStreamPrework(unittest.TestCase):

    def test_sift_and_medians_match_the_pipeline(self):
        from pipeline import stream_cache
        from pipeline.config import RigRefineConfig
        from pipeline.mask_fusion import group_medians
        from pipeline.rig_refine import _detect
        from pipeline.stream_prework import StreamPrework

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            photos = rig_photos(levels=(1, 2), per_level=3)
            cache = d / "stream"
            pw = StreamPrework(str(d / "in"), str(cache), {"step_deg": 120}, workers=2, level_masks=False)
            for rel, data in photos.items():
                p = d / "in" / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                pw.photo_arrived(rel, str(p))
            pw.capture_done()
            info = pw.finish(timeout=60)
            self.assertEqual(info["sift_photos"], len(photos))
            self.assertEqual(sorted(info["levels"]), ["1", "2"])         # both levels complete by count (3 shots)

            paths = [str(d / "in" / r) for r in sorted(photos)]
            groups = [(r.split("/")[1], Path(r).stem[2]) for r in sorted(photos)]
            stream_cache.stats(reset=True)
            cached = group_medians(paths, groups, d / "a", cache_dir=str(cache))
            fresh = group_medians(paths, groups, d / "b")
            self.assertEqual(stream_cache.stats().get("median_hit"), 4)
            for g in fresh:
                self.assertTrue(np.array_equal(np.load(cached[g]), np.load(fresh[g])))
            K = np.array([[150.0, 0, 80], [0, 150.0, 60], [0, 0, 1]])
            cfg = RigRefineConfig()
            for p in paths[:3]:
                a, b = _detect(p, K, np.zeros(5), cfg), _detect(p, K, np.zeros(5), cfg, str(cache))
                for x, y in zip(a, b):
                    self.assertTrue(np.array_equal(x, y))

    def test_incomplete_level_waits(self):
        from pipeline.stream_prework import StreamPrework

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pw = StreamPrework(str(d), str(d / "stream"), {"step_deg": 120, "rig_refine": False}, level_masks=False)
            photos = rig_photos(levels=(1,), per_level=3)
            for rel, data in list(photos.items())[:4]:                     # cam 1 has 1 of 3 shots
                p = d / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                pw.photo_arrived(rel, str(p))
            self.assertFalse(pw.levels_started)
            pw.level_marker(1, [])                                          # the scanner says it is complete
            self.assertEqual(pw.levels_started, {1})
            pw.finish(timeout=30)


@unittest.skipUnless(HAS_CV2, "OpenCV required")
class TestPodStreamingEndToEnd(unittest.TestCase):
    def test_pod_streams_photos_then_runs_the_mock_pipeline(self):
        import runpod_job

        s3 = LockedS3()
        photos = rig_photos(levels=(1,), per_level=3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = {"JOB_ID": "js", "R2_BUCKET": BUCKET, "POD_MAX_RUNTIME_S": "3600",
                   "JOB_JSON": json.dumps({"mock_openmvs": True, "rig_mode": True, "stream": True, "step_deg": 120,
                                           "rig_refine": False})}
            t = scanner(s3, "js", photos, gap_s=0.05)
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(r2, "client_from_env", lambda prefix="R2_": s3), \
                    mock.patch.object(runpod_job, "INPUT_DIR", tmp / "in"), \
                    mock.patch.object(runpod_job, "OUTPUT_DIR", tmp / "out"), \
                    mock.patch.object(runpod_job, "WORK_DIR", tmp / "work"), \
                    mock.patch.object(runpod_job, "STREAM_CACHE_DIR", tmp / "work" / "stream"), \
                    mock.patch.object(runpod_job, "STREAM_POLL_S", 0.01):
                code = runpod_job.main()
            t.join()
            status = r2.get_json(r2.output_key("js", "status.json"), BUCKET, client=s3)
            self.assertEqual(code, 0, status)
            self.assertEqual(status["state"], "ok")
            self.assertEqual(status["stream"]["photos"], len(photos))
            self.assertIn("wait_after_capture_s", status["timings_s"])
            progress = r2.get_json(r2.output_key("js", "progress.json"), BUCKET, client=s3)
            self.assertIsNotNone(progress)
            stats = json.loads(zipfile.ZipFile(io.BytesIO(s3.objects[(BUCKET, r2.output_key("js", "output.zip"))]))
                               .read("stats.json"))
            self.assertEqual(stats["run_info"]["stream"]["photos"], len(photos))

    def test_pod_reports_cancel(self):
        import runpod_job

        s3 = LockedS3()
        s3.put_object(Bucket=BUCKET, Key=r2.input_key("jc", "cancel.json"), Body="{}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = {"JOB_ID": "jc", "R2_BUCKET": BUCKET, "POD_MAX_RUNTIME_S": "3600",
                   "JOB_JSON": json.dumps({"mock_openmvs": True, "stream": True})}
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(r2, "client_from_env", lambda prefix="R2_": s3), \
                    mock.patch.object(runpod_job, "INPUT_DIR", tmp / "in"), \
                    mock.patch.object(runpod_job, "OUTPUT_DIR", tmp / "out"), \
                    mock.patch.object(runpod_job, "WORK_DIR", tmp / "work"), \
                    mock.patch.object(runpod_job, "STREAM_CACHE_DIR", tmp / "work" / "stream"), \
                    mock.patch.object(runpod_job, "STREAM_POLL_S", 0.01):
                code = runpod_job.main()
            status = r2.get_json(r2.output_key("jc", "status.json"), BUCKET, client=s3)
        self.assertEqual(code, 1)
        self.assertEqual(status["state"], "cancelled")


if __name__ == "__main__":
    unittest.main()
