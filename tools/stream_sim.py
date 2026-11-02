"""
stream_sim.py — Local only: the streaming mode of the pod on the photos of an earlier scan (plans/PLAN_STREAMING_DAG.md
§5 step 7). No OpenMVS needed.

1. A "scanner" thread puts the photos into an in-memory S3 in the order and at the pace they were taken (file mtimes,
   divided by --speed), then level_done_<L>.json and capture_done.json, like the Pi in streaming mode.
2. The pod side lists / downloads them (runpod_job.stream_inputs) and does the pre-work (pipeline/stream_prework.py:
   SIFT, medians, preliminary poses + AI crops of every finished level).
3. The front of the pipeline (camera init, turntable_poses, rig_refine, fused masks) runs twice on the same photos:
   without any cache, and with the streaming cache. Poses, rig_refine.json and every mask must be identical (byte for
   byte); the report gives the times of both runs, the AI crops reused and how long after capture_done the
   pre-work of each level ended.

    python tools/stream_sim.py rpi_results/20261101-084007-a89a0e --speed 2 --workers 2 --out /tmp/sim
    TURNTABLE_MASK_MODEL=/path/BiRefNet.onnx (the model of the image) must be set for the AI masks.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloud import cloudflare_r2 as r2  # noqa: E402

BUCKET = "sim"


class MemS3:
    """The S3 calls of runpod_job.stream_inputs, in memory, safe for one writer thread."""

    def __init__(self):
        self.objects: Dict[str, bytes] = {}
        self.lock = threading.Lock()

    def put_object(self, Bucket, Key, Body, **_):
        with self.lock:
            self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError
        with self.lock:
            if Key not in self.objects:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, Bucket, Key, Filename):
        with self.lock:
            data = self.objects[Key]
        Path(Filename).write_bytes(data)

    def get_paginator(self, _name):
        outer = self

        class Pager:
            def paginate(self, Bucket, Prefix=""):
                with outer.lock:
                    keys = sorted(k for k in outer.objects if k.startswith(Prefix))
                yield {"Contents": [{"Key": k} for k in keys]}
        return Pager()


def scanner(s3: MemS3, job_id: str, photos_dir: Path, speed: float, log: Dict[str, Any]) -> threading.Thread:
    files = sorted((p for p in photos_dir.rglob("*.jpg")), key=lambda p: (p.stat().st_mtime, str(p)))
    t_first = files[0].stat().st_mtime

    def run():
        t0 = time.time()
        levels: Dict[str, list] = {}
        for p in files:
            due = (p.stat().st_mtime - t_first) / speed
            time.sleep(max(0.0, due - (time.time() - t0)))
            rel = p.relative_to(photos_dir).as_posix()
            s3.put_object(Bucket=BUCKET, Key=r2.input_key(job_id, rel), Body=p.read_bytes())
            L = p.stem[2]
            levels.setdefault(L, []).append(rel)
            # the level is uploaded when the first photo of the next one arrives (the Pi writes the marker then)
            for done_L in [k for k in levels if k < L and f"marker_{k}" not in log]:
                s3.put_object(Bucket=BUCKET, Key=r2.input_key(job_id, f"level_done_{done_L}.json"),
                              Body=json.dumps({"level": int(done_L), "files": levels[done_L]}))
                log[f"marker_{done_L}"] = round(time.time() - t0, 2)
        log["last_photo_s"] = round(time.time() - t0, 2)
        all_files = [f for v in levels.values() for f in v]
        s3.put_object(Bucket=BUCKET, Key=r2.input_key(job_id, "capture_done.json"),
                      Body=json.dumps({"n_images": len(all_files), "files": sorted(all_files)}))
        log["capture_done_s"] = round(time.time() - t0, 2)
        log["t0"] = t0

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def front_of_pipeline(input_dir: Path, out: Path, job: Dict[str, Any], workers: int, cache_dir,
                      poses_from: Path = None) -> Dict[str, Any]:
    """
    Camera init, turntable_poses, rig_refine, fused masks: as run_pipeline does them. `poses_from`: the turntable_poses
    output of another run is used (turntable_poses itself differs in the 9th digit from run to run: the comparison of
    what the caches change needs the same tag poses).
    """
    import main
    from pipeline import run_metrics, stream_cache
    from pipeline.camera_init import generate_camera_init_sfm, run_turntable_poses, scan_input_directory
    from pipeline.config import PipelineConfig
    from pipeline.telemetry import SystemTelemetry

    out.mkdir(parents=True, exist_ok=True)
    tel = SystemTelemetry(str(out))
    cfg = PipelineConfig(input_dir=str(input_dir), output_dir=str(out), rig_mode=True,
                         turntable=main.turntable_config(job), masking=main.masking_config(job),
                         stream_cache_dir=cache_dir)
    stream_cache.stats(reset=True)
    times = {}
    t0 = time.time()
    views, intr, rx = scan_input_directory(str(input_dir), rig_mode=True)
    cfg.turntable.name_regex = rx
    init = out / "cameraInit.sfm"
    generate_camera_init_sfm(views, intr, str(init))
    n_levels = int((run_metrics.input_stats(str(init), rx) or {}).get("n_levels") or 1)
    cfg.turntable.estimate_lift = n_levels >= 2
    known = out / "cameras_known.sfm"
    if poses_from is not None:
        src = poses_from / "cameras_known_tags.sfm"
        shutil.copy(src if src.is_file() else poses_from / "cameras_known.sfm", known)
    else:
        run_turntable_poses(str(init), str(known), cfg.turntable, report_path=str(out / "turntable_report.txt"),
                            calib_out_path=str(out / "turntable_calib.json"))
    times["poses_s"] = round(time.time() - t0, 1)
    t1 = time.time()
    main._refine_rig(str(known), cfg, out, workers, tel)
    times["rig_refine_s"] = round(time.time() - t1, 1)
    t2 = time.time()
    ai_dir = stream_cache.ai_dir(cache_dir) or str(out / "ai_masks_cache")
    if not cache_dir:
        shutil.rmtree(ai_dir, ignore_errors=True)
    main._generate_masks(str(known), str(out / "masks"), cfg, workers, tel, ai_dir)
    times["masks_s"] = round(time.time() - t2, 1)
    masks = tel.run_info.get("masks") or {}
    return {"times": times, "model_runs": masks.get("model_runs"), "ai_cache_hits": masks.get("ai_cache_hits"),
            "cache": stream_cache.stats()}


def compare(a: Path, b: Path) -> Dict[str, Any]:
    res = {"cameras_known.sfm": (a / "cameras_known.sfm").read_bytes() == (b / "cameras_known.sfm").read_bytes()}
    ra = json.loads((a / "rig_refine.json").read_text())
    rb = json.loads((b / "rig_refine.json").read_text())
    for r in (ra, rb):
        r.pop("timings_s", None)
    res["rig_refine.json"] = ra == rb
    ma = sorted(p.relative_to(a / "masks") for p in (a / "masks").rglob("*.png"))
    mb = sorted(p.relative_to(b / "masks") for p in (b / "masks").rglob("*.png"))
    diff = [str(p) for p in ma if (a / "masks" / p).read_bytes() != (b / "masks" / p).read_bytes()] if ma == mb else ["file lists differ"]
    res["masks_identical"] = f"{len(ma) - len(diff)}/{len(ma)}"
    res["masks_different"] = diff[:10]
    return res


def main_cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scan", help="results folder of a scan (with photos/rig/0, photos/rig/1)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay the capture this many times faster")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="stream_sim_out")
    ap.add_argument("--skip-reference", action="store_true", help="only the streaming run")
    ap.add_argument("--determinism", action="store_true",
                    help="also a second run without cache on the same tag poses (what differs by itself)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    import runpod_job
    from pipeline.stream_prework import StreamPrework

    scan = Path(args.scan)
    out = Path(args.out).resolve() / scan.name
    shutil.rmtree(out, ignore_errors=True)
    job = {"rig_mode": True, "step_deg": 18.0, "workers": args.workers}
    s3, log = MemS3(), {}
    pw = StreamPrework(str(out / "in"), str(out / "stream"), job, workers=args.workers)
    level_end: Dict[str, float] = {}
    orig_level = pw._level

    def timed_level(level, groups):
        r = orig_level(level, groups)
        level_end[str(level)] = time.time()
        return r

    pw._level = timed_level
    t = scanner(s3, "sim", scan / "photos", args.speed, log)
    got = runpod_job.stream_inputs("sim", BUCKET, s3, out / "in", on_photo=pw.photo_arrived,
                                   on_level=pw.level_marker, poll_s=0.5)
    t.join()
    t_done = time.time()
    pw.capture_done()
    info = pw.finish(timeout=900)
    report: Dict[str, Any] = {
        "scan": scan.name, "photos": got["photos"], "speed": args.speed,
        "capture_s": log.get("capture_done_s"),
        "prework": info,
        "level_prework_end_after_capture_done_s": {L: round(v - t_done, 1) for L, v in level_end.items()},
    }
    if args.skip_reference:
        report["streaming_run"] = front_of_pipeline(out / "in", out / "streamed", job, args.workers,
                                                    str(out / "stream"))
    else:
        # the reference first (it does not touch the streaming cache), then the streamed run on its tag poses
        report["reference_run"] = front_of_pipeline(out / "in", out / "reference", job, args.workers, None)
        report["streaming_run"] = front_of_pipeline(out / "in", out / "streamed", job, args.workers,
                                                    str(out / "stream"), poses_from=out / "reference")
        report["identical"] = compare(out / "reference", out / "streamed")
        if args.determinism:
            report["reference_again"] = front_of_pipeline(out / "in", out / "reference2", job, args.workers, None,
                                                          poses_from=out / "reference")
            report["identical_without_cache"] = compare(out / "reference", out / "reference2")
    print(json.dumps(report, indent=1, default=str))
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
