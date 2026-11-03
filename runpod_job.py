"""
runpod_job.py — Entry point inside the RunPod pod (docker_entrypoint.sh starts it when JOB_ID is set).

Environment (set by cloud/job.py):
  JOB_ID, JOB_JSON (run_pipeline options), R2_ENDPOINT_URL, R2_BUCKET,
  R2_IN_*  temporary read-only credentials on jobs/<id>/input/,
  R2_OUT_* temporary read-write credentials on jobs/<id>/output/,
  POD_MAX_RUNTIME_S (watchdog; the credentials expire after the same time),
  optional R2_DEBUG_BUCKET + R2_DEBUG_* temporary credentials on jobs/<id>/ of the debug bucket
  (cloudflare_r2.debug_credentials_env): pipeline/debug_lite.py archive uploaded there as jobs/<id>/debug_lite.zip.

Streaming (JOB_JSON "stream": true, plans/PLAN_STREAMING_DAG.md §2): the pod is started when the scan starts. It lists
jobs/<id>/input/ every STREAM_POLL_S, downloads each new photo and pre-computes what depends on the photos alone
(pipeline/stream_prework.py) until the scanner writes input/capture_done.json ({"files": [...], ...}: every photo of
the scan), then runs the pipeline with those caches. input/level_done_<L>.json (optional) marks a level as uploaded;
input/cancel.json, or no new photo for STREAM_IDLE_S, ends the job without a model. Without "stream" the photos are
all there at the start, as before.

Writes to jobs/<id>/output/: progress.json (the last finished step, for the scanner), output.zip, stats.json,
run_log.txt and, LAST, status.json:
  {"state": "ok" | "ram_insufficient" | "pipeline_error" | "input_error" | "timeout" | "cancelled", ...}
  With "ok", "warnings" lists what the client should show ({"code", "message", ...}: model not watertight, object
  at the disc rim, masks disagree, ...): the model was delivered but may need a rescan.
The pod never stops itself (it has no RunPod key): the client terminates it after reading status.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from cloud import cloudflare_r2 as r2

logger = logging.getLogger("TurntableStudio.PodJob")

INPUT_DIR = Path(os.environ.get("POD_INPUT_DIR", "/workspace/input_images"))
OUTPUT_DIR = Path(os.environ.get("POD_OUTPUT_DIR", "/workspace/output"))
WORK_DIR = Path(os.environ.get("TURNTABLE_WORK_DIR", "/tmp/turntable_work"))
WATCHDOG_MARGIN_S = 180   # leave time to upload the log and status.json before the credentials expire
STREAM_POLL_S = float(os.environ.get("STREAM_POLL_S", "2"))
# Longer than the scanner's upload queue drain (UPLOAD_DRAIN_TIMEOUT_S 300 s) before it writes capture_done.json
STREAM_IDLE_S = float(os.environ.get("STREAM_IDLE_S", "420"))
CAPTURE_DONE, CANCEL = "capture_done.json", "cancel.json"
LEVEL_DONE_RX = re.compile(r"^level_done_(\d+)\.json$")
PHOTO_LEVEL_RX = re.compile(r"(\d)[_]?\d{1,4}$")      # f_1003 -> level 1 (TurntableConfig.name_regex)


class StreamCancelled(Exception):
    pass


class StreamInputError(Exception):
    pass


class ProgressWriter:
    """jobs/<id>/output/progress.json ({"step": ..., ...}) from a thread: the pipeline never waits for R2."""

    def __init__(self, client, bucket: str, job_id: str):
        self.client, self.bucket, self.key = client, bucket, r2.output_key(job_id, "progress.json")
        self._latest = None
        self._cv = threading.Condition()
        self._closed = False
        threading.Thread(target=self._run, daemon=True).start()

    def set(self, step: str, **extra) -> None:
        with self._cv:
            self._latest = dict(extra, step=step, at=time.time())
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._latest is None and not self._closed:
                    self._cv.wait()
                if self._latest is None:
                    return
                data, self._latest = self._latest, None
            try:
                r2.put_json(data, self.key, self.bucket, client=self.client)
            except Exception as e:
                logger.debug(f"progress.json not written: {e}")

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify()


def _download_atomic(key: str, target: Path, bucket: str, client) -> bool:
    """To <target>.part, then renamed: a half-written photo is never seen under its name."""
    part = target.with_name(target.name + ".part")
    ok = r2.download_file(key, str(part), bucket, client=client)
    if ok:
        os.replace(str(part), str(target))
    else:
        part.unlink(missing_ok=True)
    return ok


def stream_inputs(job_id: str, bucket: str, client, dest: Path, on_photo=None, on_level=None,
                  progress: Optional[ProgressWriter] = None, poll_s: Optional[float] = None,
                  idle_s: Optional[float] = None, max_wait_s: Optional[float] = None) -> Dict[str, Any]:
    """
    Downloads the photos of jobs/<id>/input/ as they appear, until capture_done.json lists them all. Calls
    on_photo(relative key, local path) for each new photo, on_level(level, files) for each level_done marker.
    Returns {"manifest", "photos", "first_photo_at", "last_photo_at", "capture_done_at"}.
    Raises StreamCancelled (cancel.json) or StreamInputError (idle too long, a listed photo missing).
    """
    poll_s = STREAM_POLL_S if poll_s is None else poll_s
    idle_s = STREAM_IDLE_S if idle_s is None else idle_s
    prefix = r2.input_prefix(job_id)
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    have: Dict[str, Path] = {}
    markers_seen: set = set()
    first_at = last_at = None
    last_new = t_start = time.time()
    pool = ThreadPoolExecutor(max_workers=8)

    def fetch(keys: List[str]) -> List[str]:
        jobs = []
        for key in keys:
            rel = key[len(prefix):]
            target = (dest / rel).resolve()
            if dest not in target.parents:
                continue                                  # a key like ../../x must not escape dest
            target.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((key, rel, target, pool.submit(_download_atomic, key, target, bucket, client)))
        got = []
        for key, rel, target, fut in jobs:
            if fut.result():
                have[key] = target
                got.append(rel)
                if on_photo is not None:
                    on_photo(rel, str(target))
        return got

    try:
        while True:
            keys = r2.list_keys(prefix=prefix, bucket=bucket, client=client)
            names = {k[len(prefix):]: k for k in keys}
            if CANCEL in names:
                raise StreamCancelled("cancel.json")
            images = sorted(k for k in keys if Path(k).suffix.lower() in r2.IMAGE_SUFFIXES and k not in have)
            got = fetch(images) if images else []
            now = time.time()
            if got:
                first_at = first_at or now
                last_at = last_new = now
                if progress is not None:
                    levels = [m.group(1) for m in (PHOTO_LEVEL_RX.search(Path(r).stem) for r in got) if m]
                    progress.set("0. Waiting for photos", photos=len(have), level=int(max(levels)) if levels else None)
            for rel, key in sorted(names.items()):
                m = LEVEL_DONE_RX.match(rel)
                if m and key not in markers_seen:
                    markers_seen.add(key)
                    marker = r2.get_json(key, bucket, client=client) or {}
                    if on_level is not None:
                        on_level(int(m.group(1)), [str(f) for f in marker.get("files") or []])
            if CAPTURE_DONE in names:
                manifest = r2.get_json(names[CAPTURE_DONE], bucket, client=client) or {}
                done_at = time.time()
                files = [str(f).lstrip("/") for f in manifest.get("files") or []]
                missing = [f for f in files if prefix + f not in have]
                if missing:
                    present = r2.list_keys(prefix=prefix, bucket=bucket, client=client)
                    gone = [f for f in missing if prefix + f not in present]
                    if gone:
                        raise StreamInputError(f"{len(gone)} photo(s) of capture_done.json not in R2: {gone[:5]}")
                    fetch([prefix + f for f in missing])
                    still = [f for f in missing if prefix + f not in have]
                    if still:
                        raise StreamInputError(f"{len(still)} photo(s) could not be downloaded: {still[:5]}")
                if files:
                    # exactly the photos of the manifest (what the scanner verified), nothing older or extra
                    wanted = {prefix + f for f in files}
                    for key in [k for k in have if k not in wanted]:
                        have.pop(key).unlink(missing_ok=True)
                if not have:
                    raise StreamInputError("capture_done.json but no photo")
                return {"manifest": manifest, "photos": len(have), "first_photo_at": first_at,
                        "last_photo_at": last_at, "capture_done_at": done_at}
            if now - last_new > idle_s:
                raise StreamInputError(f"capture abandoned: no new photo for {idle_s:.0f} s "
                                       f"({len(have)} received, no capture_done.json)")
            if max_wait_s is not None and now - t_start > max_wait_s:
                raise StreamInputError(f"no capture_done.json after {max_wait_s:.0f} s")
            time.sleep(poll_s)
    finally:
        pool.shutdown(wait=True)


def upload_outputs(out_client, bucket: str, job_id: str, status: dict, before_status=None) -> None:
    """
    output.zip, run_log.txt, stats.json, then status.json last: the completion signal.
    `before_status()`: runs before uploading stats.json and status.json (waits for debug archive upload and finalizes timings).
    """
    for name in ("output.zip", "run_log.txt"):
        path = OUTPUT_DIR / name
        if path.is_file():
            ok = r2.upload_file(str(path), bucket, r2.output_key(job_id, name), client=out_client)
            logger.info(f"Upload {name}: {'ok' if ok else 'FAILED'}")
            if not ok and name == "output.zip" and status.get("state") == "ok":
                status.update(state="pipeline_error", error="output.zip upload failed")
    if before_status is not None:
        before_status()
    stats_path = OUTPUT_DIR / "stats.json"
    if stats_path.is_file():
        ok = r2.upload_file(str(stats_path), bucket, r2.output_key(job_id, "stats.json"), client=out_client)
        logger.info(f"Upload stats.json: {'ok' if ok else 'FAILED'}")
    r2.put_json(status, r2.output_key(job_id, "status.json"), bucket, client=out_client)


DEBUG_PARTS_ZIP = WORK_DIR / "debug_lite_parts.zip"


def debug_upload_possible() -> bool:
    debug_bucket = os.environ.get("R2_DEBUG_BUCKET", "")
    return bool(debug_bucket and os.environ.get("R2_DEBUG_ACCESS_KEY_ID") and debug_bucket != r2.bucket_from_env())


def upload_debug_lite(job_id: str) -> dict:
    """
    pipeline/debug_lite.py archive -> jobs/<id>/debug_lite.zip of the DEBUG bucket, with the R2_DEBUG_* temporary
    credentials the launcher passes (cloudflare_r2.debug_credentials_env); skipped without them. Never fatal.
    The large parts were built by the pipeline during the run (DEBUG_PARTS_ZIP) when it got that far.
    """
    debug_bucket = os.environ.get("R2_DEBUG_BUCKET", "")
    if not debug_bucket or not os.environ.get("R2_DEBUG_ACCESS_KEY_ID"):
        return {"uploaded": False, "reason": "no debug bucket credentials"}
    if debug_bucket == r2.bucket_from_env():
        return {"uploaded": False, "reason": "debug bucket is the delivery bucket"}
    try:
        from pipeline.debug_lite import build_debug_lite

        info = build_debug_lite(str(OUTPUT_DIR), str(WORK_DIR), str(WORK_DIR / "debug_lite.zip"),
                                parts_zip=str(DEBUG_PARTS_ZIP) if DEBUG_PARTS_ZIP.is_file() else None)
        ok = r2.upload_file(info["path"], debug_bucket, r2.job_prefix(job_id) + "debug_lite.zip",
                            client=r2.client_from_env("R2_DEBUG_"))
        logger.info(f"Debug archive upload: {'ok' if ok else 'FAILED'} ({info['size_mb']} MB)")
        return {"uploaded": bool(ok), "size_mb": info["size_mb"], "parts": info["parts"]}
    except Exception as e:
        logger.warning(f"Debug archive not uploaded: {e}")
        return {"uploaded": False, "reason": str(e)[:200]}


STREAM_CACHE_DIR = WORK_DIR / "stream"


def _end_prework(prework) -> Optional[dict]:
    if prework is None:
        return None
    try:
        prework.capture_done()
        return prework.finish(timeout=30)
    except Exception as e:
        return {"error": str(e)[:200]}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    job_id = os.environ["JOB_ID"]
    bucket = r2.bucket_from_env()
    job = json.loads(os.environ.get("JOB_JSON") or "{}")
    max_runtime = float(os.environ.get("POD_MAX_RUNTIME_S", "3600"))
    in_client = r2.client_from_env("R2_IN_")
    out_client = r2.client_from_env("R2_OUT_")
    started = time.time()
    timings = {}

    # RunPod may restart the container after it exits: a finished job must not run (and bill) twice
    if r2.get_json(r2.output_key(job_id, "status.json"), bucket, client=out_client):
        logger.info(f"Job {job_id} already has a status.json: nothing to do.")
        return 0

    # First sign of life for the client: without it the pod is replaced after RUNPOD_BOOT_TIMEOUT_S
    r2.put_json({"job_id": job_id, "started_at": started}, r2.output_key(job_id, "started.json"), bucket,
                client=out_client)

    done = threading.Event()

    def watchdog():
        if not done.wait(max(60.0, max_runtime - WATCHDOG_MARGIN_S - (time.time() - started))):
            logger.error("Watchdog: runtime cap reached, reporting timeout.")
            upload_outputs(out_client, bucket, job_id, {"state": "timeout", "job_id": job_id, "started_at": started,
                                                         "elapsed_s": round(time.time() - started, 1)})
            os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()

    status = {"job_id": job_id, "state": "pipeline_error", "started_at": started}
    progress = ProgressWriter(out_client, bucket, job_id)
    stream = bool(job.get("stream"))
    prework = None
    stream_info: Dict[str, Any] = {}
    try:
        t0 = time.time()
        if stream:
            from pipeline.stream_prework import StreamPrework

            progress.set("0. Waiting for photos", photos=0, level=None)
            prework = StreamPrework(str(INPUT_DIR), str(STREAM_CACHE_DIR), job,
                                    workers=max(1, (os.cpu_count() or 2) - 1),
                                    level_masks=bool(job.get("stream_level_masks", True)))
            got = stream_inputs(job_id, bucket, in_client, INPUT_DIR, on_photo=prework.photo_arrived,
                                on_level=prework.level_marker, progress=progress)
            prework.capture_done()
            capture_done_at = got["capture_done_at"]
            time_until_capture_done_s = round(capture_done_at - started, 2)
            stream_info = {"photos": got["photos"], "pod_ready_before_first_photo_s":
                           round(got["first_photo_at"] - started, 2) if got["first_photo_at"] else None,
                           "capture_s": round(got["last_photo_at"] - got["first_photo_at"], 2)
                           if got["first_photo_at"] and got["last_photo_at"] else None,
                           "capture_done_after_last_photo_s":
                           round(capture_done_at - (got["last_photo_at"] or capture_done_at), 2),
                           "time_until_capture_done_s": time_until_capture_done_s}
            timings["stream_wait"] = round(time.time() - t0, 2)
            timings["time_until_capture_done_s"] = time_until_capture_done_s
            logger.info(f"Streaming: {got['photos']} photos, capture_done after {timings['stream_wait']} s "
                        f"(until capture_done: {time_until_capture_done_s} s)")
        else:
            keys = r2.download_job_images(job_id, str(INPUT_DIR), bucket, client=in_client)
            capture_done_at = None
            timings["download_input"] = round(time.time() - t0, 2)
            logger.info(f"Downloaded {len(keys)} photos in {timings['download_input']} s")
    except StreamCancelled:
        status.update(state="cancelled", error="cancelled by the scanner")
        _end_prework(prework)
        upload_outputs(out_client, bucket, job_id, status)
        progress.close()
        done.set()
        return 1
    except Exception as e:
        status.update(state="input_error", error=str(e))
        _end_prework(prework)
        upload_outputs(out_client, bucket, job_id, status)
        progress.close()
        done.set()
        return 1

    from main import run_pipeline  # imported late: the download needs only boto3
    from pipeline.resources import InsufficientRamError, describe_resources

    pipeline_job = dict(job)
    pipeline_job.setdefault("tag_layout", str(Path(__file__).resolve().parent / "pipeline/platan_345mm_layout.json"))
    pipeline_job.update(input_dir=str(INPUT_DIR), output_dir=str(OUTPUT_DIR), work_dir=str(WORK_DIR),
                        rig_mode=bool(job.get("rig_mode", (INPUT_DIR / "rig").is_dir())),
                        ram_margin=float(os.environ.get("RAM_MARGIN", job.get("ram_margin", 1.0))))
    if job.get("debug_lite", True) and debug_upload_possible():
        pipeline_job["debug_lite_parts"] = str(DEBUG_PARTS_ZIP)     # large parts built during the run (B2)
    pipeline_job["on_step_done"] = lambda name: progress.set(name)
    if stream:
        from pipeline import stream_cache

        stream_cache.stats(reset=True)
        pipeline_job.update(stream_cache=str(STREAM_CACHE_DIR), stream_info=stream_info,
                            capture_done_at=capture_done_at, started_at=started,
                            before_masks=lambda: prework.wait_levels(timeout=300))
    status["resources"] = describe_resources()
    try:
        t0 = time.time()
        result = run_pipeline(pipeline_job)
        timings["pipeline"] = round(time.time() - t0, 2)
        status.update(state="ok", zip_size_mb=result["bundle"].get("zip_size_mb"),
                      warnings=result.get("warnings") or [])
    except InsufficientRamError as e:
        status.update(state="ram_insufficient", error=str(e),
                      required_gb=round(e.required_gb, 2), available_gb=round(e.available_gb, 2))
    except Exception as e:
        status.update(state="pipeline_error", error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc()[-4000:])
    finally:
        # The debug archive goes up beside output.zip (B2), but before status.json: the client terminates the pod
        # as soon as it reads it
        debug_thread = None
        if job.get("debug_lite", True):
            def debug_upload() -> None:
                t1 = time.time()
                status["debug_lite"] = upload_debug_lite(job_id)
                timings["debug_lite"] = round(time.time() - t1, 2)

            debug_thread = threading.Thread(target=debug_upload, daemon=True)
            debug_thread.start()

        def before_status() -> None:
            if debug_thread is not None:
                debug_thread.join(timeout=600)
            now = time.time()
            total_time_s = round(now - started, 2)
            time_after_capture_done_s = round(now - capture_done_at, 2) if capture_done_at is not None else None
            time_until_capture_done_s = round(capture_done_at - started, 2) if capture_done_at is not None else None

            timings["total"] = total_time_s
            timings["total_time_s"] = total_time_s
            if capture_done_at is not None:
                # main indicators of the streaming mode
                timings["wait_after_capture_s"] = time_after_capture_done_s
                timings["time_after_capture_done_s"] = time_after_capture_done_s
                timings["time_until_capture_done_s"] = time_until_capture_done_s
            status["timings_s"] = timings

            stats_path = OUTPUT_DIR / "stats.json"
            if stats_path.is_file():
                try:
                    s_data = json.loads(stats_path.read_text(encoding="utf-8"))
                    s_data["total_time_s"] = total_time_s
                    if time_until_capture_done_s is not None:
                        s_data["time_until_capture_done_s"] = time_until_capture_done_s
                    if time_after_capture_done_s is not None:
                        s_data["time_after_capture_done_s"] = time_after_capture_done_s
                    if "stream" in s_data.get("run_info", {}):
                        if time_until_capture_done_s is not None:
                            s_data["run_info"]["stream"]["time_until_capture_done_s"] = time_until_capture_done_s
                        if time_after_capture_done_s is not None:
                            s_data["run_info"]["stream"]["time_after_capture_done_s"] = time_after_capture_done_s
                            s_data["run_info"]["stream"]["wait_after_capture_s"] = time_after_capture_done_s
                        s_data["run_info"]["stream"]["total_time_s"] = total_time_s
                    stats_path.write_text(json.dumps(s_data, indent=2), encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Could not update stats.json with final timings: {e}")

        status["timings_s"] = timings
        if prework is not None:
            status["stream"] = dict(stream_info, prework=_end_prework(prework))
        upload_outputs(out_client, bucket, job_id, status, before_status)
        progress.close()
        done.set()
    logger.info(f"Job {job_id} finished: {status['state']}")
    return 0 if status["state"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
