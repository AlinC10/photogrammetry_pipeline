"""
runpod_job.py — Entry point inside the RunPod pod (docker_entrypoint.sh starts it when JOB_ID is set).

Environment (set by cloud/job.py):
  JOB_ID, JOB_JSON (run_pipeline options), R2_ENDPOINT_URL, R2_BUCKET,
  R2_IN_*  temporary read-only credentials on jobs/<id>/input/,
  R2_OUT_* temporary read-write credentials on jobs/<id>/output/,
  POD_MAX_RUNTIME_S (watchdog; the credentials expire after the same time),
  optional R2_DEBUG_BUCKET + R2_DEBUG_* temporary credentials on jobs/<id>/ of the debug bucket
  (cloudflare_r2.debug_credentials_env): pipeline/debug_lite.py archive uploaded there as jobs/<id>/debug_lite.zip.

Writes to jobs/<id>/output/: output.zip, stats.json, run_log.txt and, LAST, status.json:
  {"state": "ok" | "ram_insufficient" | "pipeline_error" | "input_error" | "timeout", ...}
  With "ok", "warnings" lists what the client should show ({"code", "message", ...}: model not watertight, object
  at the disc rim, masks disagree, ...): the model was delivered but may need a rescan.
The pod never stops itself (it has no RunPod key): the client terminates it after reading status.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from cloud import cloudflare_r2 as r2

logger = logging.getLogger("TurntableStudio.PodJob")

INPUT_DIR = Path(os.environ.get("POD_INPUT_DIR", "/workspace/input_images"))
OUTPUT_DIR = Path(os.environ.get("POD_OUTPUT_DIR", "/workspace/output"))
WORK_DIR = Path(os.environ.get("TURNTABLE_WORK_DIR", "/tmp/turntable_work"))
WATCHDOG_MARGIN_S = 180   # leave time to upload the log and status.json before the credentials expire


def upload_outputs(out_client, bucket: str, job_id: str, status: dict) -> None:
    """output.zip, stats.json, run_log.txt (whatever exists), then status.json last: the completion signal."""
    for name in ("output.zip", "stats.json", "run_log.txt"):
        path = OUTPUT_DIR / name
        if path.is_file():
            ok = r2.upload_file(str(path), bucket, r2.output_key(job_id, name), client=out_client)
            logger.info(f"Upload {name}: {'ok' if ok else 'FAILED'}")
            if not ok and name == "output.zip" and status.get("state") == "ok":
                status.update(state="pipeline_error", error="output.zip upload failed")
    r2.put_json(status, r2.output_key(job_id, "status.json"), bucket, client=out_client)


def upload_debug_lite(job_id: str) -> dict:
    """
    pipeline/debug_lite.py archive -> jobs/<id>/debug_lite.zip of the DEBUG bucket, with the R2_DEBUG_* temporary
    credentials the launcher passes (cloudflare_r2.debug_credentials_env); skipped without them. Never fatal.
    """
    debug_bucket = os.environ.get("R2_DEBUG_BUCKET", "")
    if not debug_bucket or not os.environ.get("R2_DEBUG_ACCESS_KEY_ID"):
        return {"uploaded": False, "reason": "no debug bucket credentials"}
    if debug_bucket == r2.bucket_from_env():
        return {"uploaded": False, "reason": "debug bucket is the delivery bucket"}
    try:
        from pipeline.debug_lite import build_debug_lite

        info = build_debug_lite(str(OUTPUT_DIR), str(WORK_DIR), str(WORK_DIR / "debug_lite.zip"))
        ok = r2.upload_file(info["path"], debug_bucket, r2.job_prefix(job_id) + "debug_lite.zip",
                            client=r2.client_from_env("R2_DEBUG_"))
        logger.info(f"Debug archive upload: {'ok' if ok else 'FAILED'} ({info['size_mb']} MB)")
        return {"uploaded": bool(ok), "size_mb": info["size_mb"], "parts": info["parts"]}
    except Exception as e:
        logger.warning(f"Debug archive not uploaded: {e}")
        return {"uploaded": False, "reason": str(e)[:200]}


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
    try:
        t0 = time.time()
        keys = r2.download_job_images(job_id, str(INPUT_DIR), bucket, client=in_client)
        timings["download_input"] = round(time.time() - t0, 2)
        logger.info(f"Downloaded {len(keys)} photos in {timings['download_input']} s")
    except Exception as e:
        status.update(state="input_error", error=str(e))
        upload_outputs(out_client, bucket, job_id, status)
        done.set()
        return 1

    from main import run_pipeline  # imported late: the download needs only boto3
    from pipeline.resources import InsufficientRamError, describe_resources

    pipeline_job = dict(job)
    pipeline_job.update(input_dir=str(INPUT_DIR), output_dir=str(OUTPUT_DIR), work_dir=str(WORK_DIR),
                        rig_mode=bool(job.get("rig_mode", (INPUT_DIR / "rig").is_dir())),
                        ram_margin=float(os.environ.get("RAM_MARGIN", job.get("ram_margin", 1.0))))
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
        # Before status.json: the client terminates the pod as soon as it reads it
        if job.get("debug_lite", True):
            t0 = time.time()
            status["debug_lite"] = upload_debug_lite(job_id)
            timings["debug_lite"] = round(time.time() - t0, 2)
        timings["total"] = round(time.time() - started, 2)
        status["timings_s"] = timings
        upload_outputs(out_client, bucket, job_id, status)
        done.set()
    logger.info(f"Job {job_id} finished: {status['state']}")
    return 0 if status["state"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
