"""
job.py — The whole cloud flow for one scan, run from the Pi or the laptop:

  preflight R2 -> upload photos to jobs/<id>/input/ -> verify every photo is there
  -> RAM / disk estimate -> pod (fresh read-only / read-write R2 credentials limited to this job,
     expiring with the pod's runtime cap) -> wait for status.json -> terminate the pod
  -> download + extract output.zip into results/<id>/ -> delete jobs/<id>/ from R2.

The R2 job folder is deleted only after output.zip was downloaded AND extracted; after a failure the photos
stay in R2, so the job can be retried with skip_upload=True.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from botocore.exceptions import BotoCoreError, ClientError

from cloud import cloudflare_r2 as r2
from cloud import runpod as rp
from pipeline.resources import estimate_ram_gb

logger = logging.getLogger("TurntableStudio.CloudJob")

DEFAULT_IMAGE_ENV = "TURNTABLE_IMAGE"
RAM_MARGIN = 1.3          # on the estimate until more runs refit the model (tools/fit_ram_model.py)
DEFAULT_MIN_VCPU = 8


def _image_size(path: str) -> tuple[int, int]:
    """(width, height) of one photo, read from its header only."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def _merge_cloud_stats(stats_path: Path, report: Dict[str, Any], runs) -> None:
    """
    Adds what only the client knows to the job's stats.json (training data for RAM / time / cost models):
    the pod actually rented, its price, the time from pod creation to the pipeline start (image pull + boot),
    upload / download times and the requested RAM.
    """
    if not stats_path.is_file() or not runs:
        return
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        last = runs[-1]
        pod_started = (report.get("status") or {}).get("started_at")
        pod_timings = report.get("pod_timings_s") or {}
        stats["cloud"] = {
            "job_id": report["job_id"],
            "plan": report.get("plan"),
            "pods": report.get("pods"),
            "pod_attempts": len(runs),
            "cost_estimate": report.get("cost_estimate"),
            "pull_and_boot_s": round(float(pod_started) - last.started_at, 1) if pod_started else None,
            "client_timings_s": report.get("timings_s"),
            "pod_timings_s": pod_timings,
        }
        t_until = pod_timings.get("time_until_capture_done_s") or pod_timings.get("stream_wait")
        t_after = pod_timings.get("time_after_capture_done_s") or pod_timings.get("wait_after_capture_s")
        t_total = pod_timings.get("total_time_s") or pod_timings.get("total")
        if t_until is not None:
            stats["time_until_capture_done_s"] = t_until
        if t_after is not None:
            stats["time_after_capture_done_s"] = t_after
        if t_total is not None:
            stats["total_time_s"] = t_total
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    except (OSError, ValueError, KeyError) as e:
        logger.warning(f"Could not add the cloud data to {stats_path}: {e}")


def cleanup_prefix(job_id: str, delete_stale_after_s: Optional[float]) -> str:
    """
    What a finished job deletes from R2. When the next job clears the old ones at its start (delete_stale_after_s
    set), only the photos: the result (output.zip, stats.json, run_log.txt) stays in R2 until then, e.g. when this
    machine slept while the pod finished. Without that cleanup nothing else would delete it: the whole job goes.
    """
    return r2.input_prefix(job_id) if delete_stale_after_s is not None else r2.job_prefix(job_id)


# Streaming
STREAM_PHOTO_SIZE = (4056, 3040)
STREAM_PHOTO_MB = 2.0
CAPTURE_DONE_NAME, CANCEL_NAME = "capture_done.json", "cancel.json"
PHOTO_LEVEL_RX = re.compile(r"(\d)[_]?\d{1,4}$")


def stream_n_images(step_deg: float, max_levels: int, cameras: int = 2) -> int:
    """Most photos a scan can take: cameras x shots per ring x rings."""
    return int(cameras) * int(round(360.0 / float(step_deg))) * max(1, int(max_levels))


def plan_stream_job(n_images: int, resolution_level: int = 1, min_vcpu: int = DEFAULT_MIN_VCPU,
                    width: int = STREAM_PHOTO_SIZE[0], height: int = STREAM_PHOTO_SIZE[1]) -> Dict[str, Any]:
    """plan_job for a scan not taken yet: the same RAM / disk formulas on the largest photo count it can reach."""
    input_gb = n_images * STREAM_PHOTO_MB / 1024
    estimate = estimate_ram_gb(n_images, width, height, resolution_level)
    return {
        "n_images": int(n_images),
        "width": width,
        "height": height,
        "input_gb": round(input_gb, 3),
        "resolution_level": resolution_level,
        "ram_estimate_gb": round(estimate, 2),
        "min_ram_gb": int(math.ceil(estimate * RAM_MARGIN)),
        "min_vcpu": int(min_vcpu),
        "disk_gb": rp.disk_gb_for(n_images, input_gb, resolution_level),
        "files": {},
        "stream": True,
    }


def write_capture_done(job_id: str, keys, step_deg: Optional[float] = None, bucket: Optional[str] = None,
                       client=None) -> Dict[str, Any]:
    """
    jobs/<id>/input/capture_done.json: the end of a streamed scan, with every photo the scanner verified in R2.
    """
    prefix = r2.input_prefix(job_id)
    files = sorted(k[len(prefix):] if k.startswith(prefix) else k.lstrip("/") for k in keys)
    levels = sorted({int(m.group(1)) for m in (PHOTO_LEVEL_RX.search(Path(f).stem) for f in files) if m})
    body = {"n_images": len(files), "levels": levels, "files": files, "step_deg": step_deg,
            "finished_at": time.time()}
    if not r2.put_json(body, r2.input_key(job_id, CAPTURE_DONE_NAME), bucket, client=client):
        raise r2.R2Error(f"Could not write {CAPTURE_DONE_NAME} for job {job_id}")
    return body


def write_cancel(job_id: str, bucket: Optional[str] = None, client=None) -> bool:
    """jobs/<id>/input/cancel.json: a streaming pod stops at once (besides being terminated). Never raises."""
    try:
        return bool(r2.put_json({"cancelled_at": time.time()}, r2.input_key(job_id, CANCEL_NAME), bucket,
                                client=client))
    except Exception as e:
        logger.warning(f"Could not write {CANCEL_NAME} for job {job_id}: {e}")
        return False


def _photo_sort_key(rel: str) -> tuple:
    m = re.search(r"(\d)[_]?(\d{1,4})$", Path(rel).stem)
    if m:
        return (int(m.group(1)), int(m.group(2)), rel)
    return (0, 0, rel)


def stream_upload_photos(
    job_id: str,
    files: Dict[str, str],
    bucket: str,
    client=None,
    step_deg: float = 18.0,
    delay_s: float = 0.0,
    cancel_event=None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Uploads photos one by one to R2 jobs/<id>/input/ in level & angle order.
    Emits level_done_<L>.json after each level, verifies all uploads, and writes capture_done.json.
    """
    say = on_progress or logger.info
    cl = r2._client(client)
    sorted_rels = sorted(files.keys(), key=_photo_sort_key)
    levels: Dict[int, List[str]] = {}
    uploaded_keys: List[str] = []
    total = len(sorted_rels)

    for i, rel in enumerate(sorted_rels, 1):
        if cancel_event is not None and cancel_event.is_set():
            say("Streaming upload cancelled.")
            write_cancel(job_id, bucket=bucket, client=cl)
            return {"uploaded": uploaded_keys, "cancelled": True}

        abs_path = files[rel]
        key = r2.input_key(job_id, rel)
        ok = r2.upload_file(abs_path, bucket, key, client=cl)
        if not ok:
            say(f"Warning: upload failed for {rel}, retrying...")
            ok = r2.upload_file(abs_path, bucket, key, client=cl)
            if not ok:
                write_cancel(job_id, bucket=bucket, client=cl)
                raise r2.R2Error(f"Failed to upload {rel} to R2")
        uploaded_keys.append(key)

        m = PHOTO_LEVEL_RX.search(Path(rel).stem)
        lvl = int(m.group(1)) if m else 1
        levels.setdefault(lvl, []).append(rel)

        say(f"Streamed photo {i}/{total}: {rel}")

        # Check if this level is complete (either next photo has different level, or this is the last photo)
        is_level_end = (i == total) or (_photo_sort_key(sorted_rels[i])[0] != lvl)
        if is_level_end:
            marker_key = r2.input_key(job_id, f"level_done_{lvl}.json")
            r2.put_json({"level": lvl, "files": levels[lvl]}, marker_key, bucket, client=cl)
            say(f"Level {lvl} finished ({len(levels[lvl])} photos): level_done_{lvl}.json written")

        if delay_s > 0:
            if cancel_event is not None:
                if cancel_event.wait(delay_s):
                    write_cancel(job_id, bucket=bucket, client=cl)
                    return {"uploaded": uploaded_keys, "cancelled": True}
            else:
                time.sleep(delay_s)

    # All photos uploaded, verify
    missing = r2.verify_keys(uploaded_keys, bucket, prefix=r2.input_prefix(job_id), client=cl)
    if missing:
        write_cancel(job_id, bucket=bucket, client=cl)
        raise r2.R2Error(f"{len(missing)} photo(s) missing in R2 after streaming upload: {missing[:5]}")

    manifest = write_capture_done(job_id, uploaded_keys, step_deg=step_deg, bucket=bucket, client=cl)
    say(f"All {total} photos verified in R2: capture_done.json written.")
    return {"uploaded": uploaded_keys, "manifest": manifest, "cancelled": False}


def plan_job(input_dir: str, resolution_level: int = 1, min_vcpu: int = DEFAULT_MIN_VCPU) -> Dict[str, Any]:
    """Photo count, sizes and the pod requirements derived from them (§2 of the performance plan)."""
    files = r2.local_image_files(input_dir)
    if not files:
        raise r2.R2Error(f"No images found under {input_dir}")
    first = next(iter(files.values()))
    width, height = _image_size(first)
    input_gb = sum(os.path.getsize(p) for p in files.values()) / 1024 ** 3
    estimate = estimate_ram_gb(len(files), width, height, resolution_level)
    return {
        "n_images": len(files),
        "width": width,
        "height": height,
        "input_gb": round(input_gb, 3),
        "resolution_level": resolution_level,
        "ram_estimate_gb": round(estimate, 2),
        "min_ram_gb": int(math.ceil(estimate * RAM_MARGIN)),
        "min_vcpu": int(min_vcpu),
        "disk_gb": rp.disk_gb_for(len(files), input_gb, resolution_level),
        "files": files,
    }


def run_cloud_job(
    input_dir: str,
    results_dir: str = "results",
    job_id: Optional[str] = None,
    pipeline_options: Optional[Dict[str, Any]] = None,
    image: Optional[str] = None,
    dry_run: bool = False,
    skip_upload: bool = False,
    keep_remote: bool = False,
    min_vcpu: int = DEFAULT_MIN_VCPU,
    delete_stale_after_s: Optional[float] = None,
    cancel_event=None,
    on_progress: Optional[Callable[[str], None]] = None,
    stream: bool = False,
    stream_delay_s: float = 0.0,
    stream_n_photos: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs one scan in the cloud. pipeline_options go to run_pipeline() in the pod (e.g. {"rig_mode": True,
    "estimate_lift": True}). Returns a report with timings, the pod used, the cost estimate and the result folder.
    delete_stale_after_s: first delete the jobs/<id>/ folders left in R2 by older jobs untouched for that long
    (r2.delete_stale_jobs; 0 = every older job, None = keep them). This job and anything outside jobs/ (e.g. the
    scanner's own folder) are never touched.
    stream: start the pod now, while photos are being taken or streamed; photos are uploaded concurrently.
    """
    say = on_progress or logger.info
    options = dict(pipeline_options or {})
    image = image or os.environ.get(DEFAULT_IMAGE_ENV)
    bucket = r2.bucket_from_env()
    job_id = job_id or r2.new_job_id()
    report: Dict[str, Any] = {"job_id": job_id, "bucket": bucket, "timings_s": {}, "dry_run": dry_run}
    t = report["timings_s"]

    if stream:
        options["stream"] = True
        options.setdefault("stream_level_masks", True)

    files = {}
    has_local_dir = Path(input_dir).is_dir()
    if stream and not has_local_dir:
        n_photos = int(stream_n_photos or stream_n_images(float(options.get("step_deg", 18.0)), 4))
        plan = plan_stream_job(n_photos, int(options.get("resolution_level", 1)), min_vcpu)
        skip_upload = True
    else:
        plan = plan_job(input_dir, int(options.get("resolution_level", 1)), min_vcpu)
        files = plan.pop("files")
        if stream:
            plan["stream"] = True

    report["plan"] = plan
    say(f"Job {job_id}: {plan['n_images']} photos ({plan['input_gb']:.2f} GB, {plan['width']}x{plan['height']}), "
        f"RAM estimate {plan['ram_estimate_gb']} GB -> min {plan['min_ram_gb']} GB, "
        f"{plan['min_vcpu']} vCPU, disk {plan['disk_gb']} GB" + (" [STREAMING]" if stream else ""))

    client = r2.client_from_env()
    t0 = time.time()
    r2.preflight(bucket, client=client, prefix=r2.job_prefix(job_id) + "_preflight/")
    t["preflight"] = round(time.time() - t0, 2)
    say(f"R2 preflight ok ({t['preflight']} s)")

    if delete_stale_after_s is not None and not dry_run:
        try:
            stale = r2.delete_stale_jobs(delete_stale_after_s, bucket, client=client, keep=[job_id])
            report["deleted_stale_jobs"] = stale
            if stale:
                say(f"Deleted {len(stale)} old job folder(s) from R2: {', '.join(stale)}")
        except r2.R2Error as e:
            say(f"Could not clean old jobs from R2 (continuing): {e}")

    expected = [r2.input_key(job_id, rel) for rel in files]
    uploader_thread = None
    uploader_box: Dict[str, Any] = {}

    if not skip_upload and not dry_run:
        if stream:
            def _stream_worker():
                try:
                    res = stream_upload_photos(
                        job_id, files, bucket, client=client,
                        step_deg=float(options.get("step_deg", 18.0)),
                        delay_s=stream_delay_s,
                        cancel_event=cancel_event,
                        on_progress=say,
                    )
                    uploader_box["res"] = res
                except Exception as e:
                    uploader_box["error"] = e
                    say(f"Streaming upload failed: {e}")
                    if cancel_event is not None:
                        cancel_event.set()

            uploader_thread = threading.Thread(target=_stream_worker, name="stream-upload", daemon=True)
            uploader_thread.start()
            say("Streaming uploader started in background.")
        else:
            t0 = time.time()
            res = r2.upload_job_images(job_id, input_dir, bucket, client=client)
            t["upload"] = round(time.time() - t0, 2)
            mb = plan["input_gb"] * 1024
            say(f"Uploaded {len(res['uploaded'])} photos ({mb:.0f} MB) in {t['upload']} s "
                f"({mb / max(t['upload'], 1e-3):.1f} MB/s), failed: {len(res['failed'])}")
            missing = r2.verify_keys(expected, bucket, prefix=r2.input_prefix(job_id), client=client)
            if missing:
                raise r2.R2Error(f"{len(missing)} photo(s) missing in R2, not starting a pod: {missing[:5]}")
            say("All photos are in R2.")

    base_ttl = int(options.get("pod_max_runtime_s") or rp.POD_MAX_RUNTIME_S)
    ttl = max(base_ttl, 3600 if stream else 1800)
    job_spec = dict(options, job_id=job_id)
    job_spec.setdefault("tag_layout", "pipeline/platan_345mm_layout.json")

    def make_env() -> Dict[str, str]:
        env = r2.pod_credentials_env(job_id, ttl) if not dry_run else {
            "R2_ENDPOINT_URL": os.environ.get("R2_ENDPOINT_URL", ""), "R2_BUCKET": bucket,
            "R2_IN_ACCESS_KEY_ID": "<temp>", "R2_IN_SECRET_ACCESS_KEY": "<temp>", "R2_IN_SESSION_TOKEN": "<temp>",
            "R2_OUT_ACCESS_KEY_ID": "<temp>", "R2_OUT_SECRET_ACCESS_KEY": "<temp>", "R2_OUT_SESSION_TOKEN": "<temp>",
        }
        env.update({
            "JOB_ID": job_id,
            "JOB_JSON": json.dumps(job_spec),
            "TURNTABLE_IMAGE": image or "",
            "POD_MAX_RUNTIME_S": str(ttl),
            "RAM_MARGIN": "1.0",
        })
        return env

    def make_body(min_ram_gb: int) -> Dict[str, Any]:
        if not image:
            raise rp.RunPodError(f"No Docker image: set {DEFAULT_IMAGE_ENV} (e.g. ghcr.io/<user>/turntable-studio:v1).")
        return rp.build_pod_request(
            image=image, env=make_env(), min_ram_gb=min_ram_gb, min_vcpu=plan["min_vcpu"],
            disk_gb=plan["disk_gb"], gpu_types=rp.gpu_types_from_env(), name=f"turntable-{job_id}",
            cloud_type=os.environ.get("RUNPOD_CLOUD_TYPE", "SECURE"),
            registry_auth_id=os.environ.get("RUNPOD_REGISTRY_AUTH_ID") or None,
        )

    if dry_run:
        report["pod_request"] = rp.redact(make_body(plan["min_ram_gb"]))
        say("Dry run: pod request that would be sent:\n" + json.dumps(report["pod_request"], indent=2))
        return report

    status_key = r2.output_key(job_id, "status.json")
    started_key = r2.output_key(job_id, "started.json")
    progress_key = r2.output_key(job_id, "progress.json")
    last_progress = [None]

    def read_status():
        prog = r2.get_json(progress_key, bucket, client=client)
        if prog and prog.get("step") != last_progress[0]:
            last_progress[0] = prog["step"]
            extra = []
            if "photos" in prog:
                extra.append(f"{prog['photos']} photos")
            if "level" in prog and prog["level"] is not None:
                extra.append(f"level {prog['level']}")
            extra_str = f" ({', '.join(extra)})" if extra else ""
            say(f"Pod progress: {last_progress[0]}{extra_str}")
        return r2.get_json(status_key, bucket, client=client)

    def read_started() -> bool:
        return r2.get_json(started_key, bucket, client=client) is not None

    def clear_status():
        for key in (status_key, started_key, progress_key):
            try:
                client.delete_object(Bucket=bucket, Key=key)
            except (ClientError, BotoCoreError) as e:
                say(f"Could not remove the previous {key.rsplit('/', 1)[-1]}: {e}")

    clear_status()  # a status.json left by an earlier attempt of this job must not end the new pod at once
    t0 = time.time()
    runs = rp.run_job_pods(rp.RunPodClient(), make_body, plan["min_ram_gb"], read_status,
                           cancel_event=cancel_event, on_progress=say, before_retry=clear_status,
                           read_started=read_started)
    t["pods"] = round(time.time() - t0, 2)
    if uploader_thread is not None:
        uploader_thread.join(timeout=30)
        if "error" in uploader_box:
            say(f"Streaming uploader note: {uploader_box['error']}")
    report["pods"] = [{
        "pod_id": r.pod_id, "outcome": r.outcome, "terminated": r.terminated, "elapsed_s": round(r.elapsed_s, 1),
        "cost_estimate": r.cost_estimate, **r.pod_info,
    } for r in runs]
    report["cost_estimate"] = round(sum(r.cost_estimate or 0.0 for r in runs), 4)
    last = runs[-1] if runs else None
    status = (last.status if last else None) or {}
    report["status"] = status
    if status.get("timings_s"):
        report["pod_timings_s"] = status["timings_s"]

    if status.get("state") != "ok":
        report["result"] = "failed"
        say(f"Job failed: {status.get('state') or (last.outcome if last else 'no pod')} {status.get('error', '')}")
        # Keep the photos in R2 for a retry; fetch the log if the pod managed to upload it
        r2.download_file(r2.output_key(job_id, "run_log.txt"),
                         str(Path(results_dir) / job_id / "run_log.txt"), bucket, client=client)
        return report

    t0 = time.time()
    dest = r2.download_job_output(job_id, str(Path(results_dir) / job_id), bucket, client=client)
    t["download"] = round(time.time() - t0, 2)
    report["result"] = "ok"
    report["results_dir"] = str(dest)
    say(f"Result extracted to {dest} ({t['download']} s)")

    _merge_cloud_stats(dest / "stats.json", report, runs)

    if not keep_remote:
        t0 = time.time()
        prefix = cleanup_prefix(job_id, delete_stale_after_s)
        report["deleted_objects"] = r2.delete_prefix(prefix, bucket, client=client)
        t["cleanup"] = round(time.time() - t0, 2)
        kept = ("" if prefix == r2.job_prefix(job_id)
                else f" Result kept in {r2.output_prefix(job_id)} until the next job.")
        say(f"Deleted {report['deleted_objects']} objects of job {job_id} from R2.{kept}")
    (dest / "cloud_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
