#!/usr/bin/env python3
"""
cloud_scan.py — One scan in the cloud, from this machine:
upload the photos to R2 -> pod on RunPod (pipeline) -> output.zip to R2 -> download + extract into
results/<name>-<job_id>/ -> delete the job from R2. Not part of the Docker image (tools/ is in .dockerignore).

  python tools/cloud_scan.py --dry_run                       # R2 check + RAM/disk estimate + the pod request, no cost
  python tools/cloud_scan.py --mock --gpu "NVIDIA RTX A4000"  # the real pod + R2 round trip, simulated OpenMVS (~minutes)
  python tools/cloud_scan.py --name fixativ                  # input_images/ -> results/fixativ-<job_id>/
  python tools/cloud_scan.py D:\\scans\\multimetru --name multimetru --set tag_scale=1.002 --set resolution_level=0
  python tools/cloud_scan.py --keep_remote                   # leave jobs/<id>/ in R2 (debugging)
  python tools/cloud_scan.py --job_id <id> --skip_upload     # retry a job whose photos are already in R2

Ctrl+C at any time stops the wait and terminates the pod (photos stay in R2 for a retry).
--set takes any run_pipeline() option (see PIPELINE_OPTIONS: lift_mm, tag_scale, data_term ...); values are read
as JSON when possible (1, 0.5, true), otherwise as text. main.py's negative flags work too: --set no_tag_refine
is tag_refine=false. Unknown keys are refused (a typo would otherwise be silently ignored by the pod).

Needs a .env in the repository root (see .env.example): R2_*, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_R2_API_TOKEN,
RUNPOD_API_KEY, TURNTABLE_IMAGE.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from cloud import cloudflare_r2 as r2  # noqa: E402
from cloud import runpod as rp  # noqa: E402
from cloud.job import plan_job, run_cloud_job  # noqa: E402

# Set by the pod / this script, not by --set
RESERVED_OPTIONS = {"job_id", "input_dir", "output_dir", "work_dir", "mock_openmvs", "openmvs_bin_dir", "texrecon_bin_dir",
                    "texrecon_bin", "intrinsics_in"}
# Keys of the job dict main.py builds for run_pipeline()
PIPELINE_OPTIONS = {
    "rig_mode", "resolution_level", "focal_length_mm", "step_deg", "resume", "cost_smoothness_ratio",
    "outlier_threshold", "empty_color", "global_seam_leveling", "local_seam_leveling", "target_low_poly_faces",
    "export_scale", "scale_stl_to_mm", "texturing_engine", "data_term", "smoothness_term", "outlier_removal",
    "texrecon_global_seam_leveling", "keep_unseen_faces", "write_intermediate_results", "mesh_texture_masks",
    "texture_mask_margin_px", "colour_normalize", "lift_mm", "estimate_lift", "tag_refine", "tag_scale", "workers",
    "keep_depth_maps", "ram_margin", "min_point_distance", "target_high_poly_faces", "visual_hull",
    "visual_hull_refine", "visual_hull_voxel_mm", "densify_masks",
}


def parse_set(items) -> dict:
    """["tag_scale=1.002", "estimate_lift=false", "data_term=area"] -> {"tag_scale": 1.002, ...}."""
    out = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        key = key.strip().lstrip("-")
        if key.startswith("no_") and key[3:] in PIPELINE_OPTIONS and (not sep or raw.strip().lower() in ("", "true", "1")):
            out[key[3:]] = False    # main.py's --no_tag_refine, --no_colour_normalize ...
            continue
        if not sep or not key:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        if key in RESERVED_OPTIONS:
            raise SystemExit(f"--set {key} is set by the cloud flow itself (paths on this machine do not exist in the pod)")
        if key not in PIPELINE_OPTIONS:
            raise SystemExit(f"--set {key}: unknown pipeline option. Known: {', '.join(sorted(PIPELINE_OPTIONS))}")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def run_interruptible(**kwargs) -> dict:
    """
    run_cloud_job in a worker thread: Ctrl+C in the main thread sets the cancel event, so the pod is
    terminated by run_pod's finally (a blocking wait in the main thread is not always interruptible on Windows).
    """
    cancel = threading.Event()
    box: dict = {}

    def work():
        try:
            box["report"] = run_cloud_job(cancel_event=cancel, **kwargs)
        except BaseException as e:  # reported in the main thread
            box["error"] = e

    worker = threading.Thread(target=work, name="cloud-job", daemon=True)
    worker.start()
    while worker.is_alive():
        try:
            worker.join(0.5)
        except KeyboardInterrupt:
            if cancel.is_set():
                print("\nStill terminating the pod, please wait (closing now could leave it running and billing).")
            else:
                print("\nCancelling: terminating the pod...")
                cancel.set()
    if "error" in box:
        raise box["error"]
    return box["report"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", nargs="?", default=str(ROOT / "input_images"),
                    help="photos (flat, or rig/0 + rig/1); default: input_images/")
    ap.add_argument("--name", default=None, help="label of the scan: job id and result folder start with it")
    ap.add_argument("--results_dir", default=str(ROOT / "results"))
    ap.add_argument("--job_id", default=None, help="reuse a job id (with --skip_upload: photos already in R2)")
    ap.add_argument("--image", default=None, help="Docker image (default: TURNTABLE_IMAGE from .env)")
    ap.add_argument("--dry_run", action="store_true", help="no upload, no pod: print the plan and the pod request")
    ap.add_argument("--skip_upload", action="store_true")
    ap.add_argument("--keep_remote", action="store_true")
    ap.add_argument("--mock", action="store_true",
                    help="pod runs the pipeline with simulated OpenMVS/texrecon: tests image, keys, R2 and termination")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask before starting the pod")
    ap.add_argument("--gpu", action="append", metavar="GPU_ID",
                    help='only these cards, e.g. --gpu "NVIDIA GeForce RTX 4090" (repeatable; default: RUNPOD_GPU_TYPES '
                         'or the list in cloud/runpod.py)')
    ap.add_argument("--min_vcpu", type=int, default=8)
    ap.add_argument("--resolution_level", type=int, default=1)
    ap.add_argument("--no_estimate_lift", action="store_true", help="use --lift_mm instead of estimating the lift")
    ap.add_argument("--lift_mm", type=float, default=100.0)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="any run_pipeline option (repeatable)")
    args = ap.parse_args(argv)

    if args.gpu:
        os.environ["RUNPOD_GPU_TYPES"] = ",".join(args.gpu)   # read by cloud.runpod.gpu_types_from_env
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    options = {
        "rig_mode": (Path(args.input_dir) / "rig").is_dir(),
        "resolution_level": args.resolution_level,
        "estimate_lift": not args.no_estimate_lift,
        "lift_mm": args.lift_mm,
        "texturing_engine": "texrecon",
    }
    options.update(parse_set(args.set))
    if args.mock:
        options["mock_openmvs"] = True

    job_id = args.job_id
    if not job_id and args.name:
        label = re.sub(r"[^A-Za-z0-9_-]+", "-", args.name).strip("-")
        job_id = f"{label}-{r2.new_job_id()}" if label else None

    if not args.dry_run and not args.yes:
        plan = plan_job(args.input_dir, int(options.get("resolution_level", 1)), args.min_vcpu)
        print(f"\n{plan['n_images']} photos ({plan['input_gb']:.2f} GB) from {args.input_dir}")
        print(f"GPU: {', '.join(rp.gpu_types_from_env())}")
        print(f"Pod: min {plan['min_ram_gb']} GB RAM, {plan['min_vcpu']} vCPU, {plan['disk_gb']} GB disk")
        print(f"Options: {json.dumps(options)}")
        if input("Start the pod (billed)? [y/N] ").strip().lower() not in ("y", "yes", "d", "da"):
            print("Nothing started.")
            return 1

    t0 = time.time()
    report = run_interruptible(
        input_dir=args.input_dir, results_dir=args.results_dir, job_id=job_id, pipeline_options=options,
        image=args.image, dry_run=args.dry_run, skip_upload=args.skip_upload, keep_remote=args.keep_remote,
        min_vcpu=args.min_vcpu,
    )
    report["timings_s"]["total"] = round(time.time() - t0, 2)

    print("\n" + "=" * 70)
    print(f" Job {report['job_id']}: {report.get('result', 'dry run' if args.dry_run else '?')}")
    for name, sec in report["timings_s"].items():
        print(f"   {name:<12} {sec:>8.1f} s")
    for pod in report.get("pods", []):
        print(f"   pod {pod['pod_id']}: {pod.get('gpu')} | {pod.get('memoryInGb')} GB RAM | {pod.get('vcpuCount')} vCPU"
              f" | {pod['outcome']} | {pod['elapsed_s'] / 60:.1f} min | ~${pod.get('cost_estimate') or 0:.3f}")
        if not pod.get("terminated"):
            print(f"   !!! pod {pod['pod_id']} could NOT be confirmed terminated: check https://console.runpod.io/pods")
    if report.get("pod_timings_s"):
        print(f"   in the pod: {json.dumps(report['pod_timings_s'])}")
    if report.get("results_dir"):
        dest = Path(report["results_dir"])
        print(f"   results: {dest}")
        for glb in sorted(dest.glob("glb/*.glb")):
            print(f"     {glb}")
    if report.get("status", {}).get("error"):
        print(f"   error: {report['status']['error']}")
    if report.get("result") == "failed" and not args.dry_run:
        print(f"   retry: python tools/cloud_scan.py --job_id {report['job_id']} --skip_upload")
    print("=" * 70)
    return 0 if report.get("result") in ("ok", None) else 1


if __name__ == "__main__":
    sys.exit(main())
