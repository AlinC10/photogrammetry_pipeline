#!/usr/bin/env python3
"""
cloud_test_run.py — Runs the whole cloud flow from this machine with the photos in input_images/:
upload to R2 -> pod on RunPod -> wait -> download results/<job_id>/ -> delete the job from R2.
Not part of the Docker image (tools/ is in .dockerignore).

  python tools/cloud_test_run.py --dry_run          # R2 preflight + RAM/disk estimate + the pod request, no pod, no cost
  python tools/cloud_test_run.py                    # everything
  python tools/cloud_test_run.py --keep_remote      # leave jobs/<id>/ in R2 (debugging)
  python tools/cloud_test_run.py --job_id <id> --skip_upload   # retry a job whose photos are already in R2

Needs a .env in the repository root (see .env.example): R2_*, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_R2_API_TOKEN,
RUNPOD_API_KEY, TURNTABLE_IMAGE.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from cloud.job import run_cloud_job  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default=str(ROOT / "input_images"))
    ap.add_argument("--results_dir", default=str(ROOT / "results"))
    ap.add_argument("--job_id", default=None, help="reuse a job id (with --skip_upload: photos already in R2)")
    ap.add_argument("--image", default=None, help="Docker image (default: TURNTABLE_IMAGE from .env)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_upload", action="store_true")
    ap.add_argument("--keep_remote", action="store_true")
    ap.add_argument("--min_vcpu", type=int, default=8)
    ap.add_argument("--resolution_level", type=int, default=1)
    ap.add_argument("--no_estimate_lift", action="store_true", help="use --lift_mm instead of estimating the lift")
    ap.add_argument("--lift_mm", type=float, default=100.0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    options = {
        "rig_mode": (Path(args.input_dir) / "rig").is_dir(),
        "resolution_level": args.resolution_level,
        "estimate_lift": not args.no_estimate_lift,
        "lift_mm": args.lift_mm,
        "texturing_engine": "texrecon",
    }
    t0 = time.time()
    report = run_cloud_job(
        input_dir=args.input_dir, results_dir=args.results_dir, job_id=args.job_id, pipeline_options=options,
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
    if report.get("pod_timings_s"):
        print(f"   in the pod: {json.dumps(report['pod_timings_s'])}")
    if report.get("results_dir"):
        print(f"   results: {report['results_dir']}")
    if report.get("status", {}).get("error"):
        print(f"   error: {report['status']['error']}")
    print("=" * 70)
    return 0 if report.get("result") in ("ok", None) else 1


if __name__ == "__main__":
    sys.exit(main())
