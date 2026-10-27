#!/usr/bin/env python3
"""
fit_ram_model.py — Refits the DensifyPointCloud RAM model RAM_GB = A + B * N * P from several stats.json.

Every run writes n_images and work_megapixels (run_info.ram_check) and the peak RAM of the step
"4.2 OpenMVS DensifyPointCloud". With two or more runs at different N * P this prints A (GB) and
B (bytes per working pixel) to copy into pipeline/resources.py (RAM_MODEL_A_GB, RAM_MODEL_B_BYTES_PER_PX).

  python tools/fit_ram_model.py results/*/stats.json output/stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

DENSIFY_STEP = "4.2 OpenMVS DensifyPointCloud"


def read_point(path: Path) -> Optional[Tuple[float, float]]:
    """(N * P in megapixels, peak Densify RAM in GB) of one stats.json, or None when it lacks the data."""
    data = json.loads(path.read_text(encoding="utf-8"))
    check = (data.get("run_info") or {}).get("ram_check") or {}
    n, p = check.get("n_images"), check.get("work_megapixels")
    peak = next((s.get("peak_ram_mb") for s in data.get("steps", []) if s.get("name") == DENSIFY_STEP), None)
    if not n or not p or not peak:
        return None
    return float(n) * float(p), float(peak) / 1024.0


def fit(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Least squares y = A + b * x; returns (A in GB, B in bytes per pixel)."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx == 0:
        raise ValueError("All runs have the same N * P: vary the photo count or the resolution level.")
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    return a, b * 1e9 / 1e6  # b is GB per megapixel -> bytes per pixel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", nargs="+", help="stats.json files")
    args = ap.parse_args(argv)

    points = []
    for name in args.stats:
        pt = read_point(Path(name))
        if pt is None:
            print(f"skip {name}: no ram_check / Densify peak (run made before the RAM check existed?)")
            continue
        print(f"{name}: N*P = {pt[0]:.1f} MP, Densify peak {pt[1]:.2f} GB")
        points.append(pt)
    if len(points) < 2:
        print("Need at least two runs with data.")
        return 1
    a, b = fit(points)
    worst = max(abs(a + b * x * 1e6 / 1e9 - y) for x, y in points)
    print(f"\nRAM_MODEL_A_GB = {a:.2f}\nRAM_MODEL_B_BYTES_PER_PX = {b:.2f}\n(largest residual {worst:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
