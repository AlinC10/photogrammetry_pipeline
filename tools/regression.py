"""
regression.py — Local only: the regression table of a folder of scan results (plans/PLAN.md 2.7 and 6), and its
comparison with a reference run of the same objects.

Every sub-folder with an object.json is one scan (the layout the Pi / tools/cloud_scan.py leave: object.json,
stats.json, status.json, cloud_report.json, stl/high_model.stl). Per scan: watertight, length x width x height (mm),
volume (cm3), patched area, high-poly faces, warnings, pipeline time, cost.

    python tools/regression.py rpi_results                         # table
    python tools/regression.py rpi_results --save-baseline v8.json # keep it as the reference
    python tools/regression.py results_v9 --baseline v8.json       # compare (exit code 1 on a regression)
    python tools/regression.py results_v9 --baseline v8.json --names names.json --csv v9.csv

Scans are matched by name: --names maps job ids to object names ({"20261101-084007-a89a0e": "telecomanda"}), the
same object scanned again or the same photos processed again get the same name. Without it the title of
object.json, or the job id.
A regression: a model no longer watertight, a dimension changed by more than max(0.5 mm, 1 %), the volume by more
than 3 %, a new warning, the time up by more than 25 %. --check-stl also loads stl/high_model.stl and checks it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DIM_ABS_MM = 0.5
DIM_REL = 0.01
VOL_REL = 0.03
TIME_REL = 0.25


def _json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def scan_row(folder: Path, names: Dict[str, str], check_stl: bool = False) -> Optional[Dict[str, Any]]:
    obj = _json(folder / "object.json")
    if not obj:
        return None
    stats, status, cloud = _json(folder / "stats.json"), _json(folder / "status.json"), _json(folder / "cloud_report.json")
    job = obj.get("job_id") or folder.name
    title = obj.get("title")
    name = names.get(job) or names.get(folder.name) or (title if title and title != job else job)
    d = obj.get("dimensions_mm") or {}
    warnings = status.get("warnings") or (stats.get("run_info") or {}).get("warnings") or []
    row = {
        "name": name,
        "job": job,
        "watertight": obj.get("watertight"),
        "length_mm": d.get("length"),
        "width_mm": d.get("width"),
        "height_mm": d.get("height"),
        "volume_cm3": obj.get("volume_cm3"),
        "patched_area_cm2": obj.get("patched_area_cm2"),
        "faces_high": obj.get("faces_high"),
        "warnings": sorted({w.get("code", "?") for w in warnings if isinstance(w, dict)}),
        "time_s": stats.get("total_execution_time_sec"),
        "cost_usd": cloud.get("cost_estimate"),
        "state": status.get("state"),
    }
    if check_stl and (folder / "stl" / "high_model.stl").is_file():
        import trimesh

        m = trimesh.load(str(folder / "stl" / "high_model.stl"), process=True)
        row["stl_watertight"] = bool(m.is_watertight)
    return row


def collect(root: Path, names: Dict[str, str], check_stl: bool = False) -> List[Dict[str, Any]]:
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        r = scan_row(folder, names, check_stl)
        if r:
            rows.append(r)
    return rows


def compare(row: Dict[str, Any], ref: Dict[str, Any]) -> List[str]:
    """Differences of one scan against its reference: 'REGRESSION: ...' or 'better: ...' / 'changed: ...'."""
    out = []
    if ref.get("watertight") and row.get("watertight") is False:
        out.append("REGRESSION: no longer watertight")
    elif ref.get("watertight") is False and row.get("watertight"):
        out.append("better: now watertight")
    for k in ("length_mm", "width_mm", "height_mm"):
        a, b = row.get(k), ref.get(k)
        if a is None or b is None:
            continue
        if abs(a - b) > max(DIM_ABS_MM, DIM_REL * abs(b)):
            out.append(f"REGRESSION: {k} {b} -> {a}")
    a, b = row.get("volume_cm3"), ref.get("volume_cm3")
    if a is not None and b:
        if abs(a - b) > VOL_REL * abs(b):
            out.append(f"REGRESSION: volume {b} -> {a} cm3")
    elif a is not None and b is None:
        out.append(f"better: volume known ({a} cm3)")
    new_w = sorted(set(row.get("warnings") or []) - set(ref.get("warnings") or []))
    if new_w:
        out.append("REGRESSION: new warning(s) " + ", ".join(new_w))
    gone = sorted(set(ref.get("warnings") or []) - set(row.get("warnings") or []))
    if gone:
        out.append("better: warning(s) gone " + ", ".join(gone))
    a, b = row.get("time_s"), ref.get("time_s")
    if a and b and a > (1 + TIME_REL) * b:
        out.append(f"REGRESSION: time {b:.0f} -> {a:.0f} s")
    elif a and b and a < (1 - TIME_REL) * b:
        out.append(f"better: time {b:.0f} -> {a:.0f} s")
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        return f"{v:.4g}" if abs(v) < 1 else f"{v:.1f}"
    if isinstance(v, list):
        return ",".join(v) or "-"
    return str(v)


def print_table(rows: List[Dict[str, Any]], diffs: Optional[Dict[str, List[str]]] = None) -> None:
    cols = ["name", "watertight", "length_mm", "width_mm", "height_mm", "volume_cm3", "patched_area_cm2",
            "faces_high", "warnings", "time_s", "cost_usd"]
    print("| " + " | ".join(cols + (["vs reference"] if diffs is not None else [])) + " |")
    print("|" + "---|" * (len(cols) + (1 if diffs is not None else 0)))
    for r in rows:
        cells = [_fmt(r.get(c)) for c in cols]
        if diffs is not None:
            cells.append("; ".join(diffs.get(r["name"], [])) or ("no reference" if r["name"] not in diffs else "same"))
        print("| " + " | ".join(cells) + " |")
    wt = [r for r in rows if r.get("watertight") is not None]
    if wt:
        print(f"\nwatertight: {sum(1 for r in wt if r['watertight'])}/{len(wt)}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Regression table of scan results, optionally against a reference.")
    ap.add_argument("results", help="folder with one sub-folder per scan")
    ap.add_argument("--baseline", help="reference JSON (from --save-baseline) to compare with")
    ap.add_argument("--save-baseline", help="write the rows of this folder as a reference JSON")
    ap.add_argument("--names", help="JSON {job id: object name} to match scans across runs")
    ap.add_argument("--csv", help="also write the rows as CSV")
    ap.add_argument("--check-stl", action="store_true", help="load stl/high_model.stl and check it is watertight")
    args = ap.parse_args(argv)

    names = _json(Path(args.names)) if args.names else {}
    rows = collect(Path(args.results), names, args.check_stl)
    if not rows:
        print(f"No scan with an object.json in {args.results}", file=sys.stderr)
        return 2
    diffs = None
    regressions = 0
    if args.baseline:
        ref = {r["name"]: r for r in _json(Path(args.baseline)).get("rows", [])}
        diffs = {}
        for r in rows:
            if r["name"] in ref:
                diffs[r["name"]] = compare(r, ref[r["name"]])
                regressions += sum(1 for d in diffs[r["name"]] if d.startswith("REGRESSION"))
    print_table(rows, diffs)
    if args.save_baseline:
        Path(args.save_baseline).write_text(json.dumps({"results": args.results, "rows": rows}, indent=1),
                                            encoding="utf-8")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            for r in rows:
                w.writerow({k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()})
    if diffs is not None:
        print(f"\n{regressions} regression(s) against {args.baseline}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
