"""
stream_prework.py — Work done on the pod while the scanner is still taking photos (plans/PLAN_STREAMING_DAG.md §2):
everything here only fills pipeline/stream_cache.py, which run_pipeline reads back; nothing computed here goes into
the model unless it is exactly what the pipeline would compute.

- every photo, as it arrives: SIFT of rig_refine (A3);
- every finished level (all its photos for every camera, or the level_done_<L>.json marker of the Pi): the background
  median of each (camera, level) group (exact: same photos), then (A4) preliminary poses of that level alone
  (turntable_poses as a subprocess, unchanged, RULE 1, in work/stream/L<n>/) and the AI on the crops they give. The
  final masks use the AI output of a crop only when their own crop box is the same; preliminary poses never reach
  the model.

After capture_done no new level work starts (the pipeline runs then); the SIFT of the last photos goes on beside
turntable_poses, which does not need it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import stream_cache
from .taskgraph import TaskGraph

logger = logging.getLogger("TurntableStudio.StreamPrework")

NAME_RX = re.compile(r"(?P<level>\d)[_]?(?P<index>\d{1,4})$")     # TurntableConfig.name_regex


def photo_level(rel: str) -> Optional[int]:
    m = NAME_RX.search(Path(rel).stem)
    return int(m.group("level")) if m else None


def photo_camera(rel: str) -> str:
    """rig/0/f_1000.jpg -> "0"; a flat layout is one camera."""
    parts = Path(rel).parts
    return parts[-2] if len(parts) >= 2 and parts[-3:-2] == ("rig",) else "0"


class StreamPrework:
    def __init__(self, input_dir: str, cache_dir: str, job: Dict[str, Any], workers: int = 2,
                 level_masks: bool = True):
        self.input_dir = Path(input_dir)
        self.cache_dir = str(cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.job = dict(job)
        self.workers = max(1, int(workers))
        self.level_masks = level_masks
        self.per_level = int(round(360.0 / float(job.get("step_deg", 18.0) or 18.0)))
        self.graph = TaskGraph(max_workers=self.workers + 1, limits={"sift": max(1, self.workers), "level": 1},
                               name="prework")
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self.photos: Dict[str, str] = {}               # relative key -> local path
        self.levels_started: set = set()
        self.level_markers: Dict[int, List[str]] = {}
        self.info: Dict[str, Any] = {"levels": {}, "sift_photos": 0}
        self.t0 = time.time()
        self._warm_model()

    # ------------------------------------------------------------------ events from the download loop

    def photo_arrived(self, rel: str, path: str) -> None:
        with self._lock:
            if rel in self.photos:
                return
            self.photos[rel] = path
        if self.job.get("rig_refine", True):
            self.graph.add(f"sift:{rel}", self._sift, path, resource="sift", cost=0.5)
        self._check_levels()

    def level_marker(self, level: int, files: Optional[List[str]] = None) -> None:
        with self._lock:
            self.level_markers[int(level)] = list(files or [])
        self._check_levels()

    def capture_done(self) -> None:
        """The last photo is in: no new level work (the pipeline starts now)."""
        self.stop.set()
        self._check_levels()               # the medians of the last level still help (cheap)

    def wait_levels(self, timeout: float = 300.0) -> float:
        """
        Waits for the level work already started (main.py calls it before its own masks: one set of AI workers at a
        time in RAM, and their crops are in the cache by then). Returns the seconds waited.
        """
        t0 = time.time()
        with self._lock:
            names = [f"level:{L}" for L in self.levels_started]
        self.graph.wait([n for n in names if self.graph.has(n)], timeout=timeout)
        waited = round(time.time() - t0, 2)
        self.info["pipeline_waited_for_levels_s"] = waited
        return waited

    def finish(self, timeout: float = 120.0) -> Dict[str, Any]:
        """Waits for what is running (bounded); returns the numbers for stats.json."""
        self.graph.wait(timeout=timeout)
        self.graph.shutdown(wait=False)
        self.info.update(tasks=[{k: r.get(k) for k in ("name", "state", "duration_s", "error") if r.get(k) is not None}
                                for r in self.graph.records() if not r["name"].startswith("sift:")],
                         sift_failed=sum(1 for r in self.graph.records()
                                         if r["name"].startswith("sift:") and r["state"] != "done"),
                         cache=stream_cache.summary(self.cache_dir))
        return self.info

    # ------------------------------------------------------------------ tasks

    def _warm_model(self) -> None:
        """Reads the AI model once so the mask workers find it in the page cache."""
        try:
            from .config import MaskingConfig
            path = MaskingConfig().ai_model_path
            if path and Path(path).is_file():
                with open(path, "rb") as f:
                    while f.read(1 << 24):
                        pass
        except Exception:
            pass

    def _sift(self, path: str) -> int:
        from .config import RigRefineConfig
        from .rig_refine import cache_keypoints

        n = cache_keypoints(path, RigRefineConfig(), self.cache_dir)
        with self._lock:
            self.info["sift_photos"] += 1
        return n

    def _check_levels(self) -> None:
        with self._lock:
            by_level: Dict[int, Dict[str, List[str]]] = {}
            for rel in self.photos:
                L = photo_level(rel)
                if L is not None:
                    by_level.setdefault(L, {}).setdefault(photo_camera(rel), []).append(rel)
            # Every camera of the layout must have its shots (the rig has two: the photos of cam 1 arrive after
            # those of cam 0 at each position, a level is not complete with cam 0 alone)
            cams = {c for groups in by_level.values() for c in groups}
            if any(Path(r).parts[:1] == ("rig",) for r in self.photos):
                cams |= {str(c) for c in self.job.get("stream_cameras", ("0", "1"))}
            ready = []
            for L, groups in sorted(by_level.items()):
                if L in self.levels_started:
                    continue
                marker = self.level_markers.get(L)
                if marker is not None:
                    complete = all(r in self.photos for r in marker) if marker else True
                else:
                    complete = bool(cams) and all(len(groups.get(c, [])) >= self.per_level for c in cams)
                if complete:
                    self.levels_started.add(L)
                    ready.append((L, {c: sorted(v) for c, v in groups.items()}))
        for L, groups in ready:
            self.graph.add(f"level:{L}", self._level, L, groups, resource="level", cost=30.0)

    def _level(self, level: int, groups: Dict[str, List[str]]) -> Dict[str, Any]:
        from .mask_fusion import median_image, read_quarter

        t0 = time.time()
        rec: Dict[str, Any] = {"photos": sum(len(v) for v in groups.values())}
        for cam, rels in groups.items():
            paths = [self.photos[r] for r in rels]
            if stream_cache.median_path(self.cache_dir, paths) is None:
                med = median_image([read_quarter(p) for p in paths])
                if med is not None:
                    stream_cache.save_median(self.cache_dir, paths, med)
        rec["median_s"] = round(time.time() - t0, 2)
        if self.stop.is_set() or not self.level_masks:
            rec["masks"] = "skipped (capture done)" if self.stop.is_set() else "disabled"
        else:
            try:
                rec.update(self._level_masks(level, groups))
            except Exception as e:                       # only a lost head start
                rec["masks"] = f"error: {str(e)[:200]}"
                logger.warning(f"Streaming: masks of level {level} not pre-computed ({e}).")
        rec["seconds"] = round(time.time() - t0, 2)
        self.info["levels"][str(level)] = rec
        logger.info(f"Streaming: level {level} pre-work done in {rec['seconds']} s ({rec.get('masks', 'masks ok')}).")
        return rec

    def _level_masks(self, level: int, groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Preliminary poses of this level alone, then the fused masks of its views (only the AI cache is kept)."""
        import main as pipeline_main
        from .camera_init import generate_camera_init_sfm, run_turntable_poses, scan_input_directory
        from .mask_fusion import batch_fused_masks
        from .mask_processor import ai_model_available, load_tag_layout
        from .mesh_masks import load_sfm_cameras
        from .rig_refine import view_groups

        m = pipeline_main.masking_config(self.job)
        if m.method != "fused" or not ai_model_available(m.ai_model_path):
            return {"masks": "not fused / no model"}
        work = Path(self.cache_dir) / f"L{level}"
        src = work / "input"
        for cam, rels in groups.items():                 # the level's photos only, as links
            d = src / "rig" / cam
            d.mkdir(parents=True, exist_ok=True)
            for r in rels:
                link = d / Path(r).name
                if not link.exists():
                    os.symlink(os.path.abspath(self.photos[r]), link)
        t0 = time.time()
        tt = pipeline_main.turntable_config(self.job)
        tt.estimate_lift = False                         # one level: nothing to estimate
        views, intrinsics, name_regex = scan_input_directory(str(src), rig_mode=(src / "rig").is_dir(),
                                                             default_focal_mm=self.job.get("focal_length_mm", 3.9))
        tt.name_regex = name_regex
        init = work / "cameraInit.sfm"
        generate_camera_init_sfm(views, intrinsics, str(init))
        known = work / "cameras_known.sfm"
        run_turntable_poses(str(init), str(known), tt, report_path=str(work / "turntable_report.txt"))
        t_poses = time.time() - t0
        cams = load_sfm_cameras(str(known))
        cam_i, lvl, _ = view_groups(cams, name_regex)
        stats: Dict[str, Any] = {}
        workers = pipeline_main.ai_mask_workers(m, self.workers)
        batch_fused_masks(
            cams, list(zip(cam_i.tolist(), lvl.tolist())), str(work / "masks"), str(work / "fusion"),
            load_tag_layout(tt.tag_layout), tag_scale=tt.tag_scale, disc_radius_m=m.disc_radius_m,
            cylinder_radius_m=m.cylinder_radius_m, cylinder_y_min_m=m.cylinder_y_min_m,
            cylinder_y_max_m=m.cylinder_y_max_m, chroma_threshold=m.chroma_threshold,
            brightness_white_threshold=m.brightness_white_threshold, tag_blackout=m.tag_blackout,
            tag_expansion_factor=m.tag_expansion_factor, workers=workers, ai_model=m.ai_model_path,
            ai_resolution=m.ai_resolution, ai_threads=m.ai_threads,
            ai_cache_dir=stream_cache.ai_dir(self.cache_dir), stats=stats, median_cache_dir=self.cache_dir)
        return {"masks": "ok", "poses_s": round(t_poses, 2), "views": len(cams), "model_runs": stats.get("model_runs"),
                "masks_s": round(time.time() - t0 - t_poses, 2)}
