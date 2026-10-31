"""
rig_refine.py — Rigid per-(camera, level) correction of the turntable poses from sparse object features.

turntable_poses.py models the disc very well (axis, step, per-level offset: features on the object agree to ~0.4 px
between shots of one camera and level), but every level reuses the same camera orientation and cam1 sits where the
tags put it. The real carriage tilts ~0.9 deg on the first lift and cam1 is off by ~0.4 deg: object features disagree
by 2-3 px between levels and 1.5 px between the cameras, exactly on the view pairs DensifyPointCloud fuses.

This step (after turntable_poses, which stays untouched, RULE 1):
1. SIFT at half resolution on every photo;
2. drops booth points: the booth is static in the camera within a level while the object turns, so a keypoint found
   at the same pixel in >= 2 neighbouring shots of its level belongs to the walls (no mask needed);
3. matches neighbouring views (optical axes < 40 deg apart, across levels and cameras), keeps points that
   triangulate over the disc;
4. fits one rigid correction (3 rotations + 3 translations, camera-local) per (camera, level), the lowest level of the
   first camera fixed as reference, by minimizing the Sampson epipolar error (Huber). Disc model, intrinsics and scale
   stay those of the tags: a few tens of parameters, no per-photo freedom (RULE 2: no SfM drift);
5. keeps the correction only when it lowers the error on a third of the view pairs left out of the fit and stays
   physically small; otherwise the tag poses are kept unchanged.
Measured on the scans of 30.10.2026 (plans/PLAN_PRECIZIE_SI_MASTI.md): between levels 2.9 -> 1.0 px (box),
1.9 -> 0.34 px (spray can), cam0-cam1 1.4 -> 0.6 px; ~10-30 s on the pod.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import RigRefineConfig
from .mesh_masks import load_sfm_cameras

logger = logging.getLogger("TurntableStudio.RigRefine")

PAIR_TYPES = ("same cam, same level", "same cam, cross level", "cam0-cam1, same level", "cam0-cam1, cross level")


# ----------------------------------------------------------------------------- geometry helpers

def _rel(Ri: np.ndarray, Ci: np.ndarray, Rj: np.ndarray, Cj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Relative pose (R, t) with X_j = R X_i + t (RULE 3: X_cam = R (X_world - C))."""
    return Rj @ Ri.T, Rj @ (Ci - Cj)


def _essential(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    tx = np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])
    return tx @ R


def sampson_px(E: np.ndarray, a: np.ndarray, b: np.ndarray, f_px: float) -> np.ndarray:
    """Signed Sampson distance (px) of normalized correspondences a <-> b for essential matrices E [N,3,3]."""
    ah = np.c_[a, np.ones(len(a))]
    bh = np.c_[b, np.ones(len(b))]
    Ea = np.einsum("nij,nj->ni", E, ah)
    Etb = np.einsum("nji,nj->ni", E, bh)
    num = np.einsum("ni,ni->n", bh, Ea)
    den = np.sqrt(Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2) + 1e-18
    return num / den * f_px


def _rotvec_to_matrix(w: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(w, dtype=np.float64).reshape(3, 1))
    return R


def apply_correction(R: np.ndarray, C: np.ndarray, p: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Camera-local rigid correction p = (rotation vector [rad], translation [mm]): R' = dR R, C' = C + R'^T t."""
    R2 = _rotvec_to_matrix(np.asarray(p[:3])) @ R
    C2 = C + R2.T @ (np.asarray(p[3:6], dtype=np.float64) / 1000.0)
    return R2, C2


# ----------------------------------------------------------------------------- views

def view_groups(cams: List[Dict[str, Any]], name_regex: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(camera index, level, shot index) per view from the intrinsic id and the file name (turntable_poses regex)."""
    rx = re.compile(name_regex)
    cam_ids = sorted({c["intrinsic_id"] for c in cams})
    cam = np.array([cam_ids.index(c["intrinsic_id"]) for c in cams])
    lvl, shot = [], []
    for c in cams:
        m = rx.search(Path(c["path"]).stem)
        lvl.append(int(m.group("level")) if m else 1)
        shot.append(int(m.group("index")) if m else 0)
    return cam, np.array(lvl), np.array(shot)


def _detect(path: str, K: np.ndarray, dist: np.ndarray, cfg: RigRefineConfig):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 128), np.float32)
    if cfg.scale != 1.0:
        img = cv2.resize(img, None, fx=cfg.scale, fy=cfg.scale, interpolation=cv2.INTER_AREA)
    kp, desc = cv2.SIFT_create(nfeatures=cfg.max_features).detectAndCompute(img, None)
    if desc is None or not kp:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 128), np.float32)
    uv = (np.float64([k.pt for k in kp]) + 0.5) / cfg.scale - 0.5          # full-resolution pixel centres
    nrm = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    return uv, nrm, desc


def _match(d1: np.ndarray, d2: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    return np.array([m.queryIdx for m in good], int), np.array([m.trainIdx for m in good], int)


def static_keypoints(feats, cam: np.ndarray, lvl: np.ndarray, shot: np.ndarray, cfg: RigRefineConfig,
                     workers: int = 1) -> List[np.ndarray]:
    """Booth keypoints: at the same pixel in >= static_min_neighbours of the shots -2..+2 of the same camera and level."""
    n = len(feats)
    n_shots = {}
    for c, L, s in zip(cam, lvl, shot):
        n_shots[(c, L)] = max(n_shots.get((c, L), 0), s + 1)
    index = {(c, L, s): k for k, (c, L, s) in enumerate(zip(cam, lvl, shot))}

    def one(k: int) -> np.ndarray:
        uv, _, desc = feats[k]
        count = np.zeros(len(uv), int)
        ring = n_shots[(cam[k], lvl[k])]
        for ds in (-2, -1, 1, 2):
            o = index.get((cam[k], lvl[k], (shot[k] + ds) % ring))
            if o is None or o == k:
                continue
            qi, qj = _match(desc, feats[o][2], 0.8)
            if len(qi):
                still = np.linalg.norm(uv[qi] - feats[o][0][qj], axis=1) < cfg.static_px
                np.add.at(count, qi[still], 1)
        return count >= cfg.static_min_neighbours

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(one, range(n)))


def collect_observations(cams, feats, static, cfg: RigRefineConfig, workers: int = 1) -> Dict[str, np.ndarray]:
    """Correspondences on the turning object: view indices I, J and normalized points A, B."""
    n = len(cams)
    axes = np.array([c["R"][2] for c in cams])
    cosmax = np.cos(np.radians(cfg.max_axis_angle_deg))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if axes[i] @ axes[j] > cosmax]
    f_px = float(np.mean([c["K"][0, 0] for c in cams]))

    def one(pair):
        i, j = pair
        qi, qj = _match(feats[i][2], feats[j][2], cfg.ratio)
        if len(qi):
            keep = ~static[i][qi] & ~static[j][qj]
            qi, qj = qi[keep], qj[keep]
        if len(qi) < 8:
            return None
        a, b = feats[i][1][qi], feats[j][1][qj]
        Pi = np.hstack([cams[i]["R"], (-cams[i]["R"] @ cams[i]["C"])[:, None]])
        Pj = np.hstack([cams[j]["R"], (-cams[j]["R"] @ cams[j]["C"])[:, None]])
        X = cv2.triangulatePoints(Pi, Pj, a.T.astype(np.float64), b.T.astype(np.float64))
        X = (X[:3] / X[3]).T
        r = np.hypot(X[:, 0], X[:, 2])
        R, t = _rel(cams[i]["R"], cams[i]["C"], cams[j]["R"], cams[j]["C"])
        d = np.abs(sampson_px(np.repeat(_essential(R, t)[None], len(a), 0), a, b, f_px))
        keep = (r < cfg.disc_radius_m) & (X[:, 1] > cfg.y_min_m) & (X[:, 1] < cfg.y_max_m) & (d < cfg.max_sampson_px)
        k = np.nonzero(keep)[0][: cfg.max_obs_per_pair]
        if not len(k):
            return None
        return np.full(len(k), i), np.full(len(k), j), a[k], b[k]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        parts = [p for p in ex.map(one, pairs) if p is not None]
    if not parts:
        return {"I": np.zeros(0, int), "J": np.zeros(0, int), "A": np.zeros((0, 2)), "B": np.zeros((0, 2)),
                "pairs_tested": len(pairs), "f_px": f_px}
    return {"I": np.concatenate([p[0] for p in parts]), "J": np.concatenate([p[1] for p in parts]),
            "A": np.concatenate([p[2] for p in parts]), "B": np.concatenate([p[3] for p in parts]),
            "pairs_tested": len(pairs), "f_px": f_px}


# ----------------------------------------------------------------------------- fit

def _all_E(Rs, Cs, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    key = I.astype(np.int64) * 100000 + J
    u, inv = np.unique(key, return_inverse=True)
    Eu = np.array([_essential(*_rel(Rs[k // 100000], Cs[k // 100000], Rs[k % 100000], Cs[k % 100000])) for k in u])
    return Eu[inv]


def _pair_types(I, J, cam, lvl) -> np.ndarray:
    same_cam = cam[I] == cam[J]
    same_lvl = lvl[I] == lvl[J]
    return np.where(same_cam, np.where(same_lvl, 0, 1), np.where(same_lvl, 2, 3))


def _summary(d: np.ndarray, types: np.ndarray) -> Dict[str, Any]:
    out = {"median_px": round(float(np.median(d)), 3) if len(d) else None, "n": int(len(d))}
    for k, name in enumerate(PAIR_TYPES):
        m = types == k
        if m.any():
            out[name] = round(float(np.median(d[m])), 3)
    return out


def fit_corrections(cams, obs: Dict[str, np.ndarray], cam: np.ndarray, lvl: np.ndarray,
                    cfg: RigRefineConfig) -> Dict[str, Any]:
    """Per-(camera, level) corrections fitted on 2/3 of the view pairs, judged on the other third."""
    from scipy.optimize import least_squares

    I, J, A, B, f_px = obs["I"], obs["J"], obs["A"], obs["B"], obs["f_px"]
    R0 = [c["R"] for c in cams]
    C0 = [c["C"] for c in cams]
    types = _pair_types(I, J, cam, lvl)
    d0 = np.abs(sampson_px(_all_E(R0, C0, I, J), A, B, f_px)) if len(I) else np.zeros(0)

    rng = np.random.default_rng(cfg.seed)
    pid = I.astype(np.int64) * 100000 + J
    upairs = np.unique(pid)
    test_pairs = rng.choice(upairs, int(round(len(upairs) * cfg.holdout_fraction)), replace=False) if len(upairs) else []
    test = np.isin(pid, test_pairs)
    train = ~test & (d0 < cfg.max_sampson_px)

    groups = sorted(set(zip(cam.tolist(), lvl.tolist())))
    reference = min(groups) if groups else None                  # first camera, lowest level
    g_of = np.array([groups.index((c, L)) for c, L in zip(cam, lvl)])
    group_obs = {g: int((train & ((g_of[I] == gi) | (g_of[J] == gi))).sum()) for gi, g in enumerate(groups)}
    free = [g for g in groups if g != reference and group_obs[g] >= cfg.min_group_obs]

    result: Dict[str, Any] = {
        "observations": int(len(I)), "train": int(train.sum()), "test": int(test.sum()),
        "pairs_tested": obs.get("pairs_tested"), "reference": list(reference) if reference else None,
        "groups_fixed_low_obs": [list(g) for g in groups if g != reference and g not in free],
        "before": _summary(d0, types), "holdout_before": _summary(d0[test], types[test]),
        "accepted": False, "corrections": {},
    }
    if train.sum() < cfg.min_train_obs or not free or test.sum() < 20:
        result["reason"] = (f"too few observations on the object ({int(train.sum())} for the fit, "
                            f"{int(test.sum())} held out)")
        return result

    def poses(x):
        Rs, Cs = list(R0), list(C0)
        for gi, g in enumerate(free):
            p = x[6 * gi: 6 * gi + 6]
            for k in np.nonzero(g_of == groups.index(g))[0]:
                Rs[k], Cs[k] = apply_correction(R0[k], C0[k], p)
        return Rs, Cs

    def fit(sel, x0, huber):
        It, Jt, At, Bt = I[sel], J[sel], A[sel], B[sel]

        def residuals(x):
            Rs, Cs = poses(x)
            return sampson_px(_all_E(Rs, Cs, It, Jt), At, Bt, f_px)

        x_scale = np.tile([1e-3] * 3 + [0.5] * 3, len(free))
        return least_squares(residuals, x0, loss="huber", f_scale=huber, x_scale=x_scale).x

    # Coarse fit on everything plausible (the errors to correct can exceed the inlier gate), then a fine fit on the
    # observations that agree with it: a fixed gate on the tag poses would drop exactly the pairs carrying the signal.
    x = fit(train, np.zeros(6 * len(free)), cfg.coarse_huber_px)
    Rs, Cs = poses(x)
    d_coarse = np.abs(sampson_px(_all_E(Rs, Cs, I, J), A, B, f_px))
    train = ~test & (d_coarse < cfg.train_max_px)
    if train.sum() >= cfg.min_train_obs:
        x = fit(train, x, cfg.huber_px)
    result["train"] = int(train.sum())
    sol_x = x
    Rs, Cs = poses(sol_x)
    d1 = np.abs(sampson_px(_all_E(Rs, Cs, I, J), A, B, f_px))

    corrections = {}
    for gi, g in enumerate(free):
        p = sol_x[6 * gi: 6 * gi + 6]
        corrections[f"cam{g[0]}_L{g[1]}"] = {
            "rotation_deg": round(float(np.degrees(np.linalg.norm(p[:3]))), 3),
            "rotvec_deg": [round(float(v), 4) for v in np.degrees(p[:3])],
            "translation_mm": [round(float(v), 3) for v in p[3:]],
        }
    before, after = float(np.median(d0[test])), float(np.median(d1[test]))
    max_rot = max(c["rotation_deg"] for c in corrections.values())
    max_tr = max(float(np.linalg.norm(c["translation_mm"])) for c in corrections.values())
    result.update(corrections=corrections, after=_summary(d1, types), holdout_after=_summary(d1[test], types[test]),
                  max_rotation_deg=round(max_rot, 3), max_translation_mm=round(max_tr, 2))

    if after > before * (1.0 - cfg.min_improvement):
        result["reason"] = f"hold-out error {before:.3f} -> {after:.3f} px: improvement below {cfg.min_improvement:.0%}"
    elif max_rot > cfg.max_rotation_deg or max_tr > cfg.max_translation_mm:
        result["reason"] = (f"correction too large ({max_rot:.2f} deg, {max_tr:.1f} mm; limits "
                            f"{cfg.max_rotation_deg} deg, {cfg.max_translation_mm} mm)")
    else:
        result["accepted"] = True
        result["_poses"] = (Rs, Cs)
    return result


# ----------------------------------------------------------------------------- sfm I/O

def write_refined_sfm(src_sfm: str, dst_sfm: str, cams: List[Dict[str, Any]], Rs, Cs) -> None:
    """Same file as turntable_poses wrote, with the corrected rotation (column-major, RULE 3) and centre per view."""
    with open(src_sfm, "r", encoding="utf-8") as f:
        sfm = json.load(f)
    pose_of_view = {str(v["viewId"]): str(v["poseId"]) for v in sfm.get("views", [])}
    new = {pose_of_view[c["view_id"]]: (R, C) for c, R, C in zip(cams, Rs, Cs)}
    for p in sfm.get("poses", []):
        pid = str(p["poseId"])
        if pid in new:
            R, C = new[pid]
            p["pose"]["transform"]["rotation"] = [repr(float(v)) for v in R.reshape(-1, order="F")]
            p["pose"]["transform"]["center"] = [repr(float(v)) for v in C]
    with open(dst_sfm, "w", encoding="utf-8") as f:
        json.dump(sfm, f, indent=2)


def refine_rig_poses(cameras_known_sfm: str, name_regex: str, cfg: Optional[RigRefineConfig] = None,
                     workers: int = 1, report_path: Optional[str] = None,
                     backup_sfm: Optional[str] = None) -> Dict[str, Any]:
    """
    Fits the per-(camera, level) correction and, when accepted, rewrites cameras_known.sfm in place (the tag-only
    poses are kept in backup_sfm). Returns the report (also written to report_path as JSON).
    """
    cfg = cfg or RigRefineConfig()
    t0 = time.time()
    cams = load_sfm_cameras(cameras_known_sfm)
    cam, lvl, shot = view_groups(cams, name_regex)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        feats = list(ex.map(lambda c: _detect(c["path"], c["K"], c["dist"], cfg), cams))
    t_feat = time.time() - t0
    static = static_keypoints(feats, cam, lvl, shot, cfg, workers)
    obs = collect_observations(cams, feats, static, cfg, workers)
    t_match = time.time() - t0 - t_feat

    per_image = np.bincount(np.r_[obs["I"], obs["J"]], minlength=len(cams)) if len(obs["I"]) else np.zeros(len(cams))
    result = fit_corrections(cams, obs, cam, lvl, cfg)
    result.update(
        n_views=len(cams), n_levels=int(len(set(lvl.tolist()))),
        keypoints_per_image=int(np.median([len(f[0]) for f in feats])) if feats else 0,
        static_per_image=int(np.median([s.sum() for s in static])) if static else 0,
        observations_per_image_median=int(np.median(per_image)) if len(per_image) else 0,
        timings_s={"features": round(t_feat, 1), "matching": round(t_match, 1),
                   "total": round(time.time() - t0, 1)},
    )

    poses = result.pop("_poses", None)
    if result["accepted"] and poses is not None:
        if backup_sfm:
            Path(backup_sfm).write_bytes(Path(cameras_known_sfm).read_bytes())
        write_refined_sfm(cameras_known_sfm, cameras_known_sfm, cams, *poses)
        hb, ha = result["holdout_before"], result["holdout_after"]
        logger.info(f"Rig refinement accepted: hold-out epipolar error {hb['median_px']} -> {ha['median_px']} px "
                    f"(max correction {result['max_rotation_deg']} deg, {result['max_translation_mm']} mm; "
                    f"{result['timings_s']['total']} s).")
        for name in PAIR_TYPES:
            if name in hb and name in ha:
                logger.info(f"  {name:<24} {hb[name]:.3f} -> {ha[name]:.3f} px")
    else:
        logger.info(f"Rig refinement not applied: {result.get('reason')}. Tag poses kept "
                    f"({result['timings_s']['total']} s).")
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result
