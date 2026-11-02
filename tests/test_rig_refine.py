"""
test_rig_refine.py — Per-(camera, level) correction of the turntable poses: recovers a known carriage tilt and cam1
offset from synthetic correspondences, keeps the tag poses when the data are too few or the correction is implausible,
drops booth (static) keypoints and rewrites cameras_known.sfm in column-major order (RULE 3).
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

if HAS_CV2:
    from pipeline import rig_refine as rr
    from pipeline.config import RigRefineConfig

F = 2500.0


def look_at(C, target):
    f = (target - C) / np.linalg.norm(target - C)
    right = np.cross(f, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.stack([right, down, f])


def turntable_rig(n_shots=20, levels=(1, 2), lift=0.08):
    """Disc-frame poses of 2 cameras x levels x shots (the object turns: the cameras orbit in the disc frame)."""
    K = np.array([[F, 0, 2028.0], [0, F, 1520.0], [0, 0, 1.0]])
    cams = []
    for cam, (dist, h0) in enumerate(((0.34, 0.13), (0.34, 0.14))):
        for L in levels:
            for s in range(n_shots):
                a = np.radians(18.0 * s + 9.0 * cam)
                C = np.array([dist * np.sin(a), h0 + lift * (L - 1), dist * np.cos(a)])
                R = look_at(C, np.array([0.0, 0.06 + lift * (L - 1) * 0.6, 0.0]))
                cams.append({"view_id": f"{cam}{L}{s:03d}", "intrinsic_id": str(cam),
                             "path": f"/x/rig/{cam}/f_{L}{s:03d}.jpg", "R": R, "C": C, "K": K, "dist": np.zeros(5)})
    return cams


def synthetic_obs(true_cams, n_points=4000, noise_px=0.3, seed=1):
    """Normalized correspondences of points on a box-shaped object, projected with the TRUE poses."""
    rng = np.random.default_rng(seed)
    P = np.c_[rng.uniform(-0.035, 0.035, n_points), rng.uniform(0.01, 0.22, n_points), rng.uniform(-0.035, 0.035, n_points)]
    axes = np.array([c["R"][2] for c in true_cams])
    I, J, A, B = [], [], [], []
    for i in range(len(true_cams)):
        for j in range(i + 1, len(true_cams)):
            if axes[i] @ axes[j] < np.cos(np.radians(40)):
                continue
            sel = rng.choice(n_points, 60, replace=False)
            for k, out in ((i, A), (j, B)):
                Xc = (P[sel] - true_cams[k]["C"]) @ true_cams[k]["R"].T
                out.append(Xc[:, :2] / Xc[:, 2:3] + rng.normal(0, noise_px / F, (len(sel), 2)))
            I.append(np.full(len(sel), i))
            J.append(np.full(len(sel), j))
    return {"I": np.concatenate(I), "J": np.concatenate(J), "A": np.concatenate(A), "B": np.concatenate(B),
            "f_px": F, "pairs_tested": len(I)}


@unittest.skipUnless(HAS_CV2, "OpenCV required")
class TestRigRefine(unittest.TestCase):

    def setUp(self):
        self.model = turntable_rig()
        self.cam, self.lvl, self.shot = rr.view_groups(self.model, r"(?P<level>\d)[_]?(?P<index>\d{1,4})$")
        # truth: the carriage pitches 0.3 deg on the lift, cam1 sits 0.3 deg / 1.5 mm off its tag pose (errors up to
        # ~10 px between levels, as measured on the box once the tags have absorbed part of the real ~0.9 deg tilt)
        self.truth = {(0, 2): [0.3, 0.0, 0.0, 0.0, 1.5, 0.0], (1, 1): [0.0, 0.3, 0.0, 1.5, 0.0, 0.0],
                      (1, 2): [0.3, 0.3, 0.0, 1.5, 1.5, 0.0]}
        self.true_cams = []
        for k, c in enumerate(self.model):
            p = self.truth.get((int(self.cam[k]), int(self.lvl[k])))
            R, C = (c["R"], c["C"]) if p is None else rr.apply_correction(c["R"], c["C"], np.r_[np.radians(p[:3]), p[3:]])
            self.true_cams.append(dict(c, R=R, C=C))

    def test_view_groups_from_names(self):
        self.assertEqual(sorted(set(self.cam.tolist())), [0, 1])
        self.assertEqual(sorted(set(self.lvl.tolist())), [1, 2])
        self.assertEqual(int(self.shot.max()), 19)

    def test_recovers_carriage_tilt_and_cam1_offset(self):
        obs = synthetic_obs(self.true_cams)
        res = rr.fit_corrections(self.model, obs, self.cam, self.lvl, RigRefineConfig())
        self.assertTrue(res["accepted"], res.get("reason"))
        self.assertGreater(res["holdout_before"]["median_px"], 3 * res["holdout_after"]["median_px"])
        self.assertLess(res["holdout_after"]["median_px"], 0.5)
        Rs, Cs = res["_poses"]
        rot_err = [np.degrees(np.arccos(np.clip((np.trace(R @ t["R"].T) - 1) / 2, -1, 1))) for R, t in zip(Rs, self.true_cams)]
        pos_err = [np.linalg.norm(C - t["C"]) * 1000 for C, t in zip(Cs, self.true_cams)]
        self.assertLess(max(rot_err), 0.05)          # deg
        self.assertLess(max(pos_err), 0.5)           # mm

    def test_keeps_tag_poses_with_too_few_observations(self):
        obs = synthetic_obs(self.true_cams)
        few = {k: (v[:150] if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        res = rr.fit_corrections(self.model, few, self.cam, self.lvl, RigRefineConfig())
        self.assertFalse(res["accepted"])
        self.assertIn("too few", res["reason"])
        self.assertNotIn("_poses", res)

    def test_rejects_implausible_correction(self):
        obs = synthetic_obs(self.true_cams)
        res = rr.fit_corrections(self.model, obs, self.cam, self.lvl, RigRefineConfig(max_rotation_deg=0.3))
        self.assertFalse(res["accepted"])
        self.assertIn("too large", res["reason"])

    def test_rejects_when_holdout_does_not_improve(self):
        obs = synthetic_obs(self.model)              # tag poses already exact: nothing to gain
        res = rr.fit_corrections(self.model, obs, self.cam, self.lvl, RigRefineConfig())
        self.assertFalse(res["accepted"])
        self.assertIn("improvement", res["reason"])

    def test_static_booth_keypoints(self):
        rng = np.random.default_rng(0)
        desc_booth = rng.random((50, 128)).astype(np.float32)
        feats = []
        for k in range(len(self.model)):
            desc_obj = rng.random((50, 128)).astype(np.float32)        # object keypoints differ from shot to shot
            uv = np.r_[np.tile([[100.0, 200.0]], (50, 1)) + np.arange(50)[:, None], rng.uniform(0, 4000, (50, 2))]
            feats.append((uv, uv / F, np.r_[desc_booth, desc_obj]))
        static = rr.static_keypoints(feats, self.cam, self.lvl, self.shot, RigRefineConfig())
        self.assertTrue(all(s[:50].all() for s in static))
        self.assertFalse(any(s[50:].any() for s in static))

    def test_sift_cache_same_features(self):
        """Streaming pre-work: SIFT saved when the photo arrives = SIFT computed in the step; a changed photo is a miss."""
        import tempfile
        import time as _time
        from pipeline import stream_cache
        from pipeline.rig_refine import _detect, cache_keypoints

        with tempfile.TemporaryDirectory() as d:
            rng = np.random.default_rng(5)
            img = cv2.GaussianBlur(rng.integers(0, 255, (600, 800), dtype=np.uint8), (5, 5), 1.5)
            path = str(Path(d) / "rig" / "0" / "f_1003.jpg")
            Path(path).parent.mkdir(parents=True)
            cv2.imwrite(path, img)
            cfg = RigRefineConfig()
            K = np.array([[700.0, 0, 400], [0, 700.0, 300], [0, 0, 1]])
            dist = np.array([-0.1, 0.02, 0, 0, 0])
            cache = str(Path(d) / "stream")
            self.assertGreater(cache_keypoints(path, cfg, cache), 50)
            ref = _detect(path, K, dist, cfg)
            stream_cache.stats(reset=True)
            got = _detect(path, K, dist, cfg, cache)
            self.assertEqual(stream_cache.stats().get("sift_hit"), 1)
            for a, b in zip(ref, got):
                self.assertTrue(np.array_equal(a, b))
            _time.sleep(0.01)
            cv2.imwrite(path, 255 - img)                          # another photo under the same name
            self.assertIsNone(stream_cache.load_sift(cache, path, cfg.scale, cfg.max_features))

    def test_write_refined_sfm_column_major(self):
        cams = self.model[:2]
        sfm = {"views": [{"viewId": c["view_id"], "poseId": f"p{c['view_id']}", "intrinsicId": c["intrinsic_id"],
                          "path": c["path"]} for c in cams],
               "poses": [{"poseId": f"p{c['view_id']}", "pose": {"transform": {
                   "rotation": [str(v) for v in c["R"].reshape(-1, order="F")], "center": [str(v) for v in c["C"]]},
                   "locked": "true"}} for c in cams]}
        R_new = [rr.apply_correction(c["R"], c["C"], [0.01, 0, 0, 1, 0, 0])[0] for c in cams]
        C_new = [rr.apply_correction(c["R"], c["C"], [0.01, 0, 0, 1, 0, 0])[1] for c in cams]
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "cameras_known.sfm"
            src.write_text(json.dumps(sfm))
            rr.write_refined_sfm(str(src), str(src), cams, R_new, C_new)
            out = json.loads(src.read_text())
        for p, R, C in zip(out["poses"], R_new, C_new):
            R_read = np.array(p["pose"]["transform"]["rotation"], float).reshape((3, 3), order="F")
            np.testing.assert_allclose(R_read, R, atol=1e-12)
            np.testing.assert_allclose(np.array(p["pose"]["transform"]["center"], float), C, atol=1e-12)
            self.assertEqual(p["pose"]["locked"], "true")


if __name__ == "__main__":
    unittest.main()
