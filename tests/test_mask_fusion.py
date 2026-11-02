"""
test_mask_fusion.py — Fused masks (mask_fusion.py) on a synthetic turntable: white disc with the layout tags, grey
booth, a red sphere on the disc, cameras around it. A fake model that keeps the disc (as BiRefNet does) must give
masks of the sphere alone; plus the background model, the crop box and the per-view check against the hull.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from pipeline import mask_fusion as mf
from pipeline import mask_processor as mp
from pipeline import visual_hull as vh

W, H, F = 640, 480, 700.0
SPHERE_C, SPHERE_R = np.array([0.0, 0.03, 0.0]), 0.025
DISC_R = 0.127
BOOTH = (205, 200, 200)            # BGR, bright and colourless
RED = (30, 30, 200)


def look_at(C, target):
    f = (target - C) / np.linalg.norm(target - C)
    right = np.cross(f, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(f, right), f])


def make_cameras(n=8, height=0.14, dist=0.34):
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    cams = []
    for i, a in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        C = np.array([dist * np.cos(a), height, dist * np.sin(a)])
        cams.append({"view_id": str(1000 + i), "intrinsic_id": "0", "path": "", "R": look_at(C, np.array([0, 0.03, 0])),
                     "C": C, "K": K, "dist": np.zeros(5), "K_new": K, "w": W, "h": H})
    return cams


def render(cam, tex):
    """BGR photo and true object mask of the synthetic scene."""
    gx, gy = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d_cam = np.stack([(gx - W / 2) / F, (gy - H / 2) / F, np.ones_like(gx)], -1).reshape(-1, 3)
    d = d_cam @ cam["R"]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    C = cam["C"]
    # sphere
    oc = C - SPHERE_C
    b = d @ oc
    disc_term = b ** 2 - (oc @ oc - SPHERE_R ** 2)
    obj = (disc_term > 0) & (-b - np.sqrt(np.maximum(disc_term, 0)) > 0)
    # disc plane
    t = np.where(d[:, 1] < 0, -C[1] / np.where(d[:, 1] < 0, d[:, 1], -1), -1)
    x, z = C[0] + t * d[:, 0], C[2] + t * d[:, 2]
    on_disc = (t > 0) & (np.hypot(x, z) <= DISC_R)
    img = np.empty((H * W, 3), np.uint8)
    img[:] = BOOTH
    u = np.clip(((x + mf.DISC_TEX_HALF_M) / mf.DISC_TEX_RES_M).astype(int), 0, tex.shape[1] - 1)
    v = np.clip(((z + mf.DISC_TEX_HALF_M) / mf.DISC_TEX_RES_M).astype(int), 0, tex.shape[0] - 1)
    # printed paper ~235, printed black ~45 (a photo never has the pure 0 / 255 of the rendering)
    img[on_disc] = np.repeat((45 + tex[v[on_disc], u[on_disc]].astype(np.int32) * 190 // 255)[:, None], 3, 1)
    img[obj] = RED
    return img.reshape(H, W, 3), obj.reshape(H, W)


def fake_model(rgb, model_path, resolution=512, threads=2, cache_path=None):
    """Keeps the object AND the white disc, like BiRefNet on the turntable."""
    rgb = rgb.astype(np.int32)
    red = (rgb[..., 0] > 150) & (rgb[..., 1] < 80)
    disc = rgb.min(-1) > 225
    return (red | disc).astype(np.float32)


@unittest.skipUnless(HAS_CV2, "OpenCV required")
class TestMaskFusion(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.tags = mp.load_tag_layout(None)
        self.tex = mf.disc_texture(self.tags, 1.0, DISC_R)
        self.cams = make_cameras()
        self.truth = {}
        (self.dir / "rig" / "0").mkdir(parents=True)
        for c in self.cams:
            img, obj = render(c, self.tex)
            c["path"] = str(self.dir / "rig" / "0" / f"f_{c['view_id']}.png")
            cv2.imwrite(c["path"], img)
            self.truth[c["view_id"]] = obj

    def tearDown(self):
        self.tmp.cleanup()

    def run_fusion(self, model="fake.onnx"):
        stats = {}
        with mock.patch.object(mf, "ai_foreground_probability", side_effect=fake_model):
            out = mf.batch_fused_masks(self.cams, [(0, 1)] * len(self.cams), str(self.dir / "masks"),
                                       str(self.dir / "work"), self.tags, tag_scale=1.0, disc_radius_m=DISC_R,
                                       cylinder_radius_m=0.08, workers=1, ai_model=model, stats=stats)
        return out, stats

    def test_stream_caches_give_the_same_masks(self):
        """Medians and AI crops computed while the photos arrive (stream_cache) = the masks computed at the end."""
        from pipeline import stream_cache

        calls = []

        def caching_fake(rgb, model_path, resolution=512, threads=2, cache_path=None):
            if cache_path is not None and Path(cache_path).is_file():
                return np.load(str(cache_path))
            calls.append(1)
            y = fake_model(rgb, model_path)
            if cache_path is not None:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cache_path), y)
            return y

        def run(out, cache):
            stats = {}
            with mock.patch.object(mf, "ai_foreground_probability", side_effect=caching_fake):
                mf.batch_fused_masks(self.cams, [(0, 1)] * len(self.cams), str(self.dir / out), str(self.dir / "w"),
                                     self.tags, tag_scale=1.0, disc_radius_m=DISC_R, cylinder_radius_m=0.08,
                                     workers=1, ai_model="fake.onnx", stats=stats,
                                     ai_cache_dir=stream_cache.ai_dir(cache), median_cache_dir=cache)
            return stats

        ref = run("ref", None)
        cache = str(self.dir / "stream")
        # pre-work: the median of the level and the AI on the same crops
        med = mf.median_image([mf.read_quarter(c["path"]) for c in self.cams])
        stream_cache.save_median(cache, [c["path"] for c in self.cams], med)
        run("pre", cache)
        n_first = len(calls)
        stream_cache.stats(reset=True)
        st = run("cached", cache)
        self.assertEqual(len(calls), n_first)                     # no model run the second time
        self.assertEqual(st["ai_cache_hits"], ref["model_runs"])
        self.assertEqual(stream_cache.stats().get("median_hit"), 1)
        for f in sorted((self.dir / "ref").glob("*.mask.png")):
            self.assertEqual(f.read_bytes(), (self.dir / "cached" / f.name).read_bytes(), f.name)

    def test_disc_texture_has_white_disc_and_black_tags(self):
        n = self.tex.shape[0]
        self.assertEqual(self.tex[n // 2, n // 2], 255)                    # centre of the disc: paper
        self.assertEqual(self.tex[0, 0], 0)                                # outside the disc
        t = self.tags[0]["center"]
        u = int((t[0] + mf.DISC_TEX_HALF_M) / mf.DISC_TEX_RES_M)
        v = int((t[2] + mf.DISC_TEX_HALF_M) / mf.DISC_TEX_RES_M)
        patch = self.tex[v - 40:v + 40, u - 40:u + 40]
        self.assertGreater((patch < 50).mean(), 0.3)                       # a tag is mostly black

    def test_disc_zone_matches_the_rendered_scene(self):
        c = self.cams[0]
        rays = mf.pixel_rays(c["K"], c["dist"], W // 4, H // 4)
        zone, near, D = mf.disc_zone_and_render(rays, c["R"], c["C"], self.tex, DISC_R)
        q = cv2.imread(c["path"], cv2.IMREAD_REDUCED_COLOR_4)
        self.assertTrue(zone.any() and (~zone).any())
        self.assertTrue(np.all(near[zone]))
        # where the scene shows plain paper, the rendering does too
        white = zone & (D > 250)
        self.assertGreater((q[white].min(-1) > 200).mean(), 0.9)

    def test_fused_masks_keep_the_object_and_drop_the_disc(self):
        out, stats = self.run_fusion()
        self.assertEqual(sum(v for k, v in stats["methods"].items() if k.startswith("fused")), len(self.cams))
        for c in self.cams:
            mask = cv2.imread(str(self.dir / "masks" / f"{c['view_id']}_f_{c['view_id']}.mask.png"), 0) > 0
            truth = self.truth[c["view_id"]]
            iou = (mask & truth).sum() / (mask | truth).sum()
            self.assertGreater(iou, 0.7, c["view_id"])                 # 1/4-resolution edge + ~1 px tolerance
            self.assertGreater((mask & truth).sum() / truth.sum(), 0.97)   # nothing of the object lost
            # the disc the fake model keeps is gone: what is left is the tolerance ring around the object
            prob = fake_model(cv2.cvtColor(cv2.imread(c["path"]), cv2.COLOR_BGR2RGB), "")
            self.assertLess((mask & ~truth).sum(), 0.15 * ((prob > 0.5) & ~truth).sum())
            # the per-camera copy for rig mode is written as well
            self.assertTrue((self.dir / "masks" / "0" / f"f_{c['view_id']}.mask.png").is_file())
        # the work files of the fusion are removed
        self.assertFalse(list((self.dir / "work").glob("view_*.npz")))

    def test_without_fusion_the_fake_model_keeps_the_disc(self):
        # control: the fake model alone marks the disc, so the test above really checks the fusion
        c = self.cams[0]
        prob = fake_model(cv2.cvtColor(cv2.imread(c["path"]), cv2.COLOR_BGR2RGB), "")
        self.assertGreater((prob > 0.5).sum(), 3 * self.truth[c["view_id"]].sum())

    def test_background_model_on_a_known_difference(self):
        I = np.full((40, 40, 3), 200, np.uint8)
        M = I.copy()
        I[10:20, 10:20] = (20, 20, 20)                     # dark object in the booth zone
        zone = np.zeros((40, 40), bool)
        D = np.zeros((40, 40), np.float32)
        obj, strong, unreliable = mf.background_model(I, M, zone, D)
        self.assertTrue(obj[15, 15] and strong[15, 15])
        self.assertFalse(obj[30, 30] or unreliable[30, 30])
        M2 = M.copy()
        M2[:, :20] = (20, 120, 20)                          # median not the booth: the model decides there
        _, _, unreliable2 = mf.background_model(I, M2, zone, D)
        self.assertTrue(unreliable2[30, 5] and not unreliable2[30, 30])

    def test_crop_box_is_square_and_inside_the_photo(self):
        rough = np.zeros((120, 160), bool)
        rough[50:60, 70:80] = True
        x0, y0, x1, y1 = mf.crop_box(rough, 640, 480)
        self.assertLessEqual(abs((x1 - x0) - (y1 - y0)), 1)
        self.assertTrue(x0 <= 280 and x1 >= 320 and y0 <= 200 and y1 >= 240)
        self.assertEqual(mf.crop_box(np.zeros((120, 160), bool), 640, 480), (0, 0, 640, 480))

    def test_mask_agreement_flags_a_mask_that_misses_the_object(self):
        self.run_fusion()
        paths = vh.source_mask_paths(self.cams, str(self.dir / "masks"))
        bad = cv2.imread(paths[3], 0)
        bad[:, :] = 0
        bad[:20, :20] = 255                                   # the model "chose" something else
        cv2.imwrite(paths[3], bad)
        loose = vh.carve_visual_hull(self.cams, paths, radius_m=0.08, y_min_m=0.001, y_max_m=0.1, voxel_m=0.003,
                                     min_bg_votes=3, distorted=True)
        agree = vh.mask_agreement(loose, self.cams, paths)
        self.assertLess(agree[3], 0.5)
        self.assertGreater(min(a for i, a in enumerate(agree) if i != 3), 0.8)


if __name__ == "__main__":
    unittest.main()
