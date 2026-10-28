"""
test_visual_hull.py — Visual hull carving, proxy bounds, OpenMVS dense PLY I/O and dense-cloud refinement
(prune wrong depths inside the hull, fill bare surfaces) on a synthetic cylinder seen by synthetic cameras.
"""

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

from pipeline import visual_hull as vh

W, H, F = 320, 240, 400.0
RADIUS, HEIGHT = 0.030, 0.100      # synthetic object: vertical cylinder on the disc


def look_at(C, target):
    f = (target - C) / np.linalg.norm(target - C)
    right = np.cross(f, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.stack([right, down, f])


def make_cameras(n_angles=12, heights=(0.05, 0.15), dist=0.35):
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    cams = []
    for h in heights:
        for a in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            C = np.array([dist * np.cos(a), h, dist * np.sin(a)])
            cams.append({"view_id": str(len(cams)), "path": f"f_{len(cams)}.jpg", "R": look_at(C, np.array([0, 0.05, 0])),
                         "C": C, "K": K, "dist": np.zeros(5), "K_new": K, "w": W, "h": H})
    return cams


def render(cam):
    """Mask (255 = object) and BGR image (green object on white booth) of the synthetic cylinder."""
    t = np.linspace(0, 2 * np.pi, 180)
    ring = [np.stack([RADIUS * np.cos(t), np.full_like(t, y), RADIUS * np.sin(t)], 1) for y in (0.0, HEIGHT)]
    uv, _, _ = vh.project_points(cam, np.vstack(ring), distorted=False)
    Xc = (np.vstack(ring) - cam["C"]) @ cam["R"].T
    uv = (Xc[:, :2] / Xc[:, 2:3]) * F + [W / 2, H / 2]
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 255)
    img = np.full((H, W, 3), 200, np.uint8)
    img[mask > 0] = (40, 150, 30)
    return mask, img


@unittest.skipUnless(HAS_CV2, "OpenCV required")
class TestVisualHull(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cams = make_cameras()
        self.masks, self.images = [], []
        for c in self.cams:
            mask, img = render(c)
            self.masks.append(str(self.dir / f"{c['view_id']}.mask.png"))
            self.images.append(str(self.dir / f"{c['view_id']}.png"))
            cv2.imwrite(self.masks[-1], mask)
            cv2.imwrite(self.images[-1], img)

    def tearDown(self):
        self.tmp.cleanup()

    def carve(self, distorted=False):
        return vh.carve_visual_hull(self.cams, self.masks, radius_m=0.06, y_min_m=0.003, y_max_m=0.15,
                                    voxel_m=0.003, distorted=distorted)

    def test_carved_hull_encloses_the_object_tightly(self):
        """The hull contains the cylinder and is only slightly larger (24 views), with or without distortion model."""
        for distorted in (False, True):
            stats = vh.hull_stats(self.carve(distorted))
            self.assertGreaterEqual(stats["radius_max_mm"], RADIUS * 1000 - 1.5)
            self.assertLess(stats["radius_max_mm"], RADIUS * 1000 + 6)
            self.assertGreaterEqual(stats["y_max_mm"], HEIGHT * 1000 - 3)
            self.assertLess(stats["y_max_mm"], HEIGHT * 1000 + 12)
            true_volume = np.pi * (RADIUS * 100) ** 2 * (HEIGHT - 0.003) * 100
            self.assertGreater(stats["volume_cm3"], 0.95 * true_volume)
            self.assertLess(stats["volume_cm3"], 1.5 * true_volume)

    def test_one_bad_mask_does_not_cut_the_object(self):
        """A voxel needs 2 background votes: an empty mask in one view leaves the hull intact."""
        cv2.imwrite(self.masks[3], np.zeros((H, W), np.uint8))
        stats = vh.hull_stats(self.carve())
        self.assertGreaterEqual(stats["radius_max_mm"], RADIUS * 1000 - 1.5)

    def test_proxy_bounds(self):
        """Hull box + margin, capped by the masking cylinder; the cylinder itself without a hull."""
        hull = self.carve()
        b = vh.proxy_bounds_from_hull(hull, cylinder_radius_m=0.08, cylinder_y_max_m=0.35, margin_m=0.01)
        self.assertAlmostEqual(b["y_max"], vh.hull_stats(hull)["y_max_mm"] / 1000 + 0.01, places=6)
        self.assertLess(b["r_max"], 0.08)
        self.assertEqual(vh.proxy_bounds_from_hull(None, 0.08, 0.35), {"y_max": 0.35, "r_max": 0.08})
        tight = vh.proxy_bounds_from_hull(hull, cylinder_radius_m=0.02, cylinder_y_max_m=0.05)
        self.assertEqual(tight, {"y_max": 0.05, "r_max": 0.02})

    def test_hull_save_load_roundtrip(self):
        hull = self.carve()
        path = str(self.dir / "hull.npz")
        vh.save_hull(hull, path)
        back = vh.load_hull(path)
        np.testing.assert_array_equal(back["occupancy"], hull["occupancy"])
        np.testing.assert_allclose(back["origin"], hull["origin"])

    # -- OpenMVS dense cloud ------------------------------------------------------------------

    def write_cloud(self, path, pts, n_views):
        """A scene_dense.ply in the DensifyPointCloud layout; point i is seen by views 0..n_views[i]-1."""
        empty = {"header": ["ply", "format binary_little_endian 1.0", "element vertex 0"] + vh.DENSE_PLY_PROPERTIES,
                 "body": b"", "count": 0, "n_views": np.zeros(0, np.int64)}
        r = np.hypot(pts[:, 0], pts[:, 2])[:, None]
        normals = np.hstack([pts[:, :1], np.zeros((len(pts), 1)), pts[:, 2:]]) / np.maximum(r, 1e-9)
        views = [np.arange(n, dtype=np.uint32) for n in n_views]
        weights = [np.full(n, 2.0, np.float32) for n in n_views]
        vh.write_dense_ply(path, empty, pts.astype(np.float32), np.full((len(pts), 3), 90, np.uint8),
                           normals.astype(np.float32), views, weights)

    def test_dense_ply_roundtrip_and_pruned_rewrite(self):
        """Records of variable length are walked correctly; a pruned rewrite keeps exactly the other records."""
        path = str(self.dir / "scene_dense.ply")
        pts = np.random.default_rng(1).uniform(-0.02, 0.02, (50, 3))
        n_views = np.arange(50) % 7 + 1
        self.write_cloud(path, pts, n_views)
        d = vh.read_dense_ply(path)
        self.assertEqual(d["count"], 50)
        np.testing.assert_allclose(d["xyz"], pts.astype(np.float32))
        np.testing.assert_array_equal(d["n_views"], n_views)
        np.testing.assert_allclose(d["weight_sum"], 2.0 * n_views)
        self.assertAlmostEqual(d["median_weight"], 2.0)

        keep = n_views > 3
        out = str(self.dir / "pruned.ply")
        vh.write_dense_ply(out, d, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8),
                           np.zeros((0, 3), np.float32), [], [], keep=keep)
        back = vh.read_dense_ply(out)
        self.assertEqual(back["count"], int(keep.sum()))
        np.testing.assert_allclose(back["xyz"], pts[keep].astype(np.float32))
        np.testing.assert_array_equal(back["n_views"], n_views[keep])

    def test_refine_prunes_wrong_depths_and_fills_bare_surface(self):
        """
        Side surface points up to 60 mm (well supported), nothing above (bare: like the uniform cap), wrong depths
        deep inside from 2 views (pruned) and a real concavity deep inside seen by 6 views (kept).
        """
        rng = np.random.default_rng(0)
        hull = self.carve()
        a = rng.uniform(0, 2 * np.pi, 4000)
        y = rng.uniform(0.003, 0.060, 4000)
        surface = np.stack([RADIUS * np.cos(a), y, RADIUS * np.sin(a)], 1)
        wrong = np.stack([np.full(300, 0.005), rng.uniform(0.07, 0.09, 300), np.zeros(300)], 1)
        concave = np.stack([np.full(100, -0.008), rng.uniform(0.03, 0.04, 100), np.zeros(100)], 1)
        pts = np.vstack([surface, wrong, concave])
        n_views = np.concatenate([np.full(4000, 5), np.full(300, 2), np.full(100, 6)])
        ply = str(self.dir / "scene_dense.ply")
        self.write_cloud(ply, pts, n_views)

        stats = vh.refine_dense_cloud_with_hull(ply, hull, self.cams, self.images, self.masks, min_y_m=0.005)
        self.assertEqual(stats["pruned_inside"], 300)
        self.assertGreaterEqual(stats["deep_inside_kept"], 100)
        self.assertGreater(stats["added"], 50)

        out = vh.read_dense_ply(ply)
        added_y = out["xyz"][4100:, 1]
        self.assertGreater(np.mean(added_y > 0.055), 0.95)   # where the surface is bare (voxel corners aside)
        self.assertEqual(out["count"], 4400 - 300 + stats["added"])
        self.assertTrue(Path(self.dir / "scene_dense_mvs.ply").is_file())
        # Appended points carry valid view indices of views that see them
        raw = np.frombuffer(out["body"], np.uint8)
        for o in out["offsets"][4100:4110]:
            nv = raw[o + 27]
            idx = raw[o + 28:o + 28 + 4 * nv].view(np.uint32)
            self.assertGreaterEqual(nv, 2)
            self.assertTrue(np.all(idx < len(self.cams)))

        # A rerun starts again from the plain MVS cloud: same result, nothing added twice
        again = vh.refine_dense_cloud_with_hull(ply, hull, self.cams, self.images, self.masks, min_y_m=0.005)
        self.assertEqual(again["written_points"], stats["written_points"])

        # Restore puts the plain cloud back
        self.assertTrue(vh.restore_dense_cloud(ply))
        self.assertEqual(vh.read_dense_ply(ply)["count"], 4400)

    def test_fill_rejects_white_background(self):
        """Hull surface whose pixels show the white booth (e.g. inside a closed handle) gets no points."""
        hull = self.carve()
        for p in self.images:
            cv2.imwrite(p, np.full((H, W, 3), 200, np.uint8))     # every view: white background only
        ply = str(self.dir / "scene_dense.ply")
        self.write_cloud(ply, np.array([[0.0, 0.05, RADIUS]]), [5])
        stats = vh.refine_dense_cloud_with_hull(ply, hull, self.cams, self.images, [None] * len(self.cams),
                                                prune_depth_m=0.0)
        self.assertEqual(stats["added"], 0)
        self.assertGreater(stats["rejected_background"], 0)
        self.assertEqual(vh.read_dense_ply(ply)["count"], 1)
        self.assertFalse(Path(self.dir / "scene_dense_mvs.ply").exists())


if __name__ == "__main__":
    unittest.main()
