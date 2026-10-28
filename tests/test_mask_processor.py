"""
test_mask_processor.py — Unit tests for automated background matting and cylinder ROI masking.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

from pipeline.mask_processor import (
    convex_hull_2d,
    project_cylinder_to_2d,
    generate_hybrid_mask,
    apply_mask_to_image,
    batch_generate_masks,
    load_tag_layout,
    project_tags_to_2d,
    tag_may_be_occluded,
)


class TestMaskProcessor(unittest.TestCase):

    def test_convex_hull_2d(self):
        """Verifies 2D convex hull of a square with interior points."""
        points = [
            (0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0),
            (5.0, 5.0), (2.0, 3.0), (8.0, 7.0),
        ]
        hull = convex_hull_2d(points)
        self.assertEqual(len(hull), 4)
        for corner in [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]:
            self.assertIn(corner, hull)

    def test_project_cylinder_to_2d(self):
        """Verifies 3D cylinder projection produces valid 2D polygon in front of camera."""
        # Camera at Z=0.4m looking towards origin [0,0,0] (rotation 180 deg around Y)
        R = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)
        C = np.array([0.0, 0.15, 0.40], dtype=np.float64)
        K = np.array([
            [2000.0, 0.0, 1000.0],
            [0.0, 2000.0, 1000.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        hull = project_cylinder_to_2d(
            R=R, C=C, K=K,
            radius_m=0.08,
            y_min_m=0.003,
            y_max_m=0.30,
            n_angles=32,
        )
        self.assertGreater(len(hull), 5)
        u_vals = [p[0] for p in hull]
        v_vals = [p[1] for p in hull]
        self.assertTrue(all(0.0 < u < 2000.0 for u in u_vals))
        self.assertTrue(all(-500.0 < v < 2500.0 for v in v_vals))

    @unittest.skipUnless(HAS_PIL, "PIL required for mask generation tests")
    def test_generate_hybrid_mask(self):
        """Verifies segmentation of colored foreground tool from bright white background."""
        w, h = 400, 300
        # Background: flat bright white (RGB 160, 160, 165)
        synth_img = np.full((h, w, 3), [160, 160, 165], dtype=np.uint8)

        # Foreground tool: green rectangle in the center (RGB 40, 140, 50)
        synth_img[80:240, 160:240] = [40, 140, 50]

        mask = generate_hybrid_mask(
            image=synth_img,
            chroma_threshold=20.0,
            brightness_white_threshold=130.0,
            safety_dilation_px=1,
            fill_holes=True,
        )

        self.assertEqual(mask.shape, (h, w))
        self.assertEqual(mask.dtype, np.uint8)

        # Center should be foreground (255)
        self.assertEqual(mask[160, 200], 255)
        # Corners should be background (0)
        self.assertEqual(mask[10, 10], 0)
        self.assertEqual(mask[290, 390], 0)

    def test_apply_mask_to_image(self):
        """Verifies zeroing out background pixels to pure black RGB(0,0,0)."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        masked = apply_mask_to_image(img, mask)
        self.assertEqual(masked[0, 0, 0], 0)
        self.assertEqual(masked[0, 0, 1], 0)
        self.assertEqual(masked[0, 0, 2], 0)
        self.assertEqual(masked[25, 25, 0], 200)

    @unittest.skipUnless(HAS_PIL, "PIL required for batch mask generation tests")
    def test_batch_generate_masks(self):
        """Verifies batch mask generation and file output with strict naming convention."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            img_path = tmp_p / "frame_0001.jpg"

            # Create synthetic test frame
            arr = np.full((100, 100, 3), 160, dtype=np.uint8)
            arr[30:70, 30:70] = [30, 150, 40]
            Image.fromarray(arr).save(str(img_path))

            views = [{
                "viewId": "1",
                "path": str(img_path),
                "poseId": "1",
                "intrinsicId": "1",
            }]
            poses_dict = {
                "1": {
                    "R": np.eye(3),
                    "C": np.array([0.0, 0.1, 0.3]),
                }
            }
            intrinsics_dict = {
                "1": {
                    "K": np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]]),
                }
            }

            out_masks_dir = tmp_p / "masks"
            mask_map = batch_generate_masks(
                views=views,
                poses_dict=poses_dict,
                intrinsics_dict=intrinsics_dict,
                output_masks_dir=out_masks_dir,
            )

            self.assertIn("frame_0001.jpg", mask_map)
            saved_mask = Path(mask_map["frame_0001.jpg"])
            self.assertTrue(saved_mask.is_file())
            self.assertTrue(saved_mask.name.endswith(".mask.png"))

    @unittest.skipUnless(HAS_PIL, "PIL required for batch mask generation tests")
    def test_batch_generate_masks_parallel_matches_serial(self):
        """Masks made by several worker processes are byte-identical to the serial ones; rig cameras do not collide."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            views = []
            for cam in ("0", "1"):
                (tmp_p / "rig" / cam).mkdir(parents=True)
                for k in range(3):
                    arr = np.full((100, 100, 3), 160, dtype=np.uint8)
                    arr[20 + 5 * k:70, 30:70 + 5 * int(cam)] = [30, 150, 40]
                    img = tmp_p / "rig" / cam / f"f_100{k}.jpg"
                    Image.fromarray(arr).save(str(img))
                    views.append({"viewId": f"{cam}{k}", "path": str(img), "poseId": "1", "intrinsicId": "1"})
            poses = {"1": {"R": np.eye(3), "C": np.array([0.0, 0.1, 0.3])}}
            intr = {"1": {"K": np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]])}}

            serial = batch_generate_masks(views, poses, intr, tmp_p / "serial", tag_blackout=False, workers=1)
            parallel = batch_generate_masks(views, poses, intr, tmp_p / "parallel", tag_blackout=False, workers=3)

            for v in views:  # the viewId-prefixed key is unique per camera
                self.assertIn(f"{v['viewId']}_{Path(v['path']).name}", parallel)
            for v in views:
                name = Path(v["path"]).stem
                a = (tmp_p / "serial" / f"{v['viewId']}_{name}.mask.png").read_bytes()
                b = (tmp_p / "parallel" / f"{v['viewId']}_{name}.mask.png").read_bytes()
                self.assertEqual(a, b)
            # No flat f_1000.mask.png alias: both cameras have an f_1000 and one would overwrite the other
            self.assertFalse((tmp_p / "parallel" / "f_1000.mask.png").exists())
            self.assertTrue((tmp_p / "parallel" / "0" / "f_1000.mask.png").is_file())
            self.assertTrue((tmp_p / "parallel" / "1" / "f_1000.mask.png").is_file())

    def test_load_tag_layout_and_project_tags_to_2d(self):
        """Verifies loading 12 AprilTags and projecting into camera view."""
        tags = load_tag_layout()
        self.assertEqual(len(tags), 12)
        self.assertEqual(tags[0]["id"], 0)

        # Camera at [0, 0.15, 0.40] looking toward origin
        R = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)
        C = np.array([0.0, 0.15, 0.40], dtype=np.float64)
        K = np.array([
            [2000.0, 0.0, 1000.0],
            [0.0, 2000.0, 1000.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        polys = project_tags_to_2d(tags, R, C, K, expansion_factor=1.25)
        self.assertEqual(len(polys), 12)
        for poly in polys:
            self.assertEqual(len(poly), 4)

    @unittest.skipUnless(HAS_PIL, "PIL required for tag blackout mask test")
    def test_generate_hybrid_mask_with_tag_blackout(self):
        """Verifies dark AprilTag region is strictly blacked out (0) in final mask."""
        w, h = 400, 300
        # Background: flat bright white (RGB 160, 160, 165)
        synth_img = np.full((h, w, 3), [160, 160, 165], dtype=np.uint8)

        # Foreground tool: green rectangle (RGB 40, 140, 50) inside fallback ROI (y: 15..216, x: 60..340)
        synth_img[50:200, 150:250] = [40, 140, 50]

        # Dark AprilTag overlapping base region: black square (RGB 30, 30, 30) at y: 150..190, x: 160..200
        synth_img[150:190, 160:200] = [30, 30, 30]

        tag_poly = [(160.0, 150.0), (200.0, 150.0), (200.0, 190.0), (160.0, 190.0)]

        # Mask without tag blackout: dark tag is considered foreground
        mask_no_bo = generate_hybrid_mask(
            image=synth_img,
            tag_polygons=[tag_poly],
            tag_blackout=False,
            brightness_white_threshold=130.0,
        )
        self.assertEqual(mask_no_bo[170, 180], 255)

        # Mask with tag blackout: dark tag region must be strictly 0 (background)
        mask_bo = generate_hybrid_mask(
            image=synth_img,
            tag_polygons=[tag_poly],
            tag_blackout=True,
            brightness_white_threshold=130.0,
        )
        self.assertEqual(mask_bo[170, 180], 0)
        # Foreground outside the tag must remain 255
        self.assertEqual(mask_bo[100, 200], 255)


    def test_tag_behind_the_object_is_not_blacked_out(self):
        """Only tags whose camera ray misses the masking cylinder may be blacked out (G1)."""
        C = np.array([0.0, 0.15, 0.40])

        def tag(z, x=0.0):
            return {"center": [x, 0.0, z], "corners": [[x - 0.01, 0.0, z - 0.01], [x + 0.01, 0.0, z - 0.01],
                                                       [x + 0.01, 0.0, z + 0.01], [x - 0.01, 0.0, z + 0.01]]}

        self.assertTrue(tag_may_be_occluded(tag(-0.10), C, 0.08, 0.001, 0.35))          # behind the object
        self.assertFalse(tag_may_be_occluded(tag(0.10), C, 0.08, 0.001, 0.35))          # in front of it
        self.assertFalse(tag_may_be_occluded(tag(-0.10, x=0.06), C, 0.02, 0.001, 0.35))  # ray passes beside a thin volume

    @unittest.skipUnless(HAS_PIL, "PIL required for mask hole test")
    def test_white_label_inside_the_object_stays_in_the_mask(self):
        """A white label fails the photometric test, but a hole enclosed by the object is filled (G3)."""
        img = np.full((300, 400, 3), [160, 160, 165], dtype=np.uint8)
        img[50:200, 150:250] = [40, 140, 50]
        img[90:150, 170:230] = [235, 235, 235]                          # white label
        filled = generate_hybrid_mask(image=img, brightness_white_threshold=130.0, fill_holes=True)
        self.assertEqual(filled[120, 200], 255)
        self.assertEqual(filled[20, 20], 0)
        holed = generate_hybrid_mask(image=img, brightness_white_threshold=130.0, fill_holes=False)
        self.assertEqual(holed[120, 200], 0)


if __name__ == "__main__":
    unittest.main()
