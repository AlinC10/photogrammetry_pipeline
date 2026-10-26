"""
test_texture_prep.py — Unit tests for the texturing images rebuilt after mesh cleaning.
Verifies:
  1. Mesh silhouette rasterizer (area of a square, z-buffer ordering).
  2. Lossless PNG texturing images: background EXACTLY RGB(0,0,0), object never 0, so texrecon's
     corner flood fill (generate_validity_mask) invalidates the whole background and nothing else.
  3. The JPEG re-encoding it replaces breaks that flood fill.
  4. Background trimming only removes white pixels connected to the outside (no holes).
  5. Per-camera colour gains from the turntable paper.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure turntable_studio root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import trimesh

from pipeline.mesh_masks import render_face_ids, render_silhouette
from pipeline.texture_prep import (
    compute_camera_gains,
    prepare_texturing_images,
    texrecon_validity_mask,
    trim_background_band,
)


def _look_at(C, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera rotation (OpenCV axes: x right, y down, z forward)."""
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


class TestMeshRasterizer(unittest.TestCase):

    def test_square_silhouette_area(self):
        """A 40 mm square facing a 1000 px focal camera at 0.4 m covers 100 x 100 px."""
        V = np.array([[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0], [0.02, 0.02, 0.0], [-0.02, 0.02, 0.0]])
        F = np.array([[0, 1, 2], [0, 2, 3]])
        R = np.eye(3)
        C = np.array([0.0, 0.0, -0.4])
        K = np.array([[1000.0, 0, 199.5], [0, 1000.0, 149.5], [0, 0, 1]])
        sil = render_silhouette(V, F, R, C, K, 400, 300)
        self.assertEqual(int((sil > 0).sum()), 100 * 100)

    def test_zbuffer_keeps_front_face(self):
        near = np.array([[-0.01, -0.01, 0.0], [0.01, -0.01, 0.0], [0.0, 0.01, 0.0]])
        far = near * 3 + np.array([0.0, 0.0, 0.1])
        V = np.vstack([far, near])
        F = np.array([[0, 2, 1], [3, 5, 4]])  # both facing the camera (normal towards -Z)
        R = np.eye(3)
        C = np.array([0.0, 0.0, -0.3])
        K = np.array([[800.0, 0, 100.0], [0, 800.0, 100.0], [0, 0, 1]])
        fid, depth = render_face_ids(V, F, R, C, K, 200, 200, cull_backfaces=False)
        self.assertEqual(int(fid[100, 100]), 1)
        self.assertAlmostEqual(float(depth[100, 100]), 0.3, places=4)


class TestTexturingImages(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.w, self.h = 320, 240
        self.f = 400.0

        # Object: a box standing on the turntable plane
        box = trimesh.creation.box(extents=(0.06, 0.10, 0.06))
        box.apply_translation((0.0, 0.05, 0.0))
        self.mesh_path = self.root / "model_cleaned_watertight.obj"
        box.export(str(self.mesh_path))

        self.C = np.array([0.0, 0.08, -0.35])
        self.R = _look_at(self.C, (0.0, 0.05, 0.0))
        K = np.array([[self.f, 0, self.w / 2.0], [0, self.f, self.h / 2.0], [0, 0, 1]])

        # Photo: white booth, grey object with a white label inside (must stay on the object)
        sil = render_silhouette(box.vertices, box.faces, self.R, self.C, K, self.w, self.h)
        photo = np.full((self.h, self.w, 3), 235, np.uint8)
        photo[sil > 0] = (90, 110, 100)
        ys, xs = np.nonzero(sil)
        cy, cx = int(ys.mean()), int(xs.mean())
        photo[cy - 6:cy + 6, cx - 6:cx + 6] = 245
        self.sil = sil
        img_dir = self.root / "rig" / "0"
        img_dir.mkdir(parents=True)
        self.photo_path = img_dir / "f_1000.jpg"
        cv2.imwrite(str(self.photo_path), photo, [cv2.IMWRITE_JPEG_QUALITY, 95])

        sfm = {
            "version": ["1", "0", "0"],
            "views": [{"viewId": "1000", "poseId": "1000", "intrinsicId": "0", "path": str(self.photo_path),
                       "width": str(self.w), "height": str(self.h)}],
            "intrinsics": [{"intrinsicId": "0", "width": self.w, "height": self.h, "sensorWidth": 6.4,
                            "focalLength": self.f * 6.4 / self.w, "pixelRatio": 1.0,
                            "principalPoint": [0.0, 0.0], "distortionParams": [0.0, 0.0, 0.0]}],
            "poses": [{"poseId": "1000", "pose": {"transform": {
                "rotation": [float(x) for x in self.R.flatten(order="F")],
                "center": [float(x) for x in self.C]}}}],
        }
        self.sfm_path = self.root / "cameras_known.sfm"
        self.sfm_path.write_text(json.dumps(sfm))

    def tearDown(self):
        self.tmp.cleanup()

    def test_png_background_exactly_zero(self):
        out_dir = self.root / "texturing_images"
        summary = prepare_texturing_images(
            str(self.sfm_path), str(self.mesh_path), str(out_dir),
            margin_px=2, trim_band_px=6, colour_normalize=False, workers=1,
        )
        png = out_dir / "1000_f_1000.png"
        mask_p = out_dir / "1000_f_1000.mask.png"
        self.assertTrue(png.is_file())
        self.assertEqual(Path(summary["views"][0]["image"]).suffix, ".png")
        with open(png, "rb") as fh:
            self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")

        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        fg = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 0
        s = img.astype(np.int32).sum(axis=2)
        self.assertTrue(np.all(s[~fg] == 0), "background must be exactly RGB(0,0,0)")
        self.assertTrue(np.all(s[fg] > 0), "object pixels must never be exactly black")

        # texrecon flood fill: whole background invalid, whole object (minus 1 px erosion) valid
        valid = texrecon_validity_mask(img)
        self.assertFalse(valid[~fg].any())
        inner = cv2.erode(fg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
        self.assertTrue(valid[inner].all())

        # The white label inside the object is kept (no interior hole), the mask covers the silhouette
        self.assertGreaterEqual(int((fg & (self.sil > 0)).sum()), int(0.98 * (self.sil > 0).sum()))

    def test_jpeg_reencoding_breaks_validity(self):
        """The old path (black-out, then JPEG) leaves near-black ringing that texrecon treats as valid."""
        out_dir = self.root / "texturing_images"
        prepare_texturing_images(str(self.sfm_path), str(self.mesh_path), str(out_dir),
                                 colour_normalize=False, workers=1)
        img = cv2.imread(str(out_dir / "1000_f_1000.png"), cv2.IMREAD_COLOR)
        fg = cv2.imread(str(out_dir / "1000_f_1000.mask.png"), cv2.IMREAD_GRAYSCALE) > 0
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        jpg = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.assertGreater(int(texrecon_validity_mask(jpg)[~fg].sum()), 0)

    def test_trim_background_band_keeps_interior(self):
        img = np.full((100, 100, 3), 235, np.uint8)
        img[30:70, 30:70] = (60, 60, 60)        # object
        img[45:55, 45:55] = 250                 # white label inside the object
        mask = np.zeros((100, 100), np.uint8)
        mask[25:75, 25:75] = 255                # silhouette 5 px too large (pose error)
        trimmed = trim_background_band(img, mask, band_px=8, brightness_threshold=135, chroma_threshold=20)
        self.assertTrue(np.all(trimmed[30:70, 30:70] == 255), "object and its white label must stay")
        self.assertTrue(np.all(trimmed[25:30, 25:75] == 0), "white wall inside the silhouette is trimmed")

    def test_camera_gains_from_paper(self):
        gains = compute_camera_gains({
            "0": [np.array([156.0, 155.0, 157.0])],
            "1": [np.array([154.0, 158.0, 147.0])],
        })
        p0 = np.array([156.0, 155.0, 157.0]) * gains["0"]
        p1 = np.array([154.0, 158.0, 147.0]) * gains["1"]
        np.testing.assert_allclose(p0, p1, rtol=1e-6)
        np.testing.assert_allclose(p1, np.full(3, p1.mean()), rtol=1e-6)
        clamped = compute_camera_gains({"0": [np.array([100.0, 100.0, 100.0])],
                                        "1": [np.array([100.0, 100.0, 10.0])]}, max_deviation=0.25)
        self.assertLessEqual(float(clamped["1"].max()), 1.25 + 1e-9)


if __name__ == "__main__":
    unittest.main()
