"""
test_mesh_cleaner.py — Unit test for watertight pedestal base sealing,
2D Ear-Clipping triangulation, and normal orientation.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure turntable_studio root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from pipeline.config import MeshCleaningConfig
from pipeline.mesh_cleaner import (
    clean_and_repair_mesh,
    face_target,
    seal_watertight_pedestal,
    triangulate_2d_polygon,
)


class TestMeshCleaner(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_2d_ear_clipping_circle(self):
        # 16-sided regular polygon in 2D
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        coords = np.column_stack([np.cos(angles), np.sin(angles)])
        triangles = triangulate_2d_polygon(coords)

        # A 16-sided polygon must produce exactly 16 - 2 = 14 triangles
        self.assertEqual(len(triangles), 14)
        self.assertEqual(triangles.shape[1], 3)

    def test_seal_watertight_pedestal(self):
        # Create cylinder with height along Y axis, crossing the slice plane
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        cylinder = trimesh.creation.cylinder(radius=0.03, height=0.10, sections=24, transform=R)
        # Shift so it rests with bottom at Y = -0.02 and top at Y = 0.08
        cylinder.apply_translation([0.0, 0.03, 0.0])

        # Slicing at slice_y = 0.003 and extruding skirt down to skirt_y = 0.000
        sealed = seal_watertight_pedestal(
            input_mesh=cylinder,
            slice_y=0.003,
            skirt_y=0.000,
            max_radius=0.10,
        )

        self.assertIsNotNone(sealed)
        self.assertGreater(len(sealed.faces), 0)

        # Check lowest Y coordinate is exactly 0.000 m
        min_y = float(sealed.vertices[:, 1].min())
        self.assertAlmostEqual(min_y, 0.000, places=3)

        # Check watertightness
        self.assertTrue(sealed.is_watertight)
        self.assertGreater(sealed.volume, 0.0)

    def test_seal_watertight_pedestal_no_skirt(self):
        """Verifies direct planar capping without vertical skirt extrusion."""
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        cylinder = trimesh.creation.cylinder(radius=0.03, height=0.10, sections=24, transform=R)
        cylinder.apply_translation([0.0, 0.03, 0.0])

        sealed = seal_watertight_pedestal(
            input_mesh=cylinder,
            slice_y=0.001,
            skirt_y=0.001,
            max_radius=0.10,
            extrude_skirt=False,
            base_color_rgb=(25, 25, 25),
        )

        self.assertIsNotNone(sealed)
        self.assertTrue(sealed.is_watertight)
        self.assertGreater(sealed.volume, 0.0)
        # Lowest point must be exactly at slice_y = 0.001
        min_y = float(sealed.vertices[:, 1].min())
        self.assertAlmostEqual(min_y, 0.001, places=3)

    def test_full_clean_and_repair_flow(self):
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        cylinder = trimesh.creation.cylinder(radius=0.025, height=0.08, sections=16, transform=R)
        cylinder.apply_translation([0.0, 0.03, 0.0])

        in_obj = Path(self.test_dir) / "test_raw.obj"
        out_obj = Path(self.test_dir) / "test_cleaned.obj"
        cylinder.export(str(in_obj))

        config = MeshCleaningConfig(
            slice_y_m=0.003,
            skirt_bottom_y_m=0.000,
            max_radius_m=0.10,
        )

        res = clean_and_repair_mesh(
            input_obj_path=str(in_obj),
            output_obj_path=str(out_obj),
            config=config,
        )

        self.assertTrue(out_obj.is_file())
        self.assertGreater(res["vertex_count"], 0)
        self.assertGreater(res["face_count"], 0)
        self.assertTrue(res["is_watertight"])

    def test_high_poly_conditioning_and_normals(self):
        """Verifies high-poly decimation, Taubin smoothing, and vertex normal unitization."""
        R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        cylinder = trimesh.creation.cylinder(radius=0.03, height=0.10, sections=40, transform=R)
        cylinder.apply_translation([0.0, 0.03, 0.0])

        in_obj = Path(self.test_dir) / "dense_raw.obj"
        out_obj = Path(self.test_dir) / "dense_cleaned.obj"
        cylinder.export(str(in_obj))

        config = MeshCleaningConfig(
            slice_y_m=0.001,
            skirt_bottom_y_m=0.001,
            extrude_skirt=False,
            target_high_poly_faces=100,
            high_poly_smooth_iterations=3,
        )

        res = clean_and_repair_mesh(
            input_obj_path=str(in_obj),
            output_obj_path=str(out_obj),
            config=config,
        )

        self.assertTrue(out_obj.is_file())
        loaded = trimesh.load(str(out_obj), process=False)
        # Verify normals are unit length (||n|| ≈ 1.0)
        if hasattr(loaded, "vertex_normals") and len(loaded.vertex_normals) > 0:
            lengths = np.linalg.norm(loaded.vertex_normals, axis=1)
            valid = lengths > 1e-4
            if np.any(valid):
                np.testing.assert_allclose(lengths[valid], 1.0, atol=1e-2)


    def test_pieces_left_after_the_cut_are_dropped(self):
        """Disc rim pieces rising above the cut become separate components: only the object is kept (G4)."""
        obj = trimesh.creation.cylinder(radius=0.03, height=0.10)
        obj.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        obj.apply_translation([0.0, 0.048, 0.0])                        # y from -0.002 to 0.098
        rim = trimesh.creation.box(extents=[0.01, 0.010, 0.02])
        rim.apply_translation([0.115, 0.003, 0.0])                     # y from -0.002 to 0.008, radius ~115 mm
        both = trimesh.util.concatenate([obj, rim])

        sealed = seal_watertight_pedestal(both, slice_y=0.003, skirt_y=0.003, max_radius=0.12)
        self.assertEqual(len(sealed.split(only_watertight=False)), 1)
        self.assertLess(np.hypot(sealed.vertices[:, 0], sealed.vertices[:, 2]).max(), 0.035)
        self.assertTrue(sealed.is_watertight)

    def test_high_poly_decimation_after_closing_a_hole(self):
        """close_holes used to leave its new faces selected and the quadric decimation then did nothing (G5)."""
        sphere = trimesh.creation.icosphere(subdivisions=6, radius=0.04)
        sphere.apply_translation([0.0, 0.06, 0.0])
        sphere.update_faces(np.arange(len(sphere.faces)) != 7)         # one small hole for close_holes
        sphere.remove_unreferenced_vertices()
        in_path = Path(self.test_dir) / "holed.obj"
        out_path = Path(self.test_dir) / "clean.obj"
        sphere.export(str(in_path))

        config = MeshCleaningConfig(slice_y_m=0.003, skirt_bottom_y_m=0.003, target_high_poly_faces=5000)
        res = clean_and_repair_mesh(str(in_path), str(out_path), config)
        self.assertLessEqual(res["face_count"], 5500)
        self.assertTrue(res["is_watertight"])


    def test_face_target_ratio_and_fixed(self):
        """Decimation targets are a share of the mesh; a fixed target > 0 overrides the ratio."""
        self.assertEqual(face_target(800000, 0.8), 640000)
        self.assertEqual(face_target(800000, 0.2), 160000)
        self.assertEqual(face_target(800000, 0.8, fixed=250000), 250000)
        self.assertEqual(face_target(2000, 0.1), 1000)          # never below the minimum
        self.assertEqual(face_target(500, 0.1), 500)            # never above the mesh itself
        self.assertEqual(face_target(800000, 0.2, maximum=50000), 50000)    # low poly: min(20 %, 50 000)
        self.assertEqual(face_target(100000, 0.2, maximum=50000), 20000)

    def test_binary_copies_equal_what_the_obj_gives(self):
        """B5: <name>.ply = texrecon's own OBJ -> PLY conversion (bytes), <name>.npz = the OBJ vertices / faces."""
        from pipeline.config import TexReconConfig
        from pipeline.texrecon_runner import TexReconRunner

        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
        sphere.apply_translation([0.0, 0.06, 0.0])
        in_path = Path(self.test_dir) / "sphere.obj"
        out_path = Path(self.test_dir) / "clean.obj"
        sphere.export(str(in_path))
        res = clean_and_repair_mesh(str(in_path), str(out_path),
                                    MeshCleaningConfig(slice_y_m=0.003, skirt_bottom_y_m=0.003), binary_copies=True)
        self.assertTrue(Path(res["ply_path"]).is_file())
        conv_dir = Path(self.test_dir) / "tr"
        runner = TexReconRunner(TexReconConfig(), str(conv_dir))
        (conv_dir / "scene").mkdir(parents=True)
        runner.run_texturing("unused.sfm", str(out_path), self.test_dir, out_prefix=str(conv_dir / "m"),
                             mock_mode=True, scene_ready=True)
        self.assertEqual((conv_dir / "clean.ply").read_bytes(), Path(res["ply_path"]).read_bytes())
        ref = trimesh.load(str(out_path), process=False, force="mesh")
        arr = np.load(res["arrays_path"])
        self.assertTrue(np.array_equal(arr["vertices"], np.asarray(ref.vertices, np.float64)))
        self.assertTrue(np.array_equal(arr["faces"], np.asarray(ref.faces, np.int64)))

    def test_high_poly_ratio_decimates_every_mesh(self):
        """With the default ratio a mesh of any size loses ~25 % of its faces (no fixed cap left it untouched)."""
        sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.04)
        sphere.apply_translation([0.0, 0.06, 0.0])
        in_path = Path(self.test_dir) / "sphere.obj"
        out_path = Path(self.test_dir) / "clean.obj"
        sphere.export(str(in_path))
        res = clean_and_repair_mesh(str(in_path), str(out_path), MeshCleaningConfig(slice_y_m=0.003, skirt_bottom_y_m=0.003))
        self.assertEqual(res["sealed_faces"], len(sphere.faces))
        self.assertAlmostEqual(res["face_count"] / len(sphere.faces), 0.75, delta=0.01)


    # ------------------------------------------------------------------ closed models (01.11 audit, G1-G3, P7)

    def _clean(self, mesh, name, **cfg):
        in_path = Path(self.test_dir) / f"{name}.obj"
        out_path = Path(self.test_dir) / f"{name}_clean.obj"
        mesh.export(str(in_path))
        config = MeshCleaningConfig(slice_y_m=0.003, skirt_bottom_y_m=0.003, **cfg)
        res = clean_and_repair_mesh(str(in_path), str(out_path), config)
        return res, trimesh.load(str(out_path), process=False)

    def test_ring_foot_is_capped_as_a_polygon_with_a_hole(self):
        """A mug on its rim: two nested loops in the cut plane -> one annulus, not only the largest loop (G1)."""
        ring = trimesh.creation.annulus(r_min=0.02, r_max=0.03, height=0.06, sections=48)
        ring.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        ring.apply_translation([0.0, 0.028, 0.0])                         # y from -0.002 to 0.058
        report = {}
        sealed = seal_watertight_pedestal(ring, slice_y=0.003, skirt_y=0.003, max_radius=0.12, report=report)
        self.assertEqual(report["base"]["polygons"], 1)
        self.assertEqual(report["base"]["holes_in_polygons"], 1)
        self.assertTrue(sealed.is_watertight)
        # the cap is the annulus: its area is pi (R^2 - r^2), not pi R^2
        bottom = sealed.faces[np.all(np.abs(sealed.vertices[sealed.faces][:, :, 1] - 0.003) < 1e-6, axis=1)]
        area = trimesh.triangles.area(sealed.vertices[bottom]).sum()
        self.assertAlmostEqual(area, np.pi * (0.03 ** 2 - 0.02 ** 2), delta=0.1 * np.pi * (0.03 ** 2 - 0.02 ** 2))

    def test_three_legs_are_all_capped(self):
        legs = []
        for a in (0.0, 2.1, 4.2):
            leg = trimesh.creation.cylinder(radius=0.006, height=0.05, sections=24)
            leg.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            leg.apply_translation([0.03 * np.cos(a), 0.023, 0.03 * np.sin(a)])
            legs.append(leg)
        report = {}
        sealed = seal_watertight_pedestal(trimesh.util.concatenate(legs), slice_y=0.003, skirt_y=0.003,
                                          max_radius=0.12, report=report)
        self.assertEqual(report["base"]["polygons"], 3)
        self.assertTrue(sealed.is_watertight)

    def test_skirt_under_every_loop_stays_watertight(self):
        ring = trimesh.creation.annulus(r_min=0.02, r_max=0.03, height=0.06, sections=32)
        ring.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        ring.apply_translation([0.0, 0.028, 0.0])
        sealed = seal_watertight_pedestal(ring, slice_y=0.003, skirt_y=0.0, max_radius=0.12)
        self.assertTrue(sealed.is_watertight)
        self.assertAlmostEqual(float(sealed.vertices[:, 1].min()), 0.0, places=6)
        self.assertGreater(sealed.volume, 0.0)

    def test_hole_reaching_the_cut_is_closed_before_it(self):
        """A hole touching the cut plane used to open the base outline into a path that earcut closed with a chord (G2)."""
        cyl = trimesh.creation.cylinder(radius=0.03, height=0.06, sections=48)
        cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        cyl.apply_translation([0.0, 0.028, 0.0])
        cyl = cyl.subdivide_to_size(0.003)
        c = cyl.triangles_center
        ang = np.arctan2(c[:, 2], c[:, 0])
        side = np.hypot(c[:, 0], c[:, 2]) > 0.029
        hole = side & (np.abs(ang) < 0.4) & (c[:, 1] < 0.012)
        cyl.update_faces(~hole)
        cyl.remove_unreferenced_vertices()
        res, out = self._clean(cyl, "g2", target_high_poly_faces=0, high_poly_ratio=1.0)
        self.assertTrue(res["is_watertight"])
        self.assertTrue(out.is_watertight)
        self.assertAlmostEqual(float(out.vertices[:, 1].min()), 0.003, delta=0.0005)    # flat base kept
        self.assertGreater(res["base_band"].get("faces_added", 0), 0)

    def test_small_object_beside_a_large_disc_is_kept(self):
        """The orange piece 094944: the disc was the largest piece and the object was thrown away (P7)."""
        disc = trimesh.creation.cylinder(radius=0.127, height=0.003, sections=128)
        disc.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        disc.apply_translation([0.0, -0.0005, 0.0])                       # y from -2 to 1 mm: under the cut
        disc = disc.subdivide_to_size(0.01)
        cube = trimesh.creation.box(extents=[0.012, 0.012, 0.012])
        cube.apply_translation([0.01, 0.005, 0.0])                        # y from -1 to 11 mm
        self.assertGreater(len(disc.faces), 10 * len(cube.faces))
        res, out = self._clean(trimesh.util.concatenate([disc, cube]), "orange")
        self.assertLess(max(out.extents), 0.02)
        self.assertGreater(min(out.extents[[0, 2]]), 0.010)
        self.assertTrue(res["is_watertight"])
        self.assertEqual(res["components"]["kept"], 1)
        self.assertNotIn("model_tiny", [w["code"] for w in res["warnings"]])

    def test_large_hole_is_filled_once_on_the_high_poly(self):
        sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.04)
        sphere.apply_translation([0.0, 0.06, 0.0])
        top = sphere.triangles_center[:, 1] > 0.093                      # a cap of ~300 faces, ~100 border edges
        sphere.update_faces(~top)
        sphere.remove_unreferenced_vertices()
        res, out = self._clean(sphere, "holed_top")
        self.assertGreater(int(top.sum()), 100)
        self.assertTrue(res["is_watertight"])
        self.assertEqual(res["holes"]["boundary_edges_after"], 0)
        self.assertGreater(res["patched_area_cm2"], 1.0)
        # metres in, metres out (PyMeshLab works in mm inside)
        self.assertAlmostEqual(float(out.extents[0]), 0.08, delta=0.004)
        self.assertEqual(res["warnings"], [])

    def test_warnings_for_a_lost_object(self):
        speck = trimesh.creation.box(extents=[0.004, 0.004, 0.004])
        speck.apply_translation([0.0, 0.004, 0.0])
        res, _ = self._clean(speck, "speck")
        self.assertIn("model_tiny", [w["code"] for w in res["warnings"]])

    def test_split_pinch_vertices_are_separated(self):
        """Two cubes touching at one corner: the repair splits the pinch into coincident copies an STL merges back."""
        import pymeshlab
        from pipeline.mesh_cleaner import _separate_coincident_vertices

        a = trimesh.creation.box(extents=[10, 10, 10])
        b = trimesh.creation.box(extents=[10, 10, 10])
        b.apply_translation([10, 10, 10])                     # shares the corner (5, 5, 5), as separate vertices
        both = trimesh.util.concatenate([a, b])
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=both.vertices, face_matrix=both.faces.astype(np.int32)))
        moved = _separate_coincident_vertices(ms)
        self.assertEqual(moved, 2)
        V = ms.current_mesh().vertex_matrix()
        self.assertEqual(len(np.unique(np.round(V, 6), axis=0)), len(V))
        merged = trimesh.Trimesh(V, ms.current_mesh().face_matrix(), process=True)
        self.assertTrue(merged.is_watertight)
        self.assertLess(np.abs(V - both.vertices).max(), 0.021)


if __name__ == "__main__":
    unittest.main()
