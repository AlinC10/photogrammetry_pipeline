"""
test_main_checks.py — Automatic checks of main.py: warning when a tag correction is implausible (object on the tags),
masking cylinder widened when the visual hull reaches it (not for a few stray voxels, not for the ambiguous cone),
warning for a hull that is only the cone, cleaning warnings in status.json, hole patches snapped onto the hull.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import main
from pipeline.config import PipelineConfig

REPORT_BAD = """\
Corecția tag-urilor (denivelări ale hârtiei, față de planul discului): 0: +2.43 mm, 1: -6.79 mm, 2: +0.17 mm
Total: 2111 colțuri, RMS 2.646 px, maxim 7.913 px
"""


class FakeTelemetry:
    def __init__(self):
        self.info = {}

    def set_info(self, **kw):
        self.info.update(kw)

    def add_warning(self, code, message, **data):
        self.info.setdefault("warnings", []).append({"code": code, "message": message, **data})


class TestMainChecks(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_tag_correction_over_limit_warns_and_keeps_poses(self):
        (self.out / "turntable_report.txt").write_text(REPORT_BAD, encoding="utf-8")
        config, tel = PipelineConfig(), FakeTelemetry()
        q = main.run_metrics.pose_quality(str(self.out / "turntable_report.txt"))
        with mock.patch.object(main, "run_turntable_poses") as run:
            main._check_tag_corrections(q, config, tel)
        run.assert_not_called()
        self.assertEqual(tel.info["tag_correction_warning"]["worst_correction_mm"], 6.79)

    def test_plausible_tag_corrections_no_warning(self):
        config, tel = PipelineConfig(), FakeTelemetry()
        main._check_tag_corrections({"tag_correction_max_mm": 1.49, "rms_px": 2.2}, config, tel)
        self.assertNotIn("tag_correction_warning", tel.info)

    def test_hull_touching_cylinder(self):
        config = PipelineConfig()
        v = 0.0015
        n = int(round(0.26 / v)) + 1
        occ = np.zeros((n, 20, n), bool)
        origin = np.array([-0.13, 0.003, -0.13])
        centre = int(round(0.13 / v))
        inside = lambda r: occ.__setitem__((slice(centre - int(r / v), centre + int(r / v) + 1), slice(None),
                                            slice(centre - 2, centre + 3)), True)
        inside(0.050)
        hull = {"occupancy": occ.copy(), "origin": origin, "voxel_m": v}
        self.assertFalse(main._hull_touches_cylinder(hull, config))
        inside(0.0795)                                    # reaches the 80 mm cylinder
        hull = {"occupancy": occ, "origin": origin, "voxel_m": v}
        self.assertTrue(main._hull_touches_cylinder(hull, config))
        config.masking.cylinder_radius_m = config.masking.wide_cylinder_radius_m
        self.assertFalse(main._hull_touches_cylinder(hull, config))   # already wide: only once

    def test_a_few_stray_voxels_or_the_cone_do_not_widen_the_cylinder(self):
        config = PipelineConfig()
        v = 0.0015
        n = int(round(0.16 / v)) + 1
        occ = np.zeros((n, 20, n), bool)
        origin = np.array([-0.08, 0.003, -0.08])
        c = int(round(0.08 / v))
        occ[c - 20:c + 20, :, c - 20:c + 20] = True
        occ[1, 5, c] = True                                 # one stray voxel at the wall
        hull = {"occupancy": occ, "origin": origin, "voxel_m": v}
        self.assertFalse(main._hull_touches_cylinder(hull, config))
        occ2 = occ.copy()
        occ2[0:3, :, c - 5:c + 5] = True                     # a real part of the object at the wall ...
        amb = np.zeros_like(occ2)
        amb[0:3, :, :] = True                                # ... but inside the ambiguous cone
        self.assertFalse(main._hull_touches_cylinder({"occupancy": occ2, "ambiguous": amb, "origin": origin,
                                                      "voxel_m": v}, config))
        self.assertTrue(main._hull_touches_cylinder({"occupancy": occ2, "origin": origin, "voxel_m": v}, config))

    def test_cone_only_hull_warns_and_uses_the_cylinder(self):
        config, tel = PipelineConfig(), FakeTelemetry()
        occ = np.zeros((40, 40, 40), bool)
        occ[10:30, 0:20, 10:30] = True
        hull = {"occupancy": occ, "ambiguous": occ.copy(), "origin": np.array([-0.03, 0.003, -0.03]),
                "voxel_m": 0.0015}
        main._check_hull_cone_only(hull, config, tel)
        self.assertEqual([w["code"] for w in tel.info["warnings"]], ["hull_cone_only"])
        b = main.visual_hull.proxy_bounds_from_hull(hull, 0.08, 0.35)
        self.assertEqual(b, {"y_max": 0.35, "r_max": 0.08})          # not the 30 mm of the cone (S2)
        hull["ambiguous"][:, 10:, :] = False                          # decided voxels above the cone
        tel2 = FakeTelemetry()
        main._check_hull_cone_only(hull, config, tel2)
        self.assertNotIn("warnings", tel2.info)
        self.assertLess(main.visual_hull.proxy_bounds_from_hull(hull, 0.08, 0.35)["y_max"], 0.1)

    def test_cleaning_warnings_reach_the_status(self):
        tel = FakeTelemetry()
        main._report_cleaning({"is_watertight": False, "warnings": [
            {"code": "not_watertight", "message": "m", "boundary_edges": 12}]}, None, tel)
        self.assertEqual(tel.info["warnings"][0]["code"], "not_watertight")
        self.assertFalse(tel.info["cleaning"]["is_watertight"])

    def test_hole_patch_snapped_onto_the_hull(self):
        """A flat patch over a hole in a sphere sags inside the hull; the hook moves it onto the hull (2.11)."""
        import trimesh
        from pipeline.config import MeshCleaningConfig
        from pipeline.mesh_cleaner import clean_and_repair_mesh

        R, cy = 0.04, 0.06
        v = 0.001
        g = np.arange(-0.05, 0.05 + 1e-9, v)
        X, Y, Z = np.meshgrid(g, g + cy, g, indexing="ij")
        hull = {"occupancy": X ** 2 + (Y - cy) ** 2 + Z ** 2 <= R ** 2,
                "origin": np.array([g[0], g[0] + cy, g[0]]), "voxel_m": v}
        sphere = trimesh.creation.icosphere(subdivisions=5, radius=R)
        sphere.apply_translation([0.0, cy, 0.0])
        cap = sphere.triangles_center[:, 0] > 0.034                   # hole on the side, ~20 mm across
        sphere.update_faces(~cap)
        sphere.remove_unreferenced_vertices()
        src = self.out / "holed.obj"
        sphere.export(str(src))
        cfg = MeshCleaningConfig(slice_y_m=0.003, skirt_bottom_y_m=0.003, high_poly_smooth_iterations=0)
        config = PipelineConfig()

        def patch_radius(hook):
            out = self.out / f"clean_{hook is not None}.obj"
            res = clean_and_repair_mesh(str(src), str(out), cfg, patch_hook=hook)
            m = trimesh.load(str(out), process=False)
            side = m.vertices[:, 0] > 0.036
            return res, np.linalg.norm(m.vertices[side] - [0, cy, 0], axis=1).mean() if side.any() else 0.0

        _, r_plain = patch_radius(None)
        hook = main._hull_patch_hook(hull, config)
        res, r_snap = patch_radius(hook)
        self.assertTrue(res["is_watertight"])
        self.assertGreater(hook.stats["vertices_snapped"], 0)
        self.assertGreater(r_snap, r_plain)
        self.assertAlmostEqual(r_snap, R, delta=0.0015)


if __name__ == "__main__":
    unittest.main()
