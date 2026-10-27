"""
test_run_metrics.py — stats.json features for RAM / time / quality models: log parsers, image and mesh
metrics, telemetry step fields and the CSV collector.
"""

import csv
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
import trimesh

from pipeline import run_metrics as rm
from pipeline.telemetry import SystemTelemetry

DENSIFY_LOG = """\
22:21:14 [App     ] Depth-map for image  50 estimated using  6 images: 2028x1520 (1s459ms)
22:21:15 [App     ] Depth-map for image  90 estimated using  6 images: 2028x1520 (352ms)
22:34:39 [App     ] Depth map  29 filtered using 8 other images: 12917/135301 depths discarded (340ms)
22:34:39 [App     ] Depth map  59 filtered using 8 other images: 22061/119497 depths discarded (458ms)
22:39:26 [App     ] Depth-maps fused and filtered: 120 depth-maps, 11844454 depths, 2295091 points (19%%) (1m58s995ms)
22:39:29 [App     ] Densifying point-cloud completed: 2259040 points (18m18s633ms)
"""
RECONSTRUCT_LOG = """\
22:39:57 [App     ] Delaunay tetrahedralization completed: 2259040 points -> 2177156 vertices, 14315152 (+406) cells, 28630507 (+609) faces (14s664ms)
22:41:44 [App     ] Mesh reconstruction completed: 582119 vertices, 1164164 faces (2m2s316ms)
"""
REPORT = """\
Detecție: 4.7 s; tag-uri pe poză: min 4, median 9, max 11; poze fără tag-uri: 0
Pas: 17.9992° (nominal 18.000°), sens +
Urcare: 84.91 mm/nivel, unghi față de axă 0.686°
poza                         tag  RMS(px)  dif.liberă(mm/°)
0/f_1000.jpg                   9    6.673     1.00 / 0.55
0/f_2000.jpg                   9   13.103     3.00 / 0.61
1/f_2001.jpg                   9   12.000     5.00 / 0.61
Total: 3808 colțuri, RMS 3.108 px, maxim 9.370 px
"""


class TestRunMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_openmvs_logs(self):
        (self.tmp / "DensifyPointCloud-1.log").write_text(DENSIFY_LOG)
        (self.tmp / "ReconstructMesh-1.log").write_text(RECONSTRUCT_LOG)
        s = rm.openmvs_log_stats(str(self.tmp))
        self.assertEqual(s["densify"]["depth_map_estimations"], 2)
        self.assertAlmostEqual(s["densify"]["depth_map_gpu_seconds"], 1.8, places=1)
        self.assertEqual(s["densify"]["dense_points"], 2259040)
        self.assertEqual(s["densify"]["fused_points"], 2295091)
        self.assertAlmostEqual(s["densify"]["openmvs_seconds"], 1098.6, places=1)
        self.assertAlmostEqual(s["densify"]["depths_discarded_fraction"], (12917 + 22061) / (135301 + 119497), places=3)
        self.assertEqual(s["reconstruct"]["mesh_faces"], 1164164)
        self.assertEqual(s["reconstruct"]["tetra_cells"], 14315152)

    def test_pose_quality(self):
        (self.tmp / "r.txt").write_text(REPORT, encoding="utf-8")
        q = rm.pose_quality(str(self.tmp / "r.txt"))
        self.assertEqual((q["tags_min"], q["tags_median"]), (4, 9))
        self.assertEqual((q["rms_px"], q["lift_mm"]), (3.108, 84.91))
        self.assertEqual(q["free_diff_mm_per_level"], {"1": 1.0, "2": 4.0})

    def test_object_image_stats(self):
        img = np.full((400, 400, 3), 0, np.uint8)
        img[100:300, 150:250] = (40, 40, 200)
        img[150:160, 150:250] = 255                     # white label on the object
        mask = np.zeros((400, 400), np.uint8)
        mask[100:300, 150:250] = 255
        cv2.imwrite(str(self.tmp / "1000_f_1000.jpg"), img)
        cv2.imwrite(str(self.tmp / "1000_f_1000.mask.png"), mask)
        cv2.imwrite(str(self.tmp / "1000_f_1000.jpg.mask.png"), mask)  # alias: not counted twice
        s = rm.object_image_stats(str(self.tmp), scale=1.0)
        self.assertEqual(s["sampled_views"], 1)
        self.assertAlmostEqual(s["foreground_fraction"]["mean"], 0.125, places=3)
        self.assertAlmostEqual(s["bbox_height_fraction"]["mean"], 0.5, places=2)
        self.assertGreater(s["white_or_specular_fraction"]["mean"], 0.02)

    def test_mesh_quality(self):
        a = trimesh.creation.box()
        b = trimesh.creation.box()
        b.apply_translation([5, 0, 0])
        open_mesh = trimesh.util.concatenate([a, b])
        open_mesh.faces = open_mesh.faces[:-1]          # one hole
        open_mesh.export(self.tmp / "m.obj")
        q = rm.mesh_quality(str(self.tmp / "m.obj"), "x")
        self.assertEqual(q["components"], 2)
        self.assertEqual(q["boundary_edges"], 3)
        self.assertFalse(q["watertight"])

    def test_metrics_never_raise(self):
        self.assertIsNone(rm.pose_quality(str(self.tmp / "missing.txt")))
        self.assertIsNone(rm.mesh_quality(str(self.tmp / "missing.obj")))
        self.assertEqual(rm.openmvs_log_stats(str(self.tmp)), {})

    def test_disk_benchmark_and_versions(self):
        d = rm.disk_benchmark(str(self.tmp), size_mb=8)
        self.assertGreater(d["write_mb_s"], 0)
        self.assertGreater(d["read_mb_s"], 0)
        self.assertEqual(list(self.tmp.iterdir()), [])  # probe removed
        self.assertIn("git_commit", rm.code_version())
        self.assertIn("logical_cpus", rm.cpu_info())

    def test_input_stats(self):
        views = []
        for lvl in ("1", "2"):
            for cam in ("0", "1"):
                p = self.tmp / "rig" / cam / f"f_{lvl}000.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x" * 2048)
                views.append({"path": str(p), "intrinsicId": cam})
        sfm = {"views": views, "intrinsics": [{"width": 4056, "height": 3040}]}
        (self.tmp / "c.sfm").write_text(json.dumps(sfm))
        s = rm.input_stats(str(self.tmp / "c.sfm"))
        self.assertEqual((s["n_images"], s["n_levels"], s["n_cameras"]), (4, 2, 2))
        self.assertEqual(s["images_per_level"], {"1": 2, "2": 2})
        self.assertAlmostEqual(s["jpeg_mean_kb"], 2.0)


class TestTelemetryFields(unittest.TestCase):
    def test_step_cpu_disk_and_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = SystemTelemetry(output_dir=tmp)
            t.watch(tmp)
            with t.step("busy"):
                (Path(tmp) / "f.bin").write_bytes(b"0" * (2 * 1024 * 1024))
                end = time.time() + 0.3
                while time.time() < end:
                    pass
            t.set_outcome("ok")
            data = t.summary()
        step = data["steps"][0]
        self.assertGreater(step["cpu_seconds"], 0.1)
        self.assertGreater(step["cpu_utilization"], 0.0)
        self.assertAlmostEqual(step["peak_disk_gb"], 2 / 1024, places=3)
        self.assertEqual(data["outcome"], {"state": "ok"})
        self.assertEqual(data["schema_version"], rm.STATS_SCHEMA_VERSION)

    def test_failed_step_is_remembered(self):
        t = SystemTelemetry()
        with self.assertRaises(RuntimeError):
            with t.step("4.2 OpenMVS DensifyPointCloud"):
                raise RuntimeError("boom")
        self.assertEqual(t.current_step, "4.2 OpenMVS DensifyPointCloud")
        with t.step("ok step"):
            pass
        self.assertIsNone(t.current_step)


class TestCollectStats(unittest.TestCase):
    def test_csv_rows(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        import collect_stats
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            a = {"run_id": "a", "outcome": {"state": "ok"}, "run_info": {"inputs": {"n_images": 120}},
                 "steps": [{"name": "4.2 OpenMVS DensifyPointCloud", "duration_sec": 1109.0, "peak_ram_mb": 11305}],
                 "exported_models": [{"file_path": "high_model.glb", "face_count": 250000}]}
            b = {"run_id": "b", "outcome": {"state": "error", "failed_step": "x"},
                 "run_info": {"log_counters": {"texrecon_unseen_faces": [10, 5]}}, "steps": []}
            (tmp / "a.json").write_text(json.dumps(a))
            (tmp / "b.json").write_text(json.dumps(b))
            self.assertEqual(collect_stats.main([str(tmp / "a.json"), str(tmp / "b.json"), "-o", str(tmp / "r.csv")]), 0)
            rows = list(csv.DictReader(open(tmp / "r.csv", encoding="utf-8")))
        self.assertEqual(rows[0]["run_info.inputs.n_images"], "120")
        self.assertEqual(rows[0]["step.4_2_openmvs_densifypointcloud.peak_ram_mb"], "11305")
        self.assertEqual(rows[0]["model.high_model_glb.face_count"], "250000")
        self.assertEqual(rows[1]["outcome.failed_step"], "x")
        self.assertEqual(rows[1]["run_info.log_counters.texrecon_unseen_faces.sum"], "15")
        self.assertEqual(rows[1]["run_info.inputs.n_images"], "")


if __name__ == "__main__":
    unittest.main()
