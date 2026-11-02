"""
test_parallel.py — Work beside the main steps: worker logs reach this process, the export with a process pool writes
the same deliverables as without, the texrecon scene is written once for concurrent runs, output.zip stores
already-compressed files.
"""

import logging
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

import trimesh
from PIL import Image

from pipeline.config import PipelineConfig, TexReconConfig
from pipeline.exporter import export_multi_lod_bundle, export_print_assets, package_output_zip
from pipeline.parallel import LoggingProcessPool
from pipeline.telemetry import SystemTelemetry, mesh_file_metrics
from pipeline.texrecon_runner import TexReconRunner


def _log_and_add(a, b):
    logging.getLogger("TurntableStudio.TestWorker").info(f"worker adds {a} + {b}")
    return a + b


def _fail():
    raise RuntimeError("boom")


def _textured_source(d: Path, name: str, colour: str) -> Path:
    Image.new("RGB", (64, 64), color=colour).save(str(d / f"{name}_map_Kd.png"))
    (d / f"{name}.mtl").write_text(f"newmtl m0\nKd 1 1 1\nmap_Kd {name}_map_Kd.png\n")
    box = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    lines = [f"mtllib {name}.mtl"] + [f"v {x} {y + 0.025} {z}" for x, y, z in box.vertices]
    lines += ["vt 0 0", "vt 1 0", "vt 0.5 1", "usemtl m0"]
    lines += [f"f {a + 1}/1 {b + 1}/2 {c + 1}/3" for a, b, c in box.faces]
    obj = d / f"{name}.obj"
    obj.write_text("\n".join(lines) + "\n")
    return obj


class TestParallel(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_pool_forwards_worker_logs_and_isolates_failures(self):
        with self.assertLogs("TurntableStudio.TestWorker", level="INFO") as logs:
            with LoggingProcessPool(2) as pool:
                pool.submit("add", _log_and_add, 2, 3)
                pool.submit("fail", _fail)
                self.assertEqual(pool.result("add"), 5)
                self.assertIsNone(pool.result("fail"))           # a failed background job does not stop the run
        self.assertTrue(any("worker adds 2 + 3" in line for line in logs.output))

    def test_metrics_run_in_workers(self):
        """The @_safe metrics of run_metrics must pickle into the workers (P2: stats.json metrics in the background)."""
        from pipeline import run_metrics
        mesh = self.tmp / "m.obj"
        trimesh.creation.box(extents=(0.05, 0.05, 0.05)).export(str(mesh))
        with LoggingProcessPool(1) as pool:
            pool.submit("q", run_metrics.mesh_quality, str(mesh), "cleaned")
            q = pool.result("q", raise_errors=True)
        self.assertEqual(q["faces"], 12)
        self.assertTrue(q["watertight"])

    def test_export_with_pool_matches_sequential(self):
        src = self.tmp / "src"
        src.mkdir()
        high = _textured_source(src, "model_textured", "purple")
        (src / "low").mkdir()
        low = _textured_source(src / "low", "model_textured_low", "orange")
        clean = self.tmp / "clean.obj"
        trimesh.creation.box(extents=(0.05, 0.05, 0.05)).export(str(clean))
        cfg = PipelineConfig()
        cfg.glb_draco = False

        seq = self.tmp / "seq"
        export_multi_lod_bundle(str(clean), str(high), str(seq), SystemTelemetry(str(seq)), cfg,
                                textured_low_obj_path=str(low), make_zip=False)
        par = self.tmp / "par"
        printed = self.tmp / "printed"
        printed.mkdir()
        pre = export_print_assets(str(clean), str(printed / "high_model.stl"), str(printed / "high_model.3mf"),
                                  cfg.export_scale)
        tel = SystemTelemetry(str(par))
        with LoggingProcessPool(2) as pool:
            export_multi_lod_bundle(str(clean), str(high), str(par), tel, cfg, textured_low_obj_path=str(low),
                                    make_zip=False, pool=pool, print_high=pre)

        files = lambda root: sorted(str(p.relative_to(root)) for p in root.rglob("*")
                                    if p.is_file() and p.suffix != ".json")
        self.assertEqual(files(seq), files(par))
        self.assertFalse((par / "output.zip").exists())
        for rel in files(seq):
            if Path(rel).suffix in (".obj", ".mtl", ".png", ".jpg", ".stl"):
                self.assertEqual((seq / rel).read_bytes(), (par / rel).read_bytes(), rel)
        # The deliverables' metrics were measured in the workers, same numbers as record_mesh's own parse
        by_name = {m.file_path: m for m in tel.mesh_metrics}
        self.assertEqual(by_name["high_model.stl"].face_count,
                         mesh_file_metrics(str(par / "stl" / "high_model.stl"))["f_count"])
        self.assertGreater(by_name["model_low.obj"].face_count, 0)

    def test_early_exports_match_the_export_at_the_end(self):
        """B3: high / low chains started when their texturing ended give the same files; a different source is redone."""
        from pipeline.exporter import EarlyExports

        src = self.tmp / "src"
        src.mkdir()
        high = _textured_source(src, "model_textured", "purple")
        (src / "low").mkdir()
        low = _textured_source(src / "low", "model_textured_low", "orange")
        other = self.tmp / "other"
        other.mkdir()
        low_other = _textured_source(other, "model_textured_low", "green")     # e.g. the TextureMesh fallback
        clean = self.tmp / "clean.obj"
        trimesh.creation.box(extents=(0.05, 0.05, 0.05)).export(str(clean))
        cfg = PipelineConfig()
        cfg.glb_draco = False

        seq = self.tmp / "seq"
        export_multi_lod_bundle(str(clean), str(high), str(seq), SystemTelemetry(str(seq)), cfg,
                                textured_low_obj_path=str(low_other), make_zip=False)
        early_dir = self.tmp / "early"
        with LoggingProcessPool(2) as pool:
            early = EarlyExports(str(early_dir), str(clean), cfg, pool)
            early.start_high(str(high))
            early.start_low(str(low))            # started from a model the end no longer uses
            export_multi_lod_bundle(str(clean), str(high), str(early_dir), SystemTelemetry(str(early_dir)), cfg,
                                    textured_low_obj_path=str(low_other), make_zip=False, pool=pool, early=early)
            self.assertEqual(early.jobs, {})

        files = lambda root: sorted(str(p.relative_to(root)) for p in root.rglob("*")
                                    if p.is_file() and p.suffix != ".json")
        self.assertEqual(files(seq), files(early_dir))
        for rel in files(seq):
            if Path(rel).suffix in (".obj", ".mtl", ".png", ".jpg", ".stl"):
                self.assertEqual((seq / rel).read_bytes(), (early_dir / rel).read_bytes(), rel)

    def test_texrecon_scene_written_once_for_concurrent_runs(self):
        runner = TexReconRunner(TexReconConfig(), str(self.tmp / "texrecon"))
        scene = self.tmp / "texrecon" / "scene"
        scene.mkdir(parents=True)
        marker = scene / "written_by_prepare_scene.txt"
        marker.write_text("x")
        mesh = self.tmp / "m.obj"
        trimesh.creation.box(extents=(0.05, 0.05, 0.05)).export(str(mesh))
        runner.run_texturing("unused.sfm", str(mesh), str(self.tmp), out_prefix=str(self.tmp / "texrecon" / "low" / "m"),
                             mock_mode=True, scene_ready=True)
        self.assertTrue(marker.is_file())                         # not rebuilt (a parallel run would lose its scene)

    def test_zip_stores_compressed_formats(self):
        out = self.tmp / "out"
        (out / "obj" / "high").mkdir(parents=True)
        Image.new("RGB", (64, 64), color="red").save(str(out / "obj" / "high" / "t.png"))
        (out / "obj" / "high" / "m.obj").write_text("v 0 0 0\n" * 200)
        z = package_output_zip(str(out))
        with zipfile.ZipFile(z) as zf:
            kinds = {Path(i.filename).suffix: i.compress_type for i in zf.infolist()}
        self.assertEqual(kinds[".png"], zipfile.ZIP_STORED)
        self.assertEqual(kinds[".obj"], zipfile.ZIP_DEFLATED)

    def test_incremental_zip_same_files(self):
        """B4: deliverables zipped in a thread, stats.json and logs added at the end = the zip made in one go."""
        from pipeline.exporter import OutputZip

        out = self.tmp / "out"
        for sub in (("obj", "high"), ("obj", "low"), ("stl",), ("glb",)):
            out.joinpath(*sub).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color="red").save(str(out / "obj" / "high" / "t.png"))
        (out / "obj" / "high" / "m.obj").write_text("v 0 0 0\n" * 2000)
        (out / "obj" / "low" / "l.obj").write_text("v 1 1 1\n" * 500)
        (out / "stl" / "high_model.stl").write_bytes(bytes(range(256)) * 50)
        (out / "object.json").write_text('{"title": "x"}')
        (out / "stats.json").write_text('{"final": true}')
        (out / "run_log.txt").write_text("log\n")
        logs = {"logs/run_log.txt": str(out / "run_log.txt")}
        ref = package_output_zip(str(out), "ref.zip", str(out / "stats.json"), logs)
        z = OutputZip(str(out), "inc.zip")
        z.add_deliverables_async()
        inc = z.finish(str(out / "stats.json"), logs)
        with zipfile.ZipFile(ref) as a, zipfile.ZipFile(inc) as b:
            self.assertEqual(sorted(a.namelist()), sorted(b.namelist()))
            self.assertEqual(len(b.namelist()), len(set(b.namelist())))
            for n in a.namelist():
                self.assertEqual(a.read(n), b.read(n), n)
            self.assertIn("stats.json", b.namelist())


if __name__ == "__main__":
    unittest.main()
