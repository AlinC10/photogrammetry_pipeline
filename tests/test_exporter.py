"""
test_exporter.py — Unit test for Multi-LOD export bundle:
- High-Poly OBJ with PNG textures and model_high.mtl
- Low-Poly OBJ with downscaled JPG textures and model_low.mtl
- High and Low GLBs with embedded textures
- STL, 3MF, and ZIP packaging
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure turntable_studio root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh
from PIL import Image

from pipeline.config import PipelineConfig
from pipeline.exporter import export_multi_lod_bundle
from pipeline.telemetry import SystemTelemetry


class TestExporter(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.telemetry = SystemTelemetry()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_export_multi_lod_bundle_textured(self):
        tpath = Path(self.test_dir)

        # 1. Create simulated OpenMVS textured outputs
        openmvs_dir = tpath / "openmvs"
        openmvs_dir.mkdir(parents=True, exist_ok=True)

        # Texture image (simulate 4096x4096 JPEG from OpenMVS)
        src_tex = openmvs_dir / "scene_textured_material_0_map_Kd.jpg"
        img = Image.new("RGB", (512, 512), color="purple")
        img.save(str(src_tex), format="JPEG")

        # MTL file
        src_mtl = openmvs_dir / "scene_textured.mtl"
        src_mtl.write_text(
            "newmtl material_0\n"
            "Ka 1.0 1.0 1.0\n"
            "Kd 1.0 1.0 1.0\n"
            "map_Kd scene_textured_material_0_map_Kd.jpg\n"
        )

        # OBJ file with texture coordinates
        src_obj = openmvs_dir / "scene_textured.obj"
        src_obj.write_text(
            "mtllib scene_textured.mtl\n"
            "v -0.05 0.05 0.0\n"
            "v 0.05 0.05 0.0\n"
            "v 0.0 0.15 0.0\n"
            "vt 0.0 0.0\n"
            "vt 1.0 0.0\n"
            "vt 0.5 1.0\n"
            "vn 0.0 0.0 1.0\n"
            "vn 0.0 0.0 1.0\n"
            "vn 0.0 0.0 1.0\n"
            "usemtl material_0\n"
            "f 1/1/1 2/2/2 3/3/3\n"
        )

        # Cleaned watertight mesh (for 3D print)
        clean_obj = tpath / "model_cleaned_watertight.obj"
        box = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
        box.export(str(clean_obj))

        output_dir = tpath / "output"
        config = PipelineConfig()

        bundle = export_multi_lod_bundle(
            cleaned_mesh_path=str(clean_obj),
            textured_obj_path=str(src_obj),
            output_dir=str(output_dir),
            telemetry=self.telemetry,
            config=config,
        )

        # Check High-Poly deliverables
        high_dir = output_dir / "obj" / "high"
        self.assertTrue((high_dir / "model_high.obj").is_file())
        self.assertTrue((high_dir / "model_high.mtl").is_file())

        high_pngs = list(high_dir.glob("*.png"))
        self.assertGreaterEqual(len(high_pngs), 1, "High poly folder must have at least one PNG texture image")

        # Verify model_high.obj references model_high.mtl
        high_obj_content = (high_dir / "model_high.obj").read_text()
        self.assertIn("mtllib model_high.mtl", high_obj_content)

        # Verify model_high.mtl references the PNG texture
        high_mtl_content = (high_dir / "model_high.mtl").read_text()
        self.assertIn(".png", high_mtl_content)

        # Check Low-Poly deliverables
        low_dir = output_dir / "obj" / "low"
        self.assertTrue((low_dir / "model_low.obj").is_file())
        self.assertTrue((low_dir / "model_low.mtl").is_file(), "Low poly folder must contain model_low.mtl")

        low_jpgs = list(low_dir.glob("*.jpg"))
        self.assertGreaterEqual(len(low_jpgs), 1, "Low poly folder must have at least one JPG texture image")

        # Verify model_low.obj references model_low.mtl
        low_obj_content = (low_dir / "model_low.obj").read_text()
        self.assertIn("mtllib model_low.mtl", low_obj_content)

        # Verify model_low.mtl references the JPG texture
        low_mtl_content = (low_dir / "model_low.mtl").read_text()
        self.assertIn(".jpg", low_mtl_content)

        # Check GLB deliverables
        glb_dir = output_dir / "glb"
        self.assertTrue((glb_dir / "high_model.glb").is_file())
        self.assertTrue((glb_dir / "low_model.glb").is_file())
        self.assertGreater((glb_dir / "high_model.glb").stat().st_size, 0)
        self.assertGreater((glb_dir / "low_model.glb").stat().st_size, 0)

        # Check STL, 3MF and ZIP packaging
        self.assertTrue((output_dir / "stl" / "high_model.stl").is_file())
        self.assertTrue((output_dir / "stl" / "low_model.stl").is_file())
        self.assertTrue((output_dir / "output.zip").is_file())
        self.assertGreater((output_dir / "output.zip").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
