"""
test_exporter.py — Unit test for Multi-LOD export bundle:
- High-Poly OBJ with PNG textures and model_high.mtl
- Low-Poly OBJ with downscaled JPG textures and model_low.mtl
- High and Low GLBs with embedded textures
- STL, 3MF, and ZIP packaging
"""

import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

# Ensure turntable_studio root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh
from PIL import Image

from pipeline.config import PipelineConfig
from pipeline.exporter import export_multi_lod_bundle, convert_obj_to_glb, find_source_mtl_and_textures
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

        # Verify model_high.mtl references the PNG texture and has neutral Ka 0
        high_mtl_content = (high_dir / "model_high.mtl").read_text()
        self.assertIn(".png", high_mtl_content)
        self.assertIn("Ka 0.000000 0.000000 0.000000", high_mtl_content)

        # Check Low-Poly deliverables
        low_dir = output_dir / "obj" / "low"
        self.assertTrue((low_dir / "model_low.obj").is_file())
        self.assertTrue((low_dir / "model_low.mtl").is_file(), "Low poly folder must contain model_low.mtl")

        low_jpgs = list(low_dir.glob("*.jpg"))
        self.assertGreaterEqual(len(low_jpgs), 1, "Low poly folder must have at least one JPG texture image")
        low_pngs = list(low_dir.glob("*.png"))
        self.assertEqual(len(low_pngs), 0, "Low poly folder must NOT contain PNG textures")

        # Verify model_low.obj references model_low.mtl
        low_obj_content = (low_dir / "model_low.obj").read_text()
        self.assertIn("mtllib model_low.mtl", low_obj_content)

        # Verify model_low.mtl references the JPG texture and has neutral Ka 0
        low_mtl_content = (low_dir / "model_low.mtl").read_text()
        self.assertIn(".jpg", low_mtl_content)
        self.assertIn("Ka 0.000000 0.000000 0.000000", low_mtl_content)

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

    def test_export_multi_lod_bundle_branching_low_poly(self):
        """Verifies that providing a dedicated textured_low_obj_path produces textured low-poly deliverables."""
        tpath = Path(self.test_dir) / "branching"
        tpath.mkdir(parents=True, exist_ok=True)

        # High poly source
        high_dir = tpath / "high"
        high_dir.mkdir(parents=True, exist_ok=True)
        high_tex = high_dir / "texture.png"
        Image.new("RGB", (256, 256), color="red").save(str(high_tex), format="PNG")
        high_mtl = high_dir / "high.mtl"
        high_mtl.write_text("newmtl mat_high\nKa 1 1 1\nKd 1 1 1\nmap_Kd texture.png\n")
        high_obj = high_dir / "high.obj"
        high_obj.write_text(
            "mtllib high.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 0 1\n"
            "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
            "usemtl mat_high\n"
            "f 1/1/1 2/2/2 3/3/3\n"
        )

        # Dedicated low poly textured source
        low_src_dir = tpath / "low_src"
        low_src_dir.mkdir(parents=True, exist_ok=True)
        low_tex = low_src_dir / "texture_low.png"
        Image.new("RGB", (128, 128), color="blue").save(str(low_tex), format="PNG")
        low_src_mtl = low_src_dir / "low_branch.mtl"
        low_src_mtl.write_text("newmtl mat_low\nKa 1 1 1\nKd 1 1 1\nmap_Kd texture_low.png\n")
        low_src_obj = low_src_dir / "low_branch.obj"
        low_src_obj.write_text(
            "mtllib low_branch.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 0 1\n"
            "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
            "usemtl mat_low\n"
            "f 1/1/1 2/2/2 3/3/3\n"
        )

        clean_obj = tpath / "clean.obj"
        box = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
        box.export(str(clean_obj))

        output_dir = tpath / "out"
        config = PipelineConfig()

        bundle = export_multi_lod_bundle(
            cleaned_mesh_path=str(clean_obj),
            textured_obj_path=str(high_obj),
            output_dir=str(output_dir),
            telemetry=self.telemetry,
            config=config,
            textured_low_obj_path=str(low_src_obj),
        )

        low_out_dir = output_dir / "obj" / "low"
        self.assertTrue((low_out_dir / "model_low.obj").is_file())
        self.assertTrue((low_out_dir / "model_low.mtl").is_file())
        low_obj_content = (low_out_dir / "model_low.obj").read_text()
        self.assertIn("vt ", low_obj_content, "model_low.obj must have texture coordinates vt from branching!")
        self.assertIn("mtllib model_low.mtl", low_obj_content)

        # Check GLB deliverables
        self.assertTrue((output_dir / "glb" / "high_model.glb").is_file())
        self.assertTrue((output_dir / "glb" / "low_model.glb").is_file())

    def test_convert_obj_to_glb_standalone(self):
        """Verifies convert_obj_to_glb preserves materials and produces valid GLB binary."""
        folder = Path(self.test_dir) / "test_standalone_glb"
        folder.mkdir(parents=True, exist_ok=True)

        tex_path = folder / "my_tex.png"
        img = Image.new("RGB", (128, 128), color=(20, 180, 50))
        img.save(str(tex_path), format="PNG")

        mtl_path = folder / "model.mtl"
        mtl_path.write_text(
            "newmtl mat0\n"
            "Ka 0.0 0.0 0.0\n"
            "Kd 1.0 1.0 1.0\n"
            "map_Kd my_tex.png\n"
        )

        obj_path = folder / "model.obj"
        obj_path.write_text(
            "mtllib model.mtl\n"
            "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\n"
            "vt 0.0 0.0\nvt 1.0 0.0\nvt 0.0 1.0\n"
            "vn 0.0 0.0 1.0\nvn 0.0 0.0 1.0\nvn 0.0 0.0 1.0\n"
            "usemtl mat0\n"
            "f 1/1/1 2/2/2 3/3/3\n"
        )

        glb_path = folder / "model.glb"
        # Test converting by passing the folder
        convert_obj_to_glb(folder, glb_path, compress_textures=True)

        self.assertTrue(glb_path.is_file())
        self.assertGreater(glb_path.stat().st_size, 500)

        # Verify loaded glb scene has geometry
        loaded_scene = trimesh.load(str(glb_path), file_type="glb")
        self.assertIsNotNone(loaded_scene)

    def test_convert_obj_to_glb_multi_material(self):
        """texrecon writes several atlases: every GLB primitive must keep its own texture (not atlas 0)."""
        folder = Path(self.test_dir) / "multi_material"
        folder.mkdir(parents=True, exist_ok=True)
        colours = {"material0000": (220, 30, 30), "material0001": (30, 30, 220), "material0002": (30, 200, 40)}
        mtl_lines = []
        for idx, (name, rgb) in enumerate(colours.items()):
            Image.new("RGB", (64, 64), color=rgb).save(str(folder / f"model_material{idx:04d}_map_Kd.png"))
            mtl_lines.append(f"newmtl {name}\nKa 0 0 0\nKd 1 1 1\nmap_Kd model_material{idx:04d}_map_Kd.png\n")
        (folder / "model.mtl").write_text("".join(mtl_lines))
        obj = ["mtllib model.mtl\n"]
        for i in range(len(colours)):
            x = float(i)
            obj += [f"v {x} 0 0\n", f"v {x + 1} 0 0\n", f"v {x} 1 0\n"]
        obj += ["vt 0.25 0.25\n", "vt 0.75 0.25\n", "vt 0.25 0.75\n"]
        for i, name in enumerate(colours):
            obj += [f"usemtl {name}\n", f"f {3 * i + 1}/1 {3 * i + 2}/2 {3 * i + 3}/3\n"]
        obj_path = folder / "model.obj"
        obj_path.write_text("".join(obj))

        glb_path = folder / "model.glb"
        convert_obj_to_glb(obj_path, glb_path, compress_textures=False)

        js, binc = _read_glb(glb_path)
        self.assertEqual(len(js["materials"]), 3)
        self.assertEqual(len(js["images"]), 3)
        seen = {}
        for mesh in js["meshes"]:
            for prim in mesh["primitives"]:
                mat = js["materials"][prim["material"]]
                tex = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
                img = js["images"][js["textures"][tex]["source"]]
                bv = js["bufferViews"][img["bufferView"]]
                raw = binc[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
                pixel = Image.open(io.BytesIO(raw)).convert("RGB").getpixel((32, 32))
                seen[mesh["name"]] = pixel
        self.assertEqual(seen, colours)
        self.assertIn("KHR_materials_unlit", js.get("extensionsUsed", []))

    def test_textures_fallback_does_not_mix_lods(self):
        """Without map_Kd in the MTL, model_textured.obj must not pick model_textured_low_* atlases."""
        folder = Path(self.test_dir) / "fallback"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model_textured.obj").write_text("mtllib model_textured.mtl\nv 0 0 0\n")
        (folder / "model_textured.mtl").write_text("newmtl material0000\nKd 1 1 1\n")
        for name in ("model_textured_material0000_map_Kd.png", "model_textured_low_material0000_map_Kd.png"):
            Image.new("RGB", (8, 8)).save(str(folder / name))
        _, images = find_source_mtl_and_textures(folder / "model_textured.obj")
        self.assertEqual([p.name for p in images], ["model_textured_material0000_map_Kd.png"])

    def test_bundle_removes_stale_deliverables(self):
        """Atlases left by a previous run (model_high_tex_23.png ...) must not reach output.zip."""
        tpath = Path(self.test_dir) / "stale"
        src = tpath / "src"
        src.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color="red").save(str(src / "tex.png"))
        (src / "m.mtl").write_text("newmtl a\nKd 1 1 1\nmap_Kd tex.png\n")
        (src / "m.obj").write_text("mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl a\nf 1/1 2/2 3/3\n")
        clean_obj = tpath / "clean.obj"
        trimesh.creation.box(extents=(0.05, 0.05, 0.05)).export(str(clean_obj))
        out = tpath / "out"
        stale = out / "obj" / "high" / "model_high_tex_23.png"
        stale.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(str(stale))

        export_multi_lod_bundle(cleaned_mesh_path=str(clean_obj), textured_obj_path=str(src / "m.obj"),
                                output_dir=str(out), telemetry=self.telemetry, config=PipelineConfig())
        self.assertFalse(stale.exists())
        with zipfile.ZipFile(out / "output.zip") as zf:
            self.assertFalse(any(n.endswith("model_high_tex_23.png") for n in zf.namelist()))


def _read_glb(path):
    data = Path(path).read_bytes()
    _, _, length = struct.unpack_from("<4sII", data, 0)
    off, js, binc = 12, None, b""
    while off < length:
        clen, ctype = struct.unpack_from("<I4s", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == b"JSON":
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            binc = chunk
        off += 8 + clen
    return js, binc


if __name__ == "__main__":
    unittest.main()
