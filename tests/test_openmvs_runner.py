"""
test_openmvs_runner.py — TextureMesh PLY output (texcoord, texnumber) converted to OBJ + MTL.
TextureMesh v2.3.0 crashes while writing OBJ, so the runner exports PLY and converts it.
"""

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.openmvs_runner import textured_ply_to_obj


def write_textured_ply(path, vertices, faces, uvs, texnumbers, textures):
    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment TextureFile {t}" for t in textures]
    header += [f"element vertex {len(vertices)}", "property float32 x", "property float32 y", "property float32 z",
               f"element face {len(faces)}", "property list uint8 uint32 vertex_indices",
               "property list uint8 float32 texcoord", "property uint8 texnumber", "end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for v in vertices:
            f.write(struct.pack("<3f", *v))
        for face, uv, tn in zip(faces, uvs, texnumbers):
            f.write(struct.pack("<B3I", 3, *face) + struct.pack("<B6f", 6, *uv) + struct.pack("<B", tn))


class TestTexturedPlyToObj(unittest.TestCase):

    def test_two_atlases(self):
        """Faces keep their vertices, texcoords and atlas (usemtl per texture, map_Kd per material)."""
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        faces = [(0, 1, 2), (1, 3, 2)]
        uvs = [(0.0, 0.0, 1.0, 0.0, 0.0, 1.0), (0.25, 0.5, 0.75, 0.5, 0.5, 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / "scene_texture.ply"
            write_textured_ply(ply, vertices, faces, uvs, [0, 1], ["scene_texture0.png", "scene_texture1.png"])
            obj = textured_ply_to_obj(str(ply), str(Path(tmp) / "scene_texture.obj"))

            lines = Path(obj).read_text().splitlines()
            self.assertEqual(lines[0], "mtllib scene_texture.mtl")
            self.assertEqual(sum(ln.startswith("v ") for ln in lines), 4)
            vts = [tuple(map(float, ln.split()[1:])) for ln in lines if ln.startswith("vt ")]
            np.testing.assert_allclose(np.array(vts).ravel(), np.array(uvs).ravel(), atol=1e-6)
            body = [ln for ln in lines if ln.startswith(("usemtl", "f "))]
            self.assertEqual(body, ["usemtl material_0", "f 1/1 2/2 3/3", "usemtl material_1", "f 2/4 4/5 3/6"])
            mtl = (Path(tmp) / "scene_texture.mtl").read_text()
            self.assertIn("newmtl material_1", mtl)
            self.assertIn("map_Kd scene_texture1.png", mtl)


if __name__ == "__main__":
    unittest.main()
