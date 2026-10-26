"""
test_openmvs_runner.py — Command lines passed to the OpenMVS v2.3.0 binaries.
"""

import sys
import tempfile
import unittest
from pathlib import Path

# Ensure turntable_studio root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import OpenMVSConfig
from pipeline.openmvs_runner import OpenMVSRunner


class TestOpenMVSCommands(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        cfg = OpenMVSConfig(densify_point_cloud="DensifyPointCloud", refine_mesh="RefineMesh",
                            texture_mesh="TextureMesh")
        self.runner = OpenMVSRunner(config=cfg, work_dir=str(self.work))
        self.calls = []

        def fake_execute(cmd, name):
            self.calls.append(cmd)
            for i, a in enumerate(cmd):
                if a == "-o":
                    out = Path(cmd[i + 1])
                    out.write_text("x")
                    out.with_suffix(".ply").write_text("x")

        self.runner._execute_cmd = fake_execute

    def tearDown(self):
        self.tmp.cleanup()

    def test_densify_fuses_three_views(self):
        self.runner.run_densify()
        cmd = self.calls[0]
        self.assertEqual(cmd[cmd.index("--number-views-fuse") + 1], "3")

    def test_refine_mesh_uses_reconstructed_ply(self):
        mesh = self.work / "scene_mesh.ply"
        mesh.write_text("ply")
        (self.work / "scene_dense.mvs").write_text("mvs")
        out = self.runner.run_refine_mesh(input_mvs="scene_dense.mvs", mesh_file=str(mesh))
        cmd = self.calls[0]
        self.assertEqual(Path(cmd[cmd.index("--mesh-file") + 1]), mesh.resolve())
        self.assertTrue(out.endswith("scene_mesh_refined.ply"))

    def test_texture_mesh_has_no_mask_path(self):
        """OpenMVS v2.3.0 TextureMesh aborts on --mask-path."""
        masks = self.work / "masks"
        masks.mkdir()
        (self.work / "scene_dense.mvs").write_text("mvs")
        try:
            self.runner.run_texture_mesh(input_mvs="scene_dense.mvs", masks_dir=str(masks))
        except Exception:
            pass
        self.assertNotIn("--mask-path", self.calls[0])


if __name__ == "__main__":
    unittest.main()
