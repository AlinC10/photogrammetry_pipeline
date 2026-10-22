"""
Turntable Studio - Studio-grade 3D scanning pipeline with OpenMVS.
"""

from .config import PipelineConfig, OpenMVSConfig, AliceVisionConfig, TexReconConfig, TurntableConfig, MeshCleaningConfig
from .camera_init import scan_input_directory, generate_camera_init_sfm, run_turntable_poses
from .bridge_sfm import convert_sfm_to_colmap
from .openmvs_runner import OpenMVSRunner
from .alicevision_runner import AliceVisionRunner, find_alicevision_binary
from .texrecon_runner import TexReconRunner, find_texrecon_binary, export_sfm_to_cam_scene
from .mesh_cleaner import clean_and_repair_mesh, seal_watertight_pedestal
from .exporter import export_multi_lod_bundle
from .telemetry import SystemTelemetry

__version__ = "1.0.0"

__all__ = [
    "PipelineConfig",
    "OpenMVSConfig",
    "AliceVisionConfig",
    "TexReconConfig",
    "TurntableConfig",
    "MeshCleaningConfig",
    "scan_input_directory",
    "generate_camera_init_sfm",
    "run_turntable_poses",
    "convert_sfm_to_colmap",
    "OpenMVSRunner",
    "AliceVisionRunner",
    "find_alicevision_binary",
    "TexReconRunner",
    "find_texrecon_binary",
    "export_sfm_to_cam_scene",
    "clean_and_repair_mesh",
    "seal_watertight_pedestal",
    "export_multi_lod_bundle",
    "SystemTelemetry",
]
