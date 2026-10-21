"""
Turntable Studio - Studio-grade 3D scanning pipeline with OpenMVS.
"""

from .config import PipelineConfig, OpenMVSConfig, TurntableConfig, MeshCleaningConfig
from .camera_init import scan_input_directory, generate_camera_init_sfm, run_turntable_poses
from .bridge_sfm import convert_sfm_to_colmap
from .openmvs_runner import OpenMVSRunner
from .mesh_cleaner import clean_and_repair_mesh, seal_watertight_pedestal
from .exporter import export_multi_lod_bundle
from .telemetry import SystemTelemetry

__version__ = "1.0.0"

__all__ = [
    "PipelineConfig",
    "OpenMVSConfig",
    "TurntableConfig",
    "MeshCleaningConfig",
    "scan_input_directory",
    "generate_camera_init_sfm",
    "run_turntable_poses",
    "convert_sfm_to_colmap",
    "OpenMVSRunner",
    "clean_and_repair_mesh",
    "seal_watertight_pedestal",
    "export_multi_lod_bundle",
    "SystemTelemetry",
]
