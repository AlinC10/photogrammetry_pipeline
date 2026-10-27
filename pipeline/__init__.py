"""
Turntable Studio - Studio-grade 3D scanning pipeline with OpenMVS.
"""

from .config import PipelineConfig, OpenMVSConfig, TexReconConfig, TurntableConfig, MeshCleaningConfig, MaskingConfig, TexturingImagesConfig
from .camera_init import scan_input_directory, generate_camera_init_sfm, run_turntable_poses
from .bridge_sfm import convert_sfm_to_colmap
from .mask_processor import batch_generate_masks, generate_hybrid_mask, apply_mask_to_image, load_tag_layout, project_tags_to_2d
from .openmvs_runner import OpenMVSRunner
from .texrecon_runner import TexReconRunner, find_texrecon_binary, export_sfm_to_cam_scene
from .mesh_masks import render_silhouette, generate_mesh_masks
from .texture_prep import prepare_texturing_images, texrecon_validity_mask
from .mesh_cleaner import clean_and_repair_mesh, seal_watertight_pedestal
from .exporter import export_multi_lod_bundle, convert_obj_to_glb
from .telemetry import SystemTelemetry
from .resources import available_cpus, total_ram_gb, estimate_ram_gb, resolve_workers, check_ram, InsufficientRamError

__version__ = "1.0.0"

__all__ = [
    "PipelineConfig",
    "OpenMVSConfig",
    "TexReconConfig",
    "TurntableConfig",
    "MeshCleaningConfig",
    "MaskingConfig",
    "TexturingImagesConfig",
    "scan_input_directory",
    "generate_camera_init_sfm",
    "run_turntable_poses",
    "convert_sfm_to_colmap",
    "batch_generate_masks",
    "generate_hybrid_mask",
    "apply_mask_to_image",
    "OpenMVSRunner",
    "TexReconRunner",
    "find_texrecon_binary",
    "export_sfm_to_cam_scene",
    "render_silhouette",
    "generate_mesh_masks",
    "prepare_texturing_images",
    "texrecon_validity_mask",
    "clean_and_repair_mesh",
    "seal_watertight_pedestal",
    "export_multi_lod_bundle",
    "convert_obj_to_glb",
    "SystemTelemetry",
    "available_cpus",
    "total_ram_gb",
    "estimate_ram_gb",
    "resolve_workers",
    "check_ram",
    "InsufficientRamError",
]
