"""
config.py — Global configuration, hardware presets, and OpenMVS paths.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ==============================================================================
# Hardware Presets
# ==============================================================================

# Sony IMX477 (Raspberry Pi HQ Camera / Arducam 12MP)
# Sensor size: 6.287 mm (H) x 4.712 mm (V) (Type 1/2.3", diagonal 7.857 mm)
# Max resolution: 4056 x 3040
DEFAULT_SENSOR_WIDTH_MM: float = 6.287
DEFAULT_IMAGE_WIDTH: int = 4056
DEFAULT_IMAGE_HEIGHT: int = 3040
DEFAULT_FOCAL_LENGTH_MM: float = 6.0  # Common M12 lens default


# ==============================================================================
# OpenMVS Binary Locator
# ==============================================================================

def find_openmvs_binary(name: str) -> Optional[str]:
    """
    Locates an OpenMVS executable across system PATH, custom environment variables,
    and standard installation directories on Windows and Linux.
    """
    exe_name = f"{name}.exe" if sys.platform == "win32" else name

    # 1. Custom ENV directory
    env_dir = os.environ.get("OPENMVS_BIN_DIR")
    if env_dir:
        candidate = Path(env_dir) / exe_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    # 2. Check system PATH
    which_path = shutil.which(exe_name)
    if which_path:
        return str(Path(which_path).resolve())

    # 3. Known fallback paths
    fallback_dirs = [
        "/usr/local/bin/OpenMVS",
        "/opt/openmvs/bin",
        "/usr/local/bin",
        "C:\\Program Files\\OpenMVS",
        "C:\\OpenMVS",
        str(Path.home() / "openMVS" / "bin"),
        str(Path.home() / "openMVS_build" / "bin"),
    ]
    for d in fallback_dirs:
        candidate = Path(d) / exe_name
        if candidate.is_file():
            return str(candidate.resolve())

    return None


@dataclass
class OpenMVSConfig:
    """Paths to OpenMVS executables and execution settings."""
    bin_dir: Optional[str] = None
    interface_colmap: Optional[str] = None
    densify_point_cloud: Optional[str] = None
    reconstruct_mesh: Optional[str] = None
    refine_mesh: Optional[str] = None
    texture_mesh: Optional[str] = None

    # Algorithm parameters
    resolution_level: int = 1         # 1 = 2x downscale (optimal speed/quality), 0 = full, 2 = 4x
    min_resolution: int = 640
    max_resolution: int = 3200
    density_factor: float = 1.0
    number_views: int = 6             # min views for PatchMatch (6 for turntable precision)
    smooth_mesh: int = 2              # OpenMVS ReconstructMesh --smooth (default 2)
    min_point_distance: float = 0.5   # mm distance filter
    texture_resolution: int = 4096    # high-res texture atlas
    low_poly_texture_res: int = 2048  # low-poly texture atlas

    def __post_init__(self):
        if self.bin_dir:
            os.environ["OPENMVS_BIN_DIR"] = str(self.bin_dir)
        self.interface_colmap = self.interface_colmap or find_openmvs_binary("InterfaceCOLMAP")
        self.densify_point_cloud = self.densify_point_cloud or find_openmvs_binary("DensifyPointCloud")
        self.reconstruct_mesh = self.reconstruct_mesh or find_openmvs_binary("ReconstructMesh")
        self.refine_mesh = self.refine_mesh or find_openmvs_binary("RefineMesh")
        self.texture_mesh = self.texture_mesh or find_openmvs_binary("TextureMesh")

    def validate(self) -> list[str]:
        """Returns missing binaries if any."""
        missing = []
        if not self.interface_colmap: missing.append("InterfaceCOLMAP")
        if not self.densify_point_cloud: missing.append("DensifyPointCloud")
        if not self.reconstruct_mesh: missing.append("ReconstructMesh")
        if not self.texture_mesh: missing.append("TextureMesh")
        return missing


@dataclass
class TurntableConfig:
    """Turntable geometry and camera rig configuration."""
    turntable_poses_script: str = field(
        default_factory=lambda: os.environ.get("TURNTABLE_POSES_PATH")
        or str(Path(__file__).resolve().parent / "turntable_poses.py")
    )
    tag_layout: Optional[str] = field(
        default_factory=lambda: str(Path(__file__).resolve().parent / "platan_260mm_layout.json")
    )
    step_deg: float = 18.0            # 20 steps per full 360 rotation
    tag_scale: float = 1.0
    name_regex: str = r"(?P<level>\d)[_]?(?P<index>\d{1,4})$"
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM
    default_focal_mm: float = DEFAULT_FOCAL_LENGTH_MM


@dataclass
class MeshCleaningConfig:
    """Watertight base slicing and CAD pedestal sealing."""
    slice_y_m: float = 0.003          # Cut plane at Y = 3 mm above turntable
    skirt_bottom_y_m: float = 0.000   # Extrude down to Y = 0 mm (turntable plane)
    max_radius_m: float = 0.12        # Radial bounding cylinder (120 mm)
    keep_largest_only: bool = True
    remove_duplicate_faces: bool = True
    target_low_poly_faces: int = 50000


@dataclass
class PipelineConfig:
    """Master configuration for the studio scanning pipeline."""
    input_dir: str = "input_images"
    output_dir: str = "output"
    rig_mode: bool = False             # True if images are in rig/0 and rig/1
    two_sides_mode: bool = False       # True if images are in rig1/ and rig2/
    use_masks: bool = True
    masks_dir: Optional[str] = None
    openmvs: OpenMVSConfig = field(default_factory=OpenMVSConfig)
    turntable: TurntableConfig = field(default_factory=TurntableConfig)
    cleaning: MeshCleaningConfig = field(default_factory=MeshCleaningConfig)

    # Cloud & Artifact export
    export_zip: bool = True
    zip_name: str = "output.zip"
