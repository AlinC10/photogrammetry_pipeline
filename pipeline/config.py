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
DEFAULT_FOCAL_LENGTH_MM: float = 3.9  # Arducam IMX477 M12 wide-angle lens (3.9 mm)


# ==============================================================================
# OpenMVS Binary Locator
# ==============================================================================

def find_openmvs_binary(name: str, custom_dir: Optional[str] = None) -> Optional[str]:
    """
    Locates an OpenMVS executable across system PATH, custom environment variables,
    CLI argument, and standard installation directories on Windows and Linux.
    Automatically checks subfolders like 'bin/' and versioned directories.
    """
    exe_names = [f"{name}.exe", name] if sys.platform == "win32" else [name]

    # 1. Custom directory passed via argument or OPENMVS_BIN_DIR env var
    search_dirs = []
    if custom_dir:
        search_dirs.append(Path(custom_dir))
    env_dir = os.environ.get("OPENMVS_BIN_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))

    for d in search_dirs:
        for ename in exe_names:
            for cand in [
                d / ename,
                d / "bin" / ename,
                d / "vc17" / "x64" / "Release" / ename,
                d / "vc16" / "x64" / "Release" / ename,
                d / "x64" / "Release" / ename,
                d / "x64" / ename,
            ]:
                if cand.is_file():
                    return str(cand.resolve())
            # Recursive check inside custom directory
            try:
                for cand in d.glob(f"**/{ename}"):
                    if cand.is_file():
                        return str(cand.resolve())
            except Exception:
                pass

    # 2. Check system PATH
    for ename in exe_names:
        which_path = shutil.which(ename)
        if which_path:
            return str(Path(which_path).resolve())

    # 3. Known fallback paths (Windows & Linux)
    fallback_dirs = [
        # Linux
        "/usr/local/bin/OpenMVS",
        "/usr/local/bin",
        "/opt/openmvs/bin",
        str(Path.home() / "openMVS" / "bin"),
        str(Path.home() / "openMVS_build" / "bin"),
        # Windows Visual Studio Builds (vc17 = VS2022, vc16 = VS2019)
        "C:\\OpenMVS\\vc17\\x64\\Release",
        "C:\\openMVS\\vc17\\x64\\Release",
        "C:\\OpenMVS\\vc16\\x64\\Release",
        "C:\\openMVS\\vc16\\x64\\Release",
        "C:\\OpenMVS\\x64\\Release",
        "C:\\openMVS\\x64\\Release",
        # Windows Standard & Bin
        "C:\\OpenMVS",
        "C:\\OpenMVS\\bin",
        "C:\\openMVS",
        "C:\\openMVS\\bin",
        "C:\\openMVS_v2.4.0",
        "C:\\openMVS_v2.4.0\\bin",
        "C:\\openMVS_v2.4.0_win64",
        "C:\\openMVS_v2.4.0_win64\\bin",
        "C:\\openMVS_v2.4.0_win64\\vc17\\x64\\Release",
        "C:\\openMVS_v2.3.0",
        "C:\\openMVS_v2.3.0\\bin",
        "C:\\Program Files\\OpenMVS",
        "C:\\Program Files\\OpenMVS\\bin",
        # Windows Drive D (project drive)
        "D:\\OpenMVS\\vc17\\x64\\Release",
        "D:\\openMVS\\vc17\\x64\\Release",
        "D:\\OpenMVS",
        "D:\\OpenMVS\\bin",
        "D:\\openMVS",
        "D:\\openMVS\\bin",
        "D:\\openMVS_v2.4.0",
        "D:\\openMVS_v2.4.0\\bin",
        "D:\\openMVS_v2.4.0_win64",
        "D:\\openMVS_v2.4.0_win64\\bin",
    ]
    for d in fallback_dirs:
        for ename in exe_names:
            candidate = Path(d) / ename
            if candidate.is_file():
                return str(candidate.resolve())

    # 4. Glob search on drive roots (C:\ and D:\) for any openmvs folder recursively
    if sys.platform == "win32":
        for drive in ["C:\\", "D:\\"]:
            d_path = Path(drive)
            if not d_path.is_dir():
                continue
            try:
                for match_dir in d_path.glob("*[Oo]pen[Mm][Vv][Ss]*"):
                    if match_dir.is_dir():
                        for ename in exe_names:
                            # Direct check common subpaths
                            for cand in [
                                match_dir / "vc17" / "x64" / "Release" / ename,
                                match_dir / "x64" / "Release" / ename,
                                match_dir / "bin" / ename,
                                match_dir / ename,
                            ]:
                                if cand.is_file():
                                    return str(cand.resolve())
                            # Recursive fallback
                            for cand in match_dir.glob(f"**/{ename}"):
                                if cand.is_file():
                                    return str(cand.resolve())
            except Exception:
                pass

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
    empty_color: int = 0              # 0 = black (prevents neon orange on unobserved/bottom faces)
    cost_smoothness_ratio: float = 1.0 # 1.0 prevents micro-patch fragmentation; encourages large continuous charts
    outlier_threshold: float = 0.0     # 0.0 disables statistical outlier view rejection (crucial for shiny plastics on turntable)
    global_seam_leveling: int = 0      # 0 disables global Poisson solver that bleeds black background into patches
    local_seam_leveling: int = 1       # 1 preserves smooth edge blending at seams

    def __post_init__(self):
        if self.bin_dir:
            os.environ["OPENMVS_BIN_DIR"] = str(self.bin_dir)
        self.interface_colmap = self.interface_colmap or find_openmvs_binary("InterfaceCOLMAP", self.bin_dir)
        self.densify_point_cloud = self.densify_point_cloud or find_openmvs_binary("DensifyPointCloud", self.bin_dir)
        self.reconstruct_mesh = self.reconstruct_mesh or find_openmvs_binary("ReconstructMesh", self.bin_dir)
        self.refine_mesh = self.refine_mesh or find_openmvs_binary("RefineMesh", self.bin_dir)
        self.texture_mesh = self.texture_mesh or find_openmvs_binary("TextureMesh", self.bin_dir)

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
    smooth_iterations: int = 12       # Taubin volume-preserving smoothing (eliminates 2-4mm ripples)


@dataclass
class TexReconConfig:
    """Paths and settings for mvs-texturing (texrecon) from TU Darmstadt."""
    bin_dir: Optional[str] = None
    texrecon_bin: Optional[str] = None
    data_term: str = "gmi"              # "gmi" (gradient magnitude + intensity) or "color"
    smoothness_term: str = "potts"      # Potts model MRF regularizer
    outlier_removal: str = "gauss_damping" # "gauss_damping", "none", "gauss_clamping"
    tone_mapping: str = "none"          # "none"
    skip_global_seam_leveling: bool = False # False = Poisson global seam leveling enabled
    skip_local_seam_leveling: bool = False  # False = local image-space seam blending enabled
    num_threads: Optional[int] = None   # None = auto-detect all available CPU cores
    write_timings: bool = True
    write_intermediate_results: bool = False


@dataclass
class AliceVisionConfig:
    """Paths and settings for optional AliceVision multi-band texturing."""
    bin_dir: Optional[str] = None
    texturing_bin: Optional[str] = None
    texture_side: int = 4096
    unwrap_method: str = "Basic"       # Basic (fast, robust on high-poly), LSCM, ABF
    use_udim: bool = False             # False = single standard (0,1) UV atlas for maximum 3D software compatibility
    fill_holes: bool = True            # Inpainting for texture holes (eliminates black spots)
    correct_ev: bool = True            # Exposure compensation across camera angles
    color_mapping_type: str = "png"    # png (lossless) or jpg
    padding: int = 5
    multi_band_downscale: int = 2       # 2 = finer frequency decomposition (studio grade sharpness)
    visibility_remapping_method: str = "MeshItself"  # MeshItself (direct raycasting on mesh, no dense point cloud required)


@dataclass
class PipelineConfig:
    """Master configuration for the studio scanning pipeline."""
    input_dir: str = "input_images"
    output_dir: str = "output"
    rig_mode: bool = False             # True if images are in rig/0 and rig/1
    two_sides_mode: bool = False       # True if images are in rig1/ and rig2/
    use_masks: bool = True
    masks_dir: Optional[str] = None
    texturing_engine: str = "texrecon" # "texrecon" (mvs-texturing), "alicevision", or "openmvs"
    texrecon: TexReconConfig = field(default_factory=TexReconConfig)
    openmvs: OpenMVSConfig = field(default_factory=OpenMVSConfig)
    alicevision: AliceVisionConfig = field(default_factory=AliceVisionConfig)
    turntable: TurntableConfig = field(default_factory=TurntableConfig)
    cleaning: MeshCleaningConfig = field(default_factory=MeshCleaningConfig)

    # Cloud & Artifact export
    export_zip: bool = True
    zip_name: str = "output.zip"
