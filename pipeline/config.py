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
    max_threads: int = 0              # --max-threads for every OpenMVS app (0 = OpenMVS default: all cores)
    keep_depth_maps: bool = False     # False: depth*.dmap (~74 MB each at level 1) are deleted after fusion
    smooth_mesh: int = 2              # OpenMVS ReconstructMesh --smooth (default 2)
    min_point_distance: float = 0.5   # ReconstructMesh --min-point-distance, in pixels (OpenMVS default 2.5)
    texture_resolution: int = 4096    # high-res texture atlas
    low_poly_texture_res: int = 2048  # low-poly texture atlas
    empty_color: int = 0              # 0 = black (prevents neon orange on unobserved/bottom faces)
    cost_smoothness_ratio: float = 1.0 # 1.0 enforces continuous charts and prevents micro-confetti / inverted patches (RULE 8)
    patch_packing_heuristic: int = 3   # 3 = Best/tightest UV chart packing heuristic in OpenMVS
    close_holes: int = 1               # 1 = Inpaint/extrapolate unobserved/occluded texture holes
    outlier_threshold: float = 0.05    # 0.05 rejects specular reflections on semi-gloss plastics
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
    lift_mm: float = 100.0            # Z carriage travel per level (scanner z_axis.distance_mm); turntable_poses defaults to 80
    estimate_lift: bool = False       # True = turntable_poses estimates the real lift instead of trusting lift_mm
    tag_refine: bool = True           # Per-tag height/tilt (paper unevenness): RMS 4.09 -> 2.48 px on the Parkside set
    intrinsics_in: Optional[str] = None  # Optional chessboard calibration JSON from calibrate_intrinsics.py


@dataclass
class MeshCleaningConfig:
    """Watertight base slicing and CAD pedestal sealing."""
    slice_y_m: float = 0.001          # Cut plane at Y = 1.0 mm above turntable (strictly removes disc)
    skirt_bottom_y_m: float = 0.001   # Capped directly at Y = 1.0 mm (no vertical extruded skirt)
    extrude_skirt: bool = False       # Disable vertical skirt extrusion to prevent disc texture projection
    base_color_rgb: tuple[int, int, int] = (25, 25, 25) # Technical dark matte for unobserved base
    max_radius_m: float = 0.12        # Radial bounding cylinder (120 mm)
    keep_largest_only: bool = True
    remove_duplicate_faces: bool = True
    target_low_poly_faces: int = 50000
    target_high_poly_faces: int = 1000000 # High-poly cap (~500k vertices); denser meshes are decimated to it
    smooth_iterations: int = 12       # Taubin volume-preserving smoothing (eliminates 2-4mm ripples)
    high_poly_smooth_iterations: int = 5 # Taubin steps for high-poly (smooths 200um noise without loss of detail)


@dataclass
class TexReconConfig:
    """Paths and settings for mvs-texturing (texrecon) from TU Darmstadt."""
    bin_dir: Optional[str] = None
    texrecon_bin: Optional[str] = None
    data_term: str = "gmi"              # "gmi" (gradient magnitude integral) or "area" — the only two texrecon accepts
    smoothness_term: str = "potts"      # Potts model MRF regularizer
    outlier_removal: str = "gauss_damping" # "gauss_damping", "none", "gauss_clamping"
    tone_mapping: str = "none"          # "none"
    skip_global_seam_leveling: bool = True  # True disables global Poisson solver that bleeds background into patches (RULE 8)
    skip_local_seam_leveling: bool = False  # False = local image-space seam blending enabled
    keep_unseen_faces: bool = True      # Keep faces no view sees (undersides, base) instead of leaving holes in the OBJ
    num_threads: Optional[int] = None   # None = auto-detect all available CPU cores
    write_timings: bool = True
    write_intermediate_results: bool = False


@dataclass
class MaskingConfig:
    """Settings for automated background matting, silhouette masking, and turntable disc elimination."""
    enabled: bool = True
    auto_generate: bool = True               # Automatically generate masks from 3D cylinder + chroma
    cylinder_radius_m: float = 0.080         # Radial bounding cylinder (80 mm); 35 mm cut the Parkside tool (max radius 38.7 mm)
    cylinder_y_min_m: float = 0.001          # Base cut 1.0 mm above turntable (matches slice_y_m = 1.0 mm)
    cylinder_y_max_m: float = 0.350          # Maximum object height 350 mm (the Parkside tool alone is 230 mm)
    chroma_threshold: float = 20.0           # Threshold for color vs neutral background
    brightness_white_threshold: float = 135.0 # Threshold for white photo-booth wall
    safety_dilation_px: int = 1              # Edge safety margin
    tag_blackout: bool = True                # Explicit geometric blackout of all 12 AprilTags
    tag_expansion_factor: float = 1.25       # Expand tag quad by 25% to guarantee full black border suppression
    apply_to_texturing_images: bool = True   # Set background to black RGB(0,0,0) for TextureMesh / texrecon
    use_for_densify: bool = False            # False: Densify runs cleanly on full scene; turntable is sliced at Y=1mm by mesh_cleaner


@dataclass
class TexturingImagesConfig:
    """Images texrecon textures from, rebuilt after mesh cleaning (texture_prep.py)."""
    enabled: bool = True                     # False = texrecon uses the undistorted/ images of the bridge step
    margin_px: int = 2                       # Mesh silhouette dilation (px) tolerating small pose errors
    trim_band_px: int = 25                   # White background pulled inside the silhouette by pose error is trimmed within this band
    brightness_white_threshold: float = 135.0 # Same booth-white test as the photometric mask, only applied in the band
    chroma_threshold: float = 20.0
    colour_normalize: bool = True            # Per-camera gains from the turntable paper (AWB of camera 1 differs)
    max_gain_deviation: float = 0.25         # Gains are clamped to [0.75, 1.25]
    workers: int = 0                         # 0 = automatic (resources.resolve_workers)


@dataclass
class PipelineConfig:
    """Master configuration for the studio scanning pipeline."""
    input_dir: str = "input_images"
    output_dir: str = "output"
    rig_mode: bool = False             # True if images are in rig/0 and rig/1
    use_masks: bool = True
    masks_dir: Optional[str] = None
    texturing_engine: str = "texrecon" # "texrecon" (mvs-texturing) or "openmvs" (TextureMesh)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    texturing_images: TexturingImagesConfig = field(default_factory=TexturingImagesConfig)
    texrecon: TexReconConfig = field(default_factory=TexReconConfig)
    openmvs: OpenMVSConfig = field(default_factory=OpenMVSConfig)
    turntable: TurntableConfig = field(default_factory=TurntableConfig)
    cleaning: MeshCleaningConfig = field(default_factory=MeshCleaningConfig)

    # Scaling and units
    export_scale: float = 100.0        # Default 100.0 scales raw photogrammetry mesh (2.2mm) to real-world scale (22cm / 220mm)
    scale_stl_to_mm: bool = True

    # Cloud & Artifact export
    export_zip: bool = True
    zip_name: str = "output.zip"
