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
    number_views_fuse: int = 2        # min views to fuse a 3D point (2 preserves occluded undercuts/brackets; 3 requires more views)
    max_threads: int = 0              # --max-threads for every OpenMVS app (0 = OpenMVS default: all cores)
    keep_depth_maps: bool = False     # False: depth*.dmap (~74 MB each at level 1) are deleted after fusion
    smooth_mesh: int = 4              # OpenMVS ReconstructMesh --smooth (default 4 for smooth boundary contours)
    min_point_distance: float = 2.0   # ReconstructMesh --min-point-distance in pixels (2.0 eliminates micro-triangle edge stepping; OpenMVS default 2.5)
    refine_mesh_enabled: bool = False  # Disabled by default: ReconstructMesh produces high-fidelity clean surfaces; RefineMesh adds 4-7 min with excessive decimation
    refine_resolution_level: int = 2  # Fast RefineMesh image downscale when enabled (2 = 4x, runs in <40s)
    refine_max_face_area: int = 256   # Target max face area in pixels (prevents vertex explosion)
    refine_scales: int = 1            # Single scale iteration (prevents multi-scale slowdown)
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
    turntable_poses_script: str = str(Path(__file__).resolve().parent / "turntable_poses.py")
    tag_layout: Optional[str] = str(Path(__file__).resolve().parent / "platan_345mm_layout.json")
    step_deg: float = 18.0            # 20 steps per full 360 rotation
    # Print scale of the tag sheet (measured / designed). The 345mm disc is calibrated at 0.99 (2.97 cm measured tag vs 3.00 cm design).
    tag_scale: float = 0.99
    name_regex: str = r"(?P<level>\d)[_]?(?P<index>\d{1,4})$"
    sensor_width_mm: float = DEFAULT_SENSOR_WIDTH_MM
    default_focal_mm: float = DEFAULT_FOCAL_LENGTH_MM
    lift_mm: float = 80.0             # Z carriage travel per level the scanner commands (z_axis.distance_mm)
    # None = automatic: estimated by turntable_poses when the scan has >= 2 levels (with one level there is nothing
    # to estimate). The commanded 80 mm measured ~85 mm in the photos (multimeter, fixativ), so it is estimated.
    estimate_lift: Optional[bool] = None
    tag_refine: bool = True           # Per-tag height/tilt (paper unevenness): RMS 4.09 -> 2.48 px on the Parkside set
    # The sheet's own unevenness is +-1.5 mm (tag 6 ~ +1.3, tag 4 ~ -1.1 in every scan); beyond this a tag is covered
    # by the object: warning in stats.json (the poses are kept, see main._check_tag_corrections)
    max_tag_correction_mm: float = 3.0
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
    # Decimation as a share of the sealed mesh, so every object is decimated (and denoised) a little, whatever
    # its size. A fixed target > 0 overrides the ratio (old behaviour: 250 000 high / 50 000 low).
    high_poly_ratio: float = 0.75         # high-poly keeps 75 % of the faces
    low_poly_ratio: float = 0.2           # low-poly keeps 20 % of the faces of the same sealed mesh ...
    low_poly_max_faces: int = 50000       # ... but never more than this (web model); 0 = no cap
    target_high_poly_faces: int = 0       # 0 = use high_poly_ratio
    target_low_poly_faces: int = 0        # 0 = use low_poly_ratio
    smooth_iterations: int = 12       # Taubin volume-preserving smoothing (eliminates 2-4mm ripples)
    high_poly_smooth_iterations: int = 5 # Taubin steps for high-poly (smooths 200um-1mm noise without loss of detail)
    # Holes closed once, on the high poly (mesh_cleaner.fill_holes_with_fairing): up to this many edges per hole,
    # then a few Laplacian steps on the patch. 0 = old behaviour (holes up to 30 edges, no fairing)
    max_hole_edges: int = 2000
    hole_fairing_steps: int = 5
    base_band_m: float = 0.015            # holes with a border this low are closed BEFORE the cut (G2)
    rim_radius_m: float = 0.155           # pieces beyond this radius are the disc rim, not the object (for 345mm platter)


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
    # High- and low-poly texrecon at the same time (one run peaked ~9 GB on a 120-photo scan): only with this much RAM free
    parallel_low: bool = True
    parallel_min_free_gb: float = 16.0


@dataclass
class MaskingConfig:
    """Settings for automated background matting, silhouette masking, and turntable disc elimination."""
    enabled: bool = True
    auto_generate: bool = True               # Automatically generate masks from 3D cylinder + chroma
    cylinder_radius_m: float = 0.080         # Radial bounding cylinder (80 mm for 260mm platter; 138 mm for 345mm)
    # Used when the hull reaches cylinder_radius_m (object off centre or wider): out to the disc rim
    wide_cylinder_radius_m: float = 0.125
    cylinder_y_min_m: float = 0.001          # Base cut 1.0 mm above turntable (matches slice_y_m = 1.0 mm)
    cylinder_y_max_m: float = 0.350          # Maximum object height 350 mm (the Parkside tool alone is 230 mm)
    chroma_threshold: float = 20.0           # Threshold for color vs neutral background
    brightness_white_threshold: float = 135.0 # Threshold for white photo-booth wall
    safety_dilation_px: int = 1              # Edge safety margin
    tag_blackout: bool = True                # Blackout of the AprilTags the object cannot cover (camera ray misses the cylinder)
    tag_expansion_factor: float = 1.25       # Expand tag quad by 25% to guarantee full black border suppression
    apply_to_texturing_images: bool = True   # Set background to black RGB(0,0,0) for TextureMesh / texrecon
    # Foreground: "fused" = BiRefNet_lite on a crop around the object, checked against the scan's own background
    # (per camera/level median + the disc rendered from the tag layout; mask_fusion.py): mean IoU 0.83 vs 0.32 for
    # "ai" on 70 views, no missed view, the disc out of the masks. "ai" = BiRefNet_lite on the whole photo (keeps the
    # disc), "photometric" = bright and colourless pixels are the booth (cuts white parts of objects). "fused" and
    # "ai" fall back to "photometric" without the model.
    method: str = "fused"
    ai_model_path: str = field(default_factory=lambda: os.environ.get("TURNTABLE_MASK_MODEL")
                               or "/opt/models/birefnet_lite_512.onnx")
    ai_resolution: int = 512                 # input of the 512 x 512 export; 384 px missed the aluminium bottle
    ai_threads: int = 2                      # ONNX Runtime threads per worker (8 workers x 2 threads was the fastest)
    ai_ram_per_worker_gb: float = 4.0        # measured peak ~3.8 GB per worker (ORT memory arena on, full-res arrays)
    # The model runs on 1 view in ai_view_stride (alternating shots); the others get the silhouette of the visual
    # hull carved from those: half the time, masks slightly larger than the object
    ai_view_stride: int = 2
    ai_min_area_ratio: float = 0.25          # AI mask < 25 % of the photometric one: model missed the object, keep the latter
    ai_used: bool = False                    # set by the masking step: the masks keep the turntable (visual hull cone)
    # Per-view check of the masks against a loose hull of all of them (3 mm voxels, a voxel cut by >= 3 views): a
    # view whose mask covers less than this share of that hull's silhouette misses part of the object the others
    # see; it is left out of the visual hull and its mask gets the hull silhouette added (Densify does not black
    # out the object). Not applied when more than max_excluded_share of the views fail (something systematic).
    min_hull_agreement: float = 0.75
    max_excluded_share: float = 0.25
    excluded_views: list = field(default_factory=list)   # set by the masking step (view ids)
    disc_radius_m: float = 0.170             # turntable top (345 mm platter): rays landing beyond see the booth
    # How DensifyPointCloud sees the masks: "blackout" = background pixels of undistorted/ set to 0 (default),
    # "mask_path" = images unchanged + --mask-path/--ignore-mask-label 0, "none" = images unchanged, no masks
    densify_masks: str = "blackout"


@dataclass
class VisualHullConfig:
    """Volume carved from the masks (visual_hull.py): proxy-point bounds and dense-cloud correction."""
    enabled: bool = True                     # Carve the hull after masking; its box bounds the proxy points / ROI
    refine_cloud: bool = True                # Prune wrong depths inside it and fill bare surfaces before ReconstructMesh
    voxel_m: float = 0.0015                  # 1.5 mm voxels: ~15 s for 120 views
    min_bg_votes: int = 2                    # Views that must see a voxel as background to remove it
    proxy_margin_m: float = 0.050            # Margin around the hull box for the proxy points (prevents trimming thin/tall features)
    prune_depth_m: float = 0.005             # Points deeper than this inside the hull ...
    prune_max_views: int = 3                 # ... and fused from at most this many views are dropped
    fill: bool = True                        # Append hull surface points where no MVS point is left
    support_dist_m: float = 0.003            # A hull point is "bare" with no MVS point within this distance
    max_normal_y: float = 0.8                # No fill on near-horizontal hull surfaces (the hull top is least reliable)
    # Hole-filling patches (mesh_cleaner) snapped onto the hull where they sag less than this inside it (2.11)
    snap_patches: bool = True
    snap_max_dist_m: float = 0.003


@dataclass
class RigRefineConfig:
    """Settings of the rigid per-(camera, level) correction."""
    enabled: bool = True
    scale: float = 0.5                  # features on the half-resolution image (2028 px): 4x faster, same result
    max_features: int = 1000            # SIFT keypoints per image; ~100-300 land on the object
    ratio: float = 0.75                 # Lowe ratio test
    max_axis_angle_deg: float = 40.0    # view pairs compared: optical axes closer than this
    max_obs_per_pair: int = 300
    static_px: float = 1.5              # booth point: same pixel (full resolution) in neighbouring shots ...
    static_min_neighbours: int = 2      # ... of at least this many of the 4 neighbours (shots -2..+2)
    max_sampson_px: float = 20.0        # gross outliers before the fit (measured: 2-4 px median, < 15 px between levels)
    coarse_huber_px: float = 3.0        # first fit on everything below max_sampson_px ...
    train_max_px: float = 6.0           # ... second fit on what agrees with the first one within this
    holdout_fraction: float = 1.0 / 3.0
    min_train_obs: int = 200            # fewer: nothing to fit
    min_group_obs: int = 40             # a (camera, level) seen in fewer training observations stays as it is
    min_improvement: float = 0.10       # hold-out median must drop at least 10 %
    max_rotation_deg: float = 1.5       # a correction beyond these is not a carriage tilt: rejected
    max_translation_mm: float = 15.0    # measured: up to ~8 mm (camera-local translation trades off with the tilt)
    huber_px: float = 1.0
    disc_radius_m: float = 0.135        # points kept over the disc ...
    y_min_m: float = 0.006              # ... above the paper: repeated tag patterns give false matches
    y_max_m: float = 0.40
    seed: int = 0


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
    visual_hull: VisualHullConfig = field(default_factory=VisualHullConfig)
    rig_refine: RigRefineConfig = field(default_factory=RigRefineConfig)
    texturing_images: TexturingImagesConfig = field(default_factory=TexturingImagesConfig)
    texrecon: TexReconConfig = field(default_factory=TexReconConfig)
    openmvs: OpenMVSConfig = field(default_factory=OpenMVSConfig)
    turntable: TurntableConfig = field(default_factory=TurntableConfig)
    cleaning: MeshCleaningConfig = field(default_factory=MeshCleaningConfig)

    # Scaling and units
    # OBJ, STL and 3MF units per metre of the reconstruction: 1000 = millimetres (what slicers, CAD and Fusion read;
    # 100 made every print file 10x too small in Bambu / Anycubic). object.json is measured in metres either way.
    export_scale: float = 1000.0
    # GLB units per metre. glTF is defined in metres (1.0; Blender and three.js then show the real size); 100.0 keeps
    # the centimetre GLB of the earlier releases (shown 100x too big) until the viewers are checked with metres.
    glb_scale: float = 100.0
    scale_stl_to_mm: bool = True

    # GLB geometry compressed with Draco (KHR_draco_mesh_compression): high GLB ~22 -> ~6 MB. Viewers need a
    # Draco decoder (Blender, three.js DRACOLoader, model-viewer, Babylon); False for tools without one.
    glb_draco: bool = True

    # Cloud & Artifact export
    export_zip: bool = True
    zip_name: str = "output.zip"
    # Streaming pre-work (runpod_job.py "stream"): SIFT, background medians and AI crops computed while the photos
    # arrived (pipeline/stream_cache.py); None = everything computed here
    stream_cache_dir: Optional[str] = None
