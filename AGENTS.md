# AGENTS.md — Turntable Studio Autonomous Agent Operations & Specification

Welcome, Agent. This document defines the architectural principles, invariants, mathematical conventions, operational rules, and future roadmap for **Turntable Studio**. Any AI agent working in this repository **must** read, understand, and strictly follow the guidelines detailed below.

---

## 1. System Mission & Core Purpose

Turntable Studio is a high-precision, industrial-grade 3D photogrammetry pipeline tailored specifically for **automated turntable scanning rigs** (dual Sony IMX477 12MP cameras with M12 lenses scanning technical plastics, power tools, and mechanical parts).

The system replaces the slow and noisy components of Meshroom (DepthMap, Meshing, Texturing) with **OpenMVS**, enforces rigidly locked camera poses via **TurntablePoses**, slices the turntable disc to produce a 2-manifold watertight CAD-ready base, and packages production-ready multi-LOD assets matching Cloudflare R2 and RunPod serverless job specifications.

### Physical Rig Specifications & Geometry Invariants:
- **Cameras:** Dual Sony IMX477 12.3MP sensors ($4056 \times 3040\text{ px}$) with 3.9mm M12 lenses ($\approx 75^\circ$ HFOV).
- **Same Vertical Height:** Both cameras are mounted on the **same horizontal carriage bar at the same height** ($Y \approx 135\text{ mm}$ above the platter, within $\pm 1.8\text{ mm}$ mechanical tolerance). Neither camera is physically mounted above or below the other.
- **Same Downward Pitch:** Both cameras are tilted down toward the platter center at the **same nominal pitch angle of $\approx 14^\circ - 15^\circ$**.
- **Horizontal Azimuthal Separation ($27^\circ$):** The cameras are offset horizontally around the turntable rotation axis by **$27^\circ$** (stereo baseline $b \approx 160\text{ mm}$).
- **Interleaved Vernier Sampling:** The $27^\circ$ camera separation and $18^\circ$ turntable rotation step form a **$1.5\times$ Vernier ratio** ($27^\circ = 18^\circ + 9^\circ$). Camera 1 samples $9^\circ$ between Camera 0's views, doubling effective horizontal angular resolution to 40 distinct perspectives ($9^\circ$ intervals) from 20 physical stops.
- **Elevation via Carriage Lift:** Vertical elevation changes happen exclusively when the T8 lead screw lifts the entire carriage between scanning levels (nominal $80\text{ mm}$ lift per level).
- **Platter Layout:** Default active platter is **$345\text{ mm}$** (`pipeline/platan_345mm_layout.json`, 12 AprilTags of Tag36h11 family, tag scale 0.99, $307\text{ mm}$ opposite tag distance). The legacy $260\text{ mm}$ platter (`pipeline/platan_260mm_layout.json`) remains supported.

---

## 2. The Golden Rules & Non-Negotiables

### RULE 1: Never Modify `turntable_poses.py` Directly
- `pipeline/turntable_poses.py` is an external mathematical module that computes AprilTag disc poses.
- It must be called as an external module or subprocess. Do **not** modify its internal algorithm, arguments, or JSON output format.
- Only exception (approved by the owner, 01.11.2026): the per-photo check in `write_report` runs `solvePnP` only with
  >= 4 corners left after the outlier pass (a large object hiding the tags crashed the job before the .sfm was written).

### RULE 2: Rigid Lock of Camera Poses (No BA Drift)
- On dark, semi-gloss, or smooth technical plastics, standard Structure-from-Motion (SfM) bundle adjustment (BA) will drift due to specular reflections and sparse feature points.
- Camera poses computed by `TurntablePoses` are **rigidly locked**. The pipeline feeds these locked poses directly to OpenMVS as ground truth.
- The only correction allowed is `pipeline/rig_refine.py`: one rigid 6-DOF correction per (camera, level) from sparse object features (the carriage tilts ~0.9 deg on the first lift, cam1 sits ~0.4 deg off its tag pose). Disc model, intrinsics and scale stay those of the tags; never per-photo poses. It is applied only when it lowers the epipolar error on held-out view pairs and stays small; otherwise the tag poses are kept (`rig_refine.json`, `cameras_known_tags.sfm`).

### RULE 3: Coordinate System Conventions & Conversions
- **AliceVision / TurntablePoses:** Rotation matrices $R$ are stored in **Column-Major (Fortran order, `order="F"`)** in `cameras_known.sfm`. $C$ is camera center in world coordinates.
- **OpenCV / COLMAP:** Pin-hole model using world-to-camera projection:
  $$\mathbf{X}_{\text{cam}} = R \cdot (\mathbf{X}_{\text{world}} - C) = R \cdot \mathbf{X}_{\text{world}} + T \implies T = -R \cdot C$$
- **Quaternion Format:** SciPy returns $[q_x, q_y, q_z, q_w]$. COLMAP strictly requires $[q_w, q_x, q_y, q_z]$.

### RULE 4: AliceVision Principal Point Conversion
- AliceVision stores `principalPoint` as an offset from the image center (e.g. $[-1.5, +2.3]$).
- When converting to OpenCV/COLMAP intrinsic matrices, the absolute pixel coordinates **must** be calculated as:
  $$c_x = \frac{W}{2} + \Delta c_x, \quad c_y = \frac{H}{2} + \Delta c_y$$
- Passing relative offsets directly shifts camera projection rays by half the image sensor ($\approx 2000\text{ px}$), causing `TextureMesh` to miss the object and project the black background.

### RULE 5: Bidirectional Covisibility Tracks for Proxy 3D Points
- `InterfaceCOLMAP` requires valid 3D points in `points3D.txt` with multi-view observations.
- `pipeline/bridge_sfm.py` generates 1,500 synthetic proxy 3D points inside a cylindrical bounding volume: the box of the
  visual hull (`pipeline/visual_hull.py`) plus 10 mm, or the masking cylinder without a hull. DensifyPointCloud derives
  its ROI and per-view depth range from them, so the volume must enclose the whole object.
- Points are projected into each camera view; only points visible in $\ge 2$ cameras are retained.
- Triplet projections $(u, v, \text{point\_id})$ in `images.txt` and track lists $(\text{image\_id}, \text{point2d\_idx})$ in `points3D.txt` must be strictly synchronized and 0-based.

### RULE 6: Watertight Base Slicing & Manifold Repair
- Raw photogrammetry produces jagged boundary geometry near the turntable contact plane.
- `pipeline/mesh_cleaner.py` executes:
  1. The object pieces chosen by position (vertices above the cut, inside the 115 mm rim radius), not the largest piece
     (the disc of a small object used to win); every dropped piece is logged with its size, radius and height.
  2. Holes whose border lies 3-15 mm above the cut closed BEFORE it (a hole reaching the cut opened the base outline).
  3. Planar slice at $Y = 3.0\text{ mm}$ (dropping the turntable surface), optional skirt down to $Y = 0.0\text{ mm}$.
  4. Every closed loop in the cut plane capped (polygons with holes: ring feet, several legs), ear clipping, normals
     strictly along $-Y$; loops chained along directed boundary half-edges.
  5. PyMeshLab (in mm): repair, decimation to the high-poly target, then the holes closed ONCE on the high poly
     (`fill_holes_with_fairing`: up to 2000 edges, only the patch remeshed (close_holes `refinehole` cost ~20 s per
     call), Laplacian fairing of the patch, patches snapped onto the visual hull where they sag < 3 mm inside it,
     coherent outward orientation, split pinch vertices moved apart so an STL stays manifold). 8/8 non-watertight v8
     models closed, sizes unchanged. The low poly is a topology-preserving decimation of it (re-closed if a collapse
     opens an edge), STL / 3MF come from it. Poisson was tested and rejected (watertight 3/8, 34 mm bulges).
  6. Problems the client must see go to `status.json` "warnings" (not watertight, tiny model, model at the disc rim,
     most faces dropped, masks disagreeing, hull only the cone over the disc).

### RULE 7: Multi-LOD Deliverable Contract
The pipeline must always produce the following asset structure in `output/`:
- **High-Poly OBJ (`output/obj/high/`):** `model_high.obj`, `model_high.mtl`, and lossless `model_high_texture.png`.
- **Low-Poly OBJ (`output/obj/low/`):** `model_low.obj`, `model_low.mtl`, and web-optimized `model_low_texture.jpg` (downscaled, max 2048px). **Must contain valid UV texture coordinates (`vt`)**.
- **GLB Models (`output/glb/`):** `high_model.glb` (embedded PNG texture) and `low_model.glb` (embedded JPG texture), each ONE object (one primitive per atlas), geometry Draco-compressed (`--no_draco` to disable).
- **3D Printing Assets:** Watertight `high_model.stl`, `low_model.stl`, `high_model.3mf`, `low_model.3mf`.
- **Units:** OBJ / STL / 3MF in millimetres (`export_scale` 1000; STL / 3MF Z-up, base on Z = 0), GLB in `glb_scale`
  units per metre (100 = cm, the current viewers; 1 = metres, the glTF standard).
- **Object Card (`output/object.json`):** title, description, length × width × height (mm) and volume (cm³) of the cleaned mesh.
- **Package Archive:** `output.zip` containing all multi-LOD deliverables for Cloudflare R2 / S3.

### RULE 8: OpenMVS Texturing Optimization
To prevent dark patches, Poisson bleeding, and micro-confetti texture atlases:
- `--cost-smoothness-ratio 1.0`: Weights smoothness heavily, forcing adjacent faces into large, continuous charts rather than 480,000 micro-islands.
- `--global-seam-leveling 0`: Disables global Poisson gradient equalization that bleeds black photo-booth backgrounds across the surface.
- `--local-seam-leveling 1`: Smooths boundaries right at chart seams.
- `--empty-color 0`: Fills unobserved faces (e.g. flat cut bottom) and empty atlas areas with clean black (`RGB 0,0,0`), not neon orange.

### RULE 9: Testing & Documentation Integrity
- All unit tests in `tests/` (190 at the time of writing) must pass before any commit.
- Preserve existing docstrings, comments, and structure.

---

## 3. Architecture & Codebase Map

```text
turntable_studio/
├── main.py                     # Master CLI and pipeline execution orchestrator (writes output/run_log.txt itself)
├── runpod_job.py               # Inside a RunPod pod: download photos from R2 (or stream them while the scanner takes them:
│                               #   "stream" job, capture_done.json / level_done_<L>.json / cancel.json), run_pipeline,
│                               #   progress.json per step, upload output + debug archive in parallel, status.json last
├── docker_entrypoint.sh        # Container entry: runpod_job.py when JOB_ID is set, otherwise main.py
├── docker/runtime_deps.sh      # Copies the shared libraries of OpenMVS/texrecon into the slim runtime image
├── requirements-runtime.txt    # Python packages of the image (requirements.txt = runtime + local tools)
├── .env.example                # Keys for the cloud flow (.env is ignored by git and Docker)
├── cloud/
│   ├── cloudflare_r2.py        # R2 client, jobs/<id>/input|output layout, temporary prefix-scoped credentials
│   ├── runpod.py               # RunPod REST pods: GPUs fastest first under a price cap + preferred, then min RAM/vCPU, guaranteed termination, bounded retries
│   └── job.py                  # Whole cloud flow for one scan (upload -> pod -> download -> cleanup)
├── tools/                      # Local only (not in the image): cloud_scan.py, docker_scan.ps1, fit_ram_model.py, collect_stats.py,
│                               #   debug_texturing.py (texture diagnostics), debug_blender_compare.py (headless Blender),
│                               #   pull_debug.py (debug bucket -> PC, deleted from R2 after a verified download),
│                               #   regression.py (table of a results folder vs a reference run),
│                               #   stream_sim.py (streaming replay of an old scan: cached front of the pipeline = uncached)
├── Dockerfile                  # Multi-stage CUDA 12 + OpenMVS production container
├── docker-compose.yml          # Container configuration with GPU passthrough
├── requirements.txt            # Python dependencies (NumPy, SciPy, Trimesh, PyMeshLab, etc.)
├── pipeline/
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Dataclass configurations and OpenMVS binary locator
│   ├── telemetry.py            # Hardware resource profiler (Peak RAM, VRAM, per-step timing, run_info, warnings)
│   ├── resources.py            # cgroup-aware CPUs/RAM, DensifyPointCloud RAM estimate + check, worker count
│   ├── parallel.py             # Spawned process pool whose workers log into run_log.txt (background metrics, exports)
│   ├── taskgraph.py            # Small task executor: dependencies, resource limits, critical-path order, records
│   ├── stream_cache.py         # Streaming caches: SIFT per photo, background medians per group, AI crops (exact hits only)
│   ├── stream_prework.py       # Pod work while the photos arrive: SIFT, medians, preliminary poses + AI of finished levels
│   ├── run_metrics.py          # stats.json features for RAM/time/quality models (hardware, disk speed, object, logs)
│   ├── camera_init.py          # Image discovery (flat or rig/0 + rig/1) and CameraInit SFM
│   ├── turntable_poses.py      # AprilTag disc detection and rigid pose calculation (LOCKED)
│   ├── rig_refine.py           # Per-(camera, level) rigid correction from sparse SIFT features, hold-out guarded
│   ├── platan_260mm_layout.json# 260mm disc tag layout coordinates
│   ├── platan_345mm_layout.json# 345mm disc tag layout coordinates (12 tags, 30mm, 307mm opposite dist)
│   ├── bridge_sfm.py           # SFM-to-COLMAP bridge, OpenCV undistortion, proxy 3D points
│   ├── visual_hull.py          # Volume carved from the masks: proxy bounds, scene_dense.ply prune + fill before ReconstructMesh,
│   │                           #   ambiguous cone over the disc (AI masks keep it), hull silhouettes for skipped views
│   ├── openmvs_runner.py       # Cross-platform CLI runner for OpenMVS binaries (Densify, Reconstruct, RefineMesh)
│   ├── mesh_cleaner.py         # Planar slicing, skirt extrusion, Ear-Clipping base capping
│   ├── mesh_masks.py           # Vectorised rasterizer: mesh silhouette / face-id buffers from locked poses
│   ├── texture_prep.py         # texturing_images/: PNG, background exactly 0, mesh masks, per-camera colour gains
│   ├── texrecon_runner.py      # .cam scene export (rebuilt every run) and texrecon execution
│   ├── exporter.py             # Multi-LOD decimation, texture conversion, GLB embedding
│   ├── object_info.py          # object.json: title, description, real size (mm) and volume of the cleaned mesh
│   ├── mask_processor.py       # Masks: BiRefNet_lite 512 ONNX on the CPU (white parts, through-holes) or photometric
│   │                           #   fallback; masking cylinder + tag blackout
│   ├── mask_fusion.py          # Fused masks (default): AI on a crop + per-(camera, level) background median + the disc
│   │                           #   rendered from tag layout; keeps floating upper parts (screwdriver) & metallic base (spray)
│   ├── debug_lite.py           # Debug archive of a scan (raw mesh Draco, masks 1/4, hull, refine points, small files)
│   └── calibrate_intrinsics.py # Chessboard/Charuco camera matrix calibration
└── tests/
    ├── test_bridge_sfm.py      # Camera matrix, quaternion, proxy track tests, generated masks found and applied
    ├── test_object_info.py     # object.json measurements independent of the turn on the disc
    ├── test_visual_hull.py     # Hull carving, proxy bounds, dense PLY I/O, prune + fill on a synthetic cylinder,
    │                           #   ambiguous disc cone, hull silhouettes, refinement skipped on a too short hull
    ├── test_rig_refine.py      # Recovers a known carriage tilt / cam1 offset; guards keep the tag poses; sfm column-major
    ├── test_main_checks.py     # Tag-correction warning, cylinder widened only for real (decided) contact, cone-only
    │                           #   warning, cleaning warnings to status.json, hole patches snapped onto the hull
    ├── test_parallel.py        # Worker logs forwarded, export with a pool = sequential export, texrecon scene once, zip
    ├── test_mesh_cleaner.py    # Ear clipping, every base loop capped (ring foot, legs), hole at the cut, object by
    │                           #   position next to a large disc, large hole filled once, warnings
    ├── test_mask_fusion.py     # Fused masks on a synthetic turntable: disc removed, object kept, crop box, hull check
    ├── test_debug_lite.py      # Debug archive parts and budget; upload only with debug-bucket credentials
    ├── test_regression_tool.py # tools/regression.py: regressions and improvements against a reference
    ├── test_openmvs_runner.py  # TextureMesh PLY (texcoord, texnumber) -> OBJ + MTL conversion
    ├── test_exporter.py        # Multi-LOD OBJ, MTL, JPG/PNG, and GLB bundle tests (incl. multi-material GLB)
    ├── test_texrecon_runner.py # .cam export, scene/ rebuilt per run, unseen-face patch
    ├── test_texture_prep.py    # Mesh rasterizer, PNG background exactly 0, colour gains
    ├── test_camera_init.py     # turntable_poses command line (--liftMm, --tagRefine)
    ├── test_turntable_poses_report.py # report survives photos with < 4 tag corners left (no solvePnP crash)
    ├── test_pull_debug.py      # debug bucket pulled to the PC: verified before delete, never the delivery bucket
    ├── test_resources.py       # cgroup limits, RAM estimate vs measured runs, worker count
    ├── test_run_metrics.py     # log parsers, object/mesh metrics, telemetry step fields, CSV collector
    ├── test_cloud.py           # R2 layout, temporary credentials, pod lifecycle, preferred then minimum pod request (in-memory S3, fake RunPod)
    ├── test_telemetry.py       # RAM/VRAM profiler and timer tests
    ├── test_taskgraph.py       # Task executor: dependencies, partial failure, no hang, critical path, limits
    └── test_stream.py          # Streaming pod: photos as they appear, markers, cancel, scanner gone silent, pre-work
```

---

## 4. Planned Roadmap & Future Extensions

The following features are slated for upcoming agent implementations:

### Priority 1: Automated Background Matting (BiRefNet / SAM) — implemented (BiRefNet_lite, CPU)
- `masking.method = "fused"` (default, `pipeline/mask_fusion.py`): BiRefNet_lite 512 ONNX on a crop around the object
  (grown and run again while the object leaves it), accepted where the scan's own background (per-(camera, level)
  median, disc rendered from the tag layout) agrees; every view. 70 views of 7 scans: mean IoU 0.32 -> 0.83 (local, 01.11).
- Every mask is checked against a loose hull of all of them; a view that misses part of the object is left out of the
  visual hull and gets the hull silhouette added (`masking.min_hull_agreement`).
- `masking.method = "ai"`: the model on the whole photo, 1 view in `ai_view_stride`, the others the hull silhouette; it
  keeps the turntable (the visual hull marks the cone over the disc it cannot judge). Photometric fallback without the model.
- Pending: first measured pod run (time per image with the pod's vCPUs; ~4 GB RAM per worker).

### Priority 2: AI Surface Normal Refinement & High-Relief Guidance
- For smooth plastics with fine relief (buttons, logos, engraved text), raw photogrammetry can suffer from high-frequency depth noise.
- Integrate normal estimation (e.g. DSINE or depth-guided edge-aware bilateral smoothing) to guide surface planarities without flattening embossed text.

### Priority 3: Automatic Reprojection Quality Gate
- Compute root-mean-square error (RMSE) of AprilTag reprojections.
- If $\text{RMSE} > 0.8\text{ px}$, trigger automatic calibration warning or recommend lens lock adjustment.

### Priority 4: Cloud Deployment (RunPod pods + Cloudflare R2) — implemented, first real run pending
- One pod per scan (`cloud/job.py`, `runpod_job.py`); see `cloud/README.md`.
- Pending: first measured run (`tools/cloud_scan.py`), image pushed to a registry, GPU id list checked.

---

## 5. Verification & Testing Procedures

When modifying code in `turntable_studio`:

```bash
# 1. Run full unit test suite
cd turntable_studio
uv run --with numpy --with opencv-python-headless --with scipy --with trimesh --with shapely --with mapbox-earcut --with psutil --with pillow --with networkx --with pymeshlab --with boto3 --with requests --with lxml --with DracoPy python -m unittest discover -v -s tests

# 2. Test CLI argument parser
uv run --with pillow --with opencv-python-headless --with scipy --with trimesh --with shapely --with mapbox-earcut --with psutil python main.py --help

# 3. Test mock execution (smoketest without OpenMVS binaries)
python main.py --mock_openmvs --rig_mode --input_dir ./input_images --output_dir ./output_test
```
