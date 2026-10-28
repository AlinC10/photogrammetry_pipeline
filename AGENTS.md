# AGENTS.md — Turntable Studio Autonomous Agent Operations & Specification

Welcome, Agent. This document defines the architectural principles, invariants, mathematical conventions, operational rules, and future roadmap for **Turntable Studio**. Any AI agent working in this repository **must** read, understand, and strictly follow the guidelines detailed below.

---

## 1. System Mission & Core Purpose

Turntable Studio is a high-precision, industrial-grade 3D photogrammetry pipeline tailored specifically for **automated turntable scanning rigs** (e.g., dual Sony IMX477 12MP cameras with M12 lenses scanning technical plastics, power tools, and mechanical parts).

The system replaces the slow and noisy components of Meshroom (DepthMap, Meshing, Texturing) with **OpenMVS**, enforces rigidly locked camera poses via **TurntablePoses**, slices the turntable disc to produce a 2-manifold watertight CAD-ready base, and packages production-ready multi-LOD assets matching Cloudflare R2 and RunPod serverless job specifications.

---

## 2. The Golden Rules & Non-Negotiables

### RULE 1: Never Modify `turntable_poses.py` Directly
- `pipeline/turntable_poses.py` is an external mathematical module that computes AprilTag disc poses.
- It must be called as an external module or subprocess. Do **not** modify its internal algorithm, arguments, or JSON output format.

### RULE 2: Rigid Lock of Camera Poses (No BA Drift)
- On dark, semi-gloss, or smooth technical plastics, standard Structure-from-Motion (SfM) bundle adjustment (BA) will drift due to specular reflections and sparse feature points.
- Camera poses computed by `TurntablePoses` are **rigidly locked**. The pipeline feeds these locked poses directly to OpenMVS as ground truth.

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
  1. Planar slice at $Y = 3.0\text{ mm}$ (dropping the turntable surface).
  2. Extrusion of a prismatic vertical skirt down to $Y = 0.0\text{ mm}$.
  3. 2D Ear-Clipping triangulation of the base polygon with winding normals forced strictly along $-Y$.
  4. PyMeshLab manifold repair (closing small holes, removing non-manifold vertices/edges).

### RULE 7: Multi-LOD Deliverable Contract
The pipeline must always produce the following asset structure in `output/`:
- **High-Poly OBJ (`output/obj/high/`):** `model_high.obj`, `model_high.mtl`, and lossless `model_high_texture.png`.
- **Low-Poly OBJ (`output/obj/low/`):** `model_low.obj`, `model_low.mtl`, and web-optimized `model_low_texture.jpg` (downscaled, max 2048px). **Must contain valid UV texture coordinates (`vt`)**.
- **GLB Models (`output/glb/`):** `high_model.glb` (embedded PNG texture) and `low_model.glb` (embedded JPG texture).
- **3D Printing Assets:** Watertight `high_model.stl`, `low_model.stl`, `high_model.3mf`, `low_model.3mf`.
- **Package Archive:** `output.zip` containing all multi-LOD deliverables for Cloudflare R2 / S3.

### RULE 8: OpenMVS Texturing Optimization
To prevent dark patches, Poisson bleeding, and micro-confetti texture atlases:
- `--cost-smoothness-ratio 1.0`: Weights smoothness heavily, forcing adjacent faces into large, continuous charts rather than 480,000 micro-islands.
- `--global-seam-leveling 0`: Disables global Poisson gradient equalization that bleeds black photo-booth backgrounds across the surface.
- `--local-seam-leveling 1`: Smooths boundaries right at chart seams.
- `--empty-color 0`: Fills unobserved faces (e.g. flat cut bottom) and empty atlas areas with clean black (`RGB 0,0,0`), not neon orange.

### RULE 9: Testing & Documentation Integrity
- All unit tests in `tests/` (97 at the time of writing) must pass before any commit.
- Preserve existing docstrings, comments, and structure.

---

## 3. Architecture & Codebase Map

```text
turntable_studio/
├── main.py                     # Master CLI and pipeline execution orchestrator (writes output/run_log.txt itself)
├── runpod_job.py               # Inside a RunPod pod: download photos from R2, run_pipeline, upload output + status.json
├── docker_entrypoint.sh        # Container entry: runpod_job.py when JOB_ID is set, otherwise main.py
├── docker/runtime_deps.sh      # Copies the shared libraries of OpenMVS/texrecon into the slim runtime image
├── requirements-runtime.txt    # Python packages of the image (requirements.txt = runtime + local tools)
├── .env.example                # Keys for the cloud flow (.env is ignored by git and Docker)
├── cloud/
│   ├── cloudflare_r2.py        # R2 client, jobs/<id>/input|output layout, temporary prefix-scoped credentials
│   ├── runpod.py               # RunPod REST pods: GPU list + min RAM/vCPU, guaranteed termination, bounded retries
│   └── job.py                  # Whole cloud flow for one scan (upload -> pod -> download -> cleanup)
├── tools/                      # Local only (not in the image): cloud_scan.py, fit_ram_model.py, collect_stats.py,
│                               #   debug_texturing.py (texture diagnostics), debug_blender_compare.py (headless Blender)
├── Dockerfile                  # Multi-stage CUDA 12 + OpenMVS production container
├── docker-compose.yml          # Container configuration with GPU passthrough
├── requirements.txt            # Python dependencies (NumPy, SciPy, Trimesh, PyMeshLab, etc.)
├── pipeline/
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Dataclass configurations and OpenMVS binary locator
│   ├── telemetry.py            # Hardware resource profiler (Peak RAM, VRAM, per-step timing, run_info)
│   ├── resources.py            # cgroup-aware CPUs/RAM, DensifyPointCloud RAM estimate + check, worker count
│   ├── run_metrics.py          # stats.json features for RAM/time/quality models (hardware, disk speed, object, logs)
│   ├── camera_init.py          # Image discovery (flat or rig/0 + rig/1) and CameraInit SFM
│   ├── turntable_poses.py      # AprilTag disc detection and rigid pose calculation (LOCKED)
│   ├── platan_260mm_layout.json# 260mm disc tag layout coordinates
│   ├── bridge_sfm.py           # SFM-to-COLMAP bridge, OpenCV undistortion, proxy 3D points
│   ├── visual_hull.py          # Volume carved from the masks: proxy bounds, scene_dense.ply prune + fill before ReconstructMesh
│   ├── openmvs_runner.py       # Cross-platform CLI runner for OpenMVS binaries
│   ├── mesh_cleaner.py         # Planar slicing, skirt extrusion, Ear-Clipping base capping
│   ├── mesh_masks.py           # Vectorised rasterizer: mesh silhouette / face-id buffers from locked poses
│   ├── texture_prep.py         # texturing_images/: PNG, background exactly 0, mesh masks, per-camera colour gains
│   ├── texrecon_runner.py      # .cam scene export (rebuilt every run) and texrecon execution
│   ├── exporter.py             # Multi-LOD decimation, texture conversion, GLB embedding
│   ├── mask_processor.py       # Alpha mask utilities
│   └── calibrate_intrinsics.py # Chessboard/Charuco camera matrix calibration
└── tests/
    ├── test_bridge_sfm.py      # Camera matrix, quaternion, proxy track tests, generated masks found and applied
    ├── test_visual_hull.py     # Hull carving, proxy bounds, dense PLY I/O, prune + fill on a synthetic cylinder
    ├── test_mesh_cleaner.py    # 2D Ear-clipping and watertight base tests
    ├── test_exporter.py        # Multi-LOD OBJ, MTL, JPG/PNG, and GLB bundle tests (incl. multi-material GLB)
    ├── test_texrecon_runner.py # .cam export, scene/ rebuilt per run, unseen-face patch
    ├── test_texture_prep.py    # Mesh rasterizer, PNG background exactly 0, colour gains
    ├── test_camera_init.py     # turntable_poses command line (--liftMm, --tagRefine)
    ├── test_resources.py       # cgroup limits, RAM estimate vs measured runs, worker count
    ├── test_run_metrics.py     # log parsers, object/mesh metrics, telemetry step fields, CSV collector
    ├── test_cloud.py           # R2 layout, temporary credentials, pod lifecycle (in-memory S3, fake RunPod)
    └── test_telemetry.py       # RAM/VRAM profiler and timer tests
```

---

## 4. Planned Roadmap & Future Extensions

The following features are slated for upcoming agent implementations:

### Priority 1: Automated Background Matting (BiRefNet / SAM)
- Integrate a pre-processing step that automatically segments the foreground object and removes turntable background noise before `DensifyPointCloud`.
- Generate `<image_name>.mask.png` in `output/masks/` and feed `--mask-path` and `--ignore-mask-label 0` into `DensifyPointCloud`.

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
uv run --with numpy --with opencv-python-headless --with scipy --with trimesh --with shapely --with mapbox-earcut --with psutil --with pillow --with networkx --with pymeshlab --with boto3 --with requests --with lxml python -m unittest discover -v -s tests

# 2. Test CLI argument parser
uv run --with pillow --with opencv-python-headless --with scipy --with trimesh --with shapely --with mapbox-earcut --with psutil python main.py --help

# 3. Test mock execution (smoketest without OpenMVS binaries)
python main.py --mock_openmvs --rig_mode --input_dir ./input_images --output_dir ./output_test
```
