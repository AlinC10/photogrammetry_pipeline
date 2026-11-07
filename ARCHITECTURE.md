# Turntable Studio — Technical Architecture & Pipeline Specification

## 1. Executive Overview & Problem Statement

**Turntable Studio** is an industrial-grade 3D photogrammetry and surface reconstruction system purpose-built for automated turntable scanning rigs. 

### The Problem in Industrial Photogrammetry
Conventional photogrammetry suites (e.g. standard Meshroom, RealityCapture, or COLMAP pipelines) are designed primarily for aerial drones, architecture, and handheld outdoor scanning. When applied to technical turntable rigs scanning industrial plastics, machined parts, and power tools, they fail catastrophically for three reasons:
1. **Bundle Adjustment (BA) Drift on Low-Texture Surfaces:** Smooth, dark, semi-gloss, or single-color plastics provide few stable natural keypoints. Specular highlights move with the light source rather than the object, tricking feature matchers into calculating incorrect camera trajectories.
2. **Rotating Object vs. Static Environment Conflict:** Standard Structure-from-Motion (SfM) assumes a static world observed by moving cameras. On a turntable, the object rotates while background walls and the room stay fixed. This produces contradictory epipolar rays that corrupt camera poses or place duplicate ghost geometry around the scene.
3. **Ragged Non-Manifold Platter Contact Boundaries:** Photogrammetric depth estimation cannot cleanly distinguish where the bottom of the object ends and the platter surface begins. Raw reconstructions yield jagged, noisy, non-manifold fringes around the base, requiring extensive manual CAD repair before the asset can be sliced for 3D printing or loaded into simulation.

### The Turntable Studio Solution
Turntable Studio replaces stochastic pose estimation with **deterministic geometric anchoring**, leveraging an outer ring of **AprilTags** on the platter, a **23-parameter global kinematic model**, **CUDA-accelerated OpenMVS reconstruction**, **hybrid AI/photometric silhouette masking**, **watertight base slicing and ear-clipping capping**, and **MRF Potts texture optimization**.

---

## 2. Physical Rig Architecture & Invariants

```text
                             [ T8 Z-Axis Lead Screw ]
                                        │
                         ▲              │  Nominal 80 mm lift/level
                         │              ▼
             ┌──────────────────────────────────────┐
             │    Rigid Horizontal Carriage Bar     │
             └───┬──────────────────────────────┬───┘
                 │                              │
           [ Camera 0 ]                   [ Camera 1 ]
         Sony IMX477 12MP               Sony IMX477 12MP
         (3.9mm M12 Lens)               (3.9mm M12 Lens)
            Height: ~135 mm                Height: ~135 mm (Same level, ±1.8 mm)
            Pitch:  ~14°-15° down          Pitch:  ~14°-15° down
                 \                              /
                  \       ◄── 27° ──►          /
                   \  (Horizontal Azimuth)    /
                    \                        /
                     ▼                      ▼
             ════════════════════════════════════════
                   [ 345 mm Turntable Platter ]
                   12 AprilTags (Tag36h11) Outer Ring
                   Rotates in 18° Steps (20 steps/turn)
```

### Hardware Specifications:
* **Imaging Sensors:** Dual Sony IMX477 12.3MP rolling-shutter CMOS sensors ($4056 \times 3040\text{ px}$ resolution, $1.865\,\mu\text{m}$ physical pixel pitch).
* **Optics:** Dual 3.9mm focal length M12 lenses ($\approx 75^\circ$ horizontal FOV, low barrel distortion $k_1 \approx -0.04$).
* **Planar Horizontal Mounting:** Both cameras are mounted on the **same rigid horizontal carriage bar at the same vertical height** ($Y \approx 135\text{ mm}$ above the platter plane, within $\pm 1.8\text{ mm}$ mechanical assembly tolerance). Neither camera is positioned above or below the other.
* **Matched Downward Pitch:** Both cameras are tilted down toward the platter center at the **same nominal pitch angle of $\approx 14^\circ - 15^\circ$**.
* **Horizontal Azimuthal Baseline ($27^\circ$):** The cameras are separated horizontally by **$27^\circ$** around the turntable rotation axis ($159.9\text{ mm}$ direct stereo baseline).
* **Vernier Interleaved Sampling ($1.5\times$ Ratio):**
  $$\frac{27^\circ \text{ (camera separation)}}{18^\circ \text{ (turntable step)}} = 1.5 = 1 + 0.5 \implies 27^\circ = 18^\circ + 9^\circ$$
  Because Camera 1 is offset by $1.5\times$ the turntable step, at each rotation step $k$, Camera 1 observes a viewpoint shifted by $9^\circ$ relative to Camera 0's previous position. Over 20 physical motor stops, the rig captures **40 distinct horizontal perspectives at $9^\circ$ intervals**, effectively doubling the horizontal angular density without increasing mechanical rotation time.
* **Vertical Motion:** A T8 lead screw with stepper motor lifts the entire dual-camera carriage by $\approx 80\text{ mm}$ between scanning levels (Level 1 $\approx 135\text{ mm} \to$ Level 2 $\approx 217\text{ mm} \to$ Level 3 $\approx 298\text{ mm}$). Both cameras change elevation together.
* **Turntable Platter:** Default active platter is $345\text{ mm}$ diameter with 12 AprilTags (Tag36h11 family, tag scale 0.99, $307\text{ mm}$ opposite tag distance).

---

## 3. Technology Stack & Technical Rationale

| Component | Technology | Rationale & Advantage over Alternatives |
|---|---|---|
| **Pose Estimation** | AprilTags (Tag36h11) + OpenCV `solvePnP` | Eliminates feature-matching drift on dark/glossy plastics. Provides absolute sub-pixel ground truth ($\text{RMSE} < 0.45\text{ px}$) and exact 1:1 metric scale. |
| **Global Kinematics** | 23-parameter solver (`scipy.optimize.least_squares`) | Reduces 720 unconstrained SfM degrees of freedom to 23 physical constants. Enables **Zero-Tag Prediction** when large parts occlude all tags. |
| **Mechanical Correction** | `pipeline/rig_refine.py` | SIFT-based rigid 6-DOF correction per (camera, level) to compensate for carriage cantilever flex ($\approx 0.9^\circ$), guarded by hold-out validation. |
| **Foreground Masking** | BiRefNet_lite 512 ONNX CPU + Background Median + 3D Disc | Runs efficiently on CPU without GPU memory contention. Preserves thin shafts and metallic bases on white platters. |
| **Volumetric Prior** | 1.5 mm Voxel Visual Hull | Carves exact bounding volume from multi-view silhouettes. Bounds proxy points, prunes phantom depth points, and fills textureless regions. |
| **Depth Estimation** | OpenMVS `DensifyPointCloud` (CUDA PatchMatch) | Massively parallel GPU depth map computation; 10x faster than CPU-based AliceVision DepthMap. |
| **Surface Meshing** | OpenMVS `ReconstructMesh` (Delaunay + Graph-Cut) | Extracts 2-manifold surface with `--smooth 4` and `--min-point-distance 2.0`, eliminating boundary jaggedness in 20s without heavy RefineMesh. |
| **Base Sealing** | `pipeline/mesh_cleaner.py` (Ear-Clipping + Taubin) | Cuts at $Y = 3.0\text{ mm}$, caps all boundary loops flat with 2D ear clipping, and applies volume-preserving Taubin smoothing for direct 3D printing. |
| **UV Texturing** | mvs-texturing (`texrecon`) MRF Potts Model | Assembles large, continuous texture charts; eliminates confetti micro-patches and Poisson gradient bleeding. |
| **Packaging & Export** | Trimesh + PyMeshLab + DracoPy | Multi-LOD outputs (High-Poly and Low-Poly OBJ, Draco GLB, STL, 3MF) and physical dimension profiling in `object.json`. |
| **Cloud Execution** | RunPod Serverless GPU + Cloudflare R2 | Rents high-end GPUs on demand (~8 cents/scan). Streams pre-computations while scanner captures photos. |

---

## 4. End-to-End Pipeline Walkthrough

### Step 1: Camera Discovery & Init (`pipeline/camera_init.py`)
* Scans `input_images/rig/0/` and `input_images/rig/1/`.
* Parses filename metadata (`f_<level><step:03d>.jpg`) via regex.
* Validates image resolutions and generates AliceVision `cameraInit.sfm` format with nominal focal length ($3.9\text{ mm}$).

### Step 2: Deterministic Pose Estimation (`pipeline/turntable_poses.py`)
1. **AprilTag Detection:** Detects Tag36h11 markers using OpenCV's AprilTag detector, extracting sub-pixel corners.
2. **Stage 1 (Per-View PnP with RANSAC):** Computes initial camera poses $(R_i, C_i)$ for frames with $\ge 2$ visible tags using `cv2.solvePnPRansac`.
3. **Stage 2 (Geometric Deduction):**
   * Fits a 3D circle (`fit_circle_3d`) through camera centers via Singular Value Decomposition (SVD) to identify the rotation axis $\mathbf{a}$ and center $\mathbf{p}$.
   * Performs linear regression on unrolled azimuth angles to verify direction $\sigma \in \{+1, -1\}$ and angular step $\gamma \approx 18.0^\circ$.
   * Derotates camera centers to step 0 to obtain base centers $(C_0, C_1)$ via medians.
4. **Stage 3 (Joint Nonlinear Optimization):**
   * Assembles 23 physical parameters: step $\gamma$, platter center $(p_x, p_z)$, axis tilt $(t_x, t_z)$, carriage lift scale $s_{\text{lift}}$, lift tilt $(u_x, u_z)$, level offsets $\delta_L$, camera centers $(C_0, C_1)$, rotations $(R_0, R_1)$, and camera intrinsics $(f, c_x, c_y, k_1)$.
   * Optimizes via `scipy.optimize.least_squares` using the robust **`soft_l1`** loss function:
     $$\min_{\boldsymbol{\Theta}} \sum_{k} \sum_{j} \rho\left( \|\pi(\mathbf{X}_j; \boldsymbol{\Theta}) - \mathbf{u}_{k,j}\|^2 \right)$$
   * Achieves global Root Mean Square Error (RMSE) under **0.45 pixels** ($0.84\,\mu\text{m}$ on silicon, $\approx 0.064\text{ mm}$ spatial accuracy at $300\text{ mm}$ working distance).
5. **Zero-Tag Prediction:** For frames where a large object completely covers the AprilTags, camera poses are computed deterministically from the kinematic equation:
   $$\mathbf{R}_{L, k, c} = \mathbf{R}_c \cdot \mathbf{R}_a(\theta_{L, k}), \quad \mathbf{C}_{L, k, c} = \mathbf{p} + \mathbf{R}_a^\top(\theta_{L, k}) \cdot \left[ \mathbf{C}_c + (L - 1) \cdot h \cdot s_{\text{lift}} \cdot \mathbf{u} - \mathbf{p} \right]$$

### Step 2.1: Rig Refinement (`pipeline/rig_refine.py`)
* Solves the physical deflection ($\approx 0.8^\circ - 0.9^\circ$) of the cantilever carriage arm when lifted $80\text{ mm}$ on the Z-axis.
* Extracts ~150 SIFT features on the object across adjacent views.
* Computes a single rigid 6-DOF correction $(\Delta \mathbf{R}, \Delta \mathbf{C})$ per (camera, level).
* **Hold-out Guardrail:** Accepted only if the Sampson epipolar error decreases by $\ge 10\%$ on held-out test view pairs and rotation correction is $< 1.5^\circ$. Otherwise, pure tag poses are preserved.

### Step 2.5: Hybrid Mask Fusion (`pipeline/mask_fusion.py`)
Produces crisp object silhouettes on every frame:
1. **AI Crop Matting:** Crops tightly around the object based on multi-view bounding box projections and runs **BiRefNet_lite 512 ONNX** on CPU.
2. **Background Median Subtraction:** Computes the temporal median pixel value per $(camera, level)$ across all 20 rotational stops. Pixels matching the static background are stripped.
3. **Synthetic Platter Blackout:** Renders the 3D platter disc from known tag layout coordinates, projecting a clean black disc into every view.
4. **Vertical Morphological Closing:** A $7 \times 1\text{ px}$ 1D vertical kernel preserves thin vertical shafts (e.g. stepper motor shafts, screwdriver tips).

### Step 2.6: Visual Hull Carving (`pipeline/visual_hull.py`)
* Constructs a regular 3D voxel grid ($1.5\text{ mm}$ resolution) bounded by the platter radius and vertical travel.
* Each voxel is projected into all camera views; voxels falling into background in $\ge 2$ views are carved away.
* The remaining hull bounds the synthetic tracking points and guides depth pruning.

### Step 3: SFM-to-COLMAP Bridge (`pipeline/bridge_sfm.py`)
* Undistorts 4K camera frames using Brown-Conrady radial distortion parameters ($k_1$).
* Converts AliceVision column-major rotations to COLMAP world-to-camera matrices:
  $$\mathbf{R}_{\text{colmap}} = \mathbf{R}_{\text{sfm}}^\top, \quad \mathbf{T}_{\text{colmap}} = -\mathbf{R}_{\text{sfm}}^\top \cdot \mathbf{C}$$
* Injects 1,500 synthetic proxy 3D points inside the visual hull bounding box, establishing synchronized 2D-3D covisibility observation tracks across $\ge 2$ views.

### Step 4: OpenMVS Dense Reconstruction (`pipeline/openmvs_runner.py`)
* **`DensifyPointCloud`:** Uses CUDA PatchMatch to compute per-pixel depth and normal maps, fusing depth hypotheses across multiple views (`--number-views-fuse 2`).
* **Hull Cloud Pruning (4.2b):** Dense points located $> 5\text{ mm}$ inside the visual hull supported by $\le 3$ views (common depth errors on uniform plastic) are pruned; bare hull surface points are added to fill untextured regions.
* **`ReconstructMesh`:** Performs 3D Delaunay tetrahedralization followed by graph-cut ($s-t$ min-cut) energy minimization. Surface smoothing parameters (`--smooth 4`, `--min-point-distance 2.0`) eliminate boundary jaggedness and micro-faceting directly during meshing in ~20 seconds.

### Step 5: Watertight Base Slicing & Repair (`pipeline/mesh_cleaner.py`)
* **Spatial Component Selection:** Keeps mesh components by position (vertices above the cut plane, within the platter rim), preventing large platter fragments from discarding small objects.
* **Planar Slicing at $Y = 3.0\text{ mm}$:** Drops the irregular contact zone between the object and the platter.
* **2D Ear-Clipping Base Capping:** Chains boundary half-edges into directed 2D polygons in the cut plane (handling complex ring feet and multi-legged geometries) and triangulates them flat with normals strictly directed along $-Y$.
* **PyMeshLab Manifold Conditioning & Taubin Smoothing:** Closes residual holes with fairing and applies volume-preserving Taubin smoothing (alternating positive and negative Laplacian passes) to clean surface noise without shrinking object dimensions.

### Step 6: Texturing via `texrecon` (`pipeline/texrecon_runner.py`)
* Prepares lossless PNG textures with background pixels set to $(0, 0, 0)$ and applies per-camera white balance gains derived from the white AprilTag disc paper.
* Executes `texrecon` using Markov Random Field (MRF) Potts optimization to assign adjacent faces to continuous photographic charts.
* Unseen faces (such as the flat cut base) are filled cleanly with neutral matte dark grey (`RGB 25,25,25`).

### Step 7: Multi-LOD Packaging & Exporters (`pipeline/exporter.py`)
* Generates High-Poly and Low-Poly OBJ packages with MTL and textures.
* Packages self-contained single-primitive GLB models with Draco geometric compression.
* Exports watertight STL and 3MF models scaled to millimetres ($1000\times$, Z-up, base on $Z=0$) for direct 3D printing.
* Writes `object.json` with physical dimensions ($L \times W \times H\text{ mm}$), volume ($\text{cm}^3$), and watertight status.
* Compresses all deliverables into a single `output.zip` archive.

---

## 5. Telemetry & Quality Standards

Every scan records structured telemetry in `stats.json`:
* **Hardware Utilization:** Peak RAM, Peak VRAM, per-step elapsed time, disk I/O throughput.
* **Pose Quality:** Global RMS error, per-photo RMS error, maximum tag displacement, and free-solvePnP drift ($dC\text{ mm}, dR^\circ$).
* **Mesh Health:** Number of vertices/faces, watertight boolean, number of closed boundary edges, patched hole area ($\text{cm}^2$), and volume ($\text{cm}^3$).
* **Automated Quality Warnings:** Any issue detected during processing (e.g. non-watertight geometry, model touching platter rim, excessive mask disagreement, camera step skipping) is recorded in `status.json` under `"warnings"`.
