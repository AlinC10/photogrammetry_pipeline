# Turntable Studio — Industrial 3D Photogrammetry Pipeline

Turntable Studio is a high-precision, studio-grade 3D photogrammetry pipeline designed for turntable scanning rigs (e.g. dual Arducam Sony IMX477 cameras). It replaces the heavy and noisy parts of Meshroom (DepthMap, Meshing, Texturing) with **OpenMVS**, enforces rigidly locked camera poses via **TurntablePoses**, seals the base into a watertight CAD-ready 3D print pedestal, and exports multi-LOD assets with full telemetry.

---

## 1. System Architecture

```mermaid
flowchart TD
    A["Input Images<br/>(Flat or rig/0 & rig/1)"] --> B["1. CameraInit<br/>(IMX477 3.9mm preset / EXIF)"]
    B --> C["2. TurntablePoses<br/>(Rigidly LOCKED Poses & Calibration)"]
    C --> D["3. Bridge SFM<br/>(OpenCV Undistort & COLMAP Exporter)"]
    D --> E["4. OpenMVS Geometry<br/>(InterfaceCOLMAP &rarr; Densify &rarr; ReconstructMesh)"]
    E --> F["5. Watertight Mesh Cleaner<br/>(Planar Slice at 3mm &rarr; Skirt &rarr; Ear-Clipping Base)"]
    F --> G{"Texturing Engine Selection"}
    G -->|Default / Recommended| H1["mvs-texturing (texrecon)<br/>• MRF Graph-Cut + GMI<br/>• Poisson Global & Local Seam Leveling"]
    G -->|Multi-Band LSCM| H2["AliceVision Texturing<br/>• Frequency decomposition"]
    G -->|Fast Built-in| H3["OpenMVS TextureMesh<br/>• Direct mesh texturing"]
    H1 --> I["6. Multi-LOD Exporter<br/>(OBJ, STL, 3MF, Draco GLB & output.zip)"]
    H2 --> I
    H3 --> I
    I --> J["7. Telemetry & Stats<br/>(Peak RAM, Peak VRAM, per-step timing, stats.json)"]
```

---

## 2. Python Environment & Versioning

> [!IMPORTANT]
> **Use Python 3.10 or 3.11.**
> Do **NOT** use Python 3.14 (or early alpha releases). Scientific packages like `pymeshlab`, `scipy`, `trimesh`, and `opencv-python` do not have pre-built wheels for Python 3.14 on Linux/Windows and will fail during installation.

### Quick Setup with `uv` (Recommended):
```bash
# Create Python 3.11 virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Standard `venv` Setup:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Input Directory Layouts

### A. Rig Mode (`--rig_mode`)
Designed for dual-camera rigs where both cameras shoot simultaneously at each turntable step:
```text
input_images/
├── rig/
│   ├── 0/               # Camera 1 (Lower angle)
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   └── ...
│   └── 1/               # Camera 2 (Upper angle)
│       ├── 0000.jpg
│       ├── 0001.jpg
│       └── ...
└── masks/               # Optional SAM / BiRefNet masks
    ├── 0000.mask.png
    └── ...
```

### B. Flat Mode
Single camera turntable scan:
```text
input_images/
├── 0000.jpg
├── 0001.jpg
├── ...
└── 0019.jpg
```

---

## 4. Running with Docker (Linux Server / RunPod / Cloud)

Docker is the recommended deployment method for cloud servers (RunPod, AWS, GCP) with NVIDIA GPUs.

### Build and Run with Docker Compose:
```bash
# Build the CUDA 12 + OpenMVS container
docker compose build

# Run the pipeline
docker compose up
```

### Run Directly via Docker CLI:
```bash
docker run --gpus all --rm \
  -v $(pwd)/input_images:/workspace/input_images:ro \
  -v $(pwd)/output:/workspace/output:rw \
  turntable-studio:latest \
  --input_dir /workspace/input_images \
  --output_dir /workspace/output \
  --rig_mode \
  --resolution_level 1
```

---

## 5. Running Locally on Windows x64 (Native)

You can run Turntable Studio directly on Windows without Docker:

1. **Install OpenMVS and Optional Texturing Engines:**
   - **OpenMVS:** Download the latest pre-compiled Windows x64 release from [OpenMVS Releases](https://github.com/cdcseacave/openMVS/releases) (e.g. `openMVS_v2.4.0_win64.zip`) and extract it to `C:\OpenMVS`. The pipeline automatically scans subfolders:
     * `C:\OpenMVS\vc17\x64\Release` (Visual Studio 2022)
     * `C:\OpenMVS\vc16\x64\Release` (Visual Studio 2019)
     * `C:\OpenMVS\bin`
   - **mvs-texturing (`texrecon` - Recommended):** Extract `texrecon.exe` and its OpenMP DLLs to `C:\texrecon\bin` or specify `--texrecon_bin "C:\texrecon\bin"`.
   - **AliceVision (Alternative):** If you have Meshroom installed (e.g. `D:\Meshroom-2025.1.0`), point to its binary folder with `--alicevision_bin "D:\Meshroom-2025.1.0\aliceVision\bin"`.

2. **Run the Python Pipeline:**
   ```powershell
   # Default Studio Pipeline (mvs-texturing + IMX477 3.9mm lens)
   python main.py --input_dir .\input_images --output_dir .\output --rig_mode --focal_length_mm 3.9

   # Run with AliceVision Texturing (Multi-Band frequency blending)
   python main.py --texturing_engine alicevision --alicevision_bin "D:\Meshroom-2025.1.0\aliceVision\bin" --rig_mode

   # Run with OpenMVS TextureMesh
   python main.py --texturing_engine openmvs --openmvs_bin_dir "C:\OpenMVS\vc17\x64\Release" --rig_mode

   # Fast resume (reuses existing dense point cloud and mesh, running only texturing and export)
   python main.py --resume
   ```

---

## 6. CLI Arguments & Configuration

### A. I/O & Execution Modes
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input_dir`, `--input`, `-i` | `str` | `./input_images` | Path to images directory |
| `--output_dir`, `--output`, `-o` | `str` | `./output` | Path for deliverables |
| `--rig_mode` | `flag` | `False` | Enables dual-camera rig layout (`rig/0`, `rig/1`) |
| `--resume` | `flag` | `False` | Skips densification/reconstruction if point cloud/mesh already exists |
| `--mock_openmvs` | `flag` | `False` | Simulates OpenMVS and texturing for fast unit/smoke testing |

### B. Camera Optics & Calibration
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--focal_length_mm`, `--focal_mm` | `float` | `3.9` | Camera lens focal length in mm (calibrated for IMX477 M12 lens) |
| `--step_deg` | `float` | `18.0` | Turntable rotation step in degrees (`18.0` for 20 steps/turn) |

### C. Texturing Engine Selection & Settings
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--texturing_engine` | `choice` | `texrecon` | Texturing engine: `texrecon` (mvs-texturing, default), `alicevision`, or `openmvs` |
| `--texrecon_bin` | `str` | `None` | Path to `texrecon` executable or directory (e.g. `C:\texrecon\bin`) |
| `--data_term` | `choice` | `gmi` | `texrecon` data term: `gmi` (gradient magnitude + intensity, default) or `color` |
| `--smoothness_term` | `str` | `potts` | `texrecon` smoothness term MRF regularizer (default: `potts`) |
| `--outlier_removal` | `choice` | `gauss_damping` | `texrecon` photometric outlier removal: `gauss_damping`, `none`, or `gauss_clamping` |
| `--alicevision_bin` | `str` | `None` | Path to AliceVision binary directory (e.g. `C:\Meshroom\aliceVision\bin`) |
| `--unwrap_method` | `choice` | `Basic` | AliceVision UV unwrapping method: `Basic` (fast & robust on high-poly), `LSCM`, or `ABF` |
| `--visibility_remapping_method` | `choice` | `MeshItself` | AliceVision visibility test: `MeshItself` (direct mesh raycasting) or `PullPush` |
| `--cost_smoothness_ratio` | `float` | `1.0` | OpenMVS texturing smoothness ratio (`1.0` prevents chart confetti) |
| `--outlier_threshold` | `float` | `0.0` | OpenMVS outlier rejection (`0.0` disabled to texture shiny plastics without holes) |
| `--empty_color` | `int` | `0` | OpenMVS empty/unobserved face color in decimal RGB (`0` = Black, `16744231` = Orange) |
| `--global_seam_leveling` | `int` | `0` | OpenMVS global Poisson seam leveling (`0` disables dark background bleeding) |
| `--local_seam_leveling` | `int` | `1` | OpenMVS local seam edge blending (`1` = enabled) |

### D. OpenMVS Geometry & Mesh Cleaning
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--openmvs_bin_dir`, `--bin_dir` | `str` | `None` | Path to OpenMVS binaries directory (e.g. `C:\OpenMVS\vc17\x64\Release`) |
| `--resolution_level` | `int` | `1` | OpenMVS densify downscale (`0` = 4K full, `1` = 2K, `2` = 1K) |
| `--target_low_poly_faces` | `int` | `50000` | Target triangle count for decimated low-poly model |

---

## 7. Deliverables & Multi-LOD Assets

Once complete, the `output/` directory contains complete production assets matching Cloudflare R2 / RunPod specifications:

```text
output/
├── output.zip                  # Compressed bundle ready for Cloudflare R2 / S3
├── stats.json                  # JSON telemetry report (RAM, VRAM, timing, step metrics)
├── turntable_report.txt        # TurntablePoses angular accuracy and calibration report
├── model_cleaned_watertight.obj# Intermediate watertight model (pedestal-sealed)
├── obj/
│   ├── high/
│   │   ├── model_high.obj      # High-poly textured OBJ (with mtllib model_high.mtl)
│   │   ├── model_high.mtl      # Material file referencing PNG texture
│   │   └── model_high_texture.png # Lossless, uncompressed master texture atlas
│   └── low/
│       ├── model_low.obj       # Decimated low-poly OBJ (texture-preserved UVs)
│       ├── model_low.mtl       # Material file referencing JPG texture
│       └── model_low_texture.jpg # Web-optimized downscaled JPG texture atlas (max 2048px)
├── stl/
│   ├── high_model.stl          # Watertight 3D printing STL (manifold sealed base)
│   └── low_model.stl           # Decimated STL for rapid slicing
├── 3mf/
│   ├── high_model.3mf          # Slicer-ready 3MF asset
│   └── low_model.3mf
└── glb/
    ├── high_model.glb          # Full-res web / AR 3D asset (embedded PNG texture)
    └── low_model.glb           # Low-poly web / AR 3D asset (embedded JPG texture)
```

---

## 8. Multi-Engine Texturing & Optics Optimization

### A. Tri-Engine Texturing Architecture

Turntable Studio supports three interchangeable texturing engines via `--texturing_engine`:

1. **`texrecon` (`mvs-texturing` - TU Darmstadt — Default & Recommended):**
   - Implements the ECCV 2014 paper *"Let There Be Color! — Large-Scale Texturing of 3D Reconstructions"*.
   - Uses Markov Random Field (MRF) Graph-Cut energy minimization with the GMI (Gradient Magnitude + Intensity) data term to choose the sharpest, least-occluded camera for each polygon chart.
   - Applies global Poisson seam leveling and local image-space edge blending without matrix singularities.
   - Natively textures watertight CAD meshes with zero black holes and clean UV charts.

2. **`alicevision` (Meshroom Multi-Band Texturing):**
   - Decomposes images into low and high frequency bands, blending low-frequency color gradients seamlessly while retaining crisp high-frequency details (branding, fine plastic texture).
   - Uses direct raycasting visibility (`visibility_remapping_method="MeshItself"`).

3. **`openmvs` (`TextureMesh`):**
   - High-speed direct texturing from the reconstruction mesh.
   - Optimized with `--cost-smoothness-ratio 1.0` (encouraging continuous charts rather than micro-confetti) and `--empty-color 0` (clean black for unobserved underside faces).

### B. Optical Calibration: IMX477 3.9mm Lens (`--focal_length_mm 3.9`)

> [!IMPORTANT]
> The Arducam / Raspberry Pi HQ Camera (Sony IMX477 12MP, sensor width $6.287\text{ mm}$, resolution $4056 \times 3040$) uses manual M12 lenses **without electronic EXIF contacts**.
> - **Lens Focal Length:** $3.9\text{ mm}$ (FOV $\approx 78^\circ$).
> - **Focal Length in Pixels:** $f_x = 3.9 \cdot \frac{4056}{6.287} \approx \mathbf{2516.19\text{ px}}$.
> - If left to generic defaults ($6.0\text{ mm} \implies 3870.8\text{ px}$), the $54\%$ focal length discrepancy throws camera projection rays completely off target, causing textures to appear scrambled or full of black voids. Turntable Studio defaults to $3.9\text{ mm}$ across all modules.

---

## 9. Telemetry & Performance Monitoring

The pipeline automatically benchmarks each stage and outputs a telemetry summary:

```text
==============================================================================
                   TURNTABLE STUDIO — EXECUTION TELEMETRY
==============================================================================
 Total Time : 1m 42.5s (102.5s)
 Peak RAM   : 4.82 GB (4935.6 MB) / 31.2 GB Total
 Peak VRAM  : 6.14 GB (6287.0 MB)
------------------------------------------------------------------------------
 STEP NAME                        | TIME (s)   | PEAK RAM (MB)  | VRAM (MB)
------------------------------------------------------------------------------
 1. Camera Discovery & Init       |      0.45s |          124.0 |        -
 2. Turntable Poses (Rigid Lock)  |      8.20s |         1420.5 |        -
 3. Bridge SFM & Undistortion     |      5.12s |         1845.2 |        -
 4.1 OpenMVS InterfaceCOLMAP      |      1.10s |          450.0 |        -
 4.2 OpenMVS DensifyPointCloud    |     48.30s |         4935.6 |     6140
 4.3 OpenMVS ReconstructMesh      |     14.80s |         3210.0 |     4200
 4.4 OpenMVS TextureMesh          |     12.40s |         2100.4 |     3800
 5. Watertight Base Slicing       |      3.20s |         1120.0 |        -
 6. Multi-LOD Export              |      8.93s |         1850.0 |        -
------------------------------------------------------------------------------
```

---

## 10. Troubleshooting & FAQs

### Q: Why did my model appear pitch black in Blender?
- **Cause 1 (Stale `--resume` cache):** If you ran `--resume` after changing camera intrinsics or script code, OpenMVS skipped `InterfaceCOLMAP` and projected textures using the old camera calibration, missing the object and sampling the black background. Run without `--resume` (or delete `output/openmvs`) to re-import the camera calibration.
- **Cause 2 (Viewport Shading in Blender):** In Blender, switch from "Solid" mode to **"Material Preview"** (Z &rarr; Material Preview) or **"Rendered"** mode. Ensure your material has the image texture connected to the *Base Color* of the Principled BSDF node.

### Q: Why did the low-poly model appear completely white?
- **Cause:** Standard quadric decimation strips away vertex texture coordinates (`vt`). Turntable Studio uses `meshing_decimation_quadric_edge_collapse_with_texture` to ensure UV mapping is preserved on the decimated mesh. Ensure you are running the latest version from `main`.

### Q: How do I re-texture without waiting 11 minutes for DensifyPointCloud?
- If your point cloud `output/openmvs/scene_dense.mvs` and mesh `output/openmvs/scene_mesh.ply` are already reconstructed with the correct camera calibration:
  ```powershell
  python main.py --resume
  ```
  The pipeline will detect the existing mesh, perform base slicing, and proceed directly to `TextureMesh` and multi-LOD packaging in seconds!
