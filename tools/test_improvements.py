"""
tools/test_improvements.py — Rulare locală sigură pe CPU (fără CUDA, fără OOM)
pentru a valida îmbunătățirile la mascare și Visual Hull pe seturile reale din v9:
1. Șurubelnița (rezolvarea retezării mânerului de sus)
2. Spray-ul de aluminiu (rezolvarea retezării bazei de 84 mm)
"""

import sys
import os
import time
import shutil
from pathlib import Path
import psutil
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import camera_init, mask_fusion, visual_hull, config, mask_processor
from pipeline.mesh_masks import load_sfm_cameras
from pipeline.rig_refine import view_groups

def log_mem(prefix=""):
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    avail_gb = vm.available / (1024 * 1024 * 1024)
    print(f"[{prefix}] RAM Process: {rss_mb:.1f} MB | System Avail: {avail_gb:.2f} GB / {vm.total / (1024**3):.1f} GB")

def get_ai_model_path():
    p = Path("/opt/models/birefnet_lite_512.onnx")
    if p.exists():
        return str(p)
    p_local = ROOT / "pipeline/birefnet_lite_512.onnx"
    if p_local.exists():
        return str(p_local)
    raise FileNotFoundError("Nu s-a găsit modelul birefnet_lite_512.onnx nici în /opt/models nici în pipeline/")

def test_screwdriver():
    print("\n" + "="*70)
    print("TEST 1: ȘURUBELNIȚĂ (Verificare păstrare capăt superior / mâner)")
    print("="*70)
    log_mem("Start")
    
    photos_dir = ROOT / "results/v9/20261102-194144-d80808/photos"
    out_dir = ROOT / "output_test_local/screwdriver"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Camera Init
    print("\n1. Rulare Camera Init...")
    camera_init_sfm = str(out_dir / "cameraInit.sfm")
    cameras_known_sfm = str(out_dir / "cameras_known.sfm")
    tt_cfg = config.TurntableConfig(
        step_deg=18.0,
        tag_scale=0.98,
        lift_mm=80.0,
        estimate_lift=True,
    )
    
    if not Path(cameras_known_sfm).exists():
        views, intrinsics, name_regex = camera_init.scan_input_directory(
            str(photos_dir), rig_mode=True, default_focal_mm=3.9
        )
        camera_init.generate_camera_init_sfm(views, intrinsics, camera_init_sfm)
        camera_init.run_turntable_poses(
            camera_init_sfm=camera_init_sfm,
            output_known_sfm=cameras_known_sfm,
            config=tt_cfg,
            report_path=str(out_dir / "turntable_report.txt"),
        )
    
    # Load SFM views/intrinsics for masks
    cams = load_sfm_cameras(cameras_known_sfm)
    cam_i, lvl, _ = view_groups(cams, tt_cfg.name_regex)
    groups = list(zip(cam_i.tolist(), lvl.tolist()))
    print(f"Camere înregistrate: {len(cams)} pe 2 nivele.")
    
    # 3. Măști fuzionate (BiRefNet pe CPU, 2 workeri pentru a proteja memoria RAM)
    masks_dir = out_dir / "masks"
    work_dir = out_dir / "work"
    masks_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    existing_masks = list(masks_dir.rglob("*.mask.png"))
    if len(existing_masks) < len(cams):
        print("3. Generare Măști Fuzionate cu noile reguli (keep_touching_or_roi)...")
        ai_model = get_ai_model_path()
        tags = mask_processor.load_tag_layout(tt_cfg.tag_layout)
        t0 = time.time()
        stats = {}
        mask_files = mask_fusion.batch_fused_masks(
            cameras=cams,
            groups=groups,
            output_masks_dir=str(masks_dir),
            work_dir=str(work_dir),
            tags=tags,
            tag_scale=0.98,
            disc_radius_m=0.127,
            cylinder_radius_m=0.080,
            cylinder_y_min_m=0.001,
            cylinder_y_max_m=0.350,
            workers=2,
            ai_model=ai_model,
            stats=stats,
        )
        t_masks = time.time() - t0
        print(f"Măști generate în {t_masks:.1f} s ({len(mask_files)} imagini).")
    else:
        print(f"3. Măști fuzionate deja existente ({len(existing_masks)} fișiere). Reutilizăm pentru viteză.")
    
    log_mem("După Măști")
    
    # 4. Visual Hull
    print("4. Carving Visual Hull pentru verificare cote...")
    mask_paths = visual_hull.source_mask_paths(cams, str(masks_dir))
    hull = visual_hull.carve_visual_hull(
        cams,
        mask_paths,
        radius_m=0.080,
        y_min_m=0.001,
        y_max_m=0.350,
        voxel_m=0.002, # 2 mm voxeli pentru verificare rapidă
        min_bg_votes=2,
        distorted=True,
    )
    
    stats = visual_hull.hull_stats(hull, decided_only=False)
    y_min_mm = stats["y_min_mm"]
    y_max_mm = stats["y_max_mm"]
    height_mm = y_max_mm - y_min_mm
    print(f"\n>>> REZULTAT VISUAL HULL ȘURUBELNIȚĂ:")
    print(f"    - Cotă Y_min: {y_min_mm:.1f} mm")
    print(f"    - Cotă Y_max: {y_max_mm:.1f} mm")
    print(f"    - Înălțime totală: {height_mm:.1f} mm")
    print(f"    - Volum hull: {stats['volume_cm3']:.1f} cm3 ({stats['voxels']} voxeli)")
    print(f"    - Referință v9 (veche, tăiată la mâner): 193.5 mm (Y_max: 193.5 mm)")
    print(f"    - Referință v8 (obiect complet): ~227.0 mm (Y_max: ~227.0 mm)")
    if y_max_mm > 215.0:
        print("    ✅ REZOLVAT! Mânerul superior NU mai este tăiat (atinge cota completă)!")
    else:
        print(f"    ⚠️ ATENȚIE: Înălțimea maximă este {y_max_mm:.1f} mm.")

def test_spray():
    print("\n" + "="*70)
    print("TEST 2: SPRAY ALUMINIU (Verificare păstrare bază la contactul cu discul)")
    print("="*70)
    log_mem("Start")
    
    photos_dir = ROOT / "results/v9/20261102-161557-e3bc38/photos"
    out_dir = ROOT / "output_test_local/spray"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Camera Init
    print("\n1. Rulare Camera Init...")
    camera_init_sfm = str(out_dir / "cameraInit.sfm")
    cameras_known_sfm = str(out_dir / "cameras_known.sfm")
    tt_cfg = config.TurntableConfig(
        step_deg=18.0,
        tag_scale=0.98,
        lift_mm=80.0,
        estimate_lift=True,
    )
    
    if not Path(cameras_known_sfm).exists():
        views, intrinsics, name_regex = camera_init.scan_input_directory(
            str(photos_dir), rig_mode=True, default_focal_mm=3.9
        )
        camera_init.generate_camera_init_sfm(views, intrinsics, camera_init_sfm)
        camera_init.run_turntable_poses(
            camera_init_sfm=camera_init_sfm,
            output_known_sfm=cameras_known_sfm,
            config=tt_cfg,
            report_path=str(out_dir / "turntable_report.txt"),
        )
    
    # Load SFM views/intrinsics for masks
    cams = load_sfm_cameras(cameras_known_sfm)
    cam_i, lvl, _ = view_groups(cams, tt_cfg.name_regex)
    groups = list(zip(cam_i.tolist(), lvl.tolist()))
    print(f"Camere înregistrate: {len(cams)} pe 2 nivele.")
    
    # 3. Măști fuzionate
    masks_dir = out_dir / "masks"
    work_dir = out_dir / "work"
    masks_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    existing_masks = list(masks_dir.rglob("*.mask.png"))
    if len(existing_masks) < len(cams):
        print("3. Generare Măști Fuzionate cu noile reguli (dark_on_paper pe hârtia albă)...")
        ai_model = get_ai_model_path()
        tags = mask_processor.load_tag_layout(tt_cfg.tag_layout)
        t0 = time.time()
        stats = {}
        mask_files = mask_fusion.batch_fused_masks(
            cameras=cams,
            groups=groups,
            output_masks_dir=str(masks_dir),
            work_dir=str(work_dir),
            tags=tags,
            tag_scale=0.98,
            disc_radius_m=0.127,
            cylinder_radius_m=0.080,
            cylinder_y_min_m=0.001,
            cylinder_y_max_m=0.250,
            workers=2,
            ai_model=ai_model,
            stats=stats,
        )
        t_masks = time.time() - t0
        print(f"Măști generate în {t_masks:.1f} s ({len(mask_files)} imagini).")
    else:
        print(f"3. Măști fuzionate deja existente ({len(existing_masks)} fișiere). Reutilizăm pentru viteză.")
    
    log_mem("După Măști")
    
    # 4. Visual Hull
    print("4. Carving Visual Hull pentru verificare cote...")
    mask_paths = visual_hull.source_mask_paths(cams, str(masks_dir))
    hull = visual_hull.carve_visual_hull(
        cams,
        mask_paths,
        radius_m=0.080,
        y_min_m=0.001,
        y_max_m=0.250,
        voxel_m=0.002,
        min_bg_votes=2,
        distorted=True,
    )
    
    stats = visual_hull.hull_stats(hull, decided_only=False)
    y_min_mm = stats["y_min_mm"]
    y_max_mm = stats["y_max_mm"]
    height_mm = y_max_mm - y_min_mm
    print(f"\n>>> REZULTAT VISUAL HULL SPRAY ALUMINIU:")
    print(f"    - Cotă Y_min: {y_min_mm:.1f} mm")
    print(f"    - Cotă Y_max: {y_max_mm:.1f} mm")
    print(f"    - Înălțime totală: {height_mm:.1f} mm")
    print(f"    - Volum hull: {stats['volume_cm3']:.1f} cm3 ({stats['voxels']} voxeli)")
    print(f"    - Referință v9 (veche, tăiată la bază): Y_min = 84.0 mm, înălțime = 115.5 mm")
    print(f"    - Referință așteptată (bază completă la disc): Y_min <= 3.0 mm, înălțime >= 190.0 mm")
    if y_min_mm <= 5.0 and height_mm >= 180.0:
        print("    ✅ REZOLVAT! Baza spray-ului coboară până la disc (NU mai este tăiată)!")
    else:
        print(f"    ⚠️ ATENȚIE: Cotă Y_min = {y_min_mm:.1f} mm, înălțime = {height_mm:.1f} mm.")

if __name__ == "__main__":
    test_screwdriver()
    test_spray()
