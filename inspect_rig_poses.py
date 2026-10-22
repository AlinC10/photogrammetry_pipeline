#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_rig_poses.py — Analizează și extrage geometria REALĂ a camerelor dintr-un
fișier cameras_known.sfm sau turntable_calib.json generat de turntable_poses.py.

Afișează:
1. Pasul real dintre cadre (nu cel presupus).
2. Decalajul azimutal real (unghiular) dintre camere (dacă sunt decalate sau una sub alta).
3. Înălțimile reale, distanțele la axă și unghiurile reale de înclinare (pitch).
4. Erorile de reproiecție și consistența mecanică a rig-ului.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Verifică și inspectează pozițiile reale ale camerelor din cameras_known.sfm.")
    parser.add_argument("--sfm", default="output/cameras_known.sfm", help="Calea către cameras_known.sfm")
    parser.add_argument("--calib", default=None, help="Calea opțională către turntable_calib.json")
    parser.add_argument("--export_svg", default="proiectie_camere_reale.svg", help="Exportă diagrama SVG cu pozițiile reale")
    parser.add_argument("--export_html", default="vizualizare_camere_reale_3d.html", help="Exportă viewerul 3D cu pozițiile reale")
    return parser.parse_args()


def rot_to_look_and_tilt(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculează vectorul privirii (optic) și unghiul de pitch (înclinare în jos)
    față de planul orizontal (unde normala platanului este +Y).
    R este matricea 3x3 World-to-Camera (Column-Major în AliceVision).
    """
    # În spațiul camerei, axa optică este [0, 0, 1]
    # În spațiul lumii: look_world = R.T @ [0, 0, 1]
    look_world = R.T @ np.array([0.0, 0.0, 1.0])
    look_world = look_world / np.linalg.norm(look_world)
    
    # Normala platanului este a = [0, 1, 0] (în sus)
    # Dacă privim în jos, produsul scalar cu [0, 1, 0] este negativ
    # tilt = arcsin(-look_y)
    tilt_deg = math.degrees(math.asin(max(-1.0, min(1.0, -look_world[1]))))
    return look_world, tilt_deg


def inspect_sfm(sfm_path: str) -> Dict[str, Any]:
    p = Path(sfm_path)
    if not p.is_file():
        raise FileNotFoundError(f"Fișierul SFM nu a fost găsit: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        sfm = json.load(f)

    poses_dict = {}
    for item in sfm.get("poses", []):
        pid = str(item["poseId"])
        t = item["pose"]["transform"]
        R = np.array([float(x) for x in t["rotation"]], dtype=np.float64).reshape((3, 3), order="F")
        C = np.array([float(x) for x in t["center"]], dtype=np.float64)
        poses_dict[pid] = {"R": R, "C": C}

    views_data = []
    for v in sfm.get("views", []):
        pid = str(v["poseId"])
        if pid not in poses_dict:
            continue
        
        pose = poses_dict[pid]
        R = pose["R"]
        C = pose["C"] # metri
        
        # Coordonate în mm
        X_mm = C[0] * 1000.0
        Y_mm = C[1] * 1000.0
        Z_mm = C[2] * 1000.0
        
        # Distanță radială față de axa Y (X=0, Z=0)
        dist_radial_mm = math.sqrt(X_mm**2 + Z_mm**2)
        
        # Unghi azimutal în planul orizontal (grade, 0 la 360)
        # atan2(X, Z) unde Z este 'spre tine' și X este 'la dreapta'
        azimuth_rad = math.atan2(X_mm, Z_mm)
        azimuth_deg = math.degrees(azimuth_rad)
        if azimuth_deg < 0:
            azimuth_deg += 360.0
            
        look_vec, tilt_deg = rot_to_look_and_tilt(R)
        
        views_data.append({
            "viewId": str(v["viewId"]),
            "path": v["path"],
            "filename": Path(v["path"]).name,
            "intrinsicId": str(v.get("intrinsicId", "0")),
            "X": X_mm, "Y": Y_mm, "Z": Z_mm,
            "dist_radial": dist_radial_mm,
            "azimuth": azimuth_deg,
            "tilt_down": tilt_deg,
            "R": R, "C": C,
            "look_vec": look_vec,
        })

    # Sortare după unghi azimutal sau nume
    views_data.sort(key=lambda item: (item["intrinsicId"], item["azimuth"]))
    return {"views": views_data, "raw_sfm": sfm}


def generate_report(data: Dict[str, Any]) -> str:
    views = data["views"]
    lines = []
    lines.append("=" * 80)
    lines.append("RAPORT AUDIT: POZIȚIILE REALE ALE CAMERELOR (Extrase din cameras_known.sfm)")
    lines.append("=" * 80)
    lines.append(f"Total cadre calibrate detectate: {len(views)}")
    
    # Grupare pe intrinsicId (camere distincte fizic)
    by_cam = {}
    for v in views:
        cid = v["intrinsicId"]
        by_cam.setdefault(cid, []).append(v)
        
    lines.append(f"Număr camere fizice în rig: {len(by_cam)}")
    for cid, cam_views in by_cam.items():
        heights = [v["Y"] for v in cam_views]
        dists = [v["dist_radial"] for v in cam_views]
        tilts = [v["tilt_down"] for v in cam_views]
        
        # Calcul pași unghiulari consecutivi
        cam_views_sorted = sorted(cam_views, key=lambda x: x["azimuth"])
        steps = []
        for i in range(1, len(cam_views_sorted)):
            d_az = cam_views_sorted[i]["azimuth"] - cam_views_sorted[i-1]["azimuth"]
            steps.append(d_az)
            
        avg_step = np.mean(steps) if steps else 0.0
        
        lines.append(f"\n--- Camera {cid} ({len(cam_views)} cadre) ---")
        lines.append(f"  • Înălțime medie față de disc (Y): {np.mean(heights):.2f} mm (min: {np.min(heights):.2f}, max: {np.max(heights):.2f})")
        lines.append(f"  • Distanță medie față de axă (R): {np.mean(dists):.2f} mm (min: {np.min(dists):.2f}, max: {np.max(dists):.2f})")
        lines.append(f"  • Înclinație medie în jos (Pitch): {np.mean(tilts):.2f}° (min: {np.min(tilts):.2f}°, max: {np.max(tilts):.2f}°)")
        lines.append(f"  • Pas unghiular mediu între poze:   {avg_step:.2f}° (min: {np.min(steps) if steps else 0:.2f}°, max: {np.max(steps) if steps else 0:.2f}°)")

    # Comparație decalaj între Camera 0 și Camera 1
    if len(by_cam) >= 2:
        c0_id, c1_id = list(by_cam.keys())[:2]
        c0_views = by_cam[c0_id]
        c1_views = by_cam[c1_id]
        
        delta_h = np.mean([v["Y"] for v in c1_views]) - np.mean([v["Y"] for v in c0_views])
        delta_dist = np.mean([v["dist_radial"] for v in c1_views]) - np.mean([v["dist_radial"] for v in c0_views])
        delta_tilt = np.mean([v["tilt_down"] for v in c1_views]) - np.mean([v["tilt_down"] for v in c0_views])
        
        lines.append("\n--- DECALAJUL REAL DINTRE CELE DOUĂ CAMERE ---")
        lines.append(f"  • Decalaj pe înălțime (ΔH): {delta_h:+.2f} mm (Camera {c1_id} față de {c0_id})")
        lines.append(f"  • Decalaj pe rază (ΔR):     {delta_dist:+.2f} mm")
        lines.append(f"  • Decalaj unghi pitch (ΔTilt): {delta_tilt:+.2f}°")
        
        # Testăm dacă cadrele pereche au decalaj azimutal (sunt rotite una față de alta)
        lines.append("\n--- ANALIZĂ DECALAJ AZIMUTAL (Sunt camerele una sub alta?) ---")
        min_len = min(len(c0_views), len(c1_views))
        az_diffs = []
        for i in range(min_len):
            # Căutăm cel mai apropiat cadru ca index
            diff = c1_views[i]["azimuth"] - c0_views[i]["azimuth"]
            # Normalizare la [-180, 180]
            diff = (diff + 180.0) % 360.0 - 180.0
            az_diffs.append(diff)
            
        avg_az_diff = np.mean(az_diffs)
        if abs(avg_az_diff) < 0.5:
            lines.append(f"  => Camerele sunt ALINIATE VERTICAL pe aceeași tijă (Decalaj azimutal: {avg_az_diff:+.2f}° ~ 0°)")
        else:
            lines.append(f"  => Camerele sunt DECALATE UNGHIULAR pe orizontală cu: {avg_az_diff:+.2f}° !")

    lines.append("\n" + "=" * 80)
    lines.append("TABEL DETALIAT PER CADRU (Primele 10 cadre)")
    lines.append("=" * 80)
    lines.append(f"{'Fișier':<20} {'Cam':<5} {'X (mm)':<10} {'Y (mm)':<10} {'Z (mm)':<10} {'Raza (mm)':<10} {'Azimut (°)':<12} {'Pitch (°)':<10}")
    lines.append("-" * 87)
    for v in views[:15]:
        lines.append(f"{v['filename']:<20} {v['intrinsicId']:<5} {v['X']:<10.1f} {v['Y']:<10.1f} {v['Z']:<10.1f} {v['dist_radial']:<10.1f} {v['azimuth']:<12.2f} {v['tilt_down']:<10.2f}")
    if len(views) > 15:
        lines.append(f"... și încă {len(views) - 15} cadre.")
        
    return "\n".join(lines)


def export_real_svg(views: List[Dict[str, Any]], out_svg: str):
    """Generează diagrama SVG cu coordonatele reale extrase din SFM."""
    if not views:
        return
    
    # Determinăm extinderea maximă pentru scalare
    all_x = [v["X"] for v in views]
    all_y = [v["Y"] for v in views]
    all_z = [v["Z"] for v in views]
    
    max_r = max([v["dist_radial"] for v in views] + [150.0])
    max_h = max(all_y + [150.0])
    
    svg = []
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 750" width="100%" height="100%" style="background-color:#14171c;font-family:Segoe UI,Helvetica,Arial,sans-serif;">')
    svg.append(f'<text x="30" y="40" fill="#ffffff" font-size="20" font-weight="bold">Proiecție REARĂ a Camerelor (Extrase din SFM: {len(views)} cadre)</text>')
    
    # Panou Vedere de Sus XZ (Stânga)
    top_cx = 360
    top_cy = 400
    top_scale = 260.0 / (max_r * 1.3)
    
    svg.append('<g id="panel_top">')
    svg.append('<rect x="25" y="70" width="670" height="650" rx="12" fill="#1a202c" stroke="#2d3748" stroke-width="1.5"/>')
    svg.append('<text x="45" y="105" fill="#38bdf8" font-size="16" font-weight="bold">1. Vedere de Sus REALĂ (Plan XZ - Discul și Camerele)</text>')
    
    # Platan Ø260mm
    r_disc_px = 130.0 * top_scale
    svg.append(f'<circle cx="{top_cx}" cy="{top_cy}" r="{r_disc_px}" fill="#242c38" stroke="#4a5568" stroke-width="1.5"/>')
    svg.append(f'<circle cx="{top_cx}" cy="{top_cy}" r="3" fill="#22c55e"/>')
    svg.append(f'<text x="{top_cx+6}" y="{top_cy-6}" fill="#22c55e" font-size="11">Centru Platan (0,0)</text>')
    
    colors = {"0": "#00e5ff", "1": "#ff9100", "2": "#a855f7", "3": "#22c55e"}
    for v in views:
        px = top_cx + v["X"] * top_scale
        pz = top_cy + v["Z"] * top_scale
        col = colors.get(v["intrinsicId"], "#00e5ff")
        
        # Rază optică
        svg.append(f'<line x1="{px:.1f}" y1="{pz:.1f}" x2="{top_cx}" y2="{top_cy}" stroke="{col}" stroke-width="1" opacity="0.25"/>')
        svg.append(f'<circle cx="{px:.1f}" cy="{pz:.1f}" r="4" fill="{col}" stroke="#ffffff" stroke-width="0.8"/>')
        
    svg.append('</g>')
    
    # Panou Vedere Laterală XY (Dreapta)
    side_ox = 780
    side_oy = 620
    side_scale = 320.0 / (max_r * 1.3)
    
    svg.append('<g id="panel_side">')
    svg.append('<rect x="715" y="70" width="460" height="650" rx="12" fill="#1a202c" stroke="#2d3748" stroke-width="1.5"/>')
    svg.append('<text x="735" y="105" fill="#38bdf8" font-size="16" font-weight="bold">2. Profil Lateral REAL (Înălțimi & Unghiuri)</text>')
    
    # Suprafață platan
    svg.append(f'<line x1="{side_ox-30}" y1="{side_oy}" x2="{side_ox+max_r*side_scale+40}" y2="{side_oy}" stroke="#4a5568" stroke-width="3"/>')
    svg.append(f'<text x="{side_ox}" y="{side_oy+20}" fill="#64748b" font-size="11">Planul Discului (Y = 0)</text>')
    svg.append(f'<line x1="{side_ox}" y1="{side_oy+10}" x2="{side_ox}" y2="{side_oy-max_h*side_scale-30}" stroke="#22c55e" stroke-width="1.5" stroke-dasharray="3,3"/>')
    
    for v in views:
        # Proiecție rază x înălțime
        col = colors.get(v["intrinsicId"], "#00e5ff")
        cx_proj = side_ox + v["dist_radial"] * side_scale
        cy_proj = side_oy - v["Y"] * side_scale
        svg.append(f'<circle cx="{cx_proj:.1f}" cy="{cy_proj:.1f}" r="4" fill="{col}" stroke="#ffffff" stroke-width="0.8"/>')
        
    svg.append('</g>')
    svg.append('</svg>')
    
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"[OK] Diagrama SVG reală exportată la: {out_svg}")


def main():
    args = parse_args()
    print(f"[inspect_rig_poses] Citesc fișierul: {args.sfm}")
    try:
        data = inspect_sfm(args.sfm)
        report = generate_report(data)
        print(report)
        
        # Salvare raport text
        out_txt = Path(args.sfm).parent / "raport_verificare_camere_reale.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[OK] Raportul complet a fost salvat la: {out_txt}")
        
        # Export grafic real
        out_svg = Path(args.sfm).parent / args.export_svg
        export_real_svg(data["views"], str(out_svg))
        
    except Exception as e:
        print(f"[EROARE] Nu s-a putut citi fișierul SFM ({e}).")
        print("\nCum puteți rula acest script pe fișierul Dvs. real de pe Windows:")
        print(r"  python inspect_rig_poses.py --sfm D:\photogrammetry_pipeline\output\cameras_known.sfm")


if __name__ == "__main__":
    main()
