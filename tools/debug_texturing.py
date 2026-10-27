#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_texturing.py — Diagnostic pentru textura finală (PAS 0), pe un folder output/ existent.

Secțiuni (fiecare scrie imagini în <output>/debug/<secțiune>/ și un rezumat în consolă):
  overlays  — proiectează mesh-ul curățat (model_cleaned_watertight.obj) cu pozele din
              cameras_known.sfm peste imaginile NEDISTORSIONATE (wireframe low-poly vizibil,
              contur high-poly, tag-uri AprilTag) și măsoară eroarea de contur față de marginile
              reale din imagine. Compară și masca folosită la texturare cu silueta mesh-ului.
  labels    — citește <prefix>_labeling.vec de la texrecon (--write_intermediate_results),
              reconstruiește ordinea vederilor exact ca texrecon (folderul scene/) și arată ce
              vedere a primit fiecare față (inclusiv vederi rămase din rulări vechi).
  validity  — simulează TextureView::generate_validity_mask din texrecon (flood fill din colțuri
              peste pixeli exact RGB 0 + eroziunea pentru GMI) pe imaginile de texturare.
  colors    — compară culoarea camerei 0 cu a camerei 1 pe hârtia albă a platanului și pe
              suprafața piesei (aceleași puncte văzute de ambele camere).
  render    — redă OBJ-ul texturat și GLB-ul din pozițiile reale ale camerelor, lângă fotografie.
  glb       — inspectează GLB-ul (materiale, imagini, ce atlas primește fiecare primitivă).
  report    — rezumă turntable_report.txt pe camere și niveluri.

Exemplu:
  python tools/debug_texturing.py all --output_dir output --input_dir input_images
  (rulat din Docker cu volumul de lucru: adaugă --work_dir /workspace/output la rulare, ca undistorted/ să rămână în output/)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tools/ -> repository root (pipeline/)

from pipeline.mask_processor import load_tag_layout
from pipeline.mesh_masks import (
    load_sfm_cameras,
    project_vertices,
    render_face_ids,
    render_silhouette,
)

SECTIONS = ["report", "overlays", "labels", "validity", "colors", "glb", "render"]


# =============================================================================================
# Utilitare
# =============================================================================================

def resolve_path(path: str, input_dir: Path) -> Path:
    """Căile din .sfm pot fi din container (/workspace/...); le caut și sub --input_dir."""
    p = Path(path)
    if p.is_file():
        return p
    parts = p.parts
    for i in range(len(parts) - 1, 0, -1):
        cand = input_dir.joinpath(*parts[i:])
        if cand.is_file() and len(parts) - i >= 3:
            return cand
    for k in (3, 2, 1):
        cand = input_dir.joinpath(*parts[-k:])
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Nu găsesc imaginea {path} nici sub {input_dir}")


def undistorted_original(cam: Dict, input_dir: Path) -> np.ndarray:
    """Imaginea originală (nemascată), nedistorsionată exact ca în bridge_sfm (K, dist → K_new)."""
    img = cv2.imread(str(resolve_path(cam["path"], input_dir)), cv2.IMREAD_COLOR)
    return cv2.undistort(img, cam["K"], cam["dist"], None, cam["K_new"])


def load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import trimesh
    m = trimesh.load(str(path), process=False, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def view_level(cam: Dict) -> int:
    m = re.search(r"(\d)[_]?(\d{1,4})$", Path(cam["path"]).stem)
    return int(m.group(1)) if m else 0


def select_views(cams: List[Dict], count: int, explicit: Optional[str]) -> List[Dict]:
    if explicit:
        wanted = [s.strip() for s in explicit.split(",") if s.strip()]
        return [c for c in cams if c["view_id"] in wanted]
    groups: Dict[Tuple[str, int], List[Dict]] = {}
    for c in cams:
        groups.setdefault((c["intrinsic_id"], view_level(c)), []).append(c)
    keys = sorted(groups)
    per = max(1, int(math.ceil(count / max(1, len(keys)))))
    out = []
    for k in keys:
        g = groups[k]
        idx = np.linspace(0, len(g) - 1, per + 2)[1:-1] if len(g) > per else np.arange(len(g))
        # decalaj pe cameră ca să nu fie aceleași unghiuri pentru ambele camere
        shift = (keys.index(k) * len(g)) // (2 * per + 1)
        out += [g[(int(round(i)) + shift) % len(g)] for i in idx]
    return out[:count]


def save_jpg(path: Path, img: np.ndarray, max_side: Optional[int] = None, quality: int = 92):
    path.parent.mkdir(parents=True, exist_ok=True)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def put_label(img: np.ndarray, text: str, org=(20, 50), scale=1.4, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def silhouette_bbox(mask: np.ndarray, pad: int = 40) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape[:2]
    return (max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1))


def unique_edges(F: np.ndarray) -> np.ndarray:
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    return np.unique(e, axis=0)


def texrecon_validity(img_bgr: np.ndarray, erode: bool = True) -> np.ndarray:
    """Re-implementare exactă a TextureView::generate_validity_mask (+ erode_validity_mask pentru GMI).
    Invalizi = pixeli cu R+G+B == 0 conectați (4-vecini) la unul din cele 4 colțuri."""
    from scipy import ndimage
    zero = img_bgr.astype(np.int32).sum(axis=2) == 0
    lab, _ = ndimage.label(zero, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    h, w = zero.shape
    corners = {int(lab[0, 0]), int(lab[h - 1, 0]), int(lab[0, w - 1]), int(lab[h - 1, w - 1])} - {0}
    invalid = np.isin(lab, list(corners)) if corners else np.zeros_like(zero)
    if erode:
        invalid = ndimage.binary_dilation(invalid, structure=np.ones((3, 3), bool))
        invalid[0, :] = invalid[-1, :] = invalid[:, 0] = invalid[:, -1] = True
    return ~invalid


def texturing_image_path(images_dir: Path, cam: Dict) -> Optional[Path]:
    """Imaginea pe care a primit-o texrecon (texturing_images/ sau undistorted/), PNG sau JPG."""
    stem = Path(cam["path"]).stem
    for name in (f"{cam['view_id']}_{stem}.png", f"{cam['view_id']}_{Path(cam['path']).name}", f"{cam['view_id']}_{stem}.jpg"):
        p = images_dir / name
        if p.is_file():
            return p
    return None


def texturing_mask_path(images_dir: Path, cam: Dict) -> Optional[Path]:
    stem = Path(cam["path"]).stem
    for p in (images_dir / f"{cam['view_id']}_{stem}.mask.png",
              images_dir.parent / "masks" / f"{cam['view_id']}_{stem}.mask.png"):
        if p.is_file():
            return p
    return None


# =============================================================================================
# report — turntable_report.txt
# =============================================================================================

def cmd_report(args, ctx):
    rep = ctx["out"] / "turntable_report.txt"
    if not rep.is_file():
        print("  [report] lipsește turntable_report.txt")
        return {}
    rows = []
    rx = re.compile(r"^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*/\s*([\d.]+)")
    total = None
    for line in rep.read_text(encoding="utf-8").splitlines():
        m = rx.match(line.strip())
        if m and "/" in m.group(1):
            name = m.group(1)
            cam_dir = name.split("/")[0]
            lm = re.search(r"(\d)[_]?(\d{1,4})\.\w+$", name)
            rows.append((cam_dir, int(lm.group(1)) if lm else 0, int(m.group(2)), float(m.group(3)),
                         float(m.group(4)), float(m.group(5))))
        if line.startswith("Total:"):
            total = line.strip()
        if line.startswith("Urcare:"):
            print("  [report]", line.strip())
    print(f"  [report] {total}")
    print(f"  [report] {'cam':>4s} {'nivel':>5s} {'poze':>4s} {'RMS med':>8s} {'RMS max':>8s} {'dif mm med/max':>15s} {'dif ° med/max':>14s}")
    summary = {}
    for key in sorted({(r[0], r[1]) for r in rows}):
        g = np.array([r[2:] for r in rows if (r[0], r[1]) == key], float)
        summary[f"{key[0]}/L{key[1]}"] = {
            "n": len(g), "rms_median": float(np.median(g[:, 1])), "rms_max": float(g[:, 1].max()),
            "free_mm_median": float(np.median(g[:, 2])), "free_mm_max": float(g[:, 2].max()),
            "free_deg_median": float(np.median(g[:, 3])), "free_deg_max": float(g[:, 3].max()),
        }
        s = summary[f"{key[0]}/L{key[1]}"]
        print(f"  [report] {key[0]:>4s} {key[1]:5d} {len(g):4d} {s['rms_median']:8.2f} {s['rms_max']:8.2f} "
              f"{s['free_mm_median']:7.2f}/{s['free_mm_max']:5.2f} {s['free_deg_median']:7.2f}/{s['free_deg_max']:5.2f}")
    return summary


# =============================================================================================
# overlays — aliniere geometrie–imagine
# =============================================================================================

def cmd_overlays(args, ctx):
    out_dir = ctx["dbg"] / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    V, F = ctx["mesh"]()
    low_path = ctx["out"] / "cleaned_mesh_low.obj"
    Vl, Fl = load_mesh(low_path) if low_path.is_file() else (V, F)
    tags = load_tag_layout()
    results = []
    for cam in ctx["views"]:
        R, C, K, w, h = cam["R"], cam["C"], cam["K_new"], cam["w"], cam["h"]
        img = undistorted_original(cam, ctx["inp"])
        fid, depth = render_face_ids(V, F, R, C, K, w, h)
        sil = (fid >= 0).astype(np.uint8) * 255

        # Eroare de contur: distanța de la conturul mesh-ului la cea mai apropiată muchie Canny
        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
        edges = cv2.Canny(gray, 30, 90)
        contour_img = cv2.morphologyEx(sil, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        band = cv2.dilate(contour_img, np.ones((61, 61), np.uint8)) > 0
        edges_band = (edges > 0) & band
        dist = cv2.distanceTransform((~edges_band).astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.nonzero(contour_img)
        d = dist[cy, cx]
        z_med = float(np.median(depth[np.isfinite(depth)]))
        gsd_mm = z_med / K[0, 0] * 1000.0

        # Masca folosită la texturare față de silueta mesh-ului
        mstats = {}
        mp = texturing_mask_path(ctx["images"], cam)
        diff_vis = None
        if mp:
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
            s = sil > 0
            s_d = cv2.dilate(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
            s_e = cv2.erode(sil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
            mstats = {
                "mask_px": int(m.sum()), "mesh_px": int(s.sum()),
                # pixeli de pe piesă (siluetă erodată 3 px) pe care masca îi face NEGRI
                "object_cut_pct": float(100.0 * (s_e & ~m).sum() / max(1, s_e.sum())),
                # pixeli de fundal (în afara siluetei dilatate 3 px) păstrați de mască
                "background_kept_pct": float(100.0 * (m & ~s_d).sum() / max(1, s.sum())),
            }
            diff_vis = (img * 0.35).astype(np.uint8)
            diff_vis[s & m] = (0.5 * diff_vis[s & m] + [0, 120, 0]).astype(np.uint8)
            diff_vis[s_e & ~m] = (0, 0, 255)      # roșu: piesă tăiată de mască (negru pe piesă)
            diff_vis[m & ~s_d] = (255, 128, 0)    # albastru: fundal păstrat (alb/tag pe piesă)

        # Desen: wireframe low-poly vizibil (galben), contur high-poly (verde), tag-uri (magenta)
        ov = img.copy()
        uvl, zl, Xcl = project_vertices(Vl, R, C, K)
        n = np.cross(Xcl[Fl[:, 1]] - Xcl[Fl[:, 0]], Xcl[Fl[:, 2]] - Xcl[Fl[:, 0]])
        front = np.einsum("ij,ij->i", n, Xcl[Fl[:, 0]]) < 0
        E = unique_edges(Fl[front])
        mid = (uvl[E[:, 0]] + uvl[E[:, 1]]) / 2.0
        zmid = (zl[E[:, 0]] + zl[E[:, 1]]) / 2.0
        mx = np.clip(np.round(mid[:, 0]).astype(int), 0, w - 1)
        my = np.clip(np.round(mid[:, 1]).astype(int), 0, h - 1)
        vis = zmid <= depth[my, mx] + 0.002
        wire = np.zeros_like(ov)
        for a, b in E[vis]:
            pa = tuple(np.round(uvl[a] * 4).astype(int))
            pb = tuple(np.round(uvl[b] * 4).astype(int))
            cv2.line(wire, pa, pb, (0, 220, 255), 1, cv2.LINE_AA, shift=2)
        ov = np.where(wire > 0, (0.45 * ov + 0.55 * wire).astype(np.uint8), ov)
        cnts, _ = cv2.findContours(sil, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2, cv2.LINE_AA)
        for t in tags:
            corners = np.array(t["corners"], dtype=np.float64)
            uvt, zt, _ = project_vertices(corners, R, C, K)
            if np.all(zt > 0):
                cv2.polylines(ov, [np.round(uvt).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)

        name = f"{cam['view_id']}_{Path(cam['path']).stem}_cam{cam['intrinsic_id']}_L{view_level(cam)}"
        x0, y0, x1, y1 = silhouette_bbox(sil, pad=80)
        crop = ov[y0:y1, x0:x1].copy()
        put_label(ov, f"{name}  contur: median {np.median(d):.1f}px  p90 {np.percentile(d, 90):.1f}px  (GSD {gsd_mm:.3f} mm/px)", (30, 70), 1.6)
        save_jpg(out_dir / f"{name}_overlay.jpg", ov, max_side=2400)
        save_jpg(out_dir / f"{name}_crop.jpg", crop)
        # zoom 3x pe vârful obiectului
        top_y = int(np.nonzero(sil.any(axis=1))[0][0])
        top_x = int(np.mean(np.nonzero(sil[top_y + 5])[0])) if sil[top_y + 5].any() else w // 2
        zx0, zy0 = max(0, top_x - 150), max(0, top_y - 60)
        zoom = cv2.resize(ov[zy0:zy0 + 300, zx0:zx0 + 300], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        save_jpg(out_dir / f"{name}_zoom_top.jpg", zoom)
        if diff_vis is not None:
            put_label(diff_vis, f"{name}  rosu: piesa taiata de masca {mstats['object_cut_pct']:.2f}%   albastru: fundal pastrat {mstats['background_kept_pct']:.2f}%", (30, 70), 1.4)
            save_jpg(out_dir / f"{name}_mask_vs_mesh.jpg", diff_vis[y0:y1, x0:x1] if args.crop_mask_diff else diff_vis, max_side=2400)

        r = {"view": name, "contour_median_px": float(np.median(d)), "contour_p90_px": float(np.percentile(d, 90)),
             "contour_median_mm": float(np.median(d) * gsd_mm), "gsd_mm_per_px": gsd_mm, **mstats}
        results.append(r)
        print(f"  [overlays] {name:32s} contur median {r['contour_median_px']:.2f}px p90 {r['contour_p90_px']:.2f}px"
              + (f" | mască: piesă tăiată {mstats['object_cut_pct']:.2f}%  fundal păstrat {mstats['background_kept_pct']:.2f}%" if mstats else ""))
    return results


# =============================================================================================
# labels — ce vedere a primit fiecare față
# =============================================================================================

def texrecon_scene_views(scene_dir: Path) -> List[Tuple[str, str, str]]:
    """Ordinea vederilor exact ca tex::from_images_and_camera_files (fișiere sortate, .cam + imaginea
    cu același prefix). Eticheta l din labeling.vec = vederea l-1 din listă (0 = față nevăzută)."""
    names = sorted(os.listdir(scene_dir))
    views = []
    for i, cam_name in enumerate(names):
        if cam_name[-4:].upper() != ".CAM" or len(cam_name) <= 4:
            continue
        prefix = cam_name[:-4]
        step, j = 1, i + 1
        while 0 <= j < len(names):
            img_name = names[j]
            if img_name[:len(prefix)] != prefix:
                if step == 1:
                    j, step = i - 1, -1
                    continue
                break
            if img_name[-4:].upper() in (".PNG", ".JPG", "TIFF", "JPEG"):
                views.append((prefix, cam_name, img_name))
                break
            j += step
    return views


def cmd_labels(args, ctx):
    out_dir = ctx["dbg"] / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = ctx["out"] / "texrecon"
    scene = tex_dir / "scene"
    if not scene.is_dir():
        print("  [labels] lipsește texrecon/scene")
        return {}
    views = texrecon_scene_views(scene)
    current = {f"{c['view_id']}_{Path(c['path']).stem}": c for c in ctx["cams"]}
    info = []
    for prefix, cam_name, img_name in views:
        c = current.get(prefix)
        info.append({"prefix": prefix, "image": img_name, "current": c is not None,
                     "camera": c["intrinsic_id"] if c else "?"})
    n_stale = sum(1 for x in info if not x["current"])
    print(f"  [labels] scene/: {len(views)} vederi încărcate de texrecon, {len(ctx['cams'])} în cameras_known.sfm, "
          f"{n_stale} vederi care NU aparțin rulării curente")

    results = {}
    for lab_file in sorted(tex_dir.glob("*_labeling.vec")):
        lab = np.fromfile(str(lab_file), dtype=np.uint64).astype(np.int64)
        hist = np.bincount(lab, minlength=len(views) + 1)
        groups = {"nevăzute (eticheta 0)": int(hist[0]), "cam0 curente": 0, "cam1 curente": 0, "vederi vechi (stale)": 0}
        for l in range(1, len(views) + 1):
            x = info[l - 1]
            key = "vederi vechi (stale)" if not x["current"] else f"cam{x['camera']} curente"
            groups[key] = groups.get(key, 0) + int(hist[l])
        print(f"  [labels] {lab_file.name}: {lab.size:,d} fețe")
        for k, v in groups.items():
            print(f"             {k:24s} {v:9,d}  ({100.0 * v / lab.size:5.1f}%)")
        results[lab_file.name] = {k: {"faces": v, "pct": 100.0 * v / lab.size} for k, v in groups.items()}

        # Hartă de etichete din câteva vederi reale (doar pentru mesh-ul high, aceeași ordine de fețe ca PLY-ul texrecon)
        mesh_ply = tex_dir / (lab_file.name.replace("_labeling.vec", "") + ".ply")
        src_ply = None
        conf = tex_dir / (lab_file.name.replace("_labeling.vec", "") + ".conf")
        if conf.is_file():
            for line in conf.read_text(encoding="utf-8").splitlines():
                if line.startswith("Input mesh:"):
                    src_ply = tex_dir / Path(line.split(":", 1)[1].strip()).name
        mesh_file = src_ply if src_ply and src_ply.is_file() else (mesh_ply if mesh_ply.is_file() else None)
        if mesh_file is None:
            continue
        V, F = load_mesh(mesh_file)
        if len(F) != lab.size:
            print(f"  [labels] {mesh_file.name} are {len(F)} fețe ≠ {lab.size} etichete, sar peste hartă")
            continue
        palette = np.zeros((len(views) + 1, 3), np.uint8)
        rng = np.random.default_rng(7)
        for l in range(1, len(views) + 1):
            x = info[l - 1]
            if not x["current"]:
                palette[l] = (0, 0, 230)                                           # roșu: vedere veche
            elif x["camera"] == "0":
                palette[l] = (rng.integers(150, 255), rng.integers(60, 140), 0)    # albastru: cam0
            else:
                palette[l] = (0, rng.integers(150, 255), rng.integers(0, 90))      # verde: cam1
        for cam in ctx["views"][: args.label_views]:
            fid, _ = render_face_ids(V, F, cam["R"], cam["C"], cam["K_new"], cam["w"], cam["h"])
            img = undistorted_original(cam, ctx["inp"])
            vis = (img * 0.3).astype(np.uint8)
            m = fid >= 0
            vis[m] = palette[lab[fid[m]]]
            x0, y0, x1, y1 = silhouette_bbox(m, pad=60)
            stale_pct = 100.0 * np.isin(lab[fid[m]], [i + 1 for i, x in enumerate(info) if not x["current"]]).mean()
            crop = vis[y0:y1, x0:x1].copy()
            put_label(crop, f"vedere {cam['view_id']}: rosu=vederi vechi {stale_pct:.0f}% din pixeli", (10, 40), 1.0)
            save_jpg(out_dir / f"{lab_file.stem}_{cam['view_id']}.jpg", crop)
    return {"views_in_scene": len(views), "stale_views": n_stale, "label_stats": results}


# =============================================================================================
# validity — ce pixeli consideră texrecon valizi
# =============================================================================================

def cmd_validity(args, ctx):
    from scipy import ndimage
    out_dir = ctx["dbg"] / "validity"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, cam in enumerate(ctx["cams"]):
        p = texturing_image_path(ctx["images"], cam)
        mp = texturing_mask_path(ctx["images"], cam)
        if p is None:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        valid = texrecon_validity(img)
        s = img.astype(np.int32).sum(axis=2)
        if mp is not None:
            fg = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
        else:
            fg = s > 0
        bg = ~fg
        band = ndimage.binary_dilation(fg, iterations=6) & bg
        holes = ndimage.binary_fill_holes(fg) & ~fg
        r = {
            "image": p.name,
            "bg_exact_zero_pct": float(100.0 * (s[bg] == 0).mean()) if bg.any() else 100.0,
            "bg_valid_pct": float(100.0 * valid[bg].mean()) if bg.any() else 0.0,
            "band6_valid_pct": float(100.0 * valid[band].mean()) if band.any() else 0.0,
            # negru valid în interiorul siluetei (găuri ale măștii) = texrecon îl folosește ca textură
            "inner_black_valid_px": int((valid & holes).sum()),
        }
        rows.append(r)
        if k < args.validity_images:
            vis = img.copy()
            vis[~valid] = (vis[~valid] * 0.25).astype(np.uint8)
            bad = valid & (s <= 15)
            vis[cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = (0, 0, 255)
            x0, y0, x1, y1 = silhouette_bbox(fg, pad=60)
            crop = vis[y0:y1, x0:x1].copy()
            put_label(crop, "rosu = pixel (aproape) negru pe care texrecon il considera VALID", (10, 40), 0.9)
            save_jpg(out_dir / f"{p.stem}_validity.jpg", crop)
    if rows:
        a = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "image"}
        for cam_id in sorted({c["intrinsic_id"] for c in ctx["cams"]}):
            sel = [r for r, c in zip(rows, ctx["cams"]) if c["intrinsic_id"] == cam_id]
            if sel:
                print(f"  [validity] cam{cam_id}: fundal exact 0 {np.mean([r['bg_exact_zero_pct'] for r in sel]):.2f}% | "
                      f"fundal valid {np.mean([r['bg_valid_pct'] for r in sel]):.3f}% | "
                      f"banda 6px valid {np.mean([r['band6_valid_pct'] for r in sel]):.1f}% | "
                      f"negru valid în siluetă {np.mean([r['inner_black_valid_px'] for r in sel]):.0f} px/imagine")
        return {"mean": a, "per_image": rows}
    return {}


# =============================================================================================
# colors — diferența de culoare dintre camere
# =============================================================================================

def remap_points(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Eșantionare biliniară în puncte arbitrare (cv2.remap acceptă hărți de cel mult 32767 rânduri)."""
    n = len(x)
    cols = 4096
    rows = max(1, int(math.ceil(n / cols)))
    mx = np.zeros(rows * cols, np.float32)
    my = np.zeros(rows * cols, np.float32)
    mx[:n], my[:n] = x, y
    out = cv2.remap(img, mx.reshape(rows, cols), my.reshape(rows, cols), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * cols, -1)[:n]


def bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return remap_points(img, uv[:, 0], uv[:, 1]).astype(np.float64)


def cmd_colors(args, ctx):
    V, F = ctx["mesh"]()
    tags = load_tag_layout()
    # Puncte pe hârtia albă a platanului: inel 45–125 mm, departe de tag-uri
    rng = np.random.default_rng(3)
    r = np.sqrt(rng.uniform(0.045 ** 2, 0.125 ** 2, 6000))
    th = rng.uniform(0, 2 * np.pi, 6000)
    P = np.stack([r * np.cos(th), np.zeros_like(r), r * np.sin(th)], axis=1)
    keep = np.ones(len(P), bool)
    for t in tags:
        c = np.array(t["center"], float)
        keep &= np.linalg.norm(P - c, axis=1) > 0.030
    P = P[keep]

    # Puncte pe piesă: subeșantion de vârfuri (cu normale)
    import trimesh
    mesh = trimesh.Trimesh(V, F, process=False)
    vn = np.asarray(mesh.vertex_normals)
    sel = rng.choice(len(V), size=min(40000, len(V)), replace=False)
    Pv, Nv = V[sel], vn[sel]

    paper = {"0": [], "1": []}
    surf = {"0": [], "1": []}       # listă de (indice punct, culoare)
    for cam in ctx["cams"]:
        R, Cc, K, w, h = cam["R"], cam["C"], cam["K_new"], cam["w"], cam["h"]
        if args.colors_from_images:
            # imaginile de texturare (după câștigurile pe cameră); hârtia e mascată, rămâne doar piesa
            img = cv2.imread(str(texturing_image_path(ctx["images"], cam)), cv2.IMREAD_COLOR)
        else:
            img = undistorted_original(cam, ctx["inp"])
        fid, depth = render_face_ids(V, F, R, Cc, K, w, h)
        # hârtie: vizibilă dacă nu e acoperită de piesă
        uv, z, _ = project_vertices(P, R, Cc, K)
        ok = (z > 0) & (uv[:, 0] > 2) & (uv[:, 0] < w - 3) & (uv[:, 1] > 2) & (uv[:, 1] < h - 3)
        ix = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        ok &= ~np.isfinite(depth[iy, ix]) | (z < depth[iy, ix])
        col = bilinear(img, uv[ok])
        bright = col.mean(axis=1)
        good = (bright > 60) & (col.max(axis=1) < 250)
        paper[cam["intrinsic_id"]].append(col[good])
        # suprafață: vizibilă (test de adâncime), frontală
        uv, z, _ = project_vertices(Pv, R, Cc, K)
        ok = (z > 0) & (uv[:, 0] > 2) & (uv[:, 0] < w - 3) & (uv[:, 1] > 2) & (uv[:, 1] < h - 3)
        ix = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
        ok &= np.abs(z - depth[iy, ix]) < 0.0015
        view_dir = Cc[None, :] - Pv
        view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True)
        ok &= np.einsum("ij,ij->i", view_dir, Nv) > 0.6
        idx = np.nonzero(ok)[0]
        col = bilinear(img, uv[idx])
        surf[cam["intrinsic_id"]].append((idx, col))

    out = {}
    p0 = np.concatenate(paper["0"]) if paper["0"] else np.zeros((0, 3))
    p1 = np.concatenate(paper["1"]) if paper["1"] else np.zeros((0, 3))
    if len(p0) and len(p1) and not args.colors_from_images:
        m0, m1 = np.median(p0, axis=0), np.median(p1, axis=0)
        g = m0 / m1
        out["paper_median_bgr_cam0"] = m0.tolist()
        out["paper_median_bgr_cam1"] = m1.tolist()
        out["paper_gain_cam1_to_cam0_bgr"] = g.tolist()
        print(f"  [colors] hârtie platan  cam0 BGR {m0.round(1)}  cam1 BGR {m1.round(1)}  câștig cam1→cam0 B/G/R {g.round(3)}")

    # pe piesă: median per punct per cameră, apoi raportul median
    acc = {}
    for cam_id in ("0", "1"):
        n = len(Pv)
        sums = [[] for _ in range(n)]
        for idx, col in surf[cam_id]:
            for i, c in zip(idx, col):
                sums[i].append(c)
        acc[cam_id] = sums
    ratios = []
    for i in range(len(Pv)):
        a, b = acc["0"][i], acc["1"][i]
        if len(a) >= 2 and len(b) >= 2:
            ma, mb = np.median(a, axis=0), np.median(b, axis=0)
            if (ma.mean() > 25) and (mb.mean() > 25) and ma.max() < 245 and mb.max() < 245:
                ratios.append(ma / np.maximum(mb, 1e-3))
    if ratios:
        ratios = np.array(ratios)
        g = np.median(ratios, axis=0)
        out["surface_gain_cam1_to_cam0_bgr"] = g.tolist()
        out["surface_points"] = int(len(ratios))
        lum = 0.114 * ratios[:, 0] + 0.587 * ratios[:, 1] + 0.299 * ratios[:, 2]
        out["surface_rel_chroma_B_R"] = float(np.median(ratios[:, 0] / ratios[:, 2]))
        print(f"  [colors] suprafață piesă ({len(ratios)} puncte văzute de ambele camere): câștig cam1→cam0 B/G/R {g.round(3)}, "
              f"luminanță {np.median(lum):.3f}, raport B/R {out['surface_rel_chroma_B_R']:.3f}")
    return out


# =============================================================================================
# glb — inspecție
# =============================================================================================

def read_glb(path: Path) -> Tuple[dict, bytes]:
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"{path} nu e GLB")
    off, js, binc = 12, None, b""
    while off < length:
        clen, ctype = struct.unpack_from("<I4s", data, off)
        chunk = data[off + 8: off + 8 + clen]
        if ctype == b"JSON":
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            binc = chunk
        off += 8 + clen
    return js, binc


def image_thumb(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)


def match_atlas(arr: np.ndarray, atlases: Dict[str, np.ndarray], tol: float = 4.0) -> Optional[str]:
    """Atlasul cu miniatura cea mai apropiată (toleranță pentru JPEG / redimensionare)."""
    t = image_thumb(arr)
    best, best_d = None, 1e9
    for name, a in atlases.items():
        d = float(np.abs(t - a).mean())
        if d < best_d:
            best, best_d = name, d
    return best if best_d < tol else None


def inspect_glb(glb: Path, atlas_files: List[Path]) -> Dict:
    js, binc = read_glb(glb)
    atlas_thumbs = {}
    for f in atlas_files:
        a = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if a is not None:
            atlas_thumbs[f.name] = image_thumb(a)
    images = []
    for im in js.get("images", []):
        bv = js["bufferViews"][im["bufferView"]]
        raw = binc[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        images.append({"mime": im.get("mimeType"), "bytes": bv["byteLength"],
                       "size": None if arr is None else [arr.shape[1], arr.shape[0]],
                       "matches_atlas": match_atlas(arr, atlas_thumbs) if arr is not None else None})
    prims = []
    for mesh in js.get("meshes", []):
        for pr in mesh.get("primitives", []):
            mat = js["materials"][pr["material"]] if "material" in pr else {}
            tex = (mat.get("pbrMetallicRoughness", {}).get("baseColorTexture") or {}).get("index")
            src = js["textures"][tex]["source"] if tex is not None else None
            n_idx = js["accessors"][pr["indices"]]["count"] // 3 if "indices" in pr else None
            prims.append({"mesh": mesh.get("name"), "material": pr.get("material"), "image": src, "faces": n_idx})
    return {"file": glb.name, "materials": len(js.get("materials", [])), "images": images,
            "textures": len(js.get("textures", [])), "primitives": prims,
            "extensions": js.get("extensionsUsed", [])}


def obj_materials(obj: Path) -> Tuple[Dict[str, int], Dict[str, str]]:
    counts: Dict[str, int] = {}
    cur = None
    mtllib = None
    with open(obj, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("f "):
                counts[cur] = counts.get(cur, 0) + 1
            elif line.startswith("usemtl "):
                cur = line.split(maxsplit=1)[1].strip()
            elif line.startswith("mtllib "):
                mtllib = line.split(maxsplit=1)[1].strip()
    maps = {}
    if mtllib and (obj.parent / mtllib).is_file():
        cur = None
        for line in (obj.parent / mtllib).read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("newmtl "):
                cur = line.split(maxsplit=1)[1].strip()
            elif line.strip().startswith("map_Kd") and cur:
                maps[cur] = line.split(maxsplit=1)[1].strip()
    return counts, maps


def cmd_glb(args, ctx):
    res = {}
    for lod, obj_rel, glb_rel in (("high", "obj/high/model_high.obj", "glb/high_model.glb"),
                                  ("low", "obj/low/model_low.obj", "glb/low_model.glb")):
        obj, glb = ctx["out"] / obj_rel, ctx["out"] / glb_rel
        if not (obj.is_file() and glb.is_file()):
            continue
        counts, maps = obj_materials(obj)
        atlases = [obj.parent / maps[m] for m in counts if m in maps and (obj.parent / maps[m]).is_file()]
        g = inspect_glb(glb, atlases)
        used_images = {p["image"] for p in g["primitives"]}
        wrong = 0
        by_faces = {nf: m for m, nf in counts.items()}
        for k, p in enumerate(g["primitives"]):
            # primitiva GLB poartă numele materialului din OBJ (trimesh); altfel o potrivesc după nr. de fețe
            mname = p["mesh"] if p["mesh"] in counts else by_faces.get(p["faces"])
            if mname is None and len(g["primitives"]) == len(counts):
                mname = list(counts)[k]
            img = g["images"][p["image"]]["matches_atlas"] if p["image"] is not None else None
            if mname is not None and img != maps.get(mname):
                wrong += counts[mname]
        print(f"  [glb] {lod}: OBJ {len(counts)} materiale / {sum(counts.values()):,d} fețe | GLB {g['materials']} materiale, "
              f"{len(g['images'])} imagini ({', '.join(sorted({str(i['mime']) for i in g['images']}))}), "
              f"{len(g['primitives'])} primitive, imagini distincte folosite: {len(used_images)}")
        if wrong:
            print(f"  [glb] {lod}: {wrong:,d} fețe ({100.0 * wrong / max(1, sum(counts.values())):.1f}%) au în GLB alt atlas decât în OBJ")
        res[lod] = {**g, "obj_materials": len(counts), "faces_with_wrong_atlas": wrong}
    return res


# =============================================================================================
# render — OBJ texturat și GLB, din pozițiile reale ale camerelor
# =============================================================================================

def load_textured(path: Path, scale: float = 1.0):
    """(V, F, UV per vertex, id geometrie per față, imagini BGR) dintr-un OBJ/GLB texturat (trimesh)."""
    import trimesh
    scene = trimesh.load(str(path), process=False, force="scene")
    Vs, Fs, UVs, Gs, imgs = [], [], [], [], []
    off = 0
    geoms = list(scene.geometry.items())
    if hasattr(scene, "graph") and path.suffix.lower() in (".glb", ".gltf"):
        dumped = []
        for node in scene.graph.nodes_geometry:
            T, gname = scene.graph[node]
            g = scene.geometry[gname].copy()
            g.apply_transform(T)
            dumped.append((gname, g))
        geoms = dumped
    for gi, (name, g) in enumerate(geoms):
        uv = getattr(g.visual, "uv", None)
        mat = getattr(g.visual, "material", None)
        img = None
        if mat is not None:
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if uv is None or img is None:
            continue
        arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        Vs.append(np.asarray(g.vertices, np.float64) * scale)
        Fs.append(np.asarray(g.faces, np.int64) + off)
        UVs.append(np.asarray(uv, np.float64))
        Gs.append(np.full(len(g.faces), len(imgs), np.int64))
        imgs.append(arr)
        off += len(g.vertices)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(UVs), np.concatenate(Gs), imgs


def render_textured(V, F, UV, G, imgs, cam) -> np.ndarray:
    R, C, K, w, h = cam["R"], cam["C"], cam["K_new"], cam["w"], cam["h"]
    fid, _ = render_face_ids(V, F, R, C, K, w, h)
    out = np.zeros((h, w, 3), np.uint8)
    ys, xs = np.nonzero(fid >= 0)
    f = fid[ys, xs]
    uv, z, _ = project_vertices(V, R, C, K)
    tri = F[f]
    p0, p1, p2 = uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]]
    area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    l0 = ((p2[:, 0] - p1[:, 0]) * (ys - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (xs - p1[:, 0])) / area
    l1 = ((p0[:, 0] - p2[:, 0]) * (ys - p2[:, 1]) - (p0[:, 1] - p2[:, 1]) * (xs - p2[:, 0])) / area
    l2 = 1.0 - l0 - l1
    w0, w1, w2 = l0 / z[tri[:, 0]], l1 / z[tri[:, 1]], l2 / z[tri[:, 2]]
    s = w0 + w1 + w2
    t = (w0[:, None] * UV[tri[:, 0]] + w1[:, None] * UV[tri[:, 1]] + w2[:, None] * UV[tri[:, 2]]) / s[:, None]
    g = G[f]
    for gi in np.unique(g):
        sel = g == gi
        im = imgs[gi]
        th, tw = im.shape[:2]
        mx = (t[sel, 0] * tw - 0.5).astype(np.float32)
        my = ((1.0 - t[sel, 1]) * th - 0.5).astype(np.float32)
        out[ys[sel], xs[sel]] = remap_points(im, mx, my)
    return out


def cmd_render(args, ctx):
    out_dir = ctx["dbg"] / "render"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = []
    tex_obj = ctx["out"] / "texrecon" / "model_textured.obj"
    if tex_obj.is_file():
        models.append(("texrecon_obj", tex_obj, 1.0))
    glb = ctx["out"] / "glb" / "high_model.glb"
    if glb.is_file():
        models.append(("glb_high", glb, 1.0 / float(args.export_scale)))
    for extra in args.extra_models or []:
        lab, p, sc = extra.split("=", 1)[0], Path(extra.split("=", 1)[1].split("@")[0]), float(extra.split("@")[1]) if "@" in extra else 1.0
        models.append((lab, p, sc))
    loaded = []
    for lab, p, sc in models:
        print(f"  [render] încarc {lab}: {p}")
        loaded.append((lab, load_textured(p, sc)))
    for cam in ctx["views"][: args.render_views]:
        photo = undistorted_original(cam, ctx["inp"])
        panels = []
        V0, F0 = ctx["mesh"]()
        sil = render_silhouette(V0, F0, cam["R"], cam["C"], cam["K_new"], cam["w"], cam["h"])
        x0, y0, x1, y1 = silhouette_bbox(sil > 0, pad=50)
        p = photo[y0:y1, x0:x1].copy()
        put_label(p, "fotografie", (10, 40), 1.1)
        panels.append(p)
        for lab, (V, F, UV, G, imgs) in loaded:
            r = render_textured(V, F, UV, G, imgs, cam)[y0:y1, x0:x1].copy()
            put_label(r, lab, (10, 40), 1.1)
            panels.append(r)
        sheet = np.concatenate(panels, axis=1)
        save_jpg(out_dir / f"{cam['view_id']}_{Path(cam['path']).stem}_compare.jpg", sheet, max_side=3600)
        print(f"  [render] {cam['view_id']}: {' | '.join(['foto'] + [l for l, _ in loaded])}")
    return {}


# =============================================================================================
# main
# =============================================================================================

def main():
    ap = argparse.ArgumentParser(description="Diagnostic texturare (PAS 0) pentru Turntable Studio.")
    ap.add_argument("sections", nargs="*", default=["all"], help=f"secțiuni: all sau {', '.join(SECTIONS)}")
    ap.add_argument("--output_dir", default="output", help="folderul output/ al unei rulări")
    ap.add_argument("--input_dir", default="input_images", help="imaginile originale (pentru căile /workspace/...)")
    ap.add_argument("--images_dir", default=None,
                    help="imaginile de texturare analizate (implicit <output>/texturing_images dacă există, altfel <output>/undistorted)")
    ap.add_argument("--mesh", default=None, help="mesh-ul curățat (implicit <output>/model_cleaned_watertight.obj)")
    ap.add_argument("--views", default=None, help="viewId-uri separate prin virgulă (implicit: 6, ambele camere, toate nivelurile)")
    ap.add_argument("--count", type=int, default=6, help="câte vederi pentru overlay-uri")
    ap.add_argument("--label_views", type=int, default=3)
    ap.add_argument("--render_views", type=int, default=3)
    ap.add_argument("--validity_images", type=int, default=4)
    ap.add_argument("--export_scale", type=float, default=100.0, help="scara cu care e exportat GLB-ul")
    ap.add_argument("--colors_from_images", action="store_true",
                    help="'colors' eșantionează din imaginile de texturare (--images_dir) în loc de fotografiile originale")
    ap.add_argument("--crop_mask_diff", action="store_true", default=True)
    ap.add_argument("--extra_models", nargs="*", default=None, help="eticheta=cale[@scară] modele în plus pentru 'render'")
    ap.add_argument("--json", default=None, help="salvează rezultatele (implicit <output>/<debug_subdir>/diagnostic.json)")
    ap.add_argument("--debug_subdir", default="debug", help="subfolderul din <output> pentru imagini (ex. debug_after)")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    inp = Path(args.input_dir).resolve()
    dbg = out / args.debug_subdir
    dbg.mkdir(parents=True, exist_ok=True)
    cams = load_sfm_cameras(str(out / "cameras_known.sfm"))
    mesh_path = Path(args.mesh) if args.mesh else out / "model_cleaned_watertight.obj"
    cache = {}

    def mesh():
        if "m" not in cache:
            cache["m"] = load_mesh(mesh_path)
        return cache["m"]

    if args.images_dir:
        images = Path(args.images_dir).resolve()
    else:
        images = out / "texturing_images" if (out / "texturing_images").is_dir() else out / "undistorted"
    ctx = {"out": out, "inp": inp, "dbg": dbg, "cams": cams, "mesh": mesh, "images": images,
           "views": select_views(cams, args.count, args.views)}
    print(f"Diagnostic texturare: {out}  ({len(cams)} vederi; selectate: {', '.join(c['view_id'] for c in ctx['views'])}; "
          f"imagini de texturare: {images})")
    sections = SECTIONS if "all" in args.sections else args.sections
    results = {}
    for s in sections:
        print(f"== {s}")
        results[s] = globals()[f"cmd_{s}"](args, ctx)
    js = Path(args.json) if args.json else dbg / "diagnostic.json"
    js.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Rezultate: {js}")


if __name__ == "__main__":
    main()
