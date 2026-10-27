#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_intrinsics.py — calibrează o cameră (focală, centru optic, distorsiune) cu o tablă de șah.

DE CE: tag-urile de pe platan apar doar în partea de JOS a imaginii. Din ele, scriptul de poziții
estimează bine camera acolo, dar distorsiunea din partea de SUS (unde e vârful obiectelor înalte)
rămâne ghicită (erori de 1–4 px). O calibrare cu tabla de șah plimbată prin TOT cadrul rezolvă asta.
Se face o singură dată, cât timp nu schimbi focusul (LensPosition) și nu atingi obiectivul.

CUM FACI POZELE (pentru fiecare cameră separat, cu focusul exact ca la scanare):
  * printează tabla la 100% (fără „fit to page”) și lipește-o pe ceva perfect plan (carton gros, sticlă);
  * MĂSOARĂ cu șublerul latura unui pătrat și dă valoarea exactă la --square;
  * 20–30 de poze la distanța de lucru (~35 cm), tabla întreagă în cadru, înclinată diferit
    (±30–45°) și mutată prin tot cadrul: colțuri, margini, centru — mai ales SUS, unde nu ajung tag-urile;
  * fără blur: ține tabla nemișcată sau sprijin-o.

UZ:
  python3 calibrate_intrinsics.py --images "calib/cam1/*.jpg" --serial cam1 --cols 9 --rows 6 --square 25.0 \
          --out intrinsics.json
  python3 calibrate_intrinsics.py --images "calib/cam2/*.jpg" --serial cam2 ... --out intrinsics.json
  (al doilea apel adaugă camera în același fișier)

  --serial trebuie să fie EXACT serialNumber-ul camerei din cameraInit.sfm (intrinsics),
  adică Exif:BodySerialNumber din poze (la tine: cam1 / cam2).
  --cols/--rows = numărul de COLȚURI INTERIOARE (o tablă de 10×7 pătrate are 9×6 colțuri interioare).

Rezultatul (intrinsics.json) se dă scriptului turntable_poses.py cu --intrinsicsIn.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# Windows: când ieșirea e redirecționată într-un fișier de log, consola folosește cp1252 și
# diacriticele (ă, ș, ț) dau UnicodeEncodeError. Forțăm UTF-8 pentru ieșire.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    import cv2
except ImportError:
    sys.exit("[eroare] lipsește OpenCV: pip install opencv-python-headless")


def main():
    ap = argparse.ArgumentParser(description="Calibrare cameră cu tablă de șah (model radialk3, ca AliceVision).")
    ap.add_argument("--images", required=True, help='pozele, cu ghilimele: "folder/*.jpg"')
    ap.add_argument("--serial", required=True, help="serialNumber-ul camerei din cameraInit.sfm (ex: cam1)")
    ap.add_argument("--cols", type=int, default=9, help="colțuri interioare pe orizontală")
    ap.add_argument("--rows", type=int, default=6, help="colțuri interioare pe verticală")
    ap.add_argument("--square", type=float, default=25.0, help="latura MĂSURATĂ a unui pătrat, în mm")
    ap.add_argument("--model", default="radialk3", choices=["radialk3", "radialk1"])
    ap.add_argument("--out", default="intrinsics.json", help="fișierul JSON (se completează dacă există)")
    ap.add_argument("--debugDir", default="", help="salvează imaginile cu colțurile găsite")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        sys.exit("[eroare] nicio imagine pentru %s" % args.images)
    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * (args.square / 1000.0)

    objs, imgs, used, size = [], [], [], None
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            print("  nu pot citi %s" % p)
            continue
        if size is None:
            size = (g.shape[1], g.shape[0])
        elif size != (g.shape[1], g.shape[0]):
            sys.exit("[eroare] imaginile au rezoluții diferite (%s vs %s)" % (size, g.shape[::-1]))
        ok, corners = cv2.findChessboardCornersSB(
            g, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        print("  %-40s %s" % (os.path.basename(p), "tablă găsită" if ok else "NU găsesc tabla"))
        if ok:
            objs.append(objp)
            imgs.append(corners.reshape(-1, 2).astype(np.float32))
            used.append(p)
    if len(used) < 8:
        sys.exit("[eroare] doar %d poze cu tabla detectată; fă cel puțin 15." % len(used))

    w, h = size
    flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
    if args.model == "radialk1":
        flags |= cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    K0 = np.array([[w * 0.62, 0, w / 2.0], [0, w * 0.62, h / 2.0], [0, 0, 1]])   # ~ obiectiv de 75°
    flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    def calib(o, i):
        return cv2.calibrateCamera(o, i, size, K0.copy(), np.zeros(5), flags=flags)

    rms, K, dist, rvecs, tvecs = calib(objs, imgs)
    # elimină pozele mult mai proaste decât restul (tablă mișcată, detecție greșită) și recalibrează
    per = []
    for o, i, rv, tv in zip(objs, imgs, rvecs, tvecs):
        pr, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per.append(float(np.sqrt(((pr.reshape(-1, 2) - i) ** 2).sum(axis=1).mean())))
    thr = max(1.0, 3.0 * float(np.median(per)))
    keep = [k for k, e in enumerate(per) if e <= thr]
    if len(keep) < len(per):
        print("Exclud %d poze cu eroare > %.2f px: %s" %
              (len(per) - len(keep), thr, [os.path.basename(used[k]) for k in range(len(per)) if k not in keep]))
        objs = [objs[k] for k in keep]
        imgs = [imgs[k] for k in keep]
        used = [used[k] for k in keep]
        rms, K, dist, rvecs, tvecs = calib(objs, imgs)

    # acoperirea cadrului: în câte din cele 6×4 zone ale imaginii au căzut colțuri
    allp = np.concatenate(imgs)
    gx = np.clip((allp[:, 0] / w * 6).astype(int), 0, 5)
    gy = np.clip((allp[:, 1] / h * 4).astype(int), 0, 3)
    cover = np.zeros((4, 6), bool)
    cover[gy, gx] = True
    d = dist.ravel()
    print("\nCamera %s: %d poze folosite, RMS %.3f px" % (args.serial, len(used), rms))
    print("  focală %.2f px, centru optic (%+.2f, %+.2f) px față de mijloc, k = [%.5f, %.5f, %.5f]" %
          (K[0, 0], K[0, 2] - w / 2, K[1, 2] - h / 2, d[0], d[1], d[4]))
    print("  acoperirea cadrului (# = are colțuri):")
    for row in cover:
        print("     " + "".join("#" if c else "." for c in row))
    if cover.mean() < 0.8 or not (cover[0].all() and cover[:, 0].all() and cover[:, -1].all()):
        print("  [atenție] tabla n-a trecut prin tot cadrul; adaugă poze cu ea în zonele cu '.' "
              "(mai ales marginea de SUS și colțurile).")
    if rms > 1.0:
        print("  [atenție] RMS peste 1 px: tabla nu e plană, poze mișcate sau --square greșit.")

    data = {}
    if os.path.isfile(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[args.serial] = {"width": w, "height": h, "f": float(K[0, 0]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                         "k": [float(d[0]), float(d[1]), float(d[4])], "model": args.model,
                         "rms_px": float(rms), "images": len(used), "coverage": float(cover.mean()),
                         "square_mm": args.square}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("\nScris: %s (camere: %s)" % (args.out, ", ".join(data)))

    if args.debugDir:
        os.makedirs(args.debugDir, exist_ok=True)
        for p, i in zip(used, imgs):
            im = cv2.imread(p)
            cv2.drawChessboardCorners(im, pattern, i.reshape(-1, 1, 2), True)
            cv2.imwrite(os.path.join(args.debugDir, os.path.basename(p)), cv2.resize(im, None, fx=0.5, fy=0.5))


if __name__ == "__main__":
    main()
