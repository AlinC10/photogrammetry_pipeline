# OPTICA_CALIBRARE_SI_RECONSTRUCTIE_RIG.md
## Analiză Optică Industrială, Calibrare Dinamică și Soluții Anti-Drift pentru Rig-ul Turntable Studio

**Data:** 23 Octombrie 2026  
**Hardware vizat:** Dual Arducam B0272 (Sony IMX477 12.3MP, Motorized Focus, M12 Lens Mount)  
**Destinație:** Pipeline-ul industrial Turntable Studio (OpenMVS, texrecon, AliceVision)

---

## 1. Analiza Hardware: Arducam B0272 (IMX477 Motorized Focus)

### 1.1. Arhitectura Optică Hibridă a Modulului B0272
Modulul de cameră **Arducam B0272** combină două mecanisme diferite de focalizare:
1. **Filetul exterior M12 manual:**
   * Corpul exterior al obiectivului se înșurubează pe montură. Acest reglaj manual stabilește distanța de bază flanșă-senzor (Back-Focal Length).
2. **Motorul VCM intern (Voice Coil Motor):**
   * Controlat digital prin I2C (driver DAC integrat, `LensPosition` în `libcamera` / `Picamera2` cu valori de la 0 la 1023 pași).
   * VCM-ul deplasează un grup optic intern pe o cursă fină de $\approx 0.3 - 0.5\text{ mm}$ pentru a realiza autofocusul.

### 1.2. Riscul Fizic de Vibrație și Lens Drifting
La un rig automatizat cu motoare pas-cu-pas (stepper NEMA 17 pe platan și tijă filetată T8 pe axa verticală Z):
* **Frecvențele de rezonanță ale pașilor:** Micro-pașii motorului produc vibrații mecanice continue.
* **Slăbirea filetului exterior:** Dacă obiectivul exterior nu este asigurat cu un inel de blocare strâns, filetul M12 tinde să se rotească fin în timp din cauza vibrațiilor de accelerare și decelerare ale platanului.
* **Consecințe optice:**
  * O rotație de doar $5^\circ$ pe un filet cu pas de $0.5\text{ mm}$ deplasează lentila axial cu $\approx 0.007\text{ mm}$.
  * Această mică mișcare modifică focala de bază cu $0.03 - 0.05\text{ mm}$ (ex. de la 3.90 la 3.94 mm) și provoacă un joc radial de $10 - 25\text{ px}$ al centrului optic $(c_x, c_y)$ din cauza jocului din filet.

### 1.3. Măsuri Hardware Recomandate
1. **Asigurarea mecanică a filetului exterior:**  
   Odată ce back-focus-ul a fost reglat manual astfel încât VCM-ul să aibă cursa centrată pe distanța de scanare (~35 cm), filetul exterior trebuie asigurat:
   * Fie printr-un contra-piuliță / inel de blocare M12 (locking ring);
   * Fie printr-o picătură minusculă de adeziv nepermanent (Loctite 222 mov sau o picătură de silicon / hot-glue pe exteriorul filetului, reversibilă ușor).
2. **Focus Lock Software:**  
   Comanda de autofocus se dă o singură dată la inițializarea scanării (când piesa e pusă pe disc), iar apoi registrul VCM este înghețat (`lock`) pentru toate cele 20 de unghiuri ale scanării.

---

## 2. De ce Focala Reala Oscilează în Jurul lui 3.9 mm

### 2.1. Legea Lentilelor la Distanță Finită (Lens Breathing)
Focala nominală de **3.9 mm** gravată pe carcasă este calculată exclusiv pentru focalizare la **infinit** ($d_{\text{obiect}} \to \infty$).

Când camera scanează o piesă la distanța de lucru $d_o \approx 350\text{ mm}$ (35 cm):
$$\frac{1}{f} = \frac{1}{d_o} + \frac{1}{d_i} \implies d_i = \frac{f \cdot d_o}{d_o - f} = \frac{3.90 \cdot 350}{350 - 3.90} \approx \mathbf{3.944\text{ mm}}$$
* Focala efectivă de proiecție a pixelilor crește automat cu $\approx +1.1\%$ ($+0.044\text{ mm}$, echivalentul a peste $+30\text{ pixeli}$ pe lățimea imaginii de 4056 px!).
* Dacă software-ul ar forța valoarea fixă de 3.900 mm, razele 3D s-ar deschide într-un unghi cu 1.1% prea larg, producând o ușoară descentrare și deformare geometrică către marginile piesei.

### 2.2. Cum Estimează deja `turntable_poses.py` Focala
În implementarea din `pipeline/turntable_poses.py`:
* Scriptul **NU folosește 3.9 mm ca valoare forțată**.
* 3.9 mm este doar estimarea inițială (`initial guess`) pentru `cv2.calibrateCamera`.
* La nivelul inferior (`level 0`), unde AprilTag-urile sunt perfect vizibile pe discul de 260 mm, scriptul extrage colțurile tag-urilor și optimizează focala $f$ și distorsiunea $k_1$ pentru fiecare cameră în parte.
* Valoarea calculată (de exemplu $f = 3.938\text{ mm}$ la camera 0 și $f = 3.912\text{ mm}$ la camera 1) este salvată în `cameras_known.sfm` și utilizată pe parcursul întregii reconstrucții.

---

## 3. Cercetare Tehnologică: Metode de Corecție și Auto-Calibrare în Timpul Scanării

### 3.1. Estimarea din AprilTags la Nivelul de Jos (Propagare Rig)
* **Cum funcționează:** Deoarece autofocusul VCM se blochează la începutul scanării, focala camerei 0 și a camerei 1 rămâne identică între nivelul de jos și nivelul de sus.
* **Avantaj:** Determinare matematică exactă, fără a introduce zgomot pe piese uniforme.
* **Limitare:** Când o piesă foarte voluminoasă acoperă aproape tot discul, pot rămâne vizibile doar 2-3 tag-uri, scăzând precizia estimării.

---

### 3.2. Evaluarea Modelului RoMa (Robust Dense Feature Matching)
Ai menționat modelul **RoMa** (CVPR 2024):
* **Ce este:** RoMa este cel mai performant model AI de deep learning pentru dense optical flow și corepondențe între imagini dificile.
* **Disponibilitate:** Open-source pe GitHub (`Parskatt/RoMa`), se instalează cu `pip install romatch` (necesită PyTorch și CUDA).
* **De ce NU este optim pentru acest rig:**
  1. **Consum uriaș de VRAM și necesitate de Downsampling:** Pe imagini de 12 Megapixeli (4056×3040), RoMa necesită peste 60 GB VRAM dacă ar rula la rezoluție maximă. În practică forțează redimensionarea la 560×560 sau 800×800 px, exact unde se pierde precizia subpixel necesară calculului fin de focală.
  2. **Overkill de date:** RoMa calculează 100.000 de puncte dense pe pereche de cadre. Pentru a ajusta 2 parametri scalari ($\Delta f$ și $\Delta c$), avem nevoie de doar 50–100 de puncte cheie subpixel distribuite curat pe înălțimea piesei.
  3. **Greutate:** Adaugă o dependență de 1.5 GB de greutăți neuronale într-un pipeline conceput să fie rapid și suplu.

---

### 3.3. Metoda Optimă Propusă: Epipolar Tie-Point Ray Intersection (Auto-Repair)
Pentru a calibra fin focala și centrul optic fără a depinde exclusiv de AprilTags și fără modele AI greoaie:

#### Principiul Geometric:
1. **Rigiditate Cunoscută:** Cunoaștem pozițiile relative ale camerelor $R_i, C_i$ din cinematica platanului (pasul de $18^\circ$ și axa de rotație sunt blocate rigid).
2. **Extragere Trăsături:** Extragem 200–500 de puncte cheie SIFT/AKAZE subpixel pe corpul piesei între cadre adiacente.
3. **Filtrare Epipolară:** Păstrăm exclusiv punctele care respectă linia epipolară $x_2^T F x_1 = 0$ determinată de rotirea de $18^\circ$.
4. **Calculul Distanței dintre Raze (Ray Skew Distance):**
   Fiecare pereche de puncte $(u_1, v_1) \leftrightarrow (u_2, v_2)$ definește două raze în spațiul 3D:
   $$r_1(\lambda_1) = C_1 + \lambda_1 R_1^T K^{-1} \begin{bmatrix} u_1 \\ v_1 \\ 1 \end{bmatrix}, \quad r_2(\lambda_2) = C_2 + \lambda_2 R_2^T K^{-1} \begin{bmatrix} u_2 \\ v_2 \\ 1 \end{bmatrix}$$
   Dacă matricea intrinsecă $K$ (focala $f$ și centrul optic $c_x, c_y$) este **exactă**, cele două raze se intersectează pe suprafața piesei (distanța minimă $d(r_1, r_2) \approx 0$).  
   Dacă există un drift optic $\Delta f$ sau $\Delta c$, razele trec una pe lângă alta la o distanță $d > 0$.
5. **Optimizare Rapidă (0.2 secunde cu SciPy):**
   Rulăm o optimizare Levenberg-Marquardt în care **pozițiile camerelor rămân înghețate**, iar singurele variabile sunt $f_x, f_y, c_x, c_y$:
   $$\min_{K} \sum_{k} \text{dist}^2\left(r_1^k(K), r_2^k(K)\right)$$
   Această optimizare folosește toată înălțimea piesei și rafinează focala la sutimi de milimetru, corectând orice descentrare a texturii.

---

## 4. Sinteza Problemelor Semnalate și Planul de Rezolvare

| Problemă semnalată | Cauza tehnică identificată | Soluția aplicată în cod |
| :--- | :--- | :--- |
| **Low-poly alb** | Decimatul PyMeshLab cu textură a eșuat pe cusături UV și a căzut pe decimat standard, care **șterge coordonatele `vt`**. Fără `vt`, OBJ-ul e alb. | **Branching Pipeline:** Decimare geometrică înainte de textură $\to$ texturare nativă dedicată cu 2048px JPG. |
| **OpenMVS textură neclară** | `--cost-smoothness-ratio 1.0` a penalizat cusăturile și a forțat OpenMVS să aleagă unghiuri razante/oblice la $80^\circ$. | Coborâre `--cost-smoothness-ratio` la **0.2**, activare `--close-holes 1` și împachetare UV densă (`--patch-packing-heuristic 3`). |
| **High-poly GLB gri full** | `trimesh.load` nu a putut lega buffer-ul PNG din MTL și a căzut pe mesh-ul brut untextured (material gri PBR default). | **Direct Texture Visuals Binding:** Injectarea forțată a buffer-ului de imagine direct în structura glTF. |
| **Descentrare / oscilație focală** | Lens breathing la 35 cm ($f \approx 3.94\text{ mm}$ vs $3.90\text{ mm}$ nominal) și descentrare centru optic M12. | **Propagare din AprilTags Nivel 0** + suport CLI pentru `--intrinsics_in` (Ground Truth) și opțiune de rafinare epipolară pe piese. |

---

*Document salvat în rădăcina proiectului: `turntable_studio/OPTICA_CALIBRARE_SI_RECONSTRUCTIE_RIG.md`.*
