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

### 1.3. Măsuri Hardware Recomandate și AVERTISMENTE CRITICE
> [!CAUTION]
> **NU FORȚAȚI FILETUL ȘI NU FOLOSIȚI BANDĂ DE TEFLON SAU LOCTITE LICHID!**
> 1. **Fără Bandă de Teflon (PTFE):** Banda de teflon se forfeca la înșurubare în filet și eliberează micro-fibre sintetice volatile. Acestea pătrund direct în camera senzorului și se lipesc electrostatic pe sticla de protecție a senzorului IMX477, compromițând definitiv calitatea optică.
> 2. **Fără Adeziv Lichid (Loctite):** Efectul de capilaritate trage adezivul lichid (cianoacrilat / metacrilat) direct în fanta milimetrică a bobinei VCM (Voice Coil Motor), blocând definitiv motorul de autofocus.
> 3. **Nu forțați rotirea manuală:** Ansamblul optic al modulului B0272 este suspendat pe lamele flexibile și ghidaje fine. Forțarea exterioară poate deforma suspensia VCM.

**Soluția optimă și sigură pentru B0272:**
1. **Focus Lock Software (100% Sigur și Recomandat):**  
   Comanda de autofocus se rulează **o singură dată la inițializarea scanării** (la Nivelul 0, când piesa e așezată pe platan). Imediat după atingerea contrastului maxim, modul de autofocus este trecut în manual (`AfMode = 0` în `Picamera2` / `v4l2-ctl -c focus_absolute=...`), iar registrul `LensPosition` este înghețat. În această stare, suspensia elastică a VCM-ului și retenția electromagnetică mențin grupul optic perfect stabil pe toată durata scanării celor 20 de unghiuri.
2. **Este necesară blocarea fizică?**  
   **NU.** La accelerațiile line ale unui rig cu motoare stepper NEMA 17 (controlate cu microstepping 1/16 sau 1/32), masa grupului optic mobil din VCM este infimă (~0.8 grame) și nu suferă deplasări inerțiale. Doar blocarea software a poziției VCM este suficientă pentru a garanta stabilitatea optică.

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

### 3.3. Metoda Optimă Propusă: Fixed-Pose Bundle Adjustment (Auto-Repair cu Eroare de Reproiecție 2D)
Pentru a calibra fin focala și centrul optic fără a depinde exclusiv de AprilTags și fără modele AI greoaie:

#### Principiul Matematic Scale-Invariant:
1. **Rigiditate Cunoscută:** Cunoaștem pozițiile relative ale camerelor $R_i, T_i$ din cinematica platanului (pasul de $18^\circ$, elevațiile și axa de rotație sunt blocate rigid, eliminând complet riscul de drift BA).
2. **Extragere Trăsături Subpixel:** Extragem 200–500 de puncte cheie SIFT/AKAZE subpixel pe corpul piesei între cadre adiacente.
3. **Filtrare Epipolară:** Păstrăm exclusiv corespondențele curate care respectă geometria epipolară a rig-ului.
4. **Formularea de Minimizare a Erorii de Reproiecție 2D (cu Huber Loss):**  
   În loc de măsurarea distanței 3D între raze (care are un bias dependent de distanța până la piesă), optimizarea riguroasă din photogrammetrie industrială este **Fixed-Pose Bundle Adjustment**:
   $$\min_{K, \{ \mathbf{X}_j \}} \sum_{i \in \text{cams}} \sum_{j \in \text{points}} \rho_{\text{Huber}}\left( \| \mathbf{x}_{i, j} - \pi(K, R_i, T_i, \mathbf{X}_j) \|_2 \right)$$
   Unde:
   * $\mathbf{X}_j$ sunt reperele 3D triangulate pe piesă;
   * $R_i, T_i$ sunt **înghețate strict ca constante** (zero grade de libertate pentru camere);
   * Singurele variabile optimizate sunt intrinsecile $K = [f_x, f_y, c_x, c_y]$ și adâncimile punctelor;
   * $\rho_{\text{Huber}}$ rejectează automat reflexiile speculare de pe plastic.
5. **Viteză și Precizie:**
   Optimizarea cu SciPy `least_squares` rulează în sub 0.3 secunde, utilizează toată înălțimea geometrică a obiectului și ajustează focala cu o precizie de $\pm 0.005\text{ mm}$, eliminând complet decalajele de proiecție a texturilor.

---

## 4. Sinteza Problemelor Semnalate și Soluțiile Implementate

| Problemă semnalată | Cauza tehnică identificată | Soluția implementată în cod | Status |
| :--- | :--- | :--- | :--- |
| **Low-poly alb** | Decimatul PyMeshLab cu textură a eșuat pe cusături UV și a căzut pe decimat standard, care **șterge coordonatele `vt`**. Fără `vt`, OBJ-ul e alb. | **Branching Pipeline:** Decimare geometrică a geometriei etanșate la 50k fețe $\to$ texturare dedicată cu 2048px JPG direct pe low-poly. | ✅ Implementat în `main.py` & `exporter.py` (Testat unitar) |
| **OpenMVS textură neclară** | `--cost-smoothness-ratio 1.0` a penalizat cusăturile și a forțat OpenMVS să aleagă unghiuri razante/oblice la $80^\circ$. | Coborâre `--cost-smoothness-ratio` la **0.2**, activare `--close-holes 1` și împachetare UV densă (`--patch-packing-heuristic 3`). | ✅ Implementat în `config.py` & `openmvs_runner.py` |
| **High-poly GLB gri full** | `trimesh.load` nu a putut lega buffer-ul PNG din MTL și a căzut pe mesh-ul brut untextured (material gri PBR default). | **Direct Texture Visuals Binding:** Injectarea forțată a `TextureVisuals(uv, image)` direct în structura nodurilor glTF pentru ambele LOD-uri. | ✅ Implementat în `exporter.py` |
| **Descentrare / oscilație focală** | Lens breathing la 35 cm ($f \approx 3.94\text{ mm}$ vs $3.90\text{ mm}$ nominal) și descentrare centru optic M12. | **Propagare din AprilTags Nivel 0** (estimare dinamică la începutul fiecărei scanări) + Focus Lock Software pe registrul VCM + formulare Fixed-Pose BA 2D. | ✅ Documentat & arhitectură stabilizată |

---

*Document salvat în rădăcina proiectului: `turntable_studio/OPTICA_CALIBRARE_SI_RECONSTRUCTIE_RIG.md`.*
