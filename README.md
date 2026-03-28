# QuantumVision
### AI + Quantum Digital Twin Microscope for Material Analysis

An integrated AI–Quantum Digital Twin system for real-time material analysis and defect prediction, combining OpenCV-based image processing, Three.js 3D visualization, and Qiskit-powered quantum simulation. The system supports multiple input sources including webcams, embedded camera modules, and image uploads, and follows a pipeline of encoding, superposition, entanglement, phase transformation, and measurement to predict structural stability, crack propagation, and failure probabilities.

![Uploading Screenshot 2026-03-28 212917.png…]()

---

## Quick Start (3 steps)

```bash
# 1 — Install Python packages
pip install -r requirements.txt

# 2 — Start the server  (run from THIS folder)
python app.py

# 3 — Open browser
http://localhost:5050
```

---

## Folder Structure

```
QuantumVision/
├── app.py               ← Flask backend  (run this)
├── requirements.txt     ← pip packages
├── README.md
└── templates/
    └── index.html       ← Full frontend (Three.js UI)
```

> **Do NOT rename or move `templates/index.html`.**
> The backend reads it by absolute path relative to `app.py`.

---

## How to Use

| Button | Action |
|--------|--------|
| **CAPTURE** | Snap webcam frame (or use uploaded image) |
| **ENHANCE** | AI denoising + CLAHE sharpening |
| **DETECT** | Canny edge crack detection with overlay |
| **GEN 3D** | Sobel depth map → Three.js mesh |
| **SIMULATE** | Digital twin: stress / temp / time physics |
| **QUANTUM** | 3-qubit Qiskit circuit → probabilistic futures |
| **FULL DEMO** | Runs entire pipeline automatically (no camera needed) |

---

## Input Sources

| Source | How |
|--------|-----|
| Webcam | Browser asks permission automatically |
| Upload | Click **UPLOAD** tab → drag or browse any image |
| ESP32-CAM | Click **ESP32** tab → paste `http://<IP>/capture` |

---

## ⚛ Quantum Circuit

```
q0: ─[Ry(σ)]──■──────────[Rz]──M
q1: ─[Ry(τ)]──X──■───────[Rz]──M
q2: ─[H]─────────X──■──────────M
```
- **1024 shots** run on Qiskit Aer simulator
- Bit counts mapped → Path A (stable) / B (crack growth) / C (failure)
- Falls back to classical approximation if Qiskit unavailable

---

## Offline / Demo Mode

All buttons degrade gracefully if the backend is unreachable:
- **FULL DEMO** works 100% client-side — no backend needed
- Local JS physics simulation
- Classical quantum probability approximation

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Cannot connect to backend" | Make sure `python app.py` is running |
| Webcam permission denied | Allow camera in browser settings |
| `cv2` not found | Run `pip install opencv-python-headless` |
| Qiskit errors | Quantum uses classical fallback automatically |
| Port 5050 in use | Edit last line of `app.py`: change `port=5050` |
