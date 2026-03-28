"""
QuantumVision - app.py
Run:  python app.py
Open: http://localhost:5050
"""

import os, base64, random, math, traceback
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Absolute paths (works on Windows regardless of cwd) ──────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def serve_index():
    """Read and return index.html as a plain HTTP response."""
    path = os.path.join(TEMPLATES_DIR, 'index.html')
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}

def b64_to_cv2(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    arr = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def cv2_to_b64(img):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()

# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
@app.route('/index')
@app.route('/index.html')
def index():
    return serve_index()

@app.errorhandler(404)
def not_found(e):
    # Catch everything else and serve the SPA
    return serve_index()

# ─────────────────────────────────────────────────────────────────────────────
# API: /enhance
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        img      = b64_to_cv2(request.json['image'])
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        sharp    = cv2.filter2D(denoised, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
        lab      = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        l        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
        result   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return jsonify({'enhanced': cv2_to_b64(result), 'status': 'ok'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# API: /detect
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/detect', methods=['POST'])
def detect():
    try:
        img      = b64_to_cv2(request.json['image'])
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges    = cv2.Canny(blurred, 50, 150)
        dilated  = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

        overlay           = img.copy()
        overlay[dilated > 0] = [0, 0, 255]
        result            = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        contours, _  = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crack_ratio  = float(np.sum(dilated > 0)) / (gray.shape[0] * gray.shape[1])
        lap_var      = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        texture      = 'Rough' if lap_var > 500 else ('Moderate' if lap_var > 100 else 'Smooth')
        risk         = min(100, int(crack_ratio * 2000 + lap_var * 0.05))

        cv2.drawContours(result, contours[:20], -1, (0, 255, 255), 1)

        return jsonify({
            'detected':    cv2_to_b64(result),
            'cracks':      len(contours),
            'crack_ratio': round(crack_ratio * 100, 2),
            'texture':     texture,
            'risk_score':  risk,
            'laplacian':   round(lap_var, 1),
            'status':      'ok'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# API: /depth  (intensity + Sobel → 32×32 height mesh)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/depth', methods=['POST'])
def depth():
    try:
        img    = b64_to_cv2(request.json['image'])
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w   = gray.shape

        gray_f = gray.astype(np.float32) / 255.0
        sx     = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sy     = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gm     = np.sqrt(sx**2 + sy**2)
        gn     = (gm / gm.max()).astype(np.float32) if gm.max() > 0 else gm.astype(np.float32)
        hmap   = cv2.GaussianBlur(gray_f * 0.6 + gn * 0.4, (7, 7), 0)

        samp   = hmap[::max(1, h//32), ::max(1, w//32)][:32, :32]
        sh, sw = samp.shape

        verts, faces, colors = [], [], []
        for r in range(sh):
            for c in range(sw):
                x = (c / sw - 0.5) * 2
                z = (r / sh - 0.5) * 2
                y = float(samp[r, c]) * 0.8
                verts.append([round(x,4), round(y,4), round(z,4)])
                t = y / 0.8
                colors.append([
                    round(min(1.0, t * 2), 3),
                    round(min(1.0, 1 - abs(t - 0.5) * 2), 3),
                    round(max(0.0, 1 - t * 2), 3)
                ])

        for r in range(sh - 1):
            for c in range(sw - 1):
                i = r * sw + c
                faces.append([i, i+1, i+sw])
                faces.append([i+1, i+sw+1, i+sw])

        return jsonify({
            'vertices': verts, 'faces': faces, 'colors': colors,
            'width': sw, 'height': sh, 'status': 'ok'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# API: /simulate  (digital twin physics)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        verts = data['vertices']
        s  = float(data.get('stress',      50)) / 100.0
        t  = float(data.get('temperature', 50)) / 100.0
        tf = float(data.get('time',        50)) / 100.0

        deformed = []
        for v in verts:
            x, y, z = v
            x += x * t * 0.05
            z += z * t * 0.05
            y += s  * 0.15 * math.sin(x * math.pi) * math.cos(z * math.pi)
            y -= tf * s * 0.08 * (abs(x) + abs(z)) * 0.5
            y += random.gauss(0, s * tf * 0.02)
            deformed.append([round(x,4), round(y,4), round(z,4)])

        crack_growth = round(s * tf * 100, 1)
        integrity    = max(0.0, round(100 - crack_growth * 0.8 - t * 30, 1))

        return jsonify({
            'vertices':     deformed,
            'crack_growth': crack_growth,
            'integrity':    integrity,
            'status':       'ok'
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────────────────────────────────────────────────────
# API: /quantum  (Qiskit 3-qubit circuit, classical fallback)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/quantum', methods=['POST'])
def quantum():
    data        = request.json
    stress      = float(data.get('stress',      50)) / 100.0
    temperature = float(data.get('temperature', 50)) / 100.0

    def make_outcomes(pa, pb, pc, circuit_str):
        return jsonify({
            'outcomes': [
                {'label': 'Path A: Stable State',      'prob': round(pa*100,1), 'color': '#00ff88', 'description': 'Material remains structurally sound'},
                {'label': 'Path B: Crack Propagation',  'prob': round(pb*100,1), 'color': '#ffaa00', 'description': 'Cracks grow along grain boundaries'},
                {'label': 'Path C: Critical Failure',   'prob': round(pc*100,1), 'color': '#ff3366', 'description': 'Catastrophic structural breakdown'},
            ],
            'circuit': circuit_str,
            'qubits':  3,
            'shots':   1024,
            'status':  'ok'
        })

    try:
        from qiskit import QuantumCircuit, transpile as qk_transpile
        from qiskit_aer import AerSimulator

        ts = stress * math.pi
        tt = temperature * math.pi
        qc = QuantumCircuit(3, 3)
        qc.ry(ts, 0); qc.ry(tt, 1); qc.h(2)
        qc.cx(0, 1); qc.cx(1, 2); qc.cx(2, 0)
        qc.rz(ts * 0.5, 0); qc.rz(tt * 0.5, 1)
        qc.measure([0, 1, 2], [0, 1, 2])

        sim    = AerSimulator()
        counts = sim.run(qk_transpile(qc, sim), shots=1024).result().get_counts()
        total  = sum(counts.values())
        pa = sum(v for k, v in counts.items() if k.count('1') <= 1) / total
        pb = sum(v for k, v in counts.items() if k.count('1') == 2) / total
        pc = sum(v for k, v in counts.items() if k.count('1') == 3) / total
        return make_outcomes(pa, pb, pc, str(qc.draw()))

    except Exception as e:
        traceback.print_exc()
        # Classical fallback — never crashes the UI
        combined = (stress + temperature) / 2
        pa = max(0.05, 1 - combined * 1.2)
        pc = min(0.60, combined * 0.70)
        pb = max(0.05, 1 - pa - pc)
        return make_outcomes(pa, pb, pc, f'Classical fallback (reason: {e})')

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    idx = os.path.join(TEMPLATES_DIR, 'index.html')
    print()
    print('=' * 55)
    print('  QuantumVision  —  AI + Quantum Digital Twin Microscope')
    print('=' * 55)
    print(f'  Base dir    : {BASE_DIR}')
    print(f'  index.html  : {"FOUND ✓" if os.path.exists(idx) else "MISSING ✗ — check templates/"}')
    print(f'  URL         : http://localhost:5050')
    print('=' * 55)
    print()
    app.run(debug=True, port=5050, host='0.0.0.0')
