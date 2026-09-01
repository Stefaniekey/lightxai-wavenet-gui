"""
gui_streamlit.py - GUI LightXAI-WaveNet (Streamlit) — terhubung ke model asli
================================================================================
Jalankan:  streamlit run gui_streamlit.py

Dependencies:
    pip install streamlit torch opencv-python-headless pillow plotly pandas PyWavelets numpy

Struktur folder yang diharapkan:
    <project_root>/
      gui_streamlit.py                 <- file ini
      src/model_small.py               <- berisi class LightXAIWaveNetSmall
      experiments/lightxai_wavenet_5class_final/best_model.pth

Catatan v3 (perubahan dari versi sebelumnya):
1) HALAMAN DETEKSI kini murni cepat: hanya forward pass (tanpa backward/Grad-CAM),
   supaya tidak lagi menampilkan heatmap di sana. Semua visual Grad-CAM (overlay,
   heatmap murni, reticle area fokus, pemilihan kelas target) SEKARANG HANYA ada
   di menu "📊 Grad-CAM".
2) Ditambahkan fitur opsional "Koreksi Bias Kelas" (post-hoc prior/logit adjustment)
   di sidebar — mitigasi statistik untuk model yang cenderung bias ke satu kelas
   (mis. karena distribusi kelas training tidak seimbang). ITU BUKAN PENGGANTI
   perbaikan di level training (class-weighted loss, oversampling, dsb) — hanya
   penyesuaian probabilitas pasca-hoc yang bisa dicoba langsung dari GUI.
3) Grad-CAM tetap dihitung dari gradien nyata model (bukan sintetis), dengan:
     - target layer = Conv3d TERAKHIR (fallback Conv2d jika tak ada Conv3d)
     - heatmap dihaluskan (Gaussian blur) agar tidak blocky
     - region-of-interest pakai threshold ADAPTIF (percentile), bukan 0.5 tetap
     - bisa ditarget ke kelas mana pun, tidak melulu kelas prediksi
4) Subband wavelet memakai 3D-DWT SATU LEVEL (pywt.dwtn, sumbu Depth-Height-Width)
   — sesuai representasi input asli model 3D CNN. Label: LLL, LLH, LHL, LHH, HLL,
   HLH, HHL, HHH (huruf ke-1=Depth, ke-2=Height, ke-3=Width; L=Low, H=High).
5) Tampilan dirapikan: hero header bergradasi, sidebar nav bergaya pill, hover
   effect pada card, gauge confidence (Plotly), prob-bar dengan efek glow.

Catatan v4 (baru):
6) Ditambahkan menu "📐 Evaluasi IoU" — membandingkan anotasi ahli radiolog
   (ground truth, diinput manual sebagai simulasi karena aplikasi belum
   terhubung ke data anotasi sesungguhnya) dengan area fokus model (diestimasi
   dari Grad-CAM) memakai Intersection over Union (IoU) dan Dice Coefficient.
   Menampilkan interpretasi IoU, tabel perbandingan per nodul, serta ringkasan
   statistik IoU (mean, median, min-max, %IoU>=0.50, dice mean, distribusi).
   Halaman Deteksi, Grad-CAM, dan Subband DWT TIDAK diubah sama sekali.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import pywt

# ============================================================
# SETUP PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))
MODEL_PATH = os.path.join(BASE_DIR, "experiments", "lightxai_wavenet_5class_final", "best_model.pth")

# ============================================================
# KONFIGURASI TAMPILAN
# ============================================================
st.set_page_config(page_title="WAVENET-DX · Lung Nodule Reader", page_icon="🫁", layout="wide")

CLASS_NAMES  = ['C0 · Normal', 'C1 · Tiny Benign', 'C2 · Tiny Malignant', 'C3 · Large Benign', 'C4 · Large Malignant']
CLASS_COLORS = ['#3ddc84', '#4fc3f7', '#ffb74d', '#ff8a65', '#ef5350']
CLASS_EMOJI  = ['✅', '🟢', '🟠', '🔴', '🟣']

# Kategori interpretasi IoU (dipakai di halaman "📐 Evaluasi IoU")
IOU_CAT_COLORS = {
    "Sangat Baik": "#2e7d32",
    "Baik": "#66bb6a",
    "Cukup": "#ffb74d",
    "Kurang": "#ef5350",
}

# Metrik evaluasi statis (silakan sesuaikan dengan hasil training/eval Anda sendiri)
EVAL_METRICS = {
    "Accuracy": "47.8%", "Macro AUC": "0.797", "F1-Score": "0.259", "Total Sampel": "887",
}
AUC_PER_CLASS = {'C0': 0.7214, 'C1': 0.8085, 'C2': 0.7843, 'C3': 0.8333, 'C4': 0.8366}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --bg-app:#0a0e17; --bg-panel:#121826; --bg-panel-2:#0d1220; --bg-viewport:#04060b;
  --border:#212b3d; --text:#e7ecf5; --text-dim:#8b96ac; --text-faint:#57617a;
  --accent:#2dd4bf; --accent-dim:rgba(45,212,191,.14);
  --c0:#3ddc84; --c1:#4fc3f7; --c2:#ffb74d; --c3:#ff8a65; --c4:#ef5350;
}
html, body, .stApp { background:var(--bg-app) !important; color:var(--text) !important; font-family:'IBM Plex Sans',sans-serif !important; }
[data-testid="stSidebar"]{ background:var(--bg-panel-2) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--text) !important; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.01em; }

/* ---------- Hero header ---------- */
.hero{ padding:2px 2px 20px 2px; border-bottom:1px solid var(--border); margin-bottom:20px; }
.hero .eyebrow-top{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--accent); letter-spacing:.16em; text-transform:uppercase; margin-bottom:4px; }
.hero h1{ font-size:2.05rem; margin:0; font-weight:700;
  background:linear-gradient(90deg,#e7ecf5 10%, var(--accent) 60%, #4fc3f7 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent; display:inline-block; }
.hero p{ color:var(--text-dim); margin-top:6px; font-size:13.5px; max-width:640px; line-height:1.55; }

/* ---------- Sidebar nav pills ---------- */
[data-testid="stSidebar"] [data-testid="stRadio"] > div{ gap:5px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label{
  background:var(--bg-viewport) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; padding:9px 12px !important; width:100%;
  transition:.15s ease; cursor:pointer;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{ border-color:rgba(45,212,191,.5) !important; background:var(--accent-dim) !important; }

/* ---------- Buttons ---------- */
.stButton>button{
  background:var(--accent-dim) !important; color:var(--accent) !important;
  border:1px solid rgba(45,212,191,.35) !important; border-radius:9px !important; font-weight:600 !important;
  transition:.15s ease !important;
}
.stButton>button:hover{ background:var(--accent) !important; color:#04120f !important; box-shadow:0 4px 16px rgba(45,212,191,.25) !important; }

[data-testid="stFileUploaderDropzone"]{ background:var(--bg-viewport) !important; border:1.5px dashed #2a3548 !important; border-radius:12px !important; }
[data-testid="stVerticalBlockBorderWrapper"]{ background:linear-gradient(180deg,var(--bg-panel),var(--bg-panel-2)) !important; border:1px solid var(--border) !important; border-radius:14px !important; padding:4px; }
[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono',monospace !important; color:var(--text) !important; }

.badge{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--accent); background:var(--accent-dim); border:1px solid rgba(45,212,191,.3); padding:3px 9px; border-radius:20px; display:inline-block; }
.badge-err{ color:#ef5350; background:rgba(239,83,80,.12); border:1px solid rgba(239,83,80,.35); }
.eyebrow{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--text-faint); text-transform:uppercase; letter-spacing:.12em; margin-bottom:6px; }

.metric-card{ background:var(--bg-viewport); border:1px solid var(--border); border-radius:10px; padding:14px 12px; transition:.18s ease; }
.metric-card:hover{ transform:translateY(-2px); border-color:rgba(45,212,191,.4); box-shadow:0 8px 20px rgba(45,212,191,.08); }
.metric-card .val{ font-family:'IBM Plex Mono',monospace; font-size:24px; font-weight:600; }
.metric-card .label{ font-size:10.5px; color:var(--text-faint); text-transform:uppercase; letter-spacing:.06em; margin-top:3px; }

.pred-box{ position:relative; overflow:hidden; border-radius:12px; padding:18px; background:var(--bg-viewport); border:1px solid var(--border); }
.pred-box::before{ content:""; position:absolute; inset:0; background:radial-gradient(circle at top right, rgba(45,212,191,.10), transparent 60%); pointer-events:none; }
.pred-eyebrow{ font-family:'IBM Plex Mono',monospace; font-size:10px; color:var(--text-faint); text-transform:uppercase; letter-spacing:.12em; }
.pred-class{ font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700; margin-top:4px; position:relative; }
.pred-conf{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--text-dim); margin-top:3px; position:relative; }
.chip{ display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:10px; padding:3px 9px; border-radius:20px; margin-top:8px; margin-right:6px; font-weight:600; position:relative; }

.prob-row{ display:flex; align-items:center; gap:9px; margin-bottom:7px; }
.prob-row .lbl{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--text-dim); min-width:160px; }
.prob-track{ flex:1; height:14px; background:var(--bg-viewport); border-radius:7px; overflow:hidden; border:1px solid var(--border); }
.prob-fill{ height:100%; border-radius:7px; box-shadow:0 0 10px -2px currentColor; }
.prob-val{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; font-weight:600; min-width:48px; text-align:right; }

.readout{ font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--text-dim); background:var(--bg-viewport); border:1px solid var(--border); border-radius:9px; padding:11px 13px; line-height:1.85; }
.readout b{ color:var(--text); }
.readout .k{ color:var(--text-faint); }
.notice{ font-size:12px; color:var(--text-faint); background:var(--bg-viewport); border:1px dashed var(--border); border-radius:9px; padding:11px 13px; line-height:1.6; }
.notice b{ color:var(--accent); }
.warn{ font-size:12px; color:#ffb74d; background:rgba(255,183,77,.08); border:1px dashed rgba(255,183,77,.4); border-radius:9px; padding:11px 13px; line-height:1.6; }
.warn b{ color:#ffb74d; }

.sub-cap{ font-family:'IBM Plex Mono',monospace; font-size:10px; font-weight:600; color:var(--text-dim); text-align:center; margin-top:4px; text-transform:uppercase; letter-spacing:.06em;}
.sub-lv{ font-family:'IBM Plex Mono',monospace; font-size:8.5px; color:var(--text-faint); text-align:center; }

.cta{ display:flex; align-items:center; gap:10px; background:linear-gradient(90deg, rgba(45,212,191,.10), rgba(79,195,247,.06)); border:1px solid rgba(45,212,191,.3); border-radius:11px; padding:13px 15px; margin-top:14px; }
.cta .ico{ font-size:20px; }
.cta .txt b{ color:var(--accent); }
.cta .txt{ font-size:12.5px; color:var(--text-dim); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_header(icon, title, subtitle):
    st.markdown(f"""<div class="hero">
        <div class="eyebrow-top">WAVENET-DX</div>
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        from model_small import LightXAIWaveNetSmall
    except Exception as e:
        return None, None, False, f"Gagal import model_small.LightXAIWaveNetSmall dari src/: {e}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightXAIWaveNetSmall(num_classes=5).to(device)

    if not os.path.exists(MODEL_PATH):
        return None, device, False, f"File model tidak ditemukan: {MODEL_PATH}"
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, device, True, None
    except Exception as e:
        return None, device, False, f"Gagal memuat state_dict: {e}"


model, device, model_loaded, model_error = load_model()


# ============================================================
# GRAD-CAM (hook nyata pada layer konvolusi 3D terakhir)
# ============================================================
def get_last_conv_layer(m):
    """Prioritas Conv3d TERAKHIR (arsitektur adalah CNN 3D); fallback ke Conv2d
    terakhir hanya jika model tidak punya Conv3d sama sekali."""
    last_conv3d, last_conv2d = None, None
    for module in m.modules():
        if isinstance(module, torch.nn.Conv3d):
            last_conv3d = module
        elif isinstance(module, torch.nn.Conv2d):
            last_conv2d = module
    return last_conv3d if last_conv3d is not None else last_conv2d


class GradCAM:
    def __init__(self, target_model, target_layer):
        self.model = target_model
        self.activations = None
        self.gradients = None
        self.layer_type = type(target_layer).__name__
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]


@st.cache_resource
def build_gradcam(_model):
    if _model is None:
        return None
    layer = get_last_conv_layer(_model)
    if layer is None:
        return None
    return GradCAM(_model, layer)


gradcam_engine = build_gradcam(model) if model_loaded else None


# ============================================================
# PREPROCESSING
# ============================================================
VOLUME_DEPTH = 8  # jumlah frame pada sumbu depth (harus sama dgn yg dipakai training)


def build_volume_from_gray(gray128, depth=VOLUME_DEPTH):
    """Volume 3D (D,H,W) mentah dari citra 2D — dipakai bersama oleh preprocessing
    model (setelah dinormalisasi) DAN oleh dekomposisi 3D-DWT (skala asli 0-255)."""
    return np.stack([gray128.astype(np.float32)] * depth, axis=0)


def preprocess_tensor(gray128, depth=VOLUME_DEPTH):
    volume = build_volume_from_gray(gray128, depth=depth) / 255.0   # (D, H, W)
    volume = np.expand_dims(volume, axis=0)                          # (1, D, H, W)
    tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # (1, 1, D, H, W)
    return tensor


def predict_probs(gray128):
    """Forward pass SAJA (tanpa backward/Grad-CAM) — dipakai halaman Deteksi agar
    ringan & cepat. Grad-CAM baru dihitung terpisah di halaman Grad-CAM."""
    tensor = preprocess_tensor(gray128).to(device)
    with torch.no_grad():
        out = model(tensor)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    extra = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
    probs = torch.softmax(logits, dim=1)[0]
    pred_class = int(probs.argmax().item())
    confidence = float(probs[pred_class].item())
    return dict(pred_class=pred_class, confidence=confidence,
                probs=probs.detach().cpu().numpy().tolist(), extra=extra)


def run_inference_with_cam(gray128, target_class=None):
    """Forward + backward untuk Grad-CAM. Dipakai HANYA di halaman Grad-CAM.

    target_class : None -> Grad-CAM dihitung untuk kelas hasil prediksi model.
                   int  -> paksa Grad-CAM dihitung terhadap skor kelas tsb.
    """
    tensor = preprocess_tensor(gray128).to(device)

    if gradcam_engine is None:
        probe = predict_probs(gray128)
        cam = np.full((16, 16), 0.12, dtype=np.float32)
        return dict(**probe, cam=cam, cam_available=False, cam_target=probe["pred_class"])

    model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(True):
        out = model(tensor)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        extra = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[pred_class].item())

        cam_target = pred_class if target_class is None else int(target_class)
        score = logits[0, cam_target]
        score.backward()

    grads = gradcam_engine.gradients
    acts = gradcam_engine.activations
    cam_available = True
    if grads is None or acts is None:
        cam = np.full((16, 16), 0.12, dtype=np.float32)
        cam_available = False
    else:
        if grads.dim() == 5:      # (N, C, D, H, W)
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)
            cam_t = torch.relu((weights * acts).sum(dim=1, keepdim=True)).mean(dim=2)  # (N,1,H,W)
        elif grads.dim() == 4:    # (N, C, H, W)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam_t = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        else:
            cam_t = None

        if cam_t is None:
            cam = np.full((16, 16), 0.12, dtype=np.float32)
            cam_available = False
        else:
            cam_np = cam_t.squeeze().detach().cpu().numpy()
            if cam_np.ndim != 2:
                cam_np = np.atleast_2d(cam_np)
            cmin, cmax = cam_np.min(), cam_np.max()
            cam = (cam_np - cmin) / (cmax - cmin + 1e-8) if cmax > cmin else np.zeros_like(cam_np)

    return dict(pred_class=pred_class, confidence=confidence,
                probs=probs.detach().cpu().numpy().tolist(), extra=extra, cam=cam,
                cam_available=cam_available, cam_target=cam_target)


# ============================================================
# KOREKSI BIAS KELAS (post-hoc prior/logit adjustment) — OPSIONAL
# ============================================================
def apply_prior_correction(probs, priors):
    """Koreksi probabilitas dengan membagi terhadap estimasi prior kelas lalu
    menormalkan ulang (logit-adjustment / inverse-prior weighting).
    Ini TIDAK mengubah bobot model — hanya penyesuaian statistik pasca-hoc,
    berguna bila bias prediksi berasal dari distribusi kelas training yang
    tidak seimbang. Perbaikan sesungguhnya tetap perlu dilakukan saat training
    (class-weighted loss, oversampling minoritas, dsb)."""
    p = np.asarray(probs, dtype=np.float64)
    pr = np.clip(np.asarray(priors, dtype=np.float64), 1e-6, None)
    adjusted = p / pr
    adjusted = adjusted / adjusted.sum()
    return adjusted.tolist()


# ============================================================
# VISUALISASI: overlay heatmap, reticle, subband
# ============================================================
def make_overlay(gray_uint8, cam, size=224, smooth=True):
    base = cv2.resize(gray_uint8, (size, size), interpolation=cv2.INTER_CUBIC)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    cam_resized = cv2.resize(cam, (size, size), interpolation=cv2.INTER_CUBIC)

    if smooth:
        # Heatmap asli beresolusi kecil (mis. 16x16) sehingga setelah di-resize
        # terlihat "blocky". Gaussian blur menghaluskan transisi tanpa mengubah
        # lokasi puncak fokus model.
        k = max(3, (size // 28) | 1)
        cam_resized = cv2.GaussianBlur(cam_resized, (k, k), 0)
        cmin, cmax = cam_resized.min(), cam_resized.max()
        cam_resized = (cam_resized - cmin) / (cmax - cmin + 1e-8) if cmax > cmin else cam_resized

    cam_u8 = np.uint8(np.clip(cam_resized, 0, 1) * 255)
    heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat_rgb = heat_bgr[:, :, ::-1]
    overlay = cv2.addWeighted(base_rgb, 0.55, heat_rgb, 0.45, 0)
    return overlay, heat_rgb


def analyze_cam_region(cam, percentile=85, min_threshold=0.35):
    """Threshold ADAPTIF (percentile) alih-alih ambang tetap 0.5, agar deteksi
    area fokus tetap konsisten meski kontras heatmap berbeda-beda antar kasus."""
    h, w = cam.shape
    peak_idx = np.unravel_index(np.argmax(cam), cam.shape)
    py, px = peak_idx
    thresh = max(min_threshold, float(np.percentile(cam, percentile)))
    mask = cam >= thresh
    area = int(mask.sum())
    if area > 0:
        ys, xs = np.where(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        diameter = 2 * np.sqrt(area / np.pi)
    else:
        cy, cx, diameter = float(py), float(px), 0.0
    return dict(cx=cx, cy=cy, diameter=diameter, area=area, peak=float(cam[py, px]),
                h=h, w=w, threshold=thresh)


def draw_reticle(display_rgb, region, color=(45, 212, 191)):
    img = Image.fromarray(display_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    S = img.size[0]
    cx_px = region["cx"] / region["w"] * S
    cy_px = region["cy"] / region["h"] * S
    r_px = max(10, region["diameter"] / region["w"] * S / 2)
    draw.ellipse([cx_px - r_px, cy_px - r_px, cx_px + r_px, cy_px + r_px], outline=color, width=2)
    draw.line([cx_px, cy_px - r_px - 9, cx_px, cy_px - r_px + 3], fill=color, width=1)
    draw.line([cx_px - r_px - 9, cy_px, cx_px - r_px + 3, cy_px], fill=color, width=1)
    return np.array(img)


def diverging_colormap(arr):
    max_abs = np.max(np.abs(arr)) + 1e-6
    norm = np.clip(arr / max_abs, -1, 1)
    r = np.where(norm >= 0, 255, np.round(255 * (1 + norm)))
    g = np.where(norm >= 0, np.round(255 * (1 - norm)), np.round(255 * (1 + norm)))
    b = np.where(norm >= 0, np.round(255 * (1 - norm)), 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def gray_colormap(arr):
    mn, mx = arr.min(), arr.max()
    norm = (arr - mn) / (mx - mn + 1e-6)
    v = (norm * 255).astype(np.uint8)
    return np.stack([v, v, v], axis=-1)


# ------------------------------------------------------------
# Subband 3D-DWT (1 level) — LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
# ------------------------------------------------------------
SUBBAND_ORDER = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]

SUBBAND_LABELS_3D = {
    "LLL": "Aproksimasi penuh (D-Low, H-Low, W-Low)",
    "LLH": "Detail sumbu Width (D-Low, H-Low, W-High)",
    "LHL": "Detail sumbu Height (D-Low, H-High, W-Low)",
    "LHH": "Detail Height+Width (D-Low, H-High, W-High)",
    "HLL": "Detail sumbu Depth (D-High, H-Low, W-Low)",
    "HLH": "Detail Depth+Width (D-High, H-Low, W-High)",
    "HHL": "Detail Depth+Height (D-High, H-High, W-Low)",
    "HHH": "Detail frekuensi tinggi penuh (D-High, H-High, W-High)",
}


def compute_subbands_3d(gray128, wavelet="db4", depth=VOLUME_DEPTH):
    """Dekomposisi 3D-DWT SATU LEVEL pada volume (D,H,W) — konsisten dengan
    representasi input yang benar-benar diproses model. Menghasilkan 8 subband:
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH (huruf ke-1=Depth, ke-2=Height,
    ke-3=Width; L=Low/aproksimasi, H=High/detail)."""
    volume = build_volume_from_gray(gray128, depth=depth)  # skala asli 0..255
    coeffs = pywt.dwtn(volume, wavelet, axes=(0, 1, 2))
    mapping = {"a": "L", "d": "H"}
    decoded = {}
    for key, arr in coeffs.items():
        code = "".join(mapping[c] for c in key)  # 'aad' -> 'LLH'
        decoded[code] = arr
    return {code: decoded[code] for code in SUBBAND_ORDER}


def subband_to_image(arr3d):
    """Proyeksi rata-rata sepanjang sumbu Depth agar subband 3D bisa ditampilkan
    sebagai gambar 2D (Height x Width)."""
    return arr3d.mean(axis=0)


def generate_demo_image(seed=None):
    rng = np.random.default_rng(seed)
    size = 128
    base = rng.normal(190, 10, (size, size))
    for _ in range(40):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(2, 6)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        base[mask] -= rng.uniform(5, 20)
    demo_type = int(rng.integers(0, 5))
    if demo_type != 0:
        cx = int(rng.integers(45, 83))
        cy = int(rng.integers(45, 83))
        radius = int(rng.integers(8, 16)) if demo_type in (1, 2) else int(rng.integers(20, 32))
        intensity = rng.uniform(40, 90)
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        blob = (np.clip(1 - dist / radius, 0, 1) ** 1.5) * intensity
        base -= blob
        if demo_type in (2, 4):
            ring = (dist > radius * 0.9) & (dist < radius * 1.4)
            base[ring] -= rng.uniform(0, 15, size=int(ring.sum()))
    return np.clip(base, 0, 255).astype(np.uint8)


def render_extra_maps(extra):
    try:
        if extra is None:
            st.caption("Model tidak mengembalikan output kedua (mis. koefisien wavelet internal).")
            return
        tensors = extra if isinstance(extra, (list, tuple)) else [extra]
        shown = False
        for idx, t in enumerate(tensors):
            if not torch.is_tensor(t):
                continue
            t = t.detach().cpu()
            if t.dim() == 5:
                t = t[0].mean(dim=1)
            elif t.dim() == 4:
                t = t[0]
            elif t.dim() != 3:
                continue
            n_ch = min(8, t.shape[0])
            cols = st.columns(4)
            for c in range(n_ch):
                arr = t[c].numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                img = (arr * 255).astype(np.uint8)
                with cols[c % 4]:
                    st.image(img, caption=f"internal[{idx}] ch{c}", use_container_width=True)
                    shown = True
        if not shown:
            st.caption("Output kedua model tidak dalam format tensor citra yang bisa ditampilkan otomatis.")
    except Exception as e:
        st.caption(f"Tidak dapat menampilkan output internal model: {e}")


def prob_bars_html(probs):
    rows = ""
    for i, name in enumerate(CLASS_NAMES):
        rows += f"""<div class="prob-row">
            <span class="lbl">{name}</span>
            <div class="prob-track"><div class="prob-fill" style="width:{probs[i]*100:.1f}%;background:{CLASS_COLORS[i]};color:{CLASS_COLORS[i]};"></div></div>
            <span class="prob-val" style="color:{CLASS_COLORS[i]}">{probs[i]*100:.1f}%</span>
        </div>"""
    return rows


def confidence_gauge(confidence, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={'suffix': '%', 'font': {'size': 26, 'color': '#e7ecf5', 'family': 'IBM Plex Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#57617a', 'tickfont': {'color': '#57617a', 'size': 9}},
            'bar': {'color': color, 'thickness': 0.28},
            'bgcolor': '#04060b',
            'borderwidth': 1,
            'bordercolor': '#212b3d',
            'steps': [
                {'range': [0, 50], 'color': '#121826'},
                {'range': [50, 80], 'color': '#151d2d'},
                {'range': [80, 100], 'color': '#182236'},
            ],
        },
    ))
    fig.update_layout(height=150, margin=dict(l=14, r=14, t=6, b=6),
                       paper_bgcolor='rgba(0,0,0,0)', font_color='#e7ecf5')
    return fig


def clear_cached_results():
    """Hapus semua hasil ter-cache — dipanggil setiap kali citra input baru dimuat."""
    for k in list(st.session_state.keys()):
        if k == "probe" or k.startswith("cam_result_"):
            del st.session_state[k]


# ============================================================
# SIDEBAR (rail nav + status + koreksi bias kelas)
# ============================================================
with st.sidebar:
    st.markdown("### 🫁 WAVENET-DX")
    st.markdown('<span class="badge">Lung Nodule Reader</span>', unsafe_allow_html=True)
    st.markdown("---")
    if model_loaded:
        st.markdown('<span class="badge">✅ Model dimuat</span>', unsafe_allow_html=True)
        st.caption(f"Device: `{device}`")
        if gradcam_engine is not None:
            st.caption(f"Grad-CAM layer: `{gradcam_engine.layer_type}`")
        else:
            st.caption("Grad-CAM: layer konvolusi tidak terdeteksi.")
    else:
        st.markdown('<span class="badge badge-err">❌ Model belum dimuat</span>', unsafe_allow_html=True)
        st.caption(model_error or "Periksa MODEL_PATH.")
    st.markdown("---")
    for k, v in EVAL_METRICS.items():
        st.markdown(f"**{k}:** {v}")
    st.markdown("---")
    page = st.radio("Navigasi", ["🏠 Dashboard", "🔬 Deteksi Nodul", "📊 Grad-CAM", "🌊 Subband DWT", "📐 Evaluasi IoU"])
    st.markdown("---")

    with st.expander("⚖️ Koreksi Bias Kelas (opsional)"):
        st.caption(
            "Jika model kamu cenderung terlalu sering memprediksi kelas tertentu "
            "(mis. malignant), ini kemungkinan besar karena distribusi kelas saat "
            "training tidak seimbang. Toggle ini menerapkan **koreksi prior pasca-hoc** "
            "— bukan pengganti perbaikan di training (class-weighted loss / oversampling)."
        )
        correction_enabled = st.checkbox("Aktifkan koreksi", value=False, key="corr_enabled")
        class_priors = [1.0, 1.0, 1.0, 1.0, 1.0]
        if correction_enabled:
            st.caption("Estimasi proporsi tiap kelas di data TRAINING (dinormalisasi otomatis):")
            default_priors = [0.20, 0.20, 0.20, 0.20, 0.20]
            class_priors = []
            for i, name in enumerate(CLASS_NAMES):
                v = st.slider(name, 0.01, 1.00, default_priors[i], 0.01, key=f"prior_{i}")
                class_priors.append(v)

CORRECTION_ENABLED = correction_enabled
CLASS_PRIORS = class_priors


# ============================================================
# PAGE: DASHBOARD
# ============================================================
def page_dashboard():
    page_header("🏠", "Dashboard", "Ringkasan performa model WAVENET-DX pada data evaluasi.")

    cols = st.columns(4)
    for col, (label, value) in zip(cols, EVAL_METRICS.items()):
        with col:
            st.markdown(f"""<div class="metric-card"><div class="val">{value}</div><div class="label">{label}</div></div>""",
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="eyebrow">AUC per Kelas</div>', unsafe_allow_html=True)
        df = pd.DataFrame({"Class": list(AUC_PER_CLASS.keys()), "AUC": list(AUC_PER_CLASS.values())})
        fig = px.bar(df, x="Class", y="AUC", color="Class", color_discrete_sequence=CLASS_COLORS, range_y=[0, 1])
        fig.update_layout(showlegend=False, height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e7ecf5")
        st.plotly_chart(fig, use_container_width=True)

    if not model_loaded:
        st.markdown(f"""<div class="notice"><b>Model belum aktif.</b> {model_error}<br>
        Perbaiki path/arsitektur lalu reload halaman ini agar halaman Deteksi &amp; Grad-CAM aktif.</div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown("""<div class="warn" style="margin-top:14px;">
            <b>Catatan interpretasi metrik:</b> Macro AUC yang cukup tinggi (~0.80) namun Accuracy
            &amp; F1-Score yang rendah biasanya menandakan model punya kemampuan ranking yang oke per
            kelas, tapi keputusan akhirnya bias/tidak terkalibrasi — sering disebabkan distribusi kelas
            training yang tidak seimbang. Cek confusion matrix pada data validasi Anda untuk memastikan
            kelas mana yang paling sering "ditarik" ke malignant.
        </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE: DETEKSI  (murni klasifikasi — TANPA Grad-CAM)
# ============================================================
def page_deteksi():
    page_header("🔬", "Deteksi Nodul Paru-Paru",
                "Upload citra CT scan (crop/slice) untuk klasifikasi 5 kelas (C0–C4). "
                "Halaman ini hanya menjalankan forward pass — untuk melihat area fokus model, buka menu Grad-CAM.")

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.container(border=True):
            uploaded = st.file_uploader("Pilih citra CT scan", type=["jpg", "jpeg", "png"])
            if st.button("🎯 Gunakan Citra Demo Sintetik"):
                st.session_state["gray128"] = generate_demo_image()
                st.session_state["file_name"] = "demo_sintetik.png"
                clear_cached_results()

            if uploaded is not None:
                pil_img = Image.open(uploaded).convert("L")
                st.session_state["gray128"] = np.array(pil_img.resize((128, 128), Image.LANCZOS))
                st.session_state["file_name"] = uploaded.name
                clear_cached_results()

            if "gray128" in st.session_state:
                st.image(st.session_state["gray128"], caption=st.session_state.get("file_name", "citra"), width=260)
            else:
                st.markdown('<div class="notice">Belum ada citra dimuat. Upload citra atau gunakan citra demo.</div>',
                            unsafe_allow_html=True)

    with col2:
        if "gray128" not in st.session_state:
            st.markdown('<div class="notice">🖼️ Menunggu citra input.</div>', unsafe_allow_html=True)
            return

        if not model_loaded:
            st.markdown(f"""<div class="notice"><b>Model tidak dimuat</b> — deteksi nonaktif.<br>{model_error}</div>""",
                        unsafe_allow_html=True)
            return

        if "probe" not in st.session_state:
            with st.spinner("🔍 Menjalankan inferensi model..."):
                st.session_state["probe"] = predict_probs(st.session_state["gray128"])
        probe = st.session_state["probe"]

        raw_pred = probe["pred_class"]
        display_probs = probe["probs"]
        display_pred = raw_pred

        if CORRECTION_ENABLED:
            display_probs = apply_prior_correction(probe["probs"], CLASS_PRIORS)
            display_pred = int(np.argmax(display_probs))

        color = CLASS_COLORS[display_pred]
        confidence = display_probs[display_pred]

        correction_note = ""
        if CORRECTION_ENABLED and display_pred != raw_pred:
            correction_note = (
                f'<span class="chip" style="background:#57617a22;color:#8b96ac;border:1px solid #57617a55;">'
                f'sebelum koreksi: {CLASS_NAMES[raw_pred]}</span>'
            )
        elif CORRECTION_ENABLED:
            correction_note = (
                '<span class="chip" style="background:var(--accent-dim);color:var(--accent);'
                'border:1px solid rgba(45,212,191,.4);">koreksi prior aktif</span>'
            )

        gc1, gc2 = st.columns([1.3, 1])
        with gc1:
            st.markdown(f"""
            <div class="pred-box" style="border-left:3px solid {color};">
                <div class="pred-eyebrow">Prediksi</div>
                <div class="pred-class">{CLASS_EMOJI[display_pred]} {CLASS_NAMES[display_pred]}</div>
                <div class="pred-conf">confidence {confidence*100:.1f}%</div>
                {correction_note}
            </div>
            """, unsafe_allow_html=True)
        with gc2:
            st.plotly_chart(confidence_gauge(confidence, color), use_container_width=True)

        st.markdown(
            f'<div class="eyebrow" style="margin-top:10px;">Probabilitas per Kelas'
            f'{" (setelah koreksi prior)" if CORRECTION_ENABLED else ""}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(prob_bars_html(display_probs), unsafe_allow_html=True)

        cam_hint = "🟢 tersedia" if gradcam_engine is not None else "⚪ tidak tersedia (layer conv tak terdeteksi)"
        st.markdown(f"""<div class="cta">
            <div class="ico">📊</div>
            <div class="txt">Grad-CAM {cam_hint}. Buka menu <b>"📊 Grad-CAM"</b> di sidebar untuk melihat
            area citra yang menjadi fokus perhatian model.</div>
        </div>""", unsafe_allow_html=True)

        with st.expander("🔎 Output kedua model (opsional)"):
            render_extra_maps(probe["extra"])


# ============================================================
# PAGE: GRAD-CAM  (satu-satunya tempat heatmap ditampilkan)
# ============================================================
def page_gradcam():
    page_header("📊", "Grad-CAM Visualization",
                "Heatmap dihitung dari gradien nyata model pada layer konvolusi 3D terakhir, "
                "lalu dihaluskan (Gaussian blur) agar tidak blocky.")

    if "gray128" not in st.session_state:
        st.markdown('<div class="notice">Upload atau muat citra demo pada halaman "🔬 Deteksi Nodul" terlebih dahulu.</div>',
                    unsafe_allow_html=True)
        return
    if not model_loaded:
        st.markdown(f"""<div class="notice"><b>Model tidak dimuat</b> — Grad-CAM nonaktif.<br>{model_error}</div>""",
                    unsafe_allow_html=True)
        return

    # Tentukan default target kelas: kalau koreksi prior aktif, pakai kelas hasil
    # koreksi sebagai default supaya Grad-CAM konsisten dengan apa yang ditampilkan
    # di halaman Deteksi.
    probe = st.session_state.get("probe") or predict_probs(st.session_state["gray128"])
    st.session_state["probe"] = probe
    default_pred = probe["pred_class"]
    if CORRECTION_ENABLED:
        corrected = apply_prior_correction(probe["probs"], CLASS_PRIORS)
        default_pred = int(np.argmax(corrected))

    c_sel1, c_sel2 = st.columns([1.4, 1])
    with c_sel1:
        target_option = st.selectbox(
            "🎯 Target kelas Grad-CAM",
            ["Otomatis (kelas prediksi)"] + CLASS_NAMES,
            help="Otomatis = kelas hasil prediksi (memperhitungkan koreksi bias jika aktif). "
                 "Pilih kelas lain untuk eksplorasi: 'seandainya model memilih kelas ini, area mana yang dilihat?'",
        )
    with c_sel2:
        smooth = st.checkbox("Haluskan heatmap", value=True,
                              help="Gaussian blur agar heatmap tidak terlihat blocky.")

    target_class = default_pred if target_option.startswith("Otomatis") else CLASS_NAMES.index(target_option)

    cache_key = f"cam_result_{target_class}"
    if cache_key not in st.session_state:
        with st.spinner("🧠 Menghitung Grad-CAM (forward + backward)..."):
            st.session_state[cache_key] = run_inference_with_cam(st.session_state["gray128"], target_class=target_class)
    result = st.session_state[cache_key]

    pred = result["pred_class"]
    color = CLASS_COLORS[pred]
    cam_target = result["cam_target"]

    chip_cam = ('<span class="chip" style="background:'+color+'22;color:'+color+';border:1px solid '+color+'55;">GRAD-CAM AKTIF</span>'
                if result['cam_available'] else
                '<span class="chip" style="background:#57617a22;color:#8b96ac;border:1px solid #57617a55;">GRAD-CAM TIDAK TERSEDIA</span>')
    chip_target = (f'<span class="chip" style="background:#57617a22;color:#8b96ac;border:1px solid #57617a55;">'
                    f'ditarget ke: {CLASS_NAMES[cam_target]}</span>' if cam_target != pred else "")

    st.markdown(f"""
    <div class="pred-box" style="border-left:3px solid {color}; margin-bottom:16px;">
        <div class="pred-eyebrow">Kelas Prediksi Model</div>
        <div class="pred-class">{CLASS_EMOJI[pred]} {CLASS_NAMES[pred]}</div>
        <div class="pred-conf">confidence {result['confidence']*100:.1f}%</div>
        {chip_cam}{chip_target}
    </div>
    """, unsafe_allow_html=True)

    overlay, heat = make_overlay(st.session_state["gray128"], result["cam"], size=256, smooth=smooth)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sub-cap">Original + Overlay</div>', unsafe_allow_html=True)
        if result["cam_available"]:
            region = analyze_cam_region(result["cam"])
            overlay = draw_reticle(overlay, region)
        st.image(overlay, use_container_width=True)
    with c2:
        st.markdown('<div class="sub-cap">Heatmap Murni</div>', unsafe_allow_html=True)
        st.image(heat, use_container_width=True)

    if result["cam_available"]:
        region = analyze_cam_region(result["cam"])
        st.markdown(f"""<div class="readout" style="margin-top:12px;">
            <span class="k">peak_relatif</span> <b>x={region['cx']/region['w']*100:.0f}% y={region['cy']/region['h']*100:.0f}%</b><br>
            <span class="k">area_fokus_px</span> <b>{region['area']} px @ {region['w']}×{region['h']} (threshold adaptif {region['threshold']*100:.0f}%)</b><br>
            <span class="k">intensitas_puncak</span> <b>{region['peak']*100:.1f}%</b><br>
            <span class="k">kelas_target_cam</span> <b>{CLASS_NAMES[cam_target]}</b>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="notice">Grad-CAM tidak tersedia — layer konvolusi tidak terdeteksi otomatis di model Anda.</div>',
                    unsafe_allow_html=True)

    with st.expander("🔎 Output kedua model (opsional)"):
        render_extra_maps(result["extra"])


# ============================================================
# PAGE: SUBBAND
# ============================================================
def page_subband():
    page_header("🌊", "8 Subband — 3D-DWT (db4), 1 Level",
                "Dekomposisi wavelet 3D satu level pada volume (Depth×Height×Width) — identik dengan "
                "representasi input yang benar-benar diproses model.")

    st.markdown(
        "<div class=\"notice\">Label subband: <b>LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH</b> "
        "(huruf ke-1 = Depth, ke-2 = Height, ke-3 = Width; L = Low/aproksimasi, H = High/detail).</div>",
        unsafe_allow_html=True,
    )

    if "gray128" not in st.session_state:
        st.markdown('<div class="notice" style="margin-top:10px;">Upload atau muat citra demo pada halaman "🔬 Deteksi Nodul" terlebih dahulu.</div>',
                    unsafe_allow_html=True)
        return

    st.markdown("<br>", unsafe_allow_html=True)
    subbands = compute_subbands_3d(st.session_state["gray128"])
    cols = st.columns(4)
    for i, code in enumerate(SUBBAND_ORDER):
        arr2d = subband_to_image(subbands[code])
        is_detail = code != "LLL"
        rgb = diverging_colormap(arr2d) if is_detail else gray_colormap(arr2d)
        img = Image.fromarray(rgb).resize((160, 160), Image.NEAREST)
        with cols[i % 4]:
            st.image(img, use_container_width=True)
            st.markdown(
                f'<div class="sub-cap">{code}</div>'
                f'<div class="sub-lv">{SUBBAND_LABELS_3D[code]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(f"""<div class="readout" style="margin-top:14px;">
        <span class="k">LLL</span> = struktur/aproksimasi utama volume &nbsp;·&nbsp;
        subband lain = komponen detail/tepi pada kombinasi sumbu terkait (D/H/W).<br>
        Colormap divergen: biru = koefisien negatif, merah = positif, putih ≈ nol. Panel LLL memakai skala abu-abu.<br>
        <span class="k">Catatan</span> volume 3D dibentuk dari citra 2D yang di-stack {VOLUME_DEPTH}× pada sumbu depth,
        identik dengan preprocessing input model — sehingga subband pada sumbu Depth (huruf pertama kode)
        merefleksikan replikasi tersebut, bukan variasi CT riil antar slice.
    </div>""", unsafe_allow_html=True)


# ============================================================
# PAGE: EVALUASI IoU (Radiolog vs Model) — FITUR BARU
# ============================================================
# Ground truth (anotasi ahli radiolog) belum tersedia sebagai data nyata di
# aplikasi ini, sehingga diinput manual (slider) sebagai simulasi/evaluasi.
# Prediksi model (B) diestimasi otomatis dari area fokus Grad-CAM pada kelas
# hasil prediksi model — ini adalah proxy region-of-interest, BUKAN output
# segmentasi langsung dari model (karena model ini adalah classifier 5-kelas,
# bukan segmentation network). Nilai IoU/Dice di halaman ini bersifat
# indikatif untuk kebutuhan demonstrasi & evaluasi kualitatif.

IOU_TABLE_CSS = """
<style>
.iou-table{ width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace; font-size:12.5px; margin-top:6px;}
.iou-table th{ background:var(--bg-viewport); color:var(--text-dim); text-transform:uppercase; font-size:10px; letter-spacing:.06em; padding:9px 10px; border-bottom:1px solid var(--border); text-align:left;}
.iou-table td{ padding:9px 10px; border-bottom:1px solid var(--border); color:var(--text);}
.iou-table tr:hover td{ background:rgba(45,212,191,.05); }
.iou-mean-row td{ background:var(--bg-viewport); font-weight:700; border-top:1px solid var(--border);}
.iou-dot{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; vertical-align:middle;}
</style>
"""


def iou_category(iou):
    """Kategori interpretasi IoU sesuai skema penilaian dosen."""
    if iou >= 0.75:
        label = "Sangat Baik"
    elif iou >= 0.50:
        label = "Baik"
    elif iou >= 0.25:
        label = "Cukup"
    else:
        label = "Kurang"
    return label, IOU_CAT_COLORS[label]


def iou_keterangan(kategori):
    mapping = {
        "Sangat Baik": "Sangat sesuai",
        "Baik": "Sesuai",
        "Cukup": "Kurang sesuai",
        "Kurang": "Perlu perbaikan",
    }
    return mapping.get(kategori, "-")


def circle_mask(size, cx, cy, diam):
    """Mask boolean lingkaran pada grid size x size. cx, cy, diam dalam % lebar citra."""
    h = w = size
    cx_px = cx / 100.0 * w
    cy_px = cy / 100.0 * h
    r_px = max(1.0, (diam / 100.0 * w) / 2.0)
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx_px) ** 2 + (yy - cy_px) ** 2 <= r_px ** 2
    return mask


def compute_iou_dice(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    area_a, area_b = mask_a.sum(), mask_b.sum()
    iou = float(inter) / float(union) if union > 0 else 0.0
    dice = (2.0 * float(inter)) / float(area_a + area_b) if (area_a + area_b) > 0 else 0.0
    return float(iou), float(dice), int(inter), int(union)


def _contour_of_mask(mask_u8):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def draw_dashed_contour(img_rgb, contour, color, dash_len=7, gap_len=5, thickness=1):
    if contour is None:
        return img_rgb
    pts = contour.reshape(-1, 2)
    n = len(pts)
    draw_seg = True
    acc = 0.0
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        seg_len = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if draw_seg:
            cv2.line(img_rgb, pt1, pt2, color, thickness, cv2.LINE_AA)
        acc += seg_len
        if acc >= (dash_len if draw_seg else gap_len):
            draw_seg = not draw_seg
            acc = 0.0
    return img_rgb


def build_iou_visual(gray_uint8, gt, pred, size=260):
    """gt & pred: dict(cx=%, cy=%, diam=%). Mengembalikan citra GT-only,
    Prediksi-only, Overlap, serta mask-mask booleannya untuk perhitungan IoU."""
    base = cv2.resize(gray_uint8, (size, size), interpolation=cv2.INTER_CUBIC)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    mask_gt = circle_mask(size, gt["cx"], gt["cy"], gt["diam"])
    mask_pred = circle_mask(size, pred["cx"], pred["cy"], pred["diam"])
    mask_inter = np.logical_and(mask_gt, mask_pred)
    mask_union = np.logical_or(mask_gt, mask_pred)

    c_gt = _contour_of_mask((mask_gt * 255).astype(np.uint8))
    c_pred = _contour_of_mask((mask_pred * 255).astype(np.uint8))
    c_union = _contour_of_mask((mask_union * 255).astype(np.uint8))

    GREEN = (61, 219, 105)   # Ground Truth (Ahli Radiolog)
    RED = (239, 83, 80)      # Prediksi Model
    YELLOW = (255, 214, 51)  # Intersection
    WHITE = (255, 255, 255)  # Union (dashed)

    img_gt = base_rgb.copy()
    if c_gt is not None:
        cv2.drawContours(img_gt, [c_gt], -1, GREEN, 2, cv2.LINE_AA)

    img_pred = base_rgb.copy()
    if c_pred is not None:
        cv2.drawContours(img_pred, [c_pred], -1, RED, 2, cv2.LINE_AA)

    img_overlap = base_rgb.copy()
    if mask_inter.any():
        yellow_layer = np.zeros_like(img_overlap)
        yellow_layer[:, :] = YELLOW
        blended = cv2.addWeighted(img_overlap, 0.30, yellow_layer, 0.70, 0)
        img_overlap = np.where(mask_inter[..., None], blended, img_overlap).astype(np.uint8)
    if c_gt is not None:
        cv2.drawContours(img_overlap, [c_gt], -1, GREEN, 2, cv2.LINE_AA)
    if c_pred is not None:
        cv2.drawContours(img_overlap, [c_pred], -1, RED, 2, cv2.LINE_AA)
    if c_union is not None:
        draw_dashed_contour(img_overlap, c_union, WHITE, dash_len=6, gap_len=4, thickness=1)

    return dict(img_gt=img_gt, img_pred=img_pred, img_overlap=img_overlap,
                mask_gt=mask_gt, mask_pred=mask_pred, mask_inter=mask_inter, mask_union=mask_union)


def iou_table_html(records):
    rows = ""
    for r in records:
        color = IOU_CAT_COLORS.get(r["kategori"], "#8b96ac")
        rows += f"""<tr>
            <td>{r['case_id']}</td>
            <td>{r['nodule_id']}</td>
            <td>{r['diameter']:.1f}</td>
            <td style="color:{color};font-weight:700;">{r['iou']:.2f}</td>
            <td><span class="iou-dot" style="background:{color};"></span><span style="color:{color};font-weight:600;">{r['kategori']}</span></td>
            <td>{r['dice']:.2f}</td>
            <td style="color:var(--text-dim);">{r['keterangan']}</td>
        </tr>"""

    mean_row = ""
    if records:
        ious = [r["iou"] for r in records]
        dices = [r["dice"] for r in records]
        mean_row = f"""<tr class="iou-mean-row">
            <td colspan="3" style="text-align:right;">Rata-rata (Mean)</td>
            <td>{np.mean(ious):.2f}</td>
            <td></td>
            <td>{np.mean(dices):.2f}</td>
            <td></td>
        </tr>"""

    return f"""
    <table class="iou-table">
        <thead><tr>
            <th>Case ID</th><th>Nodule ID</th><th>Diameter (mm)</th><th>IoU</th><th>Kategori</th><th>Dice Coefficient</th><th>Keterangan</th>
        </tr></thead>
        <tbody>{rows}{mean_row}</tbody>
    </table>
    """


def render_iou_summary_stats(records):
    ious = np.array([r["iou"] for r in records], dtype=float)
    dices = np.array([r["dice"] for r in records], dtype=float)

    mean_iou, std_iou = float(ious.mean()), float(ious.std())
    median_iou = float(np.median(ious))
    min_iou, max_iou = float(ious.min()), float(ious.max())
    pct_good = float((ious >= 0.50).mean() * 100)
    mean_dice, std_dice = float(dices.mean()), float(dices.std())

    cards = [
        ("Mean IoU", f"{mean_iou:.2f} ± {std_iou:.2f}"),
        ("Median IoU", f"{median_iou:.2f}"),
        ("Min – Max IoU", f"{min_iou:.2f} – {max_iou:.2f}"),
        ("IoU ≥ 0.50", f"{pct_good:.0f}%"),
        ("Dice (Mean)", f"{mean_dice:.2f} ± {std_dice:.2f}"),
    ]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="metric-card"><div class="val" style="font-size:18px;">{value}</div><div class="label">{label}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Distribusi IoU</div>', unsafe_allow_html=True)
        buckets = ["0 – 0.25", "0.25 – 0.50", "0.50 – 0.75", "0.75 – 1.00"]
        colors = ["#ef5350", "#ffb74d", "#66bb6a", "#2e7d32"]
        counts = [0, 0, 0, 0]
        for v in ious:
            if v < 0.25:
                counts[0] += 1
            elif v < 0.50:
                counts[1] += 1
            elif v < 0.75:
                counts[2] += 1
            else:
                counts[3] += 1
        fig = go.Figure(go.Bar(x=buckets, y=counts, marker_color=colors, text=counts, textposition="outside"))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e7ecf5",
                           yaxis=dict(title="Jumlah Nodul", gridcolor="#212b3d", zeroline=False),
                           xaxis=dict(title="IoU Range"))
        st.plotly_chart(fig, use_container_width=True)


def page_iou():
    page_header("📐", "Evaluasi IoU — Radiolog vs Model",
                "Membandingkan anotasi ahli radiolog (ground truth) dengan area fokus model "
                "(diestimasi dari Grad-CAM) memakai Intersection over Union (IoU) dan Dice Coefficient.")

    st.markdown(IOU_TABLE_CSS, unsafe_allow_html=True)
    st.session_state.setdefault("iou_records", [])

    if "gray128" not in st.session_state:
        st.markdown('<div class="notice">Upload atau muat citra demo pada halaman "🔬 Deteksi Nodul" terlebih dahulu.</div>',
                    unsafe_allow_html=True)
        return
    if not model_loaded:
        st.markdown(f"""<div class="notice"><b>Model tidak dimuat</b> — evaluasi IoU nonaktif.<br>{model_error}</div>""",
                    unsafe_allow_html=True)
        return

    gray128 = st.session_state["gray128"]

    # -------------------- 1) Ringkasan citra & prediksi --------------------
    probe = st.session_state.get("probe") or predict_probs(gray128)
    st.session_state["probe"] = probe
    pred_class = probe["pred_class"]

    with st.container(border=True):
        ic1, ic2 = st.columns([1, 3])
        with ic1:
            st.image(gray128, width=140, caption=st.session_state.get("file_name", "citra"))
        with ic2:
            st.markdown(f"""<div class="readout">
                <span class="k">Sumber citra</span> <b>{st.session_state.get('file_name', '-')}</b><br>
                <span class="k">Prediksi model saat ini</span> <b>{CLASS_EMOJI[pred_class]} {CLASS_NAMES[pred_class]}</b><br>
                <span class="k">Confidence</span> <b>{probe['confidence']*100:.1f}%</b>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------- 2) Rumus & interpretasi IoU --------------------
    colA, colB = st.columns([1, 1.4])
    with colA:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Rumus IoU</div>', unsafe_allow_html=True)
            st.latex(r"IoU=\dfrac{|A\cap B|}{|A\cup B|}")
            st.markdown("""<div class="readout">
                <span class="k">A</span> = Ground Truth (Anotasi Ahli Radiolog)<br>
                <span class="k">B</span> = Prediksi Model (area fokus Grad-CAM)<br>
                <span class="k">|A ∩ B|</span> = Area Intersection (overlap)<br>
                <span class="k">|A ∪ B|</span> = Area Union (gabungan)<br>
                <span class="k">IoU ∈ [0, 1]</span> — 1 = sempurna, 0 = tak tumpang tindih
            </div>""", unsafe_allow_html=True)
    with colB:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Interpretasi IoU</div>', unsafe_allow_html=True)
            legend = [("Sangat Baik", "IoU ≥ 0.75"), ("Baik", "0.50 ≤ IoU < 0.75"),
                      ("Cukup", "0.25 ≤ IoU < 0.50"), ("Kurang", "IoU < 0.25")]
            legend_html = ""
            for label, desc in legend:
                c = IOU_CAT_COLORS[label]
                legend_html += (
                    f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:8px;">'
                    f'<span class="iou-dot" style="width:12px;height:12px;background:{c};"></span>'
                    f'<span style="min-width:96px;display:inline-block;color:{c};font-weight:700;font-family:\'IBM Plex Mono\',monospace;font-size:12px;">{label}</span>'
                    f'<span style="color:var(--text-dim);font-size:12px;">{desc}</span></div>'
                )
            st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------- 3) Prediksi model (B) via Grad-CAM --------------------
    if gradcam_engine is not None:
        cache_key = f"cam_result_{pred_class}"
        if cache_key not in st.session_state:
            with st.spinner("🧠 Mengestimasi area fokus model (Grad-CAM)..."):
                st.session_state[cache_key] = run_inference_with_cam(gray128, target_class=pred_class)
        cam_result = st.session_state[cache_key]
    else:
        cam_result = None

    if cam_result is not None and cam_result["cam_available"]:
        region = analyze_cam_region(cam_result["cam"])
        pred_cx = float(np.clip(region["cx"] / region["w"] * 100, 5, 95))
        pred_cy = float(np.clip(region["cy"] / region["h"] * 100, 5, 95))
        pred_diam = float(np.clip(region["diameter"] / region["w"] * 100, 6, 90))
        cam_note = "🟢 Diestimasi otomatis dari area fokus Grad-CAM model, pada kelas hasil prediksi."
    else:
        pred_cx, pred_cy, pred_diam = 50.0, 50.0, 22.0
        cam_note = "⚠️ Grad-CAM tidak tersedia — memakai area default (tengah citra) sebagai estimasi prediksi model."

    pred_region = dict(cx=pred_cx, cy=pred_cy, diam=pred_diam)
    st.markdown(f'<div class="notice"><b>Prediksi Model (B):</b> {cam_note}</div>', unsafe_allow_html=True)

    # -------------------- 4) Input ground truth (A) --------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Input Anotasi Ground Truth (Ahli Radiolog)</div>', unsafe_allow_html=True)
        st.caption(
            "Aplikasi ini belum terhubung ke data anotasi radiolog sesungguhnya, sehingga posisi & ukuran "
            "ground truth di bawah diinput manual sebagai simulasi untuk keperluan evaluasi IoU. Pada "
            "implementasi produksi, nilai ini idealnya diambil dari mask/bounding-box anotasi ahli (mis. dataset LIDC-IDRI)."
        )
        g1, g2, g3 = st.columns(3)
        with g1:
            gt_cx = st.slider("Posisi X ground truth (%)", 0, 100, 50, key="gt_cx")
        with g2:
            gt_cy = st.slider("Posisi Y ground truth (%)", 0, 100, 50, key="gt_cy")
        with g3:
            gt_diam = st.slider("Diameter ground truth (% lebar citra)", 5, 95, 20, key="gt_diam")

    gt_region = dict(cx=gt_cx, cy=gt_cy, diam=gt_diam)

    # -------------------- 5) Visualisasi + perhitungan --------------------
    visuals = build_iou_visual(gray128, gt_region, pred_region, size=260)
    iou_val, dice_val, inter_px, union_px = compute_iou_dice(visuals["mask_gt"], visuals["mask_pred"])
    kategori, kcolor = iou_category(iou_val)
    keterangan = iou_keterangan(kategori)

    st.markdown("<br>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown('<div class="sub-cap">Ground Truth (Ahli Radiolog)</div>', unsafe_allow_html=True)
        st.image(visuals["img_gt"], use_container_width=True)
    with v2:
        st.markdown('<div class="sub-cap">Prediksi Model (Grad-CAM)</div>', unsafe_allow_html=True)
        st.image(visuals["img_pred"], use_container_width=True)
    with v3:
        st.markdown('<div class="sub-cap">Overlap (IoU)</div>', unsafe_allow_html=True)
        st.image(visuals["img_overlap"], use_container_width=True)

    r1, r2 = st.columns([1.3, 1])
    with r1:
        st.markdown(f"""
        <div class="pred-box" style="border-left:3px solid {kcolor};">
            <div class="pred-eyebrow">Hasil IoU (Ahli vs Model)</div>
            <div class="pred-class" style="color:{kcolor};">{iou_val:.2f} · {kategori}</div>
            <div class="pred-conf">Dice Coefficient {dice_val:.2f} &nbsp;|&nbsp; Intersection {inter_px}px &nbsp;|&nbsp; Union {union_px}px</div>
            <span class="chip" style="background:{kcolor}22;color:{kcolor};border:1px solid {kcolor}55;">{keterangan}</span>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.plotly_chart(confidence_gauge(iou_val, kcolor), use_container_width=True)

    # -------------------- 6) Tambahkan ke tabel perbandingan --------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Tambahkan Hasil ke Tabel Perbandingan</div>', unsafe_allow_html=True)
        default_case = os.path.splitext(st.session_state.get("file_name", "Case-01"))[0]
        n_existing = len(st.session_state["iou_records"])
        m1, m2, m3, m4 = st.columns([1.2, 1, 1, 0.9])
        with m1:
            in_case = st.text_input("Case ID", value=default_case, key="iou_case_id")
        with m2:
            in_nodule = st.text_input("Nodule ID", value=f"Nodule {n_existing + 1}", key="iou_nodule_id")
        with m3:
            in_diam_mm = st.number_input("Diameter (mm)", min_value=0.1, max_value=100.0, value=5.0, step=0.1, key="iou_diam_mm")
        with m4:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            add_clicked = st.button("➕ Tambahkan", use_container_width=True)

        if add_clicked:
            st.session_state["iou_records"].append(dict(
                case_id=in_case, nodule_id=in_nodule, diameter=in_diam_mm,
                iou=iou_val, dice=dice_val, kategori=kategori, keterangan=keterangan,
            ))
            st.success(f"Ditambahkan: {in_case} / {in_nodule} — IoU {iou_val:.2f} ({kategori})")

    # -------------------- 7) Tabel hasil per nodul --------------------
    records = st.session_state["iou_records"]
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Hasil Perbandingan IoU per Nodul</div>', unsafe_allow_html=True)
    if not records:
        st.markdown('<div class="notice">Belum ada data. Tambahkan hasil evaluasi di atas untuk mengisi tabel.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(iou_table_html(records), unsafe_allow_html=True)
        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            if st.button("↩️ Hapus baris terakhir"):
                st.session_state["iou_records"].pop()
                st.rerun()
        with d2:
            if st.button("🗑️ Reset semua"):
                st.session_state["iou_records"] = []
                st.rerun()

        # -------------------- 8) Ringkasan statistik IoU --------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Ringkasan Statistik IoU</div>', unsafe_allow_html=True)
        render_iou_summary_stats(records)

    st.markdown(f"""<div class="notice" style="margin-top:16px;">
        <b>Kesimpulan:</b> IoU digunakan untuk mengukur tingkat kesesuaian lokalisasi/segmentasi nodul antara
        pendapat ahli radiolog (ground truth) dan estimasi area fokus model computer vision (Grad-CAM).
        Semakin tinggi IoU &amp; Dice, semakin baik model dalam meniru pendapat ahli. Karena model ini adalah
        classifier (bukan segmentation network), area prediksi bersifat estimasi/indikatif — bukan output
        segmentasi langsung.
    </div>""", unsafe_allow_html=True)


# ============================================================
# ROUTING
# ============================================================
if page == "🏠 Dashboard":
    page_dashboard()
elif page == "🔬 Deteksi Nodul":
    page_deteksi()
elif page == "📊 Grad-CAM":
    page_gradcam()
elif page == "🌊 Subband DWT":
    page_subband()
elif page == "📐 Evaluasi IoU":
    page_iou()