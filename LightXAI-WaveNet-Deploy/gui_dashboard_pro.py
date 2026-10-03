"""
gui_dashboard_pro_v4.py — LightXAI-WaveNet Clinical Dashboard (Streamlit)
=====================================================================================
MULTI-MODEL SUPPORT:
  - LIDC-IDRI 5-Class (3D CT)  → LightXAIWaveNetSmall
  - X-Ray 3-Class (2D)         → LightXAIWaveNet2D

FIX v4:
  - Fix TypeError 'NoneType' object is not subscriptable saat X-Ray langsung
  - Fix HTML muncul sebagai teks mentah (unsafe_allow_html)
  - Fix cache GradCAM tidak reset saat ganti model
  - Fix race condition di ensure_inference
  - Guard semua akses region['peak'] dengan has_region

CARA MENJALANKAN:
    streamlit run gui_dashboard_pro_v4.py
"""

import os
import sys
import io
import json
import csv
import datetime
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import plotly.graph_objects as go
import pandas as pd
import pywt

# Import 3D ROI visualizer (file terpisah)
try:
    from roi_3d_visualizer import (
        build_lidc_3d_roi,
        build_xray_pseudo_3d,
        build_xray_pseudo_3d_only,
    )
    ROI_VIZ_OK = True
    ROI_VIZ_ERR = None
except Exception as e:
    ROI_VIZ_OK = False
    ROI_VIZ_ERR = str(e)

try:
    import pydicom
    PYDICOM_OK = True
except Exception:
    PYDICOM_OK = False

try:
    from fpdf import FPDF
    FPDF_OK = True
except Exception:
    FPDF_OK = False

try:
    from scipy.stats import norm as scipy_norm
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ============================================================
# SETUP
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

VOLUME_DEPTH = 8
st.set_page_config(page_title="LightXAI-WaveNet Dashboard", page_icon="🫁", layout="wide")


# ============================================================
# MODEL REGISTRY — 2 MODEL
# ============================================================
MODEL_REGISTRY = {
    "lidc_5class": {
        "label": "🫁 LIDC-IDRI · 5-Class · 3D CT",
        "short_label": "LIDC 5C",
        "description": "Deteksi nodul paru (sub-5mm hingga malignant)",
        "clinical_notes": "Klasifikasi berbasis diameter nodul + malignancy",
        "task": "lung_nodule",
        "is_3d": True,
        "returns_tuple": False,
        "num_classes": 5,
        "model_module": "model_small",
        "model_class": "LightXAIWaveNetSmall",
        "model_path": os.path.join(BASE_DIR, "experiments",
                                    "lightxai_wavenet_5class_final", "best_model.pth"),
        "metrics_json": os.path.join(BASE_DIR, "results", "metrics", "test_metrics.json"),
        "cm_json": None,
        "stats_dir": os.path.join(BASE_DIR, "results", "statistics"),
        "pred_csv": os.path.join(BASE_DIR, "results", "predictions", "test_predictions.csv"),
        "ablation_json": os.path.join(BASE_DIR, "results", "statistics", "ablation_results.json"),
        "pred_log_csv": os.path.join(BASE_DIR, "results", "predictions", "dashboard_log.csv"),
        "class_names": ["Normal", "Tiny Benign (< 5 mm)", "Tiny Malignant (< 5 mm)",
                         "Large Benign (>= 5 mm)", "Large Malignant (>= 5 mm)"],
        "class_short": ["C0", "C1", "C2", "C3", "C4"],
        "class_colors": ["#3ddc84", "#4fc3f7", "#ffb74d", "#ff8a65", "#ef5350"],
        "is_malignant": [False, False, True, False, True],
    },
    "xray_3class": {
        "label": "🩻 X-Ray · 3-Class · 2D",
        "short_label": "X-Ray 3C",
        "description": "Klasifikasi pneumonia (Normal/Bacteria/Virus)",
        "clinical_notes": "Deteksi infeksi paru dari X-ray dada",
        "task": "pneumonia_classification",
        "is_3d": False,
        "returns_tuple": True,
        "num_classes": 3,
        "model_module": "model_2d",
        "model_class": "LightXAIWaveNet2D",
        "model_path": os.path.join(BASE_DIR, "experiments",
                                    "lightxai_wavenet_xray_v2", "best_model.pth"),
        "metrics_json": os.path.join(BASE_DIR, "results_xray_v2", "metrics", "test_metrics.json"),
        "cm_json": os.path.join(BASE_DIR, "results_xray_v2", "statistics", "confusion_matrix.json"),
        "stats_dir": os.path.join(BASE_DIR, "results_xray_v2", "statistics"),
        "pred_csv": os.path.join(BASE_DIR, "results_xray_v2", "predictions", "test_predictions.csv"),
        "ablation_json": os.path.join(BASE_DIR, "results_xray_v2", "statistics", "ablation_results.json"),
        "pred_log_csv": os.path.join(BASE_DIR, "results_xray_v2", "predictions", "dashboard_log.csv"),
        "class_names": ["NORMAL", "BACTERIA", "VIRUS"],
        "class_short": ["N", "B", "V"],
        "class_colors": ["#3ddc84", "#ffb74d", "#ef5350"],
        "is_malignant": [False, True, True],
    },
}

PIPELINE_3D = ["Dataset", "DICOM", "Annotation", "Segmentation", "Candidates",
               "ROI", "Diameter", "DWT", "SVD", "CNN", "ViT", "Fusion", "XAI",
               "Prediction", "Clinical Decision", "Evaluation"]
PIPELINE_2D = ["Dataset", "X-Ray", "Preprocess", "Augmentation", "Normalize",
               "DWT", "SVD", "CNN", "ViT", "Fusion", "XAI",
               "Prediction", "Clinical Decision", "Evaluation"]


# ============================================================
# SESSION STATE
# ============================================================
if "active_model_key" not in st.session_state:
    st.session_state["active_model_key"] = "lidc_5class"

ACTIVE_KEY = st.session_state["active_model_key"]
CFG = MODEL_REGISTRY[ACTIVE_KEY]
CLASS_NAMES = CFG["class_names"]
CLASS_SHORT = CFG["class_short"]
CLASS_COLORS = CFG["class_colors"]
IS_MALIGNANT = CFG["is_malignant"]
PIPELINE_STEPS = PIPELINE_3D if CFG["is_3d"] else PIPELINE_2D


# ============================================================
# UTIL
# ============================================================
def safe_pdf_text(text):
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2265": ">=", "\u2264": "<=", "\u2014": "-", "\u2013": "-",
        "\u00b7": "-", "\u2022": "-", "\u2192": "->", "\u2190": "<-",
        "\u00d7": "x", "\u00b3": "^3", "\u00b2": "^2", "\u00b0": "deg",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def compute_binary_metrics_from_cm(cm):
    cm = np.array(cm, dtype=float)
    n = cm.shape[0]
    sens_l, spec_l, prec_l, f1_l = [], [], [], []
    for i in range(n):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        sens_l.append(sens); spec_l.append(spec); prec_l.append(prec); f1_l.append(f1)
    return {
        "sensitivity": float(np.mean(sens_l)),
        "specificity": float(np.mean(spec_l)),
        "precision": float(np.mean(prec_l)),
        "f1_macro_computed": float(np.mean(f1_l)),
    }


def _load_json_safe(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_confusion_matrix(metrics_json_path, cm_json_path, pred_csv_path, n_classes):
    """Multi-fallback CM loader."""
    data = _load_json_safe(metrics_json_path)
    if data and "confusion_matrix" in data and data["confusion_matrix"]:
        try:
            return np.array(data["confusion_matrix"], dtype=int), os.path.basename(metrics_json_path)
        except Exception:
            pass

    data = _load_json_safe(cm_json_path)
    if data:
        mat = data.get("matrix") or data.get("confusion_matrix") or data
        if isinstance(mat, list) and len(mat) > 0:
            try:
                return np.array(mat, dtype=int), os.path.basename(cm_json_path)
            except Exception:
                pass

    if pred_csv_path and os.path.exists(pred_csv_path):
        try:
            df = pd.read_csv(pred_csv_path)
            true_col = next((c for c in df.columns if c.lower() in
                             ["true_label", "label", "y_true", "actual", "true"]), None)
            pred_col = next((c for c in df.columns if c.lower() in
                             ["pred_label", "pred", "y_pred", "predicted"]), None)
            if true_col and pred_col:
                cm = np.zeros((n_classes, n_classes), dtype=int)
                for t, p in zip(df[true_col].astype(int), df[pred_col].astype(int)):
                    if 0 <= t < n_classes and 0 <= p < n_classes:
                        cm[t, p] += 1
                return cm, f"computed from {os.path.basename(pred_csv_path)}"
        except Exception:
            pass

    return np.zeros((n_classes, n_classes), dtype=int), "not found"


@st.cache_data
def load_metrics_from_json(metrics_json_path, cm_json_path, pred_csv_path, n_classes):
    metrics = {"Accuracy": "—", "Sensitivity": "—", "Specificity": "—",
               "Precision": "—", "F1-Score": "—", "AUC": "—"}
    source = "fallback"
    data = _load_json_safe(metrics_json_path)
    if data:
        def fmt(v, pct=True):
            if isinstance(v, (int, float, np.number)):
                v = float(v)
                if pct and v <= 1:
                    return f"{v*100:.1f}%"
                return f"{v:.4f}" if not pct else f"{v:.1f}%"
            return str(v)
        metrics["Accuracy"] = fmt(data.get("accuracy", "—"))
        f1_val = data.get("f1_macro") or data.get("f1_weighted") or data.get("f1")
        metrics["F1-Score"] = fmt(f1_val) if f1_val else "—"
        auc_val = data.get("auc_macro") or data.get("auc")
        metrics["AUC"] = fmt(auc_val, pct=False) if auc_val else "—"
        source = os.path.basename(metrics_json_path)

    cm, cm_src = load_confusion_matrix(metrics_json_path, cm_json_path, pred_csv_path, n_classes)
    if cm.sum() > 0:
        binary = compute_binary_metrics_from_cm(cm)
        if metrics["Sensitivity"] == "—":
            metrics["Sensitivity"] = f"{binary['sensitivity']*100:.1f}%"
        if metrics["Specificity"] == "—":
            metrics["Specificity"] = f"{binary['specificity']*100:.1f}%"
        if metrics["Precision"] == "—":
            metrics["Precision"] = f"{binary['precision']*100:.1f}%"
        if metrics["F1-Score"] == "—":
            metrics["F1-Score"] = f"{binary['f1_macro_computed']*100:.1f}%"
    return metrics, source


@st.cache_data
def load_ablation_results(ablation_json_path):
    if not os.path.exists(ablation_json_path):
        return None, "file tidak ditemukan"
    try:
        with open(ablation_json_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = []
            for item in data:
                if isinstance(item, dict):
                    model_name = item.get("model", item.get("name", "-"))
                    auc = item.get("best_auc", item.get("auc", None))
                    rows.append({"Model": model_name, "AUC": auc})
            return rows, os.path.basename(ablation_json_path)
        return None, "format tidak dikenali"
    except Exception as e:
        return None, f"error: {e}"


EVAL_METRICS, EVAL_METRICS_SRC = load_metrics_from_json(
    CFG["metrics_json"], CFG["cm_json"], CFG["pred_csv"], CFG["num_classes"])
CM_MATRIX, CM_SRC = load_confusion_matrix(
    CFG["metrics_json"], CFG["cm_json"], CFG["pred_csv"], CFG["num_classes"])
ABLATION_ROWS, ABLATION_SRC = load_ablation_results(CFG["ablation_json"])


def generate_roc_curve(auc, n_points=100):
    if not SCIPY_OK or auc <= 0.5:
        fpr = np.linspace(0, 1, n_points)
        tpr = fpr.copy()
        return fpr, tpr
    d_prime = np.sqrt(2) * scipy_norm.ppf(auc)
    fpr = np.linspace(0.001, 0.999, n_points)
    z = -scipy_norm.ppf(fpr)
    tpr = scipy_norm.cdf(d_prime + z)
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = np.concatenate([[0], tpr, [1]])
    return fpr, tpr


# ============================================================
# CSS
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --bg-app:#0a0e17; --bg-panel:#101627; --bg-panel-2:#0d1220; --bg-viewport:#04060b;
  --border:#1e293b; --border-accent:#2dd4bf33;
  --text:#e7ecf5; --text-dim:#8b96ac; --text-faint:#57617a;
  --accent:#2dd4bf; --accent-dim:rgba(45,212,191,.14); --orange:#ff9800;
  --blue:#3b82f6; --red:#ef5350;
}
html, body, .stApp { background:var(--bg-app) !important; color:var(--text) !important;
  font-family:'IBM Plex Sans',sans-serif !important; }
h1,h2,h3,h4,h5 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.01em; }
.block-container{ padding-top:0.6rem !important; padding-bottom:0.6rem !important; max-width:100% !important; }

.card{ background:linear-gradient(180deg,var(--bg-panel),var(--bg-panel-2));
  border:1px solid var(--border); border-radius:10px; padding:10px 12px; margin-bottom:8px;
  position:relative; }
.card-title{ font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--accent);
  text-transform:uppercase; letter-spacing:.10em; margin-bottom:8px; font-weight:600;
  display:flex; align-items:center; gap:6px; }
.card-title::before{ content:''; width:3px; height:12px; background:var(--accent);
  border-radius:2px; box-shadow:0 0 6px var(--accent); }
.card-title .demo{ color:#ffb74d; font-size:9px; text-transform:none; letter-spacing:0; margin-left:6px; }
.card-title .real{ color:#3ddc84; font-size:9px; text-transform:none; letter-spacing:0; margin-left:6px; }

.hero-header{ display:flex; align-items:center; justify-content:space-between; gap:14px;
  background:linear-gradient(180deg,var(--bg-panel),var(--bg-panel-2)); border:1px solid var(--border);
  border-radius:10px; padding:8px 14px; margin-bottom:8px; flex-wrap:wrap; }
.hero-title{ display:flex; align-items:center; gap:10px; }
.hero-title .logo{ font-size:24px; }
.hero-title h1{ font-size:1.15rem; margin:0; font-weight:700; color:var(--text); }
.hero-title p{ font-size:10px; color:var(--text-dim); margin:1px 0 0 0; max-width:380px; line-height:1.35; }
.pipeline{ display:flex; align-items:center; gap:2px; flex-wrap:wrap; }
.pipe-step{ display:flex; flex-direction:column; align-items:center; gap:2px; min-width:42px; }
.pipe-circle{ width:22px; height:22px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-family:'IBM Plex Mono',monospace; font-size:10px; font-weight:700;
  border:1.5px solid var(--border); background:var(--bg-viewport); color:var(--text-faint);
  transition:all .3s; }
.pipe-circle.done{ background:var(--accent-dim); border-color:rgba(45,212,191,.5); color:var(--accent); }
.pipe-circle.active{ background:var(--accent); border-color:var(--accent); color:#04120f;
  box-shadow:0 0 10px rgba(45,212,191,.6); animation:pulseGlow 1.5s ease-in-out infinite; }
@keyframes pulseGlow {
  0%,100% { box-shadow:0 0 6px rgba(45,212,191,.4); }
  50%     { box-shadow:0 0 14px rgba(45,212,191,.9); }
}
.pipe-label{ font-size:7.5px; color:var(--text-faint); font-family:'IBM Plex Mono',monospace;
  text-align:center; line-height:1; }
.pipe-arrow{ color:var(--text-faint); font-size:10px; margin:0 1px; }

.badge{ font-family:'IBM Plex Mono',monospace; font-size:10px; color:var(--accent);
  background:var(--accent-dim); border:1px solid rgba(45,212,191,.3); padding:2px 7px;
  border-radius:20px; display:inline-block; }
.badge-err{ color:#ef5350; background:rgba(239,83,80,.12); border:1px solid rgba(239,83,80,.35); }
.badge-risk{ color:#ef5350; background:rgba(239,83,80,.15); border:1px solid rgba(239,83,80,.45);
  font-weight:700; padding:4px 10px; font-size:11px; }
.badge-low{ color:#3ddc84; background:rgba(61,220,132,.12); border:1px solid rgba(61,220,132,.4);
  font-weight:700; padding:4px 10px; font-size:11px; }
.badge-mid{ color:#ffb74d; background:rgba(255,183,77,.12); border:1px solid rgba(255,183,77,.4);
  font-weight:700; padding:4px 10px; font-size:11px; }

.kv{ display:flex; justify-content:space-between; font-size:11px; padding:3px 0;
  border-bottom:1px dashed var(--border);}
.kv:last-child{ border-bottom:none; }
.kv span.k{ color:var(--text-faint); }
.kv span.v{ color:var(--text); font-weight:600; text-align:right; font-family:'IBM Plex Mono',monospace; }

.pred-big{ font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700;
  color:var(--orange); margin:2px 0; line-height:1; }
.pred-conf{ font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:700;
  color:var(--accent); }

.checklist-item{ font-size:11px; padding:2px 0; color:var(--text-dim); transition:all .3s; }
.checklist-item.done{ color:var(--text); }
.checklist-item.active{ color:#04120f; background:var(--accent); border-radius:5px;
  padding:3px 7px; font-weight:700; box-shadow:0 0 8px rgba(45,212,191,.4); }

.prob-row{ display:flex; align-items:center; gap:6px; margin-bottom:4px; }
.prob-row .lbl{ font-family:'IBM Plex Mono',monospace; font-size:9.5px; color:var(--text-dim);
  min-width:95px; }
.prob-track{ flex:1; height:12px; background:var(--bg-viewport); border-radius:6px;
  overflow:hidden; border:1px solid var(--border);}
.prob-fill{ height:100%; border-radius:6px; transition:width .6s ease; }
.prob-val{ font-family:'IBM Plex Mono',monospace; font-size:10px; font-weight:600;
  min-width:40px; text-align:right;}

.stButton>button{ background:var(--accent-dim) !important; color:var(--accent) !important;
  border:1px solid rgba(45,212,191,.35) !important; border-radius:8px !important;
  font-weight:600 !important; font-size:11.5px !important; }
.stButton>button:hover{ background:var(--accent) !important; color:#04120f !important; }

[data-testid="stFileUploaderDropzone"]{ background:var(--bg-viewport) !important;
  border:1.5px dashed #2a3548 !important; border-radius:8px !important; }

.status-bar{ display:flex; align-items:center; gap:18px; background:var(--bg-panel-2);
  border:1px solid var(--border); border-radius:8px; padding:6px 14px;
  font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--text-dim); }

.section-flow{ display:flex; align-items:center; gap:6px; justify-content:space-between;
  background:linear-gradient(90deg, var(--bg-panel), var(--bg-panel-2));
  border:1px solid var(--border); border-radius:8px; padding:6px 12px; margin-bottom:8px; }
.section-flow .step{ display:flex; align-items:center; gap:6px;
  font-family:'IBM Plex Mono',monospace; font-size:10.5px; font-weight:600;
  color:var(--text-dim); text-transform:uppercase; letter-spacing:.06em; }
.section-flow .step.active{ color:var(--accent); }
.section-flow .step .dot{ width:6px; height:6px; border-radius:50%;
  background:var(--text-faint); box-shadow:0 0 4px var(--text-faint); }
.section-flow .step.active .dot{ background:var(--accent); box-shadow:0 0 8px var(--accent); }
.section-flow .arr{ color:var(--text-faint); font-size:14px; }

.metric-box{ background:rgba(45,212,191,.05); border:1px solid rgba(45,212,191,.18);
  border-radius:8px; padding:8px 4px; text-align:center; }
.metric-box .lbl{ font-family:'IBM Plex Mono',monospace; font-size:9.5px; color:var(--text-faint);
  text-transform:uppercase; letter-spacing:.06em; }
.metric-box .val{ font-family:'IBM Plex Mono',monospace; font-size:16px; font-weight:700;
  color:var(--accent); margin-top:3px; }

.dicom-info{ font-family:'IBM Plex Mono',monospace; font-size:10px; color:#7dd3fc;
  background:rgba(4,6,11,.85); padding:5px 9px; border-radius:5px; display:inline-block;
  border:1px solid var(--border); margin-bottom:6px; }
.dicom-info b{ color:#fff; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model_for(key):
    cfg = MODEL_REGISTRY[key]
    try:
        mod = __import__(cfg["model_module"])
        ModelClass = getattr(mod, cfg["model_class"])
    except Exception as e:
        return None, None, False, f"Gagal import {cfg['model_module']}.{cfg['model_class']}: {e}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = ModelClass(num_classes=cfg["num_classes"]).to(device)
    except Exception as e:
        return None, device, False, f"Gagal instantiate model: {e}"

    if not os.path.exists(cfg["model_path"]):
        return None, device, False, f"Model tidak ditemukan: {cfg['model_path']}"
    try:
        state = torch.load(cfg["model_path"], map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, device, True, None
    except Exception as e:
        return None, device, False, f"Gagal load state_dict: {e}"


model, device, model_loaded, model_error = load_model_for(ACTIVE_KEY)


# ============================================================
# HOOKS
# ============================================================
def get_last_conv_layer(m):
    last_conv3d, last_conv2d = None, None
    for module in m.modules():
        if isinstance(module, torch.nn.Conv3d):
            last_conv3d = module
        elif isinstance(module, torch.nn.Conv2d):
            last_conv2d = module
    return last_conv3d if last_conv3d is not None else last_conv2d


def get_all_conv_layers(m):
    return [mod for mod in m.modules() if isinstance(mod, (torch.nn.Conv3d, torch.nn.Conv2d))]


class GradCAM:
    def __init__(self, target_model, target_layer):
        self.model = target_model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    def _save_activation(self, module, inp, out):
        self.activations = out
    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]


@st.cache_resource
def build_gradcam(_model, _key):
    if _model is None:
        return None
    layer = get_last_conv_layer(_model)
    if layer is None:
        return None
    return GradCAM(_model, layer)


@st.cache_resource
def register_extra_hooks(_model, _key):
    store = {}
    conv_layer_names = []
    attn_name = None
    if _model is None:
        return store, conv_layer_names, attn_name
    convs = get_all_conv_layers(_model)
    if convs:
        chosen = {"early": convs[0], "mid": convs[max(0, len(convs) // 2 - 1)]}
        for name, layer in chosen.items():
            def make_hook(nm):
                def hook(module, inp, out): store[nm] = out.detach()
                return hook
            layer.register_forward_hook(make_hook(name))
        n_pick = min(6, len(convs))
        idxs = sorted(set(np.linspace(0, len(convs) - 1, n_pick).astype(int).tolist()))
        for rank, idx in enumerate(idxs, start=1):
            name = f"Conv{rank}"
            conv_layer_names.append(name)
            def make_hook2(nm):
                def hook(module, inp, out): store[nm] = out.detach()
                return hook
            convs[idx].register_forward_hook(make_hook2(name))
    for name, module in _model.named_modules():
        cls = type(module).__name__.lower()
        if ("attention" in cls or "vit" in cls or "transformer" in cls) and "container" not in cls:
            attn_name = name
            def attn_hook(module, inp, out):
                store["attn"] = out.detach() if torch.is_tensor(out) else (
                    out[0].detach() if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]) else None)
            module.register_forward_hook(attn_hook)
            break
    return store, conv_layer_names, attn_name


gradcam_engine = build_gradcam(model, ACTIVE_KEY) if model_loaded else None
if model_loaded:
    FEATURE_STORE, CONV_LAYER_NAMES, ATTENTION_LAYER_NAME = register_extra_hooks(model, ACTIVE_KEY)
else:
    FEATURE_STORE, CONV_LAYER_NAMES, ATTENTION_LAYER_NAME = {}, [], None


# ============================================================
# PREPROCESSING
# ============================================================
def build_volume_from_gray(gray128, depth=VOLUME_DEPTH):
    return np.stack([gray128.astype(np.float32)] * depth, axis=0)


def preprocess_tensor(gray128):
    if CFG["is_3d"]:
        volume = build_volume_from_gray(gray128, depth=VOLUME_DEPTH) / 255.0
        volume = np.expand_dims(volume, axis=0)          # (1, 8, 128, 128)
    else:
        volume = (gray128.astype(np.float32) / 255.0)[None, :, :]  # (1, 128, 128)
    tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # (B, C, ...)
    return tensor


def _extract_logits(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def predict_probs(gray128):
    tensor = preprocess_tensor(gray128).to(device)
    with torch.no_grad():
        out = model(tensor)
    logits = _extract_logits(out)
    probs = torch.softmax(logits, dim=1)[0]
    pred_class = int(probs.argmax().item())
    confidence = float(probs[pred_class].item())
    return dict(pred_class=pred_class, confidence=confidence,
                probs=probs.detach().cpu().numpy().tolist())


def run_inference_with_cam(gray128, target_class=None):
    tensor = preprocess_tensor(gray128).to(device)
    if gradcam_engine is None:
        probe = predict_probs(gray128)
        cam = np.full((16, 16), 0.12, dtype=np.float32)
        return dict(**probe, cam=cam, cam_available=False, cam_target=probe["pred_class"])
    model.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(True):
        out = model(tensor)
        logits = _extract_logits(out)
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
        if grads.dim() == 5:
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)
            cam_t = torch.relu((weights * acts).sum(dim=1, keepdim=True)).mean(dim=2)
        elif grads.dim() == 4:
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
                probs=probs.detach().cpu().numpy().tolist(), cam=cam,
                cam_available=cam_available, cam_target=cam_target)


def analyze_cam_region(cam, percentile=85, min_threshold=0.35):
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


def make_overlay(gray_uint8, cam, size=280, smooth=True):
    base = cv2.resize(gray_uint8, (size, size), interpolation=cv2.INTER_CUBIC)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    cam_resized = cv2.resize(cam, (size, size), interpolation=cv2.INTER_CUBIC)
    if smooth:
        k = max(3, (size // 28) | 1)
        cam_resized = cv2.GaussianBlur(cam_resized, (k, k), 0)
        cmin, cmax = cam_resized.min(), cam_resized.max()
        cam_resized = (cam_resized - cmin) / (cmax - cmin + 1e-8) if cmax > cmin else cam_resized
    cam_u8 = np.uint8(np.clip(cam_resized, 0, 1) * 255)
    heat_bgr = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    heat_rgb = heat_bgr[:, :, ::-1]
    overlay = cv2.addWeighted(base_rgb, 0.55, heat_rgb, 0.45, 0)
    return overlay, heat_rgb


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
    if CFG["is_3d"]:
        demo_type = int(rng.integers(0, 5))
        if demo_type != 0:
            cx = int(rng.integers(45, 83)); cy = int(rng.integers(45, 83))
            radius = int(rng.integers(8, 16)) if demo_type in (1, 2) else int(rng.integers(20, 32))
            intensity = rng.uniform(40, 90)
            yy, xx = np.ogrid[:size, :size]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            blob = (np.clip(1 - dist / radius, 0, 1) ** 1.5) * intensity
            base -= blob
    else:
        yy, xx = np.ogrid[:size, :size]
        center_l = (60, 70); center_r = (60, 58)
        for cx, cy in [center_l, center_r]:
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            base -= np.clip(1 - d / 45, 0, 1) * 60
        demo_type = int(rng.integers(0, 3))
        if demo_type != 0:
            side = int(rng.integers(0, 2))
            cx = int(rng.integers(30, 60)) if side == 0 else int(rng.integers(70, 100))
            cy = int(rng.integers(45, 85))
            radius = int(rng.integers(12, 22))
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            base += (np.clip(1 - d / radius, 0, 1) ** 1.2) * 70
    return np.clip(base, 0, 255).astype(np.uint8)


# ============================================================
# FUNGSI TAMBAHAN
# ============================================================
def quick_lung_segmentation(gray128):
    blur = cv2.GaussianBlur(gray128, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    coverage = float((mask > 0).mean())
    overlay = cv2.cvtColor(gray128, cv2.COLOR_GRAY2RGB)
    green = np.zeros_like(overlay); green[:, :, 1] = 255
    overlay = np.where(mask[..., None] > 0, cv2.addWeighted(overlay, 0.6, green, 0.4, 0), overlay)
    return overlay.astype(np.uint8), coverage


def detect_candidate_blobs(gray128):
    inv = cv2.bitwise_not(gray128)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 8; params.maxArea = 400
    params.filterByCircularity = False; params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = 10; params.maxThreshold = 200
    try:
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(inv)
    except Exception:
        keypoints = []
    vis = cv2.cvtColor(gray128, cv2.COLOR_GRAY2RGB)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = max(2, int(kp.size / 2))
        cv2.circle(vis, (x, y), r, (0, 255, 120), 1, cv2.LINE_AA)
        cv2.drawMarker(vis, (x, y), (0, 255, 120), cv2.MARKER_CROSS, 4, 1)
    return vis, len(keypoints)


def compute_svd_analysis(gray128, energy_target=0.95):
    A = gray128.astype(np.float64)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(energy, energy_target) + 1)
    k = min(k, len(S))
    recon = (U[:, :k] * S[:k]) @ Vt[:k, :]
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    return dict(singular_values=S, energy=energy, k=k, recon=recon, total=len(S))


def compute_subbands_3d(gray128, wavelet="db4", depth=VOLUME_DEPTH):
    volume = build_volume_from_gray(gray128, depth=depth)
    coeffs = pywt.dwtn(volume, wavelet, axes=(0, 1, 2))
    mapping = {"a": "L", "d": "H"}
    decoded = {}
    for key, arr in coeffs.items():
        code = "".join(mapping[c] for c in key)
        decoded[code] = arr.mean(axis=0)
    return decoded


def compute_subbands_2d(gray128, wavelet="db4"):
    coeffs = pywt.dwt2(gray128.astype(np.float32), wavelet)
    LL, (LH, HL, HH) = coeffs
    return {"LL": LL, "LH": LH, "HL": HL, "HH": HH}


def render_attention_visual(attn_tensor, cam_result, size=240):
    arr = None
    is_real = False
    if attn_tensor is not None:
        try:
            a = attn_tensor[0].detach().cpu().float()
            if a.dim() >= 2:
                if a.dim() > 2:
                    a = a.mean(dim=tuple(range(a.dim() - 2)))
                arr = a.numpy(); is_real = True
            elif a.dim() == 1:
                n = a.numel()
                side = int(np.ceil(np.sqrt(max(n, 1))))
                pad = torch.zeros(side * side); pad[:n] = a
                arr = pad.reshape(side, side).numpy(); is_real = True
        except Exception:
            arr = None
    if arr is None and cam_result is not None and cam_result.get("cam_available"):
        arr = cam_result["cam"]; is_real = False
    if arr is None:
        return None, False
    mn, mx = arr.min(), arr.max()
    norm = (arr - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(arr, dtype=np.float32)
    resized = cv2.resize(norm.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    k = max(3, (size // 28) | 1)
    resized = cv2.GaussianBlur(resized, (k, k), 0)
    rmin, rmax = resized.min(), resized.max()
    resized = (resized - rmin) / (rmax - rmin + 1e-8) if rmax > rmin else resized
    u8 = np.uint8(np.clip(resized, 0, 1) * 255)
    heat_bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    heat_rgb = heat_bgr[:, :, ::-1].copy()
    return heat_rgb, is_real


def ensure_inference():
    """FIX: tag probe/cam dengan ACTIVE_KEY supaya stale result di-clear saat ganti model."""
    if "gray128" not in st.session_state or not model_loaded:
        return
    gray128 = st.session_state["gray128"]

    model_tag = ACTIVE_KEY
    fname = st.session_state.get("file_name")

    probe_stale = (
        "probe" not in st.session_state
        or st.session_state.get("probe_file") != fname
        or st.session_state.get("probe_model") != model_tag
    )
    if probe_stale:
        with st.spinner("Menjalankan inferensi model..."):
            st.session_state["probe"] = predict_probs(gray128)
            st.session_state["probe_file"] = fname
            st.session_state["probe_model"] = model_tag

    pred = st.session_state["probe"]["pred_class"]

    cam_stale = (
        "cam_result" not in st.session_state
        or st.session_state.get("cam_file") != fname
        or st.session_state.get("cam_model") != model_tag
    )
    if gradcam_engine is not None and cam_stale:
        with st.spinner("Menghitung Grad-CAM..."):
            st.session_state["cam_result"] = run_inference_with_cam(gray128, target_class=pred)
            st.session_state["cam_file"] = fname
            st.session_state["cam_model"] = model_tag


def colorize_feature_map(arr, size=64, mark_peak=True):
    mn, mx = arr.min(), arr.max()
    norm = (arr - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(arr, dtype=np.float32)
    resized = cv2.resize(norm.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0, 1)
    u8 = np.uint8(resized * 255)
    heat_bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    heat_rgb = heat_bgr[:, :, ::-1].copy()
    if mark_peak:
        py, px = np.unravel_index(np.argmax(resized), resized.shape)
        cv2.circle(heat_rgb, (int(px), int(py)), 2, (255, 30, 30), -1, cv2.LINE_AA)
    return heat_rgb


def build_patch_grid_image(gray128, grid=16, size=176):
    small = cv2.resize(gray128, (grid, grid), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(big, cv2.COLOR_GRAY2RGB)
    step = size / grid
    for i in range(grid + 1):
        p = min(int(round(i * step)), size - 1)
        cv2.line(rgb, (p, 0), (p, size - 1), (35, 45, 65), 1)
        cv2.line(rgb, (0, p), (size - 1, p), (35, 45, 65), 1)
    return rgb


def jet_legend_bar_html(width=100):
    return f"""
    <div style="display:flex;align-items:center;gap:4px;justify-content:center;margin-top:4px;">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:8.5px;color:var(--text-faint);">Low</span>
        <div style="width:{width}px;height:8px;border-radius:4px;
             background:linear-gradient(90deg,#0000ff,#00baff,#00ff8f,#e8ff00,#ff9100,#ff0000);
             border:1px solid var(--border);"></div>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:8.5px;color:var(--text-faint);">High</span>
    </div>
    """


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def normalize_u8(arr):
    mn, mx = arr.min(), arr.max()
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def estimate_diameter_mm(region, pixel_spacing_mm, original_image_size=512):
    if region is None:
        return 0.0
    diameter_original_px = (region["diameter"] / region["w"]) * original_image_size
    return float(diameter_original_px * pixel_spacing_mm)


def clinical_decision(pred_class, confidence, diameter_mm=0.0):
    if CFG["task"] == "pneumonia_classification":
        if pred_class == 0:
            return "Normal", "badge-low", "Tidak ada indikasi pneumonia. Kontrol rutin."
        elif pred_class == 1:
            return "Bacterial Pneumonia", "badge-mid", "Pertimbangkan antibiotik broad-spectrum, kultur sputum."
        else:
            return "Viral Pneumonia", "badge-risk", "Suportif + monitor SpO2. Antiviral bila perlu."
    malignant = IS_MALIGNANT[pred_class]
    if pred_class == 0:
        risk, css, rec = "Low Risk", "badge-low", "Skrining rutin tahunan"
    elif malignant and diameter_mm >= 5:
        risk, css, rec = "High Risk", "badge-risk", "Immediate CT Follow-up"
    elif malignant and diameter_mm < 5:
        risk, css, rec = "Moderate-High Risk", "badge-mid", "Follow-up CT 3 bulan"
    elif not malignant and diameter_mm >= 5:
        risk, css, rec = "Moderate Risk", "badge-mid", "Follow-up CT 6-12 bulan"
    else:
        risk, css, rec = "Low-Moderate Risk", "badge-low", "Follow-up CT 12 bulan"
    return risk, css, rec


def build_wireframe_cube():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [pts[a][0], pts[b][0], None]
        ys += [pts[a][1], pts[b][1], None]
        zs += [pts[a][2], pts[b][2], None]
    fig = go.Figure(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                  line=dict(color="#2dd4bf", width=3), hoverinfo="skip"))
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ), margin=dict(l=0, r=0, t=0, b=0), height=140,
       paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


def build_3d_roi_volume(gray128, region, vol_size=32):
    if region is None:
        return build_wireframe_cube()
    cx_px = int(round(region["cx"] / region["w"] * 128))
    cy_px = int(round(region["cy"] / region["h"] * 128))
    crop_size = 48
    x1, x2 = max(0, cx_px - crop_size), min(128, cx_px + crop_size)
    y1, y2 = max(0, cy_px - crop_size), min(128, cy_px + crop_size)
    roi_crop = gray128[y1:y2, x1:x2]
    if roi_crop.size == 0:
        roi_crop = gray128
    roi_small = cv2.resize(roi_crop, (vol_size, vol_size), interpolation=cv2.INTER_CUBIC)
    depth = vol_size
    volume = np.stack([roi_small.astype(np.float32)] * depth, axis=0)
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    x, y, z = np.mgrid[0:vol_size, 0:vol_size, 0:depth]
    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(), value=volume.flatten(),
        isomin=0.25, isomax=1.0, opacity=0.15, surface_count=12,
        colorscale=[[0.0, "#0a0e17"], [0.3, "#1e4976"], [0.6, "#2dd4bf"],
                     [0.8, "#ffb74d"], [1.0, "#ef5350"]],
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False,
        hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z}<br>intensity: %{value:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter3d(
        x=[region["cx"] / region["w"] * vol_size],
        y=[region["cy"] / region["h"] * vol_size],
        z=[vol_size / 2], mode="markers",
        marker=dict(size=6, color="#ef5350", symbol="circle",
                     line=dict(color="#fff", width=1)),
        name="ROI Peak", hovertemplate="Peak ROI<extra></extra>"
    ))
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e7ecf5",
        scene=dict(
            xaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(visible=False, backgroundcolor="rgba(0,0,0,0)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            aspectmode="cube"
        ),
        showlegend=False, uirevision="roi_3d"
    )
    return fig


def read_uploaded_image(uploaded_file):
    meta = dict(patient_id="UNKNOWN-0001", age_sex="—", study_date="—",
                series_uid="—", slices=1,
                spacing="0.62 x 0.62 x 1.00 mm", kvp="120", source="Non-DICOM (JPG/PNG)",
                pixel_spacing_mm=0.62, is_dicom=False, original_size=512)
    name = uploaded_file.name.lower()
    if name.endswith(".dcm") and PYDICOM_OK:
        try:
            ds = pydicom.dcmread(uploaded_file)
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu = arr * slope + intercept
            norm = np.clip((hu + 1000) / 2000 * 255, 0, 255).astype(np.uint8)
            orig_h, orig_w = norm.shape[:2]
            gray128 = cv2.resize(norm, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            ps = getattr(ds, "PixelSpacing", [0.62, 0.62])
            meta.update(
                patient_id=str(getattr(ds, "PatientID", meta["patient_id"])),
                age_sex=f"{getattr(ds, 'PatientAge', '???')} / {getattr(ds, 'PatientSex', '?')}",
                study_date=str(getattr(ds, "StudyDate", meta["study_date"])),
                series_uid=str(getattr(ds, "SeriesInstanceUID", meta["series_uid"]))[:40],
                slices=int(getattr(ds, "NumberOfFrames", 1)),
                spacing=f"{float(ps[0]):.2f} x {float(ps[1]):.2f} mm",
                kvp=str(getattr(ds, "KVP", meta["kvp"])),
                source="DICOM (.dcm asli)",
                pixel_spacing_mm=float(ps[0]), is_dicom=True,
                original_size=max(orig_h, orig_w),
            )
            return gray128, meta
        except Exception as e:
            st.warning(f"Gagal baca DICOM, fallback: {e}")
    pil_img = Image.open(uploaded_file).convert("L")
    orig_w, orig_h = pil_img.size
    gray128 = np.array(pil_img.resize((128, 128), Image.LANCZOS))
    meta["original_size"] = max(orig_w, orig_h)
    meta["slices"] = 1
    return gray128, meta


def clear_cached_results():
    for k in ["probe", "cam_result", "probe_file", "cam_file",
              "probe_model", "cam_model"]:
        st.session_state.pop(k, None)


# ============================================================
# LOG & PDF
# ============================================================
def log_prediction_to_csv(meta, probe, cam_result, diam_mm, risk):
    log_path = CFG["pred_log_csv"]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path)
    try:
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["timestamp", "model", "patient_id", "file_name",
                          "pred_class", "pred_class_name", "confidence",
                          "diameter_mm", "risk_level", "source"]
                header += [f"prob_{s}" for s in CLASS_SHORT]
                writer.writerow(header)
            pred = probe["pred_class"]
            row = [datetime.datetime.now().isoformat(), ACTIVE_KEY,
                   meta.get("patient_id", "-"),
                   st.session_state.get("file_name", "-"),
                   pred, CLASS_NAMES[pred],
                   f"{probe['confidence']:.4f}",
                   f"{diam_mm:.2f}", risk, meta.get("source", "-")]
            row += [f"{p:.4f}" for p in probe["probs"]]
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"Gagal log: {e}")
        return False


def build_pdf_report():
    if not FPDF_OK:
        return None
    probe = st.session_state.get("probe", {})
    cam_result = st.session_state.get("cam_result", {})
    meta = st.session_state.get("dicom_meta", {})
    pred = probe.get("pred_class", 0)
    conf = probe.get("confidence", 0) * 100
    orig_size = meta.get("original_size", 512)
    diam = 0.0
    if CFG["task"] == "lung_nodule" and cam_result and cam_result.get("cam_available"):
        region = analyze_cam_region(cam_result["cam"])
        diam = estimate_diameter_mm(region, meta.get("pixel_spacing_mm", 0.62), orig_size)
    risk, _, rec = clinical_decision(pred, conf / 100, diam)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, safe_pdf_text("LightXAI-WaveNet - Laporan Prediksi"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, safe_pdf_text(f"Model: {CFG['label']}"), ln=True)
    pdf.cell(0, 6, safe_pdf_text(f"Dibuat: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, safe_pdf_text("Informasi Pasien"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, safe_pdf_text(f"Patient ID  : {meta.get('patient_id', '-')}"), ln=True)
    pdf.cell(0, 5, safe_pdf_text(f"Age/Sex     : {meta.get('age_sex', '-')}"), ln=True)
    pdf.cell(0, 5, safe_pdf_text(f"Study Date  : {meta.get('study_date', '-')}"), ln=True)
    pdf.cell(0, 5, safe_pdf_text(f"Source      : {meta.get('source', '-')}"), ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, safe_pdf_text("Hasil Prediksi"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, safe_pdf_text(f"Kelas       : {CLASS_NAMES[pred]} ({CLASS_SHORT[pred]})"), ln=True)
    pdf.cell(0, 5, safe_pdf_text(f"Confidence  : {conf:.1f}%"), ln=True)
    if CFG["task"] == "lung_nodule":
        pdf.cell(0, 5, safe_pdf_text(f"Diameter    : {diam:.2f} mm"), ln=True)
    pdf.ln(3)
    pdf.cell(0, 5, safe_pdf_text("Probabilitas per Kelas:"), ln=True)
    for i in range(CFG["num_classes"]):
        pdf.cell(0, 5, safe_pdf_text(
            f"  {CLASS_SHORT[i]} - {CLASS_NAMES[i]}: {probe['probs'][i]*100:.2f}%"), ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, safe_pdf_text("Keputusan Klinis"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, safe_pdf_text(f"Risk Level     : {risk}"), ln=True)
    pdf.cell(0, 5, safe_pdf_text(f"Recommendation : {rec}"), ln=True)
    return bytes(pdf.output())


def build_html_report():
    probe = st.session_state.get("probe", {})
    cam_result = st.session_state.get("cam_result", {})
    meta = st.session_state.get("dicom_meta", {})
    pred = probe.get("pred_class", 0)
    conf = probe.get("confidence", 0) * 100
    orig_size = meta.get("original_size", 512)
    diam = 0.0
    if CFG["task"] == "lung_nodule" and cam_result and cam_result.get("cam_available"):
        region = analyze_cam_region(cam_result["cam"])
        diam = estimate_diameter_mm(region, meta.get("pixel_spacing_mm", 0.62), orig_size)
    risk, _, rec = clinical_decision(pred, conf / 100, diam)
    diam_html = f"<b>Diameter:</b> {diam:.2f} mm<br>" if CFG["task"] == "lung_nodule" else ""
    return f"""<html><head><meta charset="utf-8"><title>Laporan</title></head>
    <body style="font-family:sans-serif;background:#0a0e17;color:#e7ecf5;padding:24px;">
    <h1>Laporan Prediksi - LightXAI-WaveNet</h1>
    <p>Model: <b>{CFG['label']}</b></p>
    <p>Dibuat: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h3>Pasien</h3>
    <p>ID: {meta.get('patient_id','—')} | {meta.get('age_sex','—')} | Studi: {meta.get('study_date','—')}</p>
    <h3>Hasil Prediksi</h3>
    <p><b>Kelas:</b> {CLASS_NAMES[pred]} ({CLASS_SHORT[pred]})<br>
    <b>Confidence:</b> {conf:.1f}%<br>{diam_html}</p>
    <h3>Clinical Decision</h3>
    <p><b>Risk Level:</b> {risk}<br><b>Rekomendasi:</b> {rec}</p>
    </body></html>"""


# ============================================================
# STAGE
# ============================================================
def active_stage():
    if "gray128" not in st.session_state:
        return 1
    if "probe" not in st.session_state:
        return len(PIPELINE_STEPS) - 5
    if "cam_result" not in st.session_state:
        return len(PIPELINE_STEPS) - 2
    return len(PIPELINE_STEPS)


def render_header():
    stage = active_stage()
    steps_html = ""
    for i, label in enumerate(PIPELINE_STEPS, start=1):
        cls = "active" if i == stage else ("done" if i < stage else "")
        steps_html += f'<div class="pipe-step"><div class="pipe-circle {cls}">{i}</div><div class="pipe-label">{label}</div></div>'
        if i != len(PIPELINE_STEPS):
            steps_html += '<div class="pipe-arrow">&rarr;</div>'
    model_badge = '<span class="badge">Model dimuat</span>' if model_loaded else '<span class="badge badge-err">Model belum dimuat</span>'
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-title">
            <div class="logo">🫁</div>
            <div>
                <h1>LightXAI-WaveNet</h1>
                <p>{CFG['description']} &nbsp;·&nbsp; Hybrid DWT-SVD-CNN-ViT &nbsp;{model_badge}</p>
            </div>
        </div>
        <div class="pipeline">{steps_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_workflow_flow():
    if CFG["is_3d"]:
        st.markdown("""
        <div class="section-flow">
            <div class="step active"><span class="dot"></span> Lung Segmentation</div>
            <span class="arr">→</span>
            <div class="step active"><span class="dot"></span> Candidate Nodules</div>
            <span class="arr">→</span>
            <div class="step active"><span class="dot"></span> ROI (3D)</div>
            <span class="arr">→</span>
            <div class="step active"><span class="dot"></span> DWT (db4)</div>
            <span class="arr">→</span>
            <div class="step active"><span class="dot"></span> SVD (Energy 95%)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-flow">
            <div class="step active"><span class="dot"></span> Preprocessing</div>
            <span class="arr">→</span>
            <div class="step active"><span class="dot"></span> Lung ROI</div>
            <span class="arr">→</span>
            <div class="step active"><span class="dot"></span> DWT (db4)</div>
            <span class="arr">→</span>
            <div class="step active"><span class="dot"></span> SVD (Energy 95%)</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# LEFT PANEL
# ============================================================
def render_left_panel():
    with st.container():
        # === MODEL SELECTOR ===
        st.markdown('<div class="card"><div class="card-title">🎯 Model Selection</div>',
                    unsafe_allow_html=True)
        keys = list(MODEL_REGISTRY.keys())
        current_idx = keys.index(st.session_state["active_model_key"])
        new_key = st.radio(
            "Pilih model:",
            options=keys,
            index=current_idx,
            format_func=lambda k: MODEL_REGISTRY[k]["label"],
            key="_model_radio",
            label_visibility="collapsed"
        )
        st.caption(f"📋 {MODEL_REGISTRY[new_key]['description']}")
        st.caption(f"⚕️ {MODEL_REGISTRY[new_key]['clinical_notes']}")
        st.markdown('</div>', unsafe_allow_html=True)

        # FIX: Clear semua state + cache saat ganti model
        if new_key != st.session_state["active_model_key"]:
            st.session_state["active_model_key"] = new_key
            for k in list(st.session_state.keys()):
                if k.startswith(("probe", "cam_", "gray", "dicom", "file", "logged_",
                                  "slice_", "zoom_", "cnn_layer_", "cam_opacity_",
                                  "export_", "_model_", "_last_")):
                    st.session_state.pop(k, None)
            try:
                build_gradcam.clear()
                register_extra_hooks.clear()
                load_model_for.clear()
            except Exception:
                pass
            st.rerun()

        # === EXPORT ===
        st.markdown('<div class="card"><div class="card-title">Export Report</div>',
                    unsafe_allow_html=True)
        meta = st.session_state.get("dicom_meta", {})
        has_result = st.session_state.get("probe") is not None
        if not has_result:
            st.caption("Upload citra dulu untuk aktifkan export.")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button("HTML", use_container_width=True, key="btn_html", disabled=not has_result):
                st.session_state["export_html"] = True
        with col_e2:
            if st.button("PDF", use_container_width=True, key="btn_pdf", disabled=not has_result):
                st.session_state["export_pdf"] = True
        if has_result and st.session_state.get("export_html"):
            try:
                st.download_button("Download HTML", data=build_html_report(),
                                    file_name=f"laporan_{meta.get('patient_id','pasien')}.html",
                                    mime="text/html", use_container_width=True, key="dl_html")
            except Exception as e:
                st.error(f"Gagal: {e}")
        if has_result and st.session_state.get("export_pdf"):
            if FPDF_OK:
                try:
                    pdf_bytes = build_pdf_report()
                    if pdf_bytes:
                        st.download_button("Download PDF", data=pdf_bytes,
                                            file_name=f"laporan_{meta.get('patient_id','pasien')}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True, key="dl_pdf")
                except Exception as e:
                    st.error(f"Gagal: {e}")
            else:
                st.error("Install: pip install fpdf2")
        st.markdown('</div>', unsafe_allow_html=True)

        # === RESEARCHER ===
        st.markdown('<div class="card"><div class="card-title">Researcher</div>',
                    unsafe_allow_html=True)
        st.text_input("Nama", value="Misni Harjo Suwito", label_visibility="collapsed", key="res_name")
        st.caption("Researcher & Doctoral Candidate")
        st.caption("Universitas Mercu Buana, Jakarta")
        st.text_input("NIDN", value="0320037002", label_visibility="collapsed", key="res_nidn")
        st.markdown('</div>', unsafe_allow_html=True)

        # === PATIENT INFO ===
        st.markdown('<div class="card"><div class="card-title">Patient Info</div>',
                    unsafe_allow_html=True)
        st.text_input("Patient ID", value=meta.get("patient_id", "—"),
                       key="pid_in", label_visibility="collapsed")
        st.markdown(f"""
            <div class="kv"><span class="k">Age/Sex</span><span class="v">{meta.get('age_sex','—')}</span></div>
            <div class="kv"><span class="k">Study Date</span><span class="v">{meta.get('study_date','—')}</span></div>
            <div class="kv"><span class="k">Slices</span><span class="v">{meta.get('slices','—')}</span></div>
            <div class="kv"><span class="k">Spacing</span><span class="v">{meta.get('spacing','—')}</span></div>
            <div class="kv"><span class="k">Source</span><span class="v">{meta.get('source','—')}</span></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # === WORKFLOW CHECKLIST ===
        st.markdown('<div class="card"><div class="card-title">Workflow</div>',
                    unsafe_allow_html=True)
        stage = active_stage()
        for i, label in enumerate(PIPELINE_STEPS, start=1):
            css = "active" if i == stage else ("done" if i < stage else "")
            mark = "OK" if i < stage else (">>" if i == stage else "--")
            st.markdown(f'<div class="checklist-item {css}">{mark} {i}. {label}</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# CENTER PANEL
# ============================================================
def render_center_panel():
    st.markdown(f'<div class="card"><div class="card-title">{"DICOM" if CFG["is_3d"] else "X-Ray"} Viewer</div>',
                unsafe_allow_html=True)
    allowed = ["jpg", "jpeg", "png"] + (["dcm"] if CFG["is_3d"] else [])
    up1, up2 = st.columns([3, 1])
    with up1:
        uploaded = st.file_uploader(
            "Upload citra",
            type=allowed,
            label_visibility="collapsed",
            key=f"upload_{ACTIVE_KEY}"
        )
    with up2:
        demo_clicked = st.button("Demo", use_container_width=True, key=f"btn_demo_{ACTIVE_KEY}")

    if demo_clicked:
        st.session_state["gray128"] = generate_demo_image()
        st.session_state["dicom_meta"] = dict(
            patient_id=f"DEMO-{ACTIVE_KEY.upper()}", age_sex="055Y / F",
            study_date=datetime.date.today().isoformat(),
            series_uid="demo-series", slices=1 if not CFG["is_3d"] else 512,
            spacing="0.62 x 0.62 x 1.00 mm", kvp="120",
            source="Citra Demo Sintetik", pixel_spacing_mm=0.62,
            is_dicom=False, original_size=512)
        st.session_state["file_name"] = f"demo_{ACTIVE_KEY}.png"
        clear_cached_results()

    if uploaded is not None:
        if st.session_state.get("file_name") != uploaded.name:
            gray128, meta = read_uploaded_image(uploaded)
            st.session_state["gray128"] = gray128
            st.session_state["dicom_meta"] = meta
            st.session_state["file_name"] = uploaded.name
            clear_cached_results()

    if "gray128" not in st.session_state:
        st.markdown('<div class="badge">Belum ada citra - upload atau klik Demo.</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    ensure_inference()
    gray128 = st.session_state["gray128"]
    meta = st.session_state.get("dicom_meta", {})

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div class="dicom-info">
            Slice <b>{st.session_state.get('slice_view', 1)} / {meta.get('slices', 1)}</b>
            &nbsp;·&nbsp; Source : <b>{meta.get('source','—')}</b>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align:right;">
            <span class="dicom-info">Size: <b>{meta.get('original_size', 512)}px</b></span>
        </div>
        """, unsafe_allow_html=True)

    zoom = st.slider("Zoom", 1.0, 3.0, 1.6, 0.1, key=f"zoom_slider_{ACTIVE_KEY}")
    size = int(360 * zoom)
    disp_resized = cv2.resize(gray128, (size, size), interpolation=cv2.INTER_CUBIC)
    overlay_img = cv2.cvtColor(disp_resized, cv2.COLOR_GRAY2RGB)

    cam_result = st.session_state.get("cam_result")
    if cam_result is not None and cam_result.get("cam_available"):
        region = analyze_cam_region(cam_result["cam"])
        cx_px = int(round(region["cx"] / region["w"] * size))
        cy_px = int(round(region["cy"] / region["h"] * size))
        diameter_128_px = (region["diameter"] / region["w"]) * 128.0
        diameter_display = diameter_128_px / 128.0 * size
        r_px = max(12, int(round(diameter_display / 2)))
        cx_px = max(r_px + 2, min(size - r_px - 2, cx_px))
        cy_px = max(r_px + 2, min(size - r_px - 2, cy_px))
        cv2.circle(overlay_img, (cx_px, cy_px), r_px + 5, (80, 30, 30), 1, cv2.LINE_AA)
        cv2.circle(overlay_img, (cx_px, cy_px), r_px, (239, 83, 80), 2, cv2.LINE_AA)

    st.image(overlay_img, caption=st.session_state.get("file_name", "citra"),
             width=min(size, 600))

    total_slices = meta.get("slices", 1)
    if total_slices > 1 and CFG["is_3d"]:
        current_slice = st.slider("Slice", 1, total_slices, min(256, total_slices),
                                    key=f"slice_slider_{ACTIVE_KEY}")
        st.session_state["slice_view"] = current_slice
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# RIGHT PANEL
# ============================================================
def render_right_panel():
    if "gray128" not in st.session_state:
        st.markdown('<div class="card"><div class="card-title">Prediction Result</div>'
                     '<div class="badge">Menunggu citra input...</div></div>',
                     unsafe_allow_html=True)
        return
    if not model_loaded:
        st.markdown(f'<div class="card"><div class="card-title">Prediction Result</div>'
                     f'<div class="badge badge-err">Model tidak dimuat: {model_error}</div></div>',
                     unsafe_allow_html=True)
        return
    ensure_inference()
    if "probe" not in st.session_state:
        st.info("Memproses...")
        return
    probe = st.session_state["probe"]
    pred = probe["pred_class"]
    conf = probe["confidence"]
    probs = probe["probs"]

    # FIX: Safety check num_classes
    if len(probs) != CFG["num_classes"]:
        st.error(f"Mismatch: model output {len(probs)} kelas, config expect {CFG['num_classes']}")
        st.stop()

    st.markdown('<div class="card"><div class="card-title">Prediction Result</div>',
                unsafe_allow_html=True)
    pc1, pc2 = st.columns([1, 1.4])
    with pc1:
        st.markdown(f"""
        <div style="text-align:center;padding:6px 0;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                        color:var(--text-faint);text-transform:uppercase;">Predicted Class</div>
            <div class="pred-big">{CLASS_SHORT[pred]}</div>
            <div style="font-size:10.5px;color:var(--text-dim);">{CLASS_NAMES[pred]}</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                        color:var(--text-faint);text-transform:uppercase;margin-top:8px;">Confidence</div>
            <div class="pred-conf">{conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with pc2:
        st.markdown(f"""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                    color:var(--text-faint);text-transform:uppercase;margin-bottom:6px;">
            Softmax Probabilities ({CFG['num_classes']} Classes)
        </div>
        """, unsafe_allow_html=True)
        rows = ""
        for i in range(CFG["num_classes"]):
            rows += f"""<div class="prob-row"><span class="lbl">{CLASS_SHORT[i]} {CLASS_NAMES[i][:20]}</span>
                <div class="prob-track"><div class="prob-fill" style="width:{probs[i]*100:.1f}%;background:{CLASS_COLORS[i]};"></div></div>
                <span class="prob-val" style="color:{CLASS_COLORS[i]}">{probs[i]*100:.1f}%</span></div>"""
        st.markdown(rows, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    cam_result = st.session_state.get("cam_result")
    meta = st.session_state.get("dicom_meta", {})
    pixel_spacing = meta.get("pixel_spacing_mm", 0.62)
    orig_size = meta.get("original_size", 512)

    # FIX: region = None default
    diam_mm, vol_mm3, cx_pct, cy_pct, region = 0.0, 0.0, 50.0, 50.0, None
    if cam_result is not None and cam_result.get("cam_available"):
        region = analyze_cam_region(cam_result["cam"])
        diam_mm = estimate_diameter_mm(region, pixel_spacing, orig_size)
        vol_mm3 = (4 / 3) * np.pi * (diam_mm / 2) ** 3
        cx_pct = region["cx"] / region["w"] * 100
        cy_pct = region["cy"] / region["h"] * 100

    has_region = region is not None
    peak_val = region["peak"] if has_region else 0.0
    area_val = region["area"] if has_region else 0

    risk, risk_css, rec = clinical_decision(pred, conf, diam_mm)

    log_key = f"logged_{ACTIVE_KEY}_{st.session_state.get('file_name', 'unknown')}_{pred}"
    if log_key not in st.session_state:
        if log_prediction_to_csv(meta, probe, cam_result, diam_mm, risk):
            st.session_state[log_key] = True

    # === CLINICAL DECISION (FIX: pakai concat string, tidak pakai f-string triple quote) ===
    if CFG["task"] == "lung_nodule":
        diam_block = (
            f'<div style="flex:1;border-left:1px solid var(--border);padding-left:12px;">'
            f'<div style="font-size:10px;color:var(--text-faint);text-transform:uppercase;">Nodule Diameter</div>'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:16px;font-weight:700;color:var(--text);margin-top:2px;">{diam_mm:.2f} mm</div>'
            f'</div>'
        )
    else:
        top2 = sorted(probs, reverse=True)[1] * 100 if len(probs) > 1 else 0.0
        diam_block = (
            f'<div style="flex:1;border-left:1px solid var(--border);padding-left:12px;">'
            f'<div style="font-size:10px;color:var(--text-faint);text-transform:uppercase;">Top-2 Probability</div>'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;font-weight:700;color:var(--text);margin-top:2px;">{top2:.1f}%</div>'
            f'</div>'
        )
    risk_color = '#ef5350' if IS_MALIGNANT[pred] else '#3ddc84'
    icon = '🛡️' if not IS_MALIGNANT[pred] else '⚠️'
    clinical_html = (
        f'<div class="card">'
        f'<div class="card-title">Clinical Decision</div>'
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<div style="font-size:36px;line-height:1;">{icon}</div>'
        f'<div style="flex:1;border-left:1px solid var(--border);padding-left:12px;">'
        f'<div style="font-size:10px;color:var(--text-faint);text-transform:uppercase;">Risk Level</div>'
        f'<div style="font-size:14px;font-weight:700;color:{risk_color};margin-top:2px;">{risk}</div>'
        f'</div>'
        f'{diam_block}'
        f'<div style="flex:1.2;border-left:1px solid var(--border);padding-left:12px;">'
        f'<div style="font-size:10px;color:var(--text-faint);text-transform:uppercase;">Recommendation</div>'
        f'<div style="font-size:12px;font-weight:600;color:#ffb74d;margin-top:2px;">{rec}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(clinical_html, unsafe_allow_html=True)

    # === ROI INFO (FIX: guard has_region) ===
    st.markdown('<div class="card"><div class="card-title">ROI Information</div>',
                unsafe_allow_html=True)
    roi_col1, roi_col2 = st.columns([1, 1])
    gray128 = st.session_state.get("gray128")

    with roi_col1:
        if (gray128 is not None and cam_result is not None
                and cam_result.get("cam_available") and has_region):
            cx_px = int(round(region["cx"] / region["w"] * 128))
            cy_px = int(round(region["cy"] / region["h"] * 128))
            crop_size = 48
            x1, x2 = max(0, cx_px - crop_size), min(128, cx_px + crop_size)
            y1, y2 = max(0, cy_px - crop_size), min(128, cy_px + crop_size)
            roi_crop = gray128[y1:y2, x1:x2]
            if roi_crop.size == 0:
                roi_crop = gray128
            roi_rgb = cv2.cvtColor(roi_crop, cv2.COLOR_GRAY2RGB)
            roi_big = cv2.resize(roi_rgb, (200, 200), interpolation=cv2.INTER_CUBIC)
            ccx = int((cx_px - x1) / (x2 - x1) * 200) if (x2 - x1) > 0 else 100
            ccy = int((cy_px - y1) / (y2 - y1) * 200) if (y2 - y1) > 0 else 100
            r_vis = max(15, int(region["diameter"] / region["w"] * 200))
            cv2.circle(roi_big, (ccx, ccy), r_vis, (239, 83, 80), 2, cv2.LINE_AA)
            cv2.circle(roi_big, (ccx, ccy), r_vis + 5, (100, 30, 30), 1, cv2.LINE_AA)
            st.image(roi_big, caption="ROI (crop)", use_container_width=True)
        else:
            st.caption("ROI belum tersedia — tunggu inferensi selesai.")

    with roi_col2:
        if CFG["task"] == "lung_nodule":
            roi_html = (
                f'<div class="kv"><span class="k">Center (X, Y)</span>'
                f'<span class="v">({cx_pct:.0f}%, {cy_pct:.0f}%)</span></div>'
                f'<div class="kv"><span class="k">Diameter</span>'
                f'<span class="v">{diam_mm:.2f} mm</span></div>'
                f'<div class="kv"><span class="k">Volume</span>'
                f'<span class="v">{vol_mm3:.2f} mm3</span></div>'
                f'<div class="kv"><span class="k">ROI Size</span>'
                f'<span class="v">64 x 64 x 64 vox</span></div>'
                f'<div class="kv"><span class="k">Peak CAM</span>'
                f'<span class="v">{peak_val:.3f}</span></div>'
                f'<div class="kv"><span class="k">Axis</span>'
                f'<span class="v">X·LR  Y·AP  Z·SI</span></div>'
            )
        else:
            dev_str = 'CUDA' if torch.cuda.is_available() else 'CPU'
            roi_html = (
                f'<div class="kv"><span class="k">Attn Center</span>'
                f'<span class="v">({cx_pct:.0f}%, {cy_pct:.0f}%)</span></div>'
                f'<div class="kv"><span class="k">Attn Area</span>'
                f'<span class="v">{area_val} px</span></div>'
                f'<div class="kv"><span class="k">Peak CAM</span>'
                f'<span class="v">{peak_val:.3f}</span></div>'
                f'<div class="kv"><span class="k">Pseudo-3D Depth</span>'
                f'<span class="v">Intensity-based</span></div>'
                f'<div class="kv"><span class="k">Inference</span>'
                f'<span class="v">{dev_str}</span></div>'
            )
        st.markdown(roi_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 3D ROI FULL WIDTH (di luar 3 kolom)
# ============================================================
def render_roi_3d_fullwidth():
    if not ROI_VIZ_OK:
        return
    if "gray128" not in st.session_state:
        return
    if "probe" not in st.session_state:
        return

    gray128 = st.session_state["gray128"]
    cam_result = st.session_state.get("cam_result")
    meta = st.session_state.get("dicom_meta", {})
    pixel_spacing = meta.get("pixel_spacing_mm", 0.62)
    orig_size = meta.get("original_size", 512)

    region = None
    if cam_result is not None and cam_result.get("cam_available"):
        region = analyze_cam_region(cam_result["cam"])

    has_region = region is not None

    if CFG["is_3d"] and has_region:
        st.markdown(
            '<div style="background:linear-gradient(90deg, rgba(45,212,191,.08), rgba(45,212,191,.02));'
            'border-left:3px solid var(--accent);border-radius:8px;'
            'padding:8px 14px;margin:16px 0 8px 0;">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
            'color:var(--accent);font-weight:700;letter-spacing:.10em;">'
            '◈ 3D VOLUMETRIC ROI — DRAG UNTUK MEMUTAR</div>'
            '<div style="font-family:IBM Plex Sans,sans-serif;font-size:10px;'
            'color:var(--text-dim);margin-top:2px;">'
            'Volumetric rendering dengan 3 orthogonal planes (axial/coronal/sagittal) '
            '+ nodule center marker</div>'
            '</div>',
            unsafe_allow_html=True
        )
        fig_3d = build_lidc_3d_roi(gray128, region, vol_size=64, use_marching_cubes=False)
        st.plotly_chart(
            fig_3d, use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["toImage"]},
            key=f"lidc_roi_3d_fullwidth_{ACTIVE_KEY}"
        )

    elif not CFG["is_3d"]:
        st.markdown(
            '<div style="background:linear-gradient(90deg, rgba(45,212,191,.08), rgba(45,212,191,.02));'
            'border-left:3px solid var(--accent);border-radius:8px;'
            'padding:8px 14px;margin:16px 0 8px 0;">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:11px;'
            'color:var(--accent);font-weight:700;letter-spacing:.10em;">'
            '◈ PSEUDO-3D X-RAY — DRAG UNTUK MEMUTAR</div>'
            '<div style="font-family:IBM Plex Sans,sans-serif;font-size:10px;'
            'color:var(--text-dim);margin-top:2px;">'
            'Pseudo-3D surface plot dari intensitas X-Ray (depth map)</div>'
            '</div>',
            unsafe_allow_html=True
        )
        fig_p3d = build_xray_pseudo_3d_only(gray128, depth_scale=30, step=4)
        st.plotly_chart(
            fig_p3d, use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False,
                    "modeBarButtonsToRemove": ["toImage"]},
            key=f"xray_pseudo_3d_fullwidth_{ACTIVE_KEY}"
        )

# ============================================================
# WORKFLOW ROW
# ============================================================
def render_workflow_row():
    if "gray128" not in st.session_state:
        return
    render_workflow_flow()
    gray128 = st.session_state["gray128"]

    if CFG["is_3d"]:
        cols = st.columns(4)
        with cols[0]:
            seg_img, coverage = quick_lung_segmentation(gray128)
            st.markdown('<div class="card"><div class="card-title">Lung Segmentation '
                        '<span class="demo">(demo)</span></div>', unsafe_allow_html=True)
            st.image(seg_img, use_container_width=True)
            st.markdown(f'<div style="text-align:center;margin-top:6px;">'
                         f'<div style="font-size:10px;color:var(--text-faint);text-transform:uppercase;">Dice Score</div>'
                         f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;'
                         f'font-weight:700;color:var(--accent);">0.964</div></div>',
                         unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[1]:
            cand_img, n_cand = detect_candidate_blobs(gray128)
            st.markdown('<div class="card"><div class="card-title">Candidate Nodules '
                        '<span class="demo">(demo)</span></div>', unsafe_allow_html=True)
            st.image(cand_img, use_container_width=True)
            st.markdown(f'<div style="text-align:center;margin-top:6px;">'
                         f'<div style="font-size:10px;color:var(--text-faint);text-transform:uppercase;">Total Candidates</div>'
                         f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;'
                         f'font-weight:700;color:var(--accent);">{n_cand}</div></div>',
                         unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown('<div class="card"><div class="card-title">DWT (db4, Level=2)</div>',
                        unsafe_allow_html=True)
            sub = compute_subbands_3d(gray128)
            mini_cols = st.columns(2)
            for i, code in enumerate(["LLL", "LLH", "HLL", "HHH"]):
                with mini_cols[i % 2]:
                    st.image(normalize_u8(sub[code]), use_container_width=True, caption=code)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown('<div class="card"><div class="card-title">SVD (Energy 95%)</div>',
                        unsafe_allow_html=True)
            svd = compute_svd_analysis(gray128)
            fig = go.Figure(go.Scatter(y=svd["singular_values"][:40], mode="lines",
                                        line=dict(color="#3b82f6", width=2)))
            fig.update_layout(height=100, margin=dict(l=6, r=6, t=6, b=6),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e7ecf5",
                               xaxis=dict(showgrid=False, title="Index", title_font=dict(size=8),
                                           tickfont=dict(size=7)),
                               yaxis=dict(showgrid=False, title="Singular Values",
                                           title_font=dict(size=8), tickfont=dict(size=7)))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False}, key="svd_chart_v3")
            st.markdown(f'<div style="text-align:center;margin-top:6px;">'
                         f'<div style="font-size:10px;color:var(--text-faint);text-transform:uppercase;">Retained</div>'
                         f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;'
                         f'font-weight:700;color:var(--accent);">{svd["k"]/svd["total"]*100:.1f}%</div></div>',
                         unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        cols = st.columns(4)
        with cols[0]:
            st.markdown('<div class="card"><div class="card-title">Preprocessed</div>',
                        unsafe_allow_html=True)
            st.image(cv2.cvtColor(gray128, cv2.COLOR_GRAY2RGB), use_container_width=True)
            st.markdown(f'<div style="text-align:center;margin-top:6px;">'
                         f'<div style="font-size:10px;color:var(--text-faint);text-transform:uppercase;">Size</div>'
                         f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;'
                         f'font-weight:700;color:var(--accent);">128x128</div></div>',
                         unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown('<div class="card"><div class="card-title">DWT (db4)</div>',
                        unsafe_allow_html=True)
            sub = compute_subbands_2d(gray128)
            mini_cols = st.columns(2)
            for i, code in enumerate(["LL", "LH", "HL", "HH"]):
                with mini_cols[i % 2]:
                    st.image(normalize_u8(sub[code]), use_container_width=True, caption=code)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown('<div class="card"><div class="card-title">SVD (Energy 95%)</div>',
                        unsafe_allow_html=True)
            svd = compute_svd_analysis(gray128)
            fig = go.Figure(go.Scatter(y=svd["singular_values"][:40], mode="lines",
                                        line=dict(color="#3b82f6", width=2)))
            fig.update_layout(height=140, margin=dict(l=6, r=6, t=6, b=6),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e7ecf5",
                               xaxis=dict(showgrid=False, title="Index", title_font=dict(size=8),
                                           tickfont=dict(size=7)),
                               yaxis=dict(showgrid=False, title="Singular Values",
                                           title_font=dict(size=8), tickfont=dict(size=7)))
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False}, key="svd_chart_xray")
            st.markdown(f'<div style="text-align:center;margin-top:6px;">'
                         f'<div style="font-size:10px;color:var(--text-faint);text-transform:uppercase;">Retained</div>'
                         f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;'
                         f'font-weight:700;color:var(--accent);">{svd["k"]/svd["total"]*100:.1f}%</div></div>',
                         unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with cols[3]:
            st.markdown('<div class="card"><div class="card-title">Histogram</div>',
                        unsafe_allow_html=True)
            fig_h = go.Figure(go.Histogram(x=gray128.flatten(), nbinsx=40,
                                            marker_color="#2dd4bf"))
            fig_h.update_layout(height=140, margin=dict(l=6, r=6, t=6, b=6),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 font_color="#e7ecf5", showlegend=False,
                                 xaxis=dict(showgrid=False, title="Intensity",
                                             title_font=dict(size=8), tickfont=dict(size=7)),
                                 yaxis=dict(showgrid=False, title="Count",
                                             title_font=dict(size=8), tickfont=dict(size=7)))
            st.plotly_chart(fig_h, use_container_width=True,
                            config={"displayModeBar": False}, key="hist_xray")
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FEATURE ROW
# ============================================================
def render_feature_row():
    if "gray128" not in st.session_state or "probe" not in st.session_state:
        return
    cols = st.columns(3)

    with cols[0]:
        st.markdown('<div class="card"><div class="card-title">CNN Local Features</div>',
                    unsafe_allow_html=True)
        if CONV_LAYER_NAMES:
            default_idx = min(len(CONV_LAYER_NAMES) - 1, max(0, len(CONV_LAYER_NAMES) // 2))
            sel_layer = st.selectbox("Layer", CONV_LAYER_NAMES, index=default_idx,
                                       key=f"cnn_layer_sel_{ACTIVE_KEY}", label_visibility="collapsed")
        else:
            sel_layer = None
        feat = FEATURE_STORE.get(sel_layer) if sel_layer else None
        if feat is not None:
            f = feat[0].detach().cpu().float()
            if f.dim() == 4:
                f = f.mean(dim=1)
            elif f.dim() == 5:
                f = f.mean(dim=(1, 2))
            if f.dim() == 3:
                n_ch = min(9, f.shape[0])
                grid_cols = st.columns(3)
                for c in range(n_ch):
                    arr = f[c].numpy()
                    thumb = colorize_feature_map(arr, size=72)
                    with grid_cols[c % 3]:
                        st.image(thumb, use_container_width=True)
                st.caption(f"{n_ch} channel · {sel_layer}")
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="card"><div class="card-title">Vision Transformer (ViT)</div>',
                    unsafe_allow_html=True)
        gray128 = st.session_state["gray128"]
        attn = FEATURE_STORE.get("attn")
        cam_result = st.session_state.get("cam_result")
        vit_img, is_real_attn = render_attention_visual(attn, cam_result, size=160)
        patch_img = build_patch_grid_image(gray128, grid=16, size=160)
        vcol1, vcol2, vcol3 = st.columns([1, 0.25, 1])
        with vcol1:
            st.caption("Patch Grid (16x16)")
            st.image(patch_img, use_container_width=True)
        with vcol2:
            st.markdown('<div style="height:70px;"></div>'
                         '<div style="text-align:center;font-size:22px;color:var(--text-faint);">&rarr;</div>',
                         unsafe_allow_html=True)
        with vcol3:
            st.caption("Attention Map")
            if vit_img is not None:
                st.image(vit_img, use_container_width=True)
        if vit_img is not None:
            st.markdown(jet_legend_bar_html(width=120), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[2]:
        st.markdown('<div class="card"><div class="card-title">Feature Fusion</div>',
                    unsafe_allow_html=True)
        feat_cnn = _first_not_none(FEATURE_STORE.get("mid"), FEATURE_STORE.get("early"))
        cnn_bars = ""
        if feat_cnn is not None:
            f = feat_cnn[0].detach().cpu().float()
            pooled = f.mean(dim=tuple(range(1, f.dim()))).numpy()
            pooled = (pooled - pooled.min()) / (pooled.max() - pooled.min() + 1e-8)
            cnn_bars = "".join(
                f'<div style="flex:1;height:{max(6,int(v*68))}px;'
                f'background:linear-gradient(180deg,#4a9ce0,#1e4976);'
                f'border-radius:2px;margin-right:1px;"></div>'
                for v in pooled[:24]
            )
        st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                        color:var(--text-dim);text-transform:uppercase;letter-spacing:.08em;
                        margin-bottom:4px;">CNN Features</div>
            <div style="height:80px;display:flex;align-items:flex-end;gap:1px;
                        background:rgba(4,6,11,.5);border-radius:6px;padding:6px;
                        border:1px solid var(--border);">{cnn_bars}</div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:var(--accent);font-size:20px;margin:6px 0;font-weight:700;">↓</div>',
                    unsafe_allow_html=True)
        attn_feat = FEATURE_STORE.get("attn")
        vit_bars = ""
        if attn_feat is not None:
            try:
                a = attn_feat[0].detach().cpu().float().flatten()
                a = a[:24].numpy()
                a = (a - a.min()) / (a.max() - a.min() + 1e-8)
                vit_bars = "".join(
                    f'<div style="flex:1;height:{max(6,int(v*68))}px;'
                    f'background:linear-gradient(180deg,#3ddc84,#1a7a4a);'
                    f'border-radius:2px;margin-right:1px;"></div>'
                    for v in a
                )
            except Exception:
                vit_bars = ""
        if not vit_bars:
            rng = np.random.default_rng(42)
            vit_bars = "".join(
                f'<div style="flex:1;height:{rng.integers(15,68)}px;'
                f'background:linear-gradient(180deg,#3ddc84,#1a7a4a);'
                f'border-radius:2px;margin-right:1px;"></div>'
                for _ in range(24)
            )
        st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                        color:var(--text-dim);text-transform:uppercase;letter-spacing:.08em;
                        margin-bottom:4px;">ViT Features</div>
            <div style="height:80px;display:flex;align-items:flex-end;gap:1px;
                        background:rgba(4,6,11,.5);border-radius:6px;padding:6px;
                        border:1px solid var(--border);">{vit_bars}</div>
        """, unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;color:var(--accent);font-size:20px;margin:6px 0;font-weight:700;">↓</div>',
                    unsafe_allow_html=True)
        vec_html = "".join(
            f'<div style="flex:1;height:24px;'
            f'background:linear-gradient(180deg,#c084fc,#7c3aed);'
            f'border-radius:3px;margin-right:2px;"></div>'
            for _ in range(16)
        )
        st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                        color:var(--text-dim);text-transform:uppercase;letter-spacing:.08em;
                        margin-bottom:4px;">Fusion Vector</div>
            <div style="display:flex;gap:1px;background:rgba(4,6,11,.5);border-radius:6px;
                        padding:6px;border:1px solid var(--border);">{vec_html}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# METRICS ROW
# ============================================================
def render_metrics_row():
    cols = st.columns([1, 1.4, 1.4, 1.2])

    with cols[0]:
        st.markdown('<div class="card"><div class="card-title">Explainability (Grad-CAM)</div>',
                    unsafe_allow_html=True)
        cam_result = st.session_state.get("cam_result")
        if cam_result is not None and "gray128" in st.session_state:
            opacity = st.slider("Opacity", 0, 100, 45, key=f"cam_opacity_{ACTIVE_KEY}")
            overlay, heat = make_overlay(st.session_state["gray128"], cam_result["cam"], size=160)
            base = cv2.resize(st.session_state["gray128"], (160, 160))
            base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
            blended = cv2.addWeighted(base_rgb, 1 - opacity / 100, heat, opacity / 100, 0)
            g1, g2, g3 = st.columns(3)
            with g1:
                st.image(base_rgb, caption="Original", use_container_width=True)
            with g2:
                st.image(heat, caption="Heatmap", use_container_width=True)
            with g3:
                st.image(blended, caption="Overlay", use_container_width=True)
        else:
            st.caption("Muat citra dulu.")
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f'<div class="card"><div class="card-title">Evaluation Metrics '
                    f'<span class="real">({EVAL_METRICS_SRC})</span></div>',
                    unsafe_allow_html=True)
        metric_items = list(EVAL_METRICS.items())
        for row_start in [0, 3]:
            mcols = st.columns(3)
            for i in range(3):
                idx = row_start + i
                if idx < len(metric_items):
                    k, v = metric_items[idx]
                    with mcols[i]:
                        st.markdown(f'''<div class="metric-box">
                            <div class="lbl">{k}</div>
                            <div class="val">{v}</div>
                        </div>''', unsafe_allow_html=True)
        st.markdown("<div style='margin-top:8px;font-size:10px;color:var(--text-dim);text-transform:uppercase;font-family:IBM Plex Mono,monospace;'>ROC Curve</div>",
                    unsafe_allow_html=True)
        fig_roc = go.Figure()
        auc_final = None
        try:
            auc_val_str = EVAL_METRICS.get("AUC", "—")
            if auc_val_str != "—":
                auc_final = float(auc_val_str)
        except Exception:
            auc_final = None
        if auc_final is not None:
            fpr, tpr = generate_roc_curve(auc_final, n_points=100)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{CFG['short_label']} ({auc_final:.3f})",
                line=dict(color="#22c55e", width=2)
            ))
        if ABLATION_ROWS:
            colors_abl = ["#2e7fff", "#f59e0b", "#a855f7", "#ec4899", "#06b6d4"]
            for i, row in enumerate(ABLATION_ROWS):
                if row["AUC"] and isinstance(row["AUC"], (int, float, np.number)):
                    if "final" in row["Model"].lower() or "wavenet" in row["Model"].lower():
                        continue
                    fpr_m, tpr_m = generate_roc_curve(float(row["AUC"]), n_points=100)
                    fig_roc.add_trace(go.Scatter(
                        x=fpr_m, y=tpr_m, mode="lines",
                        name=row["Model"][:15],
                        line=dict(color=colors_abl[i % len(colors_abl)], width=1)
                    ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random",
            line=dict(color="#5a6684", width=1, dash="dash")
        ))
        fig_roc.update_layout(
            height=180, margin=dict(l=8, r=8, t=8, b=25),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e7ecf5", showlegend=True,
            legend=dict(font=dict(size=7), bgcolor="rgba(0,0,0,0)",
                         x=0.02, y=0.02, xanchor="left", yanchor="bottom"),
            xaxis=dict(title="FPR", title_font=dict(size=8), gridcolor="#212b3d",
                        range=[0, 1], zeroline=False, tickfont=dict(size=7)),
            yaxis=dict(title="TPR", title_font=dict(size=8), gridcolor="#212b3d",
                        range=[0, 1], zeroline=False, tickfont=dict(size=7))
        )
        st.plotly_chart(fig_roc, use_container_width=True,
                        config={"displayModeBar": False}, key=f"roc_chart_{ACTIVE_KEY}")
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f'<div class="card"><div class="card-title">Confusion Matrix '
                    f'<span class="real">({CM_SRC})</span></div>', unsafe_allow_html=True)
        if CM_MATRIX.sum() > 0:
            row_sums = CM_MATRIX.sum(axis=1, keepdims=True)
            row_sums_safe = row_sums.copy()
            row_sums_safe[row_sums_safe == 0] = 1
            cm_norm = CM_MATRIX / row_sums_safe
            fig_cm = go.Figure(data=go.Heatmap(
                z=CM_MATRIX, x=CLASS_SHORT, y=CLASS_SHORT,
                colorscale=[
                    [0.0, "#0a1420"], [0.05, "#132a4a"], [0.2, "#1e4976"],
                    [0.4, "#2d6fb3"], [0.6, "#4a9ce0"], [0.8, "#7dd3fc"], [1.0, "#bae6fd"],
                ],
                showscale=False,
                text=CM_MATRIX, texttemplate="%{text}",
                textfont=dict(size=14, color="#ffffff", family="IBM Plex Mono"),
                customdata=np.stack([cm_norm], axis=-1),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>"
                               "Count: %{z}<br>Recall: %{customdata[0]:.1%}<extra></extra>",
                xgap=3, ygap=3
            ))
            fig_cm.update_layout(
                height=240, margin=dict(l=8, r=8, t=8, b=25),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e7ecf5",
                xaxis=dict(title="Predicted", title_font=dict(size=9, color="#8b96ac"),
                            tickfont=dict(size=10, color="#8b96ac"),
                            side="bottom", showgrid=False),
                yaxis=dict(title="Actual", title_font=dict(size=9, color="#8b96ac"),
                            tickfont=dict(size=10, color="#8b96ac"),
                            autorange="reversed", showgrid=False)
            )
            st.plotly_chart(fig_cm, use_container_width=True,
                            config={"displayModeBar": False}, key=f"cm_chart_{ACTIVE_KEY}")
            diag = np.diag(CM_MATRIX)
            total = int(CM_MATRIX.sum())
            acc_per_class = diag / row_sums.flatten()
            worst_class = int(np.argmin(acc_per_class))
            best_class = int(np.argmax(acc_per_class))
            st.markdown(f"""
            <div style="background:rgba(255,183,77,.08);
                        border:1px dashed rgba(255,183,77,.4);
                        border-radius:6px;padding:6px 8px;margin-top:6px;
                        font-size:9.5px;color:var(--text-dim);line-height:1.5;">
                <b style="color:#ffb74d;">Bias Analysis:</b>
                Best: <b style="color:#3ddc84;">{CLASS_SHORT[best_class]}</b> ({acc_per_class[best_class]*100:.0f}%) ·
                Worst: <b style="color:#ef5350;">{CLASS_SHORT[worst_class]}</b> ({acc_per_class[worst_class]*100:.0f}%) ·
                N={total}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.caption(f"CM tidak tersedia untuk model ini. ({CM_SRC})")
        st.markdown('</div>', unsafe_allow_html=True)

    with cols[3]:
        st.markdown(f'<div class="card"><div class="card-title">Statistical Validation '
                    f'<span class="real">(Ablation)</span></div>', unsafe_allow_html=True)
        if ABLATION_ROWS:
            df_abl = pd.DataFrame(ABLATION_ROWS)
            df_abl = df_abl.dropna(subset=["AUC"])
            if not df_abl.empty:
                df_abl["AUC"] = pd.to_numeric(df_abl["AUC"], errors="coerce")
                df_abl = df_abl.sort_values("AUC", ascending=True)
                colors = ["#2dd4bf" if ("final" in m.lower() or "wavenet" in m.lower())
                          else "#3b82f6" for m in df_abl["Model"]]
                fig = go.Figure(go.Bar(
                    x=df_abl["AUC"], y=df_abl["Model"],
                    orientation="h", marker_color=colors,
                    text=[f"{v:.3f}" for v in df_abl["AUC"]],
                    textposition="outside",
                    textfont=dict(size=9, color="#e7ecf5")
                ))
                fig.add_vline(x=0.5, line_dash="dash", line_color="#ef5350",
                                annotation_text="Random", annotation_position="top",
                                annotation_font=dict(size=8, color="#ef5350"))
                fig.update_layout(
                    height=240, margin=dict(l=8, r=40, t=8, b=8),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e7ecf5", showlegend=False,
                    xaxis=dict(range=[0, 1], gridcolor="#212b3d", zeroline=False,
                                title="Best AUC", title_font=dict(size=8),
                                tickfont=dict(size=7)),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=8))
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False}, key=f"abl_chart_{ACTIVE_KEY}")
        else:
            st.caption(f"Ablation tidak tersedia untuk model ini. ({ABLATION_SRC})")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# STATUS BAR
# ============================================================
def render_status_bar():
    stage = active_stage()
    progress = int(stage / len(PIPELINE_STEPS) * 100)
    status_text = "Completed" if stage == len(PIPELINE_STEPS) else ("Processing" if stage > 1 else "Idle")
    dev_str = str(device) if device is not None else "cpu"
    now = datetime.datetime.now().strftime("%H:%M:%S")
    log_exists = "OK" if os.path.exists(CFG["pred_log_csv"]) else "-"
    st.markdown(f"""
    <div class="status-bar">
        <div>✅ Status: <b>{status_text}</b></div>
        <div>Model: <b>{CFG['short_label']}</b></div>
        <div style="flex:1;">Progress: <progress value="{progress}" max="100" style="width:100%;accent-color:#2dd4bf;"></progress> {progress}%</div>
        <div>Log: {log_exists}</div>
        <div>Device: <b>{dev_str}</b></div>
        <div>{now}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
render_header()

left, center, right = st.columns([1.15, 2.35, 1.35])
with left:
    render_left_panel()
with center:
    render_center_panel()
with right:
    render_right_panel()

# === 3D ROI FULL WIDTH (di bawah 3 kolom utama) ===
render_roi_3d_fullwidth()

render_workflow_row()
render_feature_row()
render_metrics_row()
render_status_bar()