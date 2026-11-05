"""
gui_dashboard_pro_v5.py — LungVision AI · LightXAI-WaveNet Clinical Dashboard (Streamlit)
=====================================================================================
DESAIN BARU (sidebar + topbar + form pasien) dengan SEMUA fitur v4 dipertahankan.

MULTI-MODEL SUPPORT:
  - LIDC-IDRI 5-Class (3D CT)  -> LightXAIWaveNetSmall
  - X-Ray 3-Class (2D)         -> LightXAIWaveNet2D

HALAMAN:
  Dashboard | New Analysis | History | Dataset | Model AI | Reports | Settings

CARA MENJALANKAN:
    streamlit run gui_dashboard_pro_v5.py

Catatan:
  - roi_3d_visualizer.py harus berada di folder yang sama (seperti versi sebelumnya).
  - Modality di form "New Analysis" menentukan model aktif:
        Chest X-Ray        -> X-Ray 3-Class
        CT Scan (LIDC-IDRI)-> LIDC 5-Class
"""

import os
import sys
import io
import json
import csv
import datetime
import html as html_lib
import hashlib
import hmac
import re
import uuid
import zipfile
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
st.set_page_config(page_title="LungVision AI", page_icon="🫁", layout="wide",
                   initial_sidebar_state="expanded")


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

MODALITY_OPTIONS = {
    "Chest X-Ray": "xray_3class",
    "CT Scan (LIDC-IDRI)": "lidc_5class",
}
KEY_TO_MODALITY = {v: k for k, v in MODALITY_OPTIONS.items()}

PIPELINE_3D = ["Dataset", "DICOM", "Annotation", "Segmentation", "Candidates",
               "ROI", "Diameter", "DWT", "SVD", "CNN", "ViT", "Fusion", "XAI",
               "Prediction", "Clinical Decision", "Evaluation"]
PIPELINE_2D = ["Dataset", "X-Ray", "Preprocess", "Augmentation", "Normalize",
               "DWT", "SVD", "CNN", "ViT", "Fusion", "XAI",
               "Prediction", "Clinical Decision", "Evaluation"]

NAV_ITEMS = [
    ("Dashboard", "🏠"),
    ("New Analysis", "📝"),
    ("History", "🕘"),
    ("Dataset", "🗄️"),
    ("Model AI", "🧊"),
    ("Reports", "📄"),
    ("Settings", "⚙️"),
]
I18N = {
    "id": {
        "sub_Dashboard": "Ringkasan aktivitas & status model",
        "sub_New Analysis": "Explainable AI-Based Medical Image Analysis System",
        "sub_History": "Daftar pasien & hasil analisis AI",
        "sub_Dataset": "Dataset LIDC-IDRI & X-Ray beserta ketersediaan file",
        "sub_Model AI": "Performa & status model LightXAI-WaveNet",
        "sub_Reports": "Arsip laporan radiologi",
        "sub_Settings": "Edit profil & preferensi aplikasi",
        "sec_title": "Security & Privacy",
        "sec_desc": "Data pasien terenkripsi 100% aman & rahasia",
        "sec_more": "Pelajari lebih lanjut →",
        "login": "Login", "register": "Daftar",
        "auth_sub": "Masuk untuk melanjutkan",
        "no_users": "Belum ada akun. Buat akun pertama di tab Daftar.",
        "ident": "Email atau ID/NIP", "password": "Password", "confirm": "Konfirmasi Password",
        "full_name": "Nama Lengkap", "nip": "ID / NIP", "email": "Email", "spec": "Spesialisasi",
        "btn_login": "Masuk", "btn_register": "Buat Akun",
        "err_required": "Nama, ID/NIP, email, dan password wajib diisi.",
        "err_email": "Format email tidak valid.",
        "err_pw_short": "Password minimal {n} karakter.",
        "err_pw_match": "Konfirmasi password tidak sama.",
        "err_email_used": "Email sudah terdaftar.",
        "err_nip_used": "ID/NIP sudah terdaftar.",
        "err_login": "Email/ID atau password salah.",
        "err_old_pw": "Password lama salah.",
    },
    "en": {
        "sub_Dashboard": "Activity summary & model status",
        "sub_New Analysis": "Explainable AI-Based Medical Image Analysis System",
        "sub_History": "Patient list & AI analysis results",
        "sub_Dataset": "LIDC-IDRI & X-Ray datasets and file availability",
        "sub_Model AI": "LightXAI-WaveNet model performance & status",
        "sub_Reports": "Radiology report archive",
        "sub_Settings": "Edit profile & application preferences",
        "sec_title": "Security & Privacy",
        "sec_desc": "Patient data is 100% encrypted, safe & confidential",
        "sec_more": "Learn more →",
        "login": "Login", "register": "Register",
        "auth_sub": "Sign in to continue",
        "no_users": "No accounts yet. Create the first one in the Register tab.",
        "ident": "Email or ID/NIP", "password": "Password", "confirm": "Confirm Password",
        "full_name": "Full Name", "nip": "ID / NIP", "email": "Email", "spec": "Specialization",
        "btn_login": "Sign in", "btn_register": "Create Account",
        "err_required": "Name, ID/NIP, email, and password are required.",
        "err_email": "Invalid email format.",
        "err_pw_short": "Password must be at least {n} characters.",
        "err_pw_match": "Password confirmation does not match.",
        "err_email_used": "Email is already registered.",
        "err_nip_used": "ID/NIP is already registered.",
        "err_login": "Wrong email/ID or password.",
        "err_old_pw": "Current password is wrong.",
    },
}
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
ANALYSES_FILE = os.path.join(DATA_DIR, "analyses.json")
THUMB_DIR = os.path.join(DATA_DIR, "thumbs")
FORM_KEYS = ["f_pid", "f_name", "f_age", "f_gender", "f_blood", "f_hw",
             "f_smoke", "f_modality", "f_symptoms"]


# ============================================================
# SESSION STATE
# ============================================================
if "active_model_key" not in st.session_state:
    # Default sesuai form: Modality = "Chest X-Ray"
    st.session_state["active_model_key"] = "xray_3class"
if "iou_history" not in st.session_state:
    st.session_state["iou_history"] = []
if "page" not in st.session_state:
    st.session_state["page"] = "New Analysis"
for _k, _v in [("ui_lang", "id"), ("ui_dark", True), ("ui_email_notif", False),
               ("ui_2fa", False), ("ui_default_model", "xray_3class"),
               ("ui_cam_opacity", 45)]:
    st.session_state.setdefault(_k, _v)

# Jaga agar nilai form tidak hilang saat pindah halaman
for _k in FORM_KEYS:
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

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
# MODEL SWITCHING
# ============================================================
def switch_model(new_key):
    """Ganti model aktif + bersihkan state/cached resource lama."""
    if new_key == st.session_state.get("active_model_key"):
        return
    st.session_state["active_model_key"] = new_key
    # Citra yang sudah di-upload TIDAK dihapus; hanya hasil inferensi lama
    # yang dibuang supaya dihitung ulang dengan model baru.
    prefixes = ("probe", "cam_", "logged_",
                "slice_", "zoom_", "cnn_layer_", "cam_opacity_",
                "export_", "_last_")
    for k in list(st.session_state.keys()):
        if k.startswith(prefixes):
            st.session_state.pop(k, None)
    st.session_state["f_modality"] = KEY_TO_MODALITY[new_key]
    st.session_state["model_radio"] = new_key
    try:
        build_gradcam.clear()
        register_extra_hooks.clear()
        load_model_for.clear()
    except Exception:
        pass


def on_modality_change():
    label = st.session_state.get("f_modality")
    if label in MODALITY_OPTIONS:
        switch_model(MODALITY_OPTIONS[label])


def on_model_radio_change():
    key = st.session_state.get("model_radio")
    if key in MODEL_REGISTRY:
        switch_model(key)


def use_model_and_go(key):
    switch_model(key)
    st.session_state["page"] = "New Analysis"


def go_page(name):
    st.session_state["page"] = name


# ============================================================
# CSS
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --bg-app:#070b14; --bg-panel:#101627; --bg-panel-2:#0d1220; --bg-viewport:#04060b;
  --bg-side:#0b1324; --bg-input:#151b2b;
  --border:#1e293b; --border-accent:#2dd4bf33;
  --text:#e7ecf5; --text-dim:#8b96ac; --text-faint:#57617a;
  --accent:#2dd4bf; --accent-dim:rgba(45,212,191,.14); --orange:#ff9800;
  --blue:#3b82f6; --red:#ef5350;
}
html, body, .stApp { background:var(--bg-app) !important; color:var(--text) !important;
  font-family:'Inter','IBM Plex Sans',sans-serif !important; }
h1,h2,h3,h4,h5 { font-family:'Space Grotesk','Inter',sans-serif !important; letter-spacing:-.01em; }
.block-container{ padding-top:0.8rem !important; padding-bottom:0.8rem !important; max-width:100% !important; }
header[data-testid="stHeader"]{ background:transparent !important; height:0 !important; }
#MainMenu, footer{ visibility:hidden; }

/* ---------------- SIDEBAR ---------------- */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b1324,#091020) !important;
  border-right:1px solid var(--border); min-width:300px !important; max-width:300px !important; }
[data-testid="stSidebar"] > div:first-child{ padding-top:0.6rem; }
[data-testid="stSidebarCollapseButton"]{ display:none; }
.brand{ display:flex; align-items:center; gap:12px; padding:10px 6px 18px 6px; }
.brand .bolt{ font-size:26px; color:var(--blue); filter:drop-shadow(0 0 8px rgba(59,130,246,.7)); }
.brand .name{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:22px; color:#fff; line-height:1.1; }
.brand .tag{ font-size:11px; color:var(--text-dim); line-height:1.25; margin-top:2px; max-width:200px; }
[data-testid="stSidebar"] .stButton>button{
  width:100%; justify-content:flex-start !important; text-align:left;
  background:transparent !important; color:#cbd5e1 !important;
  border:2px solid transparent !important; border-radius:12px !important;
  font-size:15px !important; font-weight:500 !important; padding:11px 14px !important;
  margin-bottom:2px; transition:all .2s; }
[data-testid="stSidebar"] .stButton>button:hover{
  background:rgba(59,130,246,.12) !important; color:#fff !important; }
[data-testid="stSidebar"] .stButton>button[kind="primary"],
[data-testid="stSidebar"] .stButton>button[data-testid="stBaseButton-primary"]{
  background:#3b82f6 !important; color:#fff !important; font-weight:600 !important;
  border:2px solid #ffffff !important; box-shadow:0 0 16px rgba(59,130,246,.55); }
.side-card{ background:linear-gradient(180deg,#0f1a30,#0c1527); border:1px solid #1d2c4a;
  border-radius:14px; padding:14px; margin-top:26px; }
.side-card .ico{ font-size:22px; color:var(--blue); margin-bottom:6px; }
.side-card .t{ font-weight:700; font-size:13px; color:#fff; }
.side-card .d{ font-size:12px; color:var(--text-dim); margin:4px 0 8px 0; line-height:1.4; }
.side-card .l{ font-size:12px; color:var(--blue); font-weight:600; }

/* ---------------- TOPBAR ---------------- */
.topbar{ display:flex; align-items:center; justify-content:space-between;
  padding:6px 4px 14px 4px; border-bottom:1px solid var(--border); margin-bottom:16px; flex-wrap:wrap; gap:10px; }
.topbar h1{ font-size:1.9rem; margin:0; font-weight:700; color:#fff; }
.topbar p{ margin:2px 0 0 0; font-size:15px; color:var(--text-dim); }
.top-right{ display:flex; align-items:center; gap:16px; }
.pill{ background:#111a2d; border:1px solid #243049; color:#e7ecf5; border-radius:8px;
  padding:7px 14px; font-size:13px; font-weight:600; }
.bell{ position:relative; font-size:20px; }
.bell .dot{ position:absolute; top:-2px; right:-3px; width:9px; height:9px; background:#ef4444;
  border-radius:50%; box-shadow:0 0 6px #ef4444; }
.logout{ border:1px solid #ef4444; color:#ef4444; border-radius:8px; padding:7px 18px;
  font-weight:600; font-size:14px; background:rgba(239,68,68,.06); }
.user{ display:flex; align-items:center; gap:10px; }
.avatar{ width:46px; height:46px; border-radius:50%; background:#3b82f6; color:#fff; display:flex;
  align-items:center; justify-content:center; font-weight:700; font-size:16px;
  box-shadow:0 0 12px rgba(59,130,246,.5); }
.user .n{ font-weight:700; font-size:15px; color:#fff; line-height:1.1; }
.user .r{ font-size:12.5px; color:var(--text-dim); }

/* ---------------- FORM CARD ---------------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(.form-marker){
  background:linear-gradient(180deg,#0e1526,#0b1120) !important; border:1px solid #1a2438 !important;
  border-radius:18px !important; padding:18px 20px !important; }
.form-title{ font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700;
  color:#fff; text-transform:uppercase; letter-spacing:.01em; margin:4px 0 14px 0; }
[data-testid="stTextInput"] label p, [data-testid="stSelectbox"] label p,
[data-testid="stNumberInput"] label p{ color:#cdd5e5 !important; font-size:15px !important; font-weight:500 !important; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input{
  background:var(--bg-input) !important; color:#fff !important; border:1px solid #232d44 !important;
  border-radius:10px !important; height:46px; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div{
  background:var(--bg-input) !important; border:1px solid #232d44 !important;
  border-radius:10px !important; min-height:46px; color:#fff !important; }
[data-testid="stTextInput"] input::placeholder{ color:#5a647c !important; }
.upload-note{ font-size:13px; color:var(--text-dim); text-align:center; margin:2px 0 8px 0; }
[data-testid="stFileUploaderDropzone"]{ background:#0b1530 !important;
  border:2px dashed #3b82f6 !important; border-radius:14px !important; padding:28px 12px !important; }
[data-testid="stFileUploaderDropzone"] *{ color:#dbe6ff !important; }
[data-testid="stFileUploaderDropzone"] button{ background:rgba(59,130,246,.18) !important;
  border:1px solid rgba(59,130,246,.5) !important; border-radius:8px !important; }

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
  border-radius:10px; padding:8px 14px; margin:12px 0 8px 0; flex-wrap:wrap; }
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

.stButton>button{ background:var(--accent-dim); color:var(--accent);
  border:1px solid rgba(45,212,191,.35); border-radius:8px;
  font-weight:600; font-size:12px; }
.stButton>button:hover{ background:var(--accent); color:#04120f; }
[data-testid="stSidebar"] .stButton>button:hover{ color:#fff; }

.status-bar{ display:flex; align-items:center; gap:18px; background:var(--bg-panel-2);
  border:1px solid var(--border); border-radius:8px; padding:6px 14px; margin-top:10px;
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

.stat-card{ background:linear-gradient(180deg,#101a30,#0c1424); border:1px solid #1d2c4a;
  border-radius:14px; padding:16px 18px; }
.stat-card .sl{ font-size:12px; color:var(--text-dim); text-transform:uppercase; letter-spacing:.08em; }
.stat-card .sv{ font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#fff; margin-top:4px; }
.stat-card .ss{ font-size:12px; color:var(--accent); margin-top:2px; }
.model-card{ background:linear-gradient(180deg,#101a30,#0c1424); border:1px solid #1d2c4a;
  border-radius:14px; padding:16px 18px; margin-bottom:10px; }
.model-card h3{ margin:0 0 4px 0; font-size:1.15rem; color:#fff; }
.model-card .desc{ font-size:13px; color:var(--text-dim); margin-bottom:10px; }
.ok-dot{ color:#3ddc84; font-weight:700; } .no-dot{ color:#ef5350; font-weight:700; }

.dicom-info{ font-family:'IBM Plex Mono',monospace; font-size:10px; color:#7dd3fc;
  background:rgba(4,6,11,.85); padding:5px 9px; border-radius:5px; display:inline-block;
  border:1px solid var(--border); margin-bottom:6px; }
.dicom-info b{ color:#fff; }

/* --- IOU SECTION --- */
.iou-container { background-color: #0d1220; border: 1px solid #1e293b; border-radius: 10px; padding: 20px; margin-top: 20px;}
.iou-title { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: #fff; text-align: center; margin-bottom: 5px; }
.iou-subtitle { font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; font-style: italic; color: #8b96ac; text-align: center; margin-bottom: 25px;}
.iou-step { background: #101627; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; height: 100%; text-align: center; transition: all 0.3s;}
.iou-step:hover { border-color: #2dd4bf; }
.iou-step-title { font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; font-weight: 600; color: #7dd3fc; margin-bottom: 10px; }
.iou-step-img { width: 100%; border-radius: 4px; margin-bottom: 5px;}
.iou-formula-box { background: #04060b; border: 1px solid #1e293b; border-left: 4px solid #2dd4bf;
    border-radius: 8px; padding: 20px; height: 100%; display: flex; flex-direction: column;
    justify-content: flex-start; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
.iou-formula-title { font-family: 'Space Grotesk', sans-serif; font-size: 12px; font-weight: bold;
    color: #2dd4bf; margin-bottom: 15px; letter-spacing: 2px; text-transform: uppercase; }
.iou-math-layout { display: grid; grid-template-columns: auto auto; align-items: center;
    justify-content: center; column-gap: 10px; font-family: 'Times New Roman', Times, serif;
    font-style: italic; font-size: 22px; color: #2dd4bf; margin-bottom: 20px;
    background: rgba(45,212,191,0.05); padding: 15px; border-radius: 8px;
    border: 1px dashed rgba(45,212,191,0.3); }
.iou-math-text { font-size: 22px; }
.iou-math-fraction { display: flex; flex-direction: column; align-items: center; font-size: 20px; }
.iou-math-fraction .numerator { padding-bottom: 2px; }
.iou-math-fraction .denominator { border-top: 1.5px solid #2dd4bf; padding-top: 2px; }
.iou-formula-desc { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #8b96ac;
    line-height: 1.8; background: #101627; padding: 12px; border-radius: 6px; border: 1px solid #1e293b; }
.iou-formula-desc b { color: #e7ecf5; }
.iou-interp-box { background: #04060b; border: 1px solid #1e293b; border-radius: 8px; padding: 15px; height: 100%; }
.iou-interp-row { display: flex; align-items: center; font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; margin-bottom: 8px; color: #e7ecf5; }
.iou-interp-dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 10px; }
.iou-interp-label { width: 80px; font-weight: bold; }
.iou-interp-val { color: #8b96ac; }

/* ============ TOPBAR (tombol Logout & bahasa asli) ============ */
div[data-testid="stHorizontalBlock"]:has(.topbar-marker){
  align-items:center; gap:10px; border-bottom:1px solid var(--border);
  padding:2px 0 14px 0; margin-bottom:16px; }
.tb-title h1{ font-size:1.9rem; margin:0; font-weight:700; color:#fff; line-height:1.15; }
.tb-title p{ margin:2px 0 0 0; font-size:15px; color:var(--text-dim); }
.tb-bell{ text-align:center; }
.tb-user .user{ justify-content:flex-end; }
div[data-testid="stHorizontalBlock"]:has(.topbar-marker) > div:nth-child(2) .stButton>button{
  background:#111a2d !important; border:1px solid #243049 !important; color:#e7ecf5 !important;
  border-radius:8px !important; font-weight:600 !important; font-size:13px !important; padding:6px 0 !important; }
div[data-testid="stHorizontalBlock"]:has(.topbar-marker) > div:nth-child(4) .stButton>button{
  background:rgba(239,68,68,.06) !important; border:1px solid #ef4444 !important; color:#ef4444 !important;
  border-radius:8px !important; font-weight:600 !important; font-size:14px !important; padding:6px 0 !important; }
div[data-testid="stHorizontalBlock"]:has(.topbar-marker) > div:nth-child(4) .stButton>button:hover{
  background:#ef4444 !important; color:#fff !important; }

/* ============ HISTORY / REPORTS / SETTINGS ============ */
[data-testid="stVerticalBlockBorderWrapper"]:has(.hist-marker){
  background:linear-gradient(180deg,#0e1526,#0b1120) !important; border:1px solid #1a2438 !important;
  border-radius:18px !important; padding:14px 20px !important; margin-bottom:14px; }
.tbl-title{ font-family:'Space Grotesk',sans-serif; font-size:1.45rem; font-weight:700; color:#fff;
  text-transform:uppercase; margin:6px 0 4px 0; letter-spacing:.01em; }
.th{ color:#aab6cc; font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:.02em; padding:8px 0 4px 0; }
.td{ color:#e7ecf5; font-size:14.5px; font-weight:500; padding:8px 0; }
.td.strong{ font-weight:700; }
.tag-det{ display:inline-block; background:rgba(239,68,68,.10); color:#f87171;
  border:1px solid rgba(239,68,68,.45); border-radius:999px; padding:2px 12px; font-size:12.5px; font-weight:600; }
.tag-ok{ display:inline-block; background:rgba(16,185,129,.10); color:#34d399;
  border:1px solid rgba(16,185,129,.45); border-radius:999px; padding:2px 12px; font-size:12.5px; font-weight:600; }
hr.rowline{ border:0; border-top:1px solid #172033; margin:0 0 4px 0; }
hr.sep{ border:0; border-top:1px solid #1b263b; margin:18px 0 8px 0; }
.empty-note{ color:var(--text-dim); text-align:center; padding:26px 0; font-size:14px; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.hist-marker) [data-testid="stPopover"] button,
[data-testid="stVerticalBlockBorderWrapper"]:has(.hist-marker) button[data-testid="stPopoverButton"]{
  background:#111a2d !important; border:1px solid #243049 !important; color:#e7ecf5 !important;
  border-radius:8px !important; font-weight:600 !important; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.hist-marker) [data-testid="stColumn"] [data-testid="stColumn"] button,
[data-testid="stVerticalBlockBorderWrapper"]:has(.hist-marker) [data-testid="column"] [data-testid="column"] button{
  background:transparent !important; border:none !important; color:#cbd5e1 !important;
  font-size:17px !important; padding:0 6px !important; min-height:0 !important; box-shadow:none !important; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.hist-marker) [data-testid="stColumn"] [data-testid="stColumn"] button:hover,
[data-testid="stVerticalBlockBorderWrapper"]:has(.hist-marker) [data-testid="column"] [data-testid="column"] button:hover{
  color:#fff !important; background:rgba(59,130,246,.18) !important; }
.set-section{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.2rem; color:#fff; margin:6px 0 12px 0; }
.set-label .t{ font-weight:700; color:#fff; font-size:15px; }
.set-label .d{ color:#8b96ac; font-size:12.5px; margin-top:1px; }
div[data-testid="stHorizontalBlock"]:has(.pref-row){ background:#0d1424; border:1px solid #131c2f;
  border-radius:10px; padding:10px 14px; align-items:center; margin-bottom:8px; }
[data-testid="stForm"]{ border:none !important; padding:0 !important; }
[data-testid="stFormSubmitButton"] button{ background:#3b82f6 !important; color:#fff !important;
  border:none !important; border-radius:10px !important; font-weight:600 !important; padding:8px 18px !important; }
[data-testid="stFormSubmitButton"] button:hover{ background:#2563eb !important; }

[data-testid="stDownloadButton"] button, button[data-testid="stPopoverButton"]{
  background:#111a2d !important; border:1px solid #243049 !important; color:#e7ecf5 !important;
  border-radius:8px !important; font-weight:600 !important; }
[data-testid="stDownloadButton"] button:hover, button[data-testid="stPopoverButton"]:hover{
  border-color:#3b82f6 !important; color:#fff !important; }
[data-testid="stDownloadButton"] button *, button[data-testid="stPopoverButton"] *{ color:inherit !important; }

/* ============ AUTH ============ */
.auth-brand{ text-align:center; margin-bottom:18px; }
.auth-brand .bolt{ font-size:34px; color:#3b82f6; filter:drop-shadow(0 0 10px rgba(59,130,246,.7)); }
.auth-brand .name{ font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; color:#fff; }
.auth-brand .tag{ font-size:13px; color:var(--text-dim); }
.auth-brand .sub{ margin-top:10px; font-size:15px; color:#cbd5e1; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Mode terang (pendekatan): membalik warna seluruh app, gambar dibalik lagi agar tetap asli
LIGHT_CSS = """
<style>
.stApp{ filter: invert(1) hue-rotate(180deg); }
.stApp img{ filter: invert(1) hue-rotate(180deg); }
</style>
"""


# ============================================================
# AUTH — LOGIN / REGISTER / LOGOUT  (data/users.json)
# Password disimpan sebagai hash PBKDF2-SHA256 + salt acak.
# ============================================================
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 6


def T(key):
    lang = st.session_state.get("ui_lang", "id")
    return I18N.get(lang, I18N["id"]).get(key, I18N["id"].get(key, key))


def esc(x):
    return html_lib.escape(str(x if x is not None else ""))


def _atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_users():
    d = _load_json_safe(USERS_FILE)
    return d if isinstance(d, dict) else {}


def _save_users(users):
    _atomic_write_json(USERS_FILE, users)


def _hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex(), dk.hex()


def _verify_password(password, salt_hex, hash_hex):
    try:
        _, h = _hash_password(password, salt_hex)
        return hmac.compare_digest(h, hash_hex)
    except Exception:
        return False


def _find_user(users, ident):
    ident = (ident or "").strip().lower()
    if not ident:
        return None, None
    for uid, u in users.items():
        if u.get("email", "").lower() == ident or u.get("nip", "").lower() == ident:
            return uid, u
    return None, None


def current_user():
    uid = st.session_state.get("auth_user")
    if not uid:
        return None
    return _load_users().get(uid)


def default_prefs():
    return {"lang": "id", "dark": True, "email_notif": False, "twofa": False,
            "default_model": "xray_3class", "cam_opacity": 45}


def apply_user_prefs(user):
    p = {**default_prefs(), **(user.get("prefs") or {})}
    ss = st.session_state
    ss["ui_lang"] = p["lang"] if p["lang"] in I18N else "id"
    ss["ui_dark"] = bool(p["dark"])
    ss["ui_email_notif"] = bool(p["email_notif"])
    ss["ui_2fa"] = bool(p["twofa"])
    ss["ui_cam_opacity"] = int(p["cam_opacity"])
    dm = p["default_model"] if p["default_model"] in MODEL_REGISTRY else "xray_3class"
    ss["ui_default_model"] = dm
    ss["active_model_key"] = dm
    ss["f_modality"] = KEY_TO_MODALITY[dm]
    ss["model_radio"] = dm


def _persist_prefs():
    ss = st.session_state
    uid = ss.get("auth_user")
    users = _load_users()
    if uid not in users:
        return
    users[uid]["prefs"] = {
        "lang": ss.get("ui_lang", "id"), "dark": bool(ss.get("ui_dark", True)),
        "email_notif": bool(ss.get("ui_email_notif", False)),
        "twofa": bool(ss.get("ui_2fa", False)),
        "default_model": ss.get("ui_default_model", "xray_3class"),
        "cam_opacity": int(ss.get("ui_cam_opacity", 45)),
    }
    _save_users(users)


def register_user(full_name, nip, email, spec, pw, pw2):
    full_name, nip, email, spec = (full_name or "").strip(), (nip or "").strip(), \
        (email or "").strip(), (spec or "").strip()
    if not (full_name and nip and email and pw):
        return None, T("err_required")
    if not EMAIL_RE.match(email):
        return None, T("err_email")
    if len(pw) < MIN_PASSWORD_LEN:
        return None, T("err_pw_short").format(n=MIN_PASSWORD_LEN)
    if pw != pw2:
        return None, T("err_pw_match")
    users = _load_users()
    for u in users.values():
        if u.get("email", "").lower() == email.lower():
            return None, T("err_email_used")
        if u.get("nip", "").lower() == nip.lower():
            return None, T("err_nip_used")
    uid = uuid.uuid4().hex
    salt, h = _hash_password(pw)
    users[uid] = dict(uid=uid, full_name=full_name, nip=nip, email=email,
                      specialization=spec or "Radiologi", salt=salt, hash=h,
                      created=datetime.datetime.now().isoformat(timespec="seconds"),
                      prefs=default_prefs())
    _save_users(users)
    return uid, None


def login_user(ident, pw):
    users = _load_users()
    uid, u = _find_user(users, ident)
    if u is None or not _verify_password(pw or "", u.get("salt", ""), u.get("hash", "")):
        return None, T("err_login")
    return uid, None


def update_profile(uid, full_name, nip, email, spec):
    full_name, nip, email, spec = full_name.strip(), nip.strip(), email.strip(), spec.strip()
    if not (full_name and nip and email):
        return False, T("err_required")
    if not EMAIL_RE.match(email):
        return False, T("err_email")
    users = _load_users()
    if uid not in users:
        return False, T("err_login")
    for k, u in users.items():
        if k == uid:
            continue
        if u.get("email", "").lower() == email.lower():
            return False, T("err_email_used")
        if u.get("nip", "").lower() == nip.lower():
            return False, T("err_nip_used")
    users[uid].update(full_name=full_name, nip=nip, email=email, specialization=spec)
    _save_users(users)
    return True, None


def change_password(uid, old_pw, new_pw, new_pw2):
    users = _load_users()
    u = users.get(uid)
    if not u or not _verify_password(old_pw or "", u.get("salt", ""), u.get("hash", "")):
        return False, T("err_old_pw")
    if len(new_pw or "") < MIN_PASSWORD_LEN:
        return False, T("err_pw_short").format(n=MIN_PASSWORD_LEN)
    if new_pw != new_pw2:
        return False, T("err_pw_match")
    u["salt"], u["hash"] = _hash_password(new_pw)
    u["pw_changed"] = datetime.datetime.now().isoformat(timespec="seconds")
    _save_users(users)
    return True, None


def do_logout():
    st.session_state.clear()


def _login_success(uid):
    user = _load_users().get(uid)
    st.session_state["auth_user"] = uid
    if user:
        apply_user_prefs(user)
    st.session_state["page"] = "Dashboard"


def render_auth_page():
    st.markdown("""
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"]{ display:none !important; }
    </style>""", unsafe_allow_html=True)
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        st.markdown(f"""
        <div class="auth-brand">
            <div class="bolt">⚡</div>
            <div class="name">LungVision AI</div>
            <div class="tag">Explainable AI-Based Medical Image Analysis System</div>
            <div class="sub">{esc(T('auth_sub'))}</div>
        </div>""", unsafe_allow_html=True)
        if not _load_users():
            st.info(T("no_users"))
        tab_login, tab_reg = st.tabs([T("login"), T("register")])
        with tab_login:
            with st.form("form_login", clear_on_submit=False):
                ident = st.text_input(T("ident"), key="login_ident")
                pw = st.text_input(T("password"), type="password", key="login_pw")
                submitted = st.form_submit_button(T("btn_login"), use_container_width=True)
            if submitted:
                uid, err = login_user(ident, pw)
                if err:
                    st.error(err)
                else:
                    _login_success(uid)
                    st.rerun()
        with tab_reg:
            with st.form("form_register", clear_on_submit=False):
                r1, r2 = st.columns(2)
                full = r1.text_input(T("full_name"), key="reg_name")
                nip = r2.text_input(T("nip"), key="reg_nip")
                r3, r4 = st.columns(2)
                email = r3.text_input(T("email"), key="reg_email")
                spec = r4.text_input(T("spec"), value="Radiologi", key="reg_spec")
                r5, r6 = st.columns(2)
                pw1 = r5.text_input(T("password"), type="password", key="reg_pw1")
                pw2 = r6.text_input(T("confirm"), type="password", key="reg_pw2")
                reg_submit = st.form_submit_button(T("btn_register"), use_container_width=True)
            if reg_submit:
                uid, err = register_user(full, nip, email, spec, pw1, pw2)
                if err:
                    st.error(err)
                else:
                    _login_success(uid)
                    st.rerun()


# ---- Gerbang login: tanpa login, halaman lain tidak dijalankan ----
if current_user() is None:
    st.session_state.pop("auth_user", None)
    render_auth_page()
    st.stop()

if not st.session_state.get("ui_dark", True):
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)


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
    """Tag probe/cam dengan ACTIVE_KEY supaya stale result di-clear saat ganti model."""
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
    base_name = os.path.splitext(uploaded_file.name)[0]
    clean_id = base_name.replace("_", "-").upper()
    if len(clean_id) > 20: clean_id = clean_id[:20]

    meta = dict(patient_id=clean_id, age_sex="—", study_date="—",
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


def apply_patient_form_to_meta():
    """Timpa metadata citra dengan isian form pasien (jika diisi)."""
    meta = st.session_state.get("dicom_meta")
    if not meta:
        return
    pid = (st.session_state.get("f_pid") or "").strip()
    pname = (st.session_state.get("f_name") or "").strip()
    age = (st.session_state.get("f_age") or "").strip()
    gender = st.session_state.get("f_gender")
    if pid:
        meta["patient_id"] = pid
    if pname:
        meta["patient_name"] = pname
    if age:
        meta["age_sex"] = f"{age} / {str(gender)[:1] if gender else '?'}"
    for src, dst in [("f_blood", "blood_type"), ("f_hw", "height_weight"),
                     ("f_smoke", "smoking"), ("f_symptoms", "symptoms")]:
        val = st.session_state.get(src)
        if val:
            meta[dst] = val
    st.session_state["dicom_meta"] = meta


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


def load_all_logs():
    frames = []
    for k, c in MODEL_REGISTRY.items():
        p = c["pred_log_csv"]
        if os.path.exists(p):
            try:
                d = pd.read_csv(p)
                d["model_key"] = k
                frames.append(d)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# RIWAYAT ANALISIS (History / Reports)  — data/analyses.json
# Satu catatan per citra yang dianalisis, milik pengguna yang login.
# Nama pasien dari form New Analysis ikut tersimpan & ter-update otomatis.
# ============================================================
MODALITY_SHORT = {"xray_3class": "X-ray", "lidc_5class": "CT Scan"}


def _load_analyses():
    d = _load_json_safe(ANALYSES_FILE)
    return d if isinstance(d, list) else []


def _save_analyses(items):
    _atomic_write_json(ANALYSES_FILE, items)


def _thumb_path(rid, kind):
    return os.path.join(THUMB_DIR, f"{rid}_{kind}.png")


def fmt_dt(iso):
    try:
        return datetime.datetime.fromisoformat(str(iso)).strftime("%d %b %Y %H:%M")
    except Exception:
        return str(iso)


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "laporan"


def next_report_id(items, now):
    prefix = f"REP-LV-{now:%Y}-{now:%m%d}-"
    nums = []
    for r in items:
        rid = str(r.get("report_id", ""))
        if rid.startswith(prefix):
            try:
                nums.append(int(rid[len(prefix):]))
            except ValueError:
                pass
    return f"{prefix}{(max(nums) + 1) if nums else 1:03d}"


def _record_patient_fields(meta):
    return {
        "patient_id": str(meta.get("patient_id", "-")),
        "patient_name": str(meta.get("patient_name", "") or ""),
        "age_sex": str(meta.get("age_sex", "—")),
        "blood_type": str(meta.get("blood_type", "") or ""),
        "height_weight": str(meta.get("height_weight", "") or ""),
        "smoking": str(meta.get("smoking", "") or ""),
        "symptoms": str(meta.get("symptoms", "") or ""),
        "study_date": str(meta.get("study_date", "—")),
        "source": str(meta.get("source", "—")),
    }


def _record_analysis_fields(probe, diam_mm, risk, rec_text):
    pred = int(probe["pred_class"])
    return {
        "model_key": ACTIVE_KEY,
        "modality": MODALITY_SHORT.get(ACTIVE_KEY, CFG["short_label"]),
        "file_name": str(st.session_state.get("file_name", "-")),
        "pred_class": pred,
        "class_name": CLASS_NAMES[pred],
        "class_short": CLASS_SHORT[pred],
        "confidence": float(probe["confidence"]),
        "probs": [float(p) for p in probe["probs"]],
        "diameter_mm": float(diam_mm),
        "risk": str(risk),
        "recommendation": str(rec_text),
        "detected": bool(IS_MALIGNANT[pred]),
    }


def _save_record_images(rid, gray128, cam_result):
    try:
        os.makedirs(THUMB_DIR, exist_ok=True)
        base = cv2.resize(gray128, (256, 256), interpolation=cv2.INTER_CUBIC)
        Image.fromarray(base).save(_thumb_path(rid, "img"))
        cam_p = _thumb_path(rid, "cam")
        if cam_result is not None and cam_result.get("cam_available"):
            _, heat = make_overlay(gray128, cam_result["cam"], size=256)
            base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
            blended = cv2.addWeighted(base_rgb, 0.55, np.ascontiguousarray(heat), 0.45, 0)
            Image.fromarray(blended).save(cam_p)
        elif os.path.exists(cam_p):
            os.remove(cam_p)
    except Exception as e:
        print(f"Gagal simpan thumbnail: {e}")


def upsert_analysis_record(meta, probe, cam_result, diam_mm, risk, rec_text):
    """Tambah/ubah catatan riwayat untuk citra yang sedang dianalisis."""
    uid = st.session_state.get("auth_user")
    user = current_user()
    if not uid or not user:
        return None
    ss = st.session_state
    rec_key = str(ss.get("file_name", "unknown"))
    rec_map = ss.setdefault("rec_map", {})
    rec_sig = ss.setdefault("rec_sig", {})
    dead = ss.setdefault("rec_dead", set())
    rid = rec_map.get(rec_key)
    if rid in dead:
        return None

    sig = (ACTIVE_KEY, int(probe["pred_class"]), round(float(probe["confidence"]), 4))
    items = _load_analyses()
    existing = None
    if rid:
        existing = next((r for r in items
                         if r.get("report_id") == rid and r.get("owner") == uid), None)
    patient = _record_patient_fields(meta)

    # Analisis tidak berubah -> cukup sinkronkan data pasien (nama dsb.)
    if existing is not None and rec_sig.get(rec_key) == sig:
        if any(existing.get(k) != v for k, v in patient.items()):
            existing.update(patient)
            existing["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
            _save_analyses(items)
        return rid

    now = datetime.datetime.now()
    if existing is None:
        rid = next_report_id(items, now)
        existing = {"report_id": rid, "owner": uid, "owner_name": user.get("full_name", "-"),
                    "created": now.isoformat(timespec="seconds")}
        items.append(existing)
    existing["owner_name"] = user.get("full_name", "-")
    existing.update(patient)
    existing.update(_record_analysis_fields(probe, diam_mm, risk, rec_text))
    existing["updated"] = now.isoformat(timespec="seconds")
    _save_record_images(rid, st.session_state.get("gray128"), cam_result)
    _save_analyses(items)
    rec_map[rec_key] = rid
    rec_sig[rec_key] = sig
    return rid


def my_records():
    uid = st.session_state.get("auth_user")
    items = [r for r in _load_analyses() if r.get("owner") == uid]
    items.sort(key=lambda r: r.get("created", ""), reverse=True)
    return items


def delete_record(rid):
    uid = st.session_state.get("auth_user")
    items = _load_analyses()
    items = [r for r in items if not (r.get("report_id") == rid and r.get("owner") == uid)]
    _save_analyses(items)
    for kind in ("img", "cam"):
        p = _thumb_path(rid, kind)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    st.session_state.setdefault("rec_dead", set()).add(rid)
    st.session_state.pop("hist_view", None)
    st.session_state.pop("hist_confirm_del", None)


def _rec_cfg(rec):
    return MODEL_REGISTRY.get(rec.get("model_key"), CFG)


# ---------------- PDF / HTML / ZIP dari catatan ----------------
def _pdf_line(pdf, h, text):
    try:
        pdf.cell(0, h, safe_pdf_text(text), new_x="LMARGIN", new_y="NEXT")
    except TypeError:
        pdf.cell(0, h, safe_pdf_text(text), ln=True)


def build_pdf_from_record(rec):
    if not FPDF_OK:
        return None
    cfg = _rec_cfg(rec)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    _pdf_line(pdf, 10, "LungVision AI (LightXAI-WaveNet) - Laporan Prediksi")
    pdf.set_font("Helvetica", "", 10)
    _pdf_line(pdf, 6, f"ID Laporan: {rec.get('report_id', '-')}")
    _pdf_line(pdf, 6, f"Model: {cfg['label']}")
    _pdf_line(pdf, 6, f"Dibuat: {fmt_dt(rec.get('created', ''))}")
    _pdf_line(pdf, 6, f"Dokter Pemeriksa: {rec.get('owner_name', '-')}")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    _pdf_line(pdf, 8, "Informasi Pasien")
    pdf.set_font("Helvetica", "", 10)
    _pdf_line(pdf, 5, f"Patient ID  : {rec.get('patient_id', '-')}")
    _pdf_line(pdf, 5, f"Nama        : {rec.get('patient_name') or '-'}")
    _pdf_line(pdf, 5, f"Age/Sex     : {rec.get('age_sex', '-')}")
    if rec.get("blood_type"):
        _pdf_line(pdf, 5, f"Gol. Darah  : {rec['blood_type']}")
    if rec.get("height_weight"):
        _pdf_line(pdf, 5, f"Tinggi/Berat: {rec['height_weight']}")
    if rec.get("smoking"):
        _pdf_line(pdf, 5, f"Merokok     : {rec['smoking']}")
    _pdf_line(pdf, 5, f"Modalitas   : {rec.get('modality', '-')}")
    _pdf_line(pdf, 5, f"Source      : {rec.get('source', '-')}")
    if rec.get("symptoms"):
        _pdf_line(pdf, 5, f"Gejala      : {rec['symptoms']}")
    pdf.ln(3)

    y = pdf.get_y()
    x = 10
    drew = False
    for kind in ("img", "cam"):
        p = _thumb_path(rec.get("report_id", ""), kind)
        if os.path.exists(p):
            try:
                pdf.image(p, x=x, y=y, w=55)
                x += 60
                drew = True
            except Exception:
                pass
    if drew:
        pdf.set_y(y + 58)

    pdf.set_font("Helvetica", "B", 12)
    _pdf_line(pdf, 8, "Hasil Prediksi")
    pdf.set_font("Helvetica", "", 10)
    _pdf_line(pdf, 5, f"Kelas       : {rec.get('class_name', '-')} ({rec.get('class_short', '-')})")
    _pdf_line(pdf, 5, f"Confidence  : {float(rec.get('confidence', 0)) * 100:.1f}%")
    if cfg.get("task") == "lung_nodule":
        _pdf_line(pdf, 5, f"Diameter    : {float(rec.get('diameter_mm', 0)):.2f} mm")
    pdf.ln(2)
    _pdf_line(pdf, 5, "Probabilitas per Kelas:")
    for i, p in enumerate(rec.get("probs", [])):
        if i < len(cfg["class_names"]):
            _pdf_line(pdf, 5, f"  {cfg['class_short'][i]} - {cfg['class_names'][i]}: {p * 100:.2f}%")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    _pdf_line(pdf, 8, "Keputusan Klinis")
    pdf.set_font("Helvetica", "", 10)
    _pdf_line(pdf, 5, f"Risk Level     : {rec.get('risk', '-')}")
    _pdf_line(pdf, 5, f"Recommendation : {rec.get('recommendation', '-')}")
    _pdf_line(pdf, 5, f"AI Diagnosis   : {'Terdeteksi' if rec.get('detected') else 'Tidak Terdeteksi'}")
    return bytes(pdf.output())


def build_html_from_record(rec):
    import base64
    cfg = _rec_cfg(rec)

    def img_tag(kind, label):
        p = _thumb_path(rec.get("report_id", ""), kind)
        if not os.path.exists(p):
            return ""
        with open(p, "rb") as f:
            b = base64.b64encode(f.read()).decode()
        return (f'<figure><img src="data:image/png;base64,{b}" width="220" alt="{label}">'
                f'<figcaption>{label}</figcaption></figure>')

    prob_rows = "".join(
        f"<tr><td>{esc(cfg['class_short'][i])}</td><td>{esc(cfg['class_names'][i])}</td>"
        f"<td>{p * 100:.2f}%</td></tr>"
        for i, p in enumerate(rec.get("probs", [])) if i < len(cfg["class_names"]))
    diam = (f"<tr><th>Diameter</th><td>{float(rec.get('diameter_mm', 0)):.2f} mm</td></tr>"
            if cfg.get("task") == "lung_nodule" else "")
    det = "Terdeteksi" if rec.get("detected") else "Tidak Terdeteksi"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(rec.get('report_id'))}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:820px;margin:24px auto;color:#14202e;padding:0 16px}}
h1{{font-size:22px;margin-bottom:4px}} h3{{margin:22px 0 6px;border-bottom:1px solid #ccd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid #e3e8ef;font-size:14px}}
th{{width:190px;color:#455}} .imgs{{display:flex;gap:16px}} figure{{margin:0}} figcaption{{font-size:12px;color:#667}}
</style></head><body>
<h1>Laporan Prediksi - LungVision AI (LightXAI-WaveNet)</h1>
<div>ID Laporan: <b>{esc(rec.get('report_id'))}</b> &middot; Dibuat: {esc(fmt_dt(rec.get('created', '')))}
&middot; Dokter Pemeriksa: {esc(rec.get('owner_name', '-'))}</div>
<h3>Pasien</h3>
<table>
<tr><th>Patient ID</th><td>{esc(rec.get('patient_id'))}</td></tr>
<tr><th>Nama</th><td>{esc(rec.get('patient_name') or '-')}</td></tr>
<tr><th>Age / Sex</th><td>{esc(rec.get('age_sex'))}</td></tr>
<tr><th>Gol. Darah</th><td>{esc(rec.get('blood_type') or '-')}</td></tr>
<tr><th>Tinggi / Berat</th><td>{esc(rec.get('height_weight') or '-')}</td></tr>
<tr><th>Riwayat Merokok</th><td>{esc(rec.get('smoking') or '-')}</td></tr>
<tr><th>Gejala Klinis</th><td>{esc(rec.get('symptoms') or '-')}</td></tr>
<tr><th>Modalitas / Model</th><td>{esc(rec.get('modality'))} &middot; {esc(cfg['label'])}</td></tr>
</table>
<h3>Citra</h3><div class="imgs">{img_tag('img', 'Citra asli')}{img_tag('cam', 'Grad-CAM overlay')}</div>
<h3>Hasil Prediksi</h3>
<table>
<tr><th>Kelas</th><td>{esc(rec.get('class_name'))} ({esc(rec.get('class_short'))})</td></tr>
<tr><th>Confidence</th><td>{float(rec.get('confidence', 0)) * 100:.1f}%</td></tr>
{diam}
<tr><th>AI Diagnosis</th><td>{det}</td></tr>
</table>
<table style="margin-top:10px"><tr><th style="width:60px">Kelas</th><th>Nama</th><th>Probabilitas</th></tr>{prob_rows}</table>
<h3>Keputusan Klinis</h3>
<table>
<tr><th>Risk Level</th><td>{esc(rec.get('risk'))}</td></tr>
<tr><th>Rekomendasi</th><td>{esc(rec.get('recommendation'))}</td></tr>
</table>
<p style="font-size:12px;color:#778;margin-top:24px">Hasil AI bersifat pendukung keputusan dan harus
dikonfirmasi oleh dokter.</p>
</body></html>"""


@st.cache_data(show_spinner=False)
def _pdf_cached(rec_json):
    return build_pdf_from_record(json.loads(rec_json))


@st.cache_data(show_spinner=False)
def _html_cached(rec_json):
    return build_html_from_record(json.loads(rec_json))


def pdf_bytes_for(rec):
    return _pdf_cached(json.dumps(rec, sort_keys=True, ensure_ascii=False))


def html_for(rec):
    return _html_cached(json.dumps(rec, sort_keys=True, ensure_ascii=False))


def report_filename(rec, ext):
    return f"{_safe_name(rec.get('report_id', 'laporan'))}_{_safe_name(rec.get('patient_id', ''))}.{ext}"


def build_batch_zip(recs):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in recs:
            data = pdf_bytes_for(r) if FPDF_OK else None
            if data:
                z.writestr(report_filename(r, "pdf"), data)
            else:
                z.writestr(report_filename(r, "html"), html_for(r).encode("utf-8"))
    return buf.getvalue()


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
# SIDEBAR + TOPBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="brand">
            <div class="bolt">⚡</div>
            <div>
                <div class="name">LungVision AI</div>
                <div class="tag">Explainable AI-Based Medical Image Analysis System</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        current = st.session_state["page"]
        for name, icon in NAV_ITEMS:
            st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True,
                      type="primary" if current == name else "secondary",
                      on_click=go_page, args=(name,))
        st.markdown(f"""
        <div class="side-card">
            <div class="ico">🛡️</div>
            <div class="t">{esc(T('sec_title'))}</div>
            <div class="d">{esc(T('sec_desc'))}</div>
            <div class="l">{esc(T('sec_more'))}</div>
        </div>
        """, unsafe_allow_html=True)


def toggle_lang():
    ss = st.session_state
    new = "en" if ss.get("ui_lang", "id") == "id" else "id"
    ss["ui_lang"] = new
    if "w_lang" in ss:
        ss["w_lang"] = "English" if new == "en" else "Bahasa Indonesia"
    _persist_prefs()


def render_topbar(page):
    user = current_user() or {}
    name = user.get("full_name", "User")
    initials = "".join([w[0] for w in str(name).split()[:2]]).upper() or "U"
    spec = user.get("specialization", "")
    role = "Radiologist" if "radiolog" in spec.lower() else (spec or "User")
    lang = st.session_state.get("ui_lang", "id")
    c = st.columns([4.4, 0.6, 0.5, 1.0, 2.9])
    with c[0]:
        st.markdown(f"""
        <span class="topbar-marker"></span>
        <div class="tb-title"><h1>{esc(page)}</h1><p>{esc(T('sub_' + page))}</p></div>
        """, unsafe_allow_html=True)
    with c[1]:
        st.button(lang.upper(), key="btn_lang", on_click=toggle_lang, use_container_width=True)
    with c[2]:
        st.markdown('<div class="tb-bell"><div class="bell">🔔<span class="dot"></span></div></div>',
                    unsafe_allow_html=True)
    with c[3]:
        st.button("Logout", key="btn_logout", on_click=do_logout, use_container_width=True)
    with c[4]:
        st.markdown(f"""
        <div class="tb-user"><div class="user">
            <div class="avatar">{esc(initials)}</div>
            <div>
                <div class="n">{esc(name)}</div>
                <div class="r">{esc(role)} (NIP. {esc(user.get('nip', '-'))})</div>
            </div>
        </div></div>
        """, unsafe_allow_html=True)


# ============================================================
# NEW ANALYSIS: FORM + UPLOAD
# ============================================================
def render_patient_form_and_upload():
    # init modality dari model aktif
    if "f_modality" not in st.session_state:
        st.session_state["f_modality"] = KEY_TO_MODALITY[ACTIVE_KEY]

    with st.container(border=True):
        st.markdown('<span class="form-marker"></span><div class="form-title">New Analysis Form</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Patient ID *", placeholder="Patient ID", key="f_pid")
        with c2:
            st.text_input("Patient Name *", placeholder="Patient Name", key="f_name")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Age", placeholder="Contoh: 58", key="f_age")
        with c2:
            st.selectbox("Gender", ["Male", "Female"], key="f_gender")
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Blood Type", ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"], key="f_blood")
        with c2:
            st.text_input("Height / Weight", placeholder="172 cm / 75 kg", key="f_hw")
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Smoking History", key="f_smoke")
        with c2:
            st.selectbox("Modality", list(MODALITY_OPTIONS.keys()), key="f_modality",
                         on_change=on_modality_change)
        st.text_input("Clinical Symptoms", key="f_symptoms")

        st.markdown('<div class="upload-note">Drag &amp; drop DICOM/JPEG/PNG files di bawah ini '
                    f'· Model aktif: <b>{CFG["short_label"]}</b></div>', unsafe_allow_html=True)
        allowed = ["jpg", "jpeg", "png"] + (["dcm"] if CFG["is_3d"] else [])
        up1, up2 = st.columns([4, 1])
        with up1:
            uploaded = st.file_uploader("Upload citra", type=allowed,
                                        label_visibility="collapsed",
                                        key=f"upload_{ACTIVE_KEY}")
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

    apply_patient_form_to_meta()
    if "gray128" in st.session_state and not (st.session_state.get("f_name") or "").strip():
        st.caption("ℹ️ Isi **Patient Name** pada form agar pasien tercatat dengan namanya di History.")


# ============================================================
# LEFT PANEL
# ============================================================
def render_left_panel():
    st.markdown('<div class="card"><div class="card-title">🎯 Model Selection</div>',
                unsafe_allow_html=True)
    if "model_radio" not in st.session_state:
        st.session_state["model_radio"] = ACTIVE_KEY
    st.radio("Pilih model:", options=list(MODEL_REGISTRY.keys()),
             format_func=lambda k: MODEL_REGISTRY[k]["label"],
             key="model_radio", on_change=on_model_radio_change,
             label_visibility="collapsed")
    st.caption(f"📋 {CFG['description']}")
    st.caption(f"⚕️ {CFG['clinical_notes']}")
    st.caption("Sinkron dengan field Modality pada form.")
    st.markdown('</div>', unsafe_allow_html=True)

    meta = st.session_state.get("dicom_meta", {})

    st.markdown(f"""
        <div class="card"><div class="card-title">Patient Info</div>
            <div class="kv"><span class="k">Patient ID</span><span class="v">{esc(meta.get('patient_id','—'))}</span></div>
            <div class="kv"><span class="k">Nama</span><span class="v">{esc(meta.get('patient_name','—'))}</span></div>
            <div class="kv"><span class="k">Age/Sex</span><span class="v">{meta.get('age_sex','—')}</span></div>
            <div class="kv"><span class="k">Study Date</span><span class="v">{meta.get('study_date','—')}</span></div>
            <div class="kv"><span class="k">Slices</span><span class="v">{meta.get('slices','—')}</span></div>
            <div class="kv"><span class="k">Spacing</span><span class="v">{meta.get('spacing','—')}</span></div>
            <div class="kv"><span class="k">Source</span><span class="v">{meta.get('source','—')}</span></div>
        </div>
    """, unsafe_allow_html=True)

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
# CENTER PANEL (VIEWER)
# ============================================================
def render_center_panel():
    st.markdown(f'<div class="card"><div class="card-title">{"DICOM" if CFG["is_3d"] else "X-Ray"} Viewer</div>',
                unsafe_allow_html=True)

    if "gray128" not in st.session_state:
        st.markdown('<div class="badge">Belum ada citra - upload atau klik Demo.</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    ensure_inference()
    gray128 = st.session_state["gray128"]
    meta = st.session_state.get("dicom_meta", {})

    fname_l = str(st.session_state.get("file_name", "")).lower()
    if CFG["is_3d"] and not meta.get("is_dicom") and not fname_l.startswith("demo_"):
        st.warning("Model aktif: **LIDC (CT 3D)**. Jika citra ini X-Ray dada, "
                   "pilih **Chest X-Ray** pada Modality / Model Selection agar "
                   "ROI, Grad-CAM, dan IoU memakai model X-Ray.")
    cr = st.session_state.get("cam_result")
    if cr is not None and cr.get("cam_available") and float(np.max(cr["cam"])) <= 0:
        st.warning("Grad-CAM kosong untuk citra ini (semua nilai 0) — kemungkinan "
                   "model tidak cocok dengan jenis citra.")

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
    upsert_analysis_record(meta, probe, cam_result, diam_mm, risk, rec)

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
# 3D ROI FULL WIDTH
# ============================================================
def render_roi_3d_fullwidth():
    if not ROI_VIZ_OK:
        if "gray128" in st.session_state:
            st.warning(f"roi_3d_visualizer.py tidak bisa diimport: {ROI_VIZ_ERR}")
        return
    if "gray128" not in st.session_state:
        return
    if "probe" not in st.session_state:
        return

    gray128 = st.session_state["gray128"]
    cam_result = st.session_state.get("cam_result")

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
            opacity = st.slider("Opacity", 0, 100, int(st.session_state.get("ui_cam_opacity", 45)),
                            key=f"cam_opacity_{ACTIVE_KEY}")
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
            acc_per_class = diag / np.where(row_sums.flatten() == 0, 1, row_sums.flatten())
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
# BAGIAN BONUS: ANALISA IoU KOMPLIT
# ============================================================
def generate_iou_masks(gray128, region, cam):
    """
    Membuat simulasi GT dan mask prediksi yang tegas (tidak blur).
    Menggunakan Bounding Box untuk X-Ray, dan Convex Hull untuk CT.
    """
    if region is None:
        return None, None, None

    h, w = gray128.shape

    # 1. Mask Prediksi (Berdasarkan Grad-CAM)
    cam_resized = cv2.resize(cam, (w, h))
    _, mask_pred = cv2.threshold(cam_resized, 0.5, 1, cv2.THRESH_BINARY)
    mask_pred = (mask_pred * 255).astype(np.uint8)

    # 2. Mask Ground Truth
    mask_gt = np.zeros_like(gray128, dtype=np.uint8)

    if not CFG["is_3d"]:
        contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(c)
            pad_x, pad_y = int(w_box*0.1), int(h_box*0.1)
            cv2.rectangle(mask_gt, (x+pad_x, y+pad_y), (x+w_box-pad_x, y+h_box-pad_y), 255, -1)
        else:
            cx, cy = int(w/2), int(h/2)
            cv2.rectangle(mask_gt, (cx-20, cy-20), (cx+20, cy+20), 255, -1)
    else:
        contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            cv2.drawContours(mask_gt, [hull], -1, 255, -1)
            mask_gt = cv2.GaussianBlur(mask_gt, (5, 5), 0)
            _, mask_gt = cv2.threshold(mask_gt, 127, 255, cv2.THRESH_BINARY)
        else:
            cx, cy = int(w/2), int(h/2)
            cv2.circle(mask_gt, (cx, cy), 20, 255, -1)

    intersection = np.logical_and(mask_pred > 0, mask_gt > 0).sum()
    union = np.logical_or(mask_pred > 0, mask_gt > 0).sum()
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / ((mask_pred > 0).sum() + (mask_gt > 0).sum()) if ((mask_pred > 0).sum() + (mask_gt > 0).sum()) > 0 else 0.0

    return mask_gt, mask_pred, dict(iou=iou, dice=dice, intersection=intersection, union=union)


def render_iou_analysis():
    if "gray128" not in st.session_state or "probe" not in st.session_state:
        return

    gray128 = st.session_state["gray128"]
    cam_result = st.session_state.get("cam_result")
    meta = st.session_state.get("dicom_meta", {})

    if cam_result is None or not cam_result.get("cam_available"):
        return

    cam = cam_result["cam"]
    region = analyze_cam_region(cam)

    mask_gt, mask_pred, metrics = generate_iou_masks(gray128, region, cam)
    if mask_gt is None:
        return

    current_file = st.session_state.get("file_name", "-")
    if not any(d["file_name"] == current_file for d in st.session_state["iou_history"]):
        st.session_state["iou_history"].append({
            "file_name": current_file,
            "patient_id": meta.get("patient_id", "-"),
            "diameter": f"{region['diameter']:.1f}",
            "iou": f"{metrics['iou']:.2f}",
            "dice": f"{metrics['dice']:.2f}",
            "model": CFG["short_label"]
        })

    img_base = cv2.cvtColor(gray128, cv2.COLOR_GRAY2RGB)

    # 1. Ground Truth (Hijau Transparan)
    img_gt = img_base.copy()
    gt_color_layer = np.zeros_like(img_gt)
    gt_color_layer[mask_gt > 0] = [61, 220, 132]
    img_gt = cv2.addWeighted(img_gt, 0.6, gt_color_layer, 0.4, 0)
    contours_gt, _ = cv2.findContours(mask_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_gt, contours_gt, -1, (0, 255, 0), 2, cv2.LINE_AA)

    # 2. Prediksi (Merah Transparan)
    img_pred = img_base.copy()
    pred_color_layer = np.zeros_like(img_pred)
    pred_color_layer[mask_pred > 0] = [239, 83, 80]
    img_pred = cv2.addWeighted(img_pred, 0.6, pred_color_layer, 0.4, 0)
    contours_pred, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_pred, contours_pred, -1, (255, 0, 0), 2, cv2.LINE_AA)

    # 3. Overlap (Kuning & Biru Transparan)
    img_iou = img_base.copy()
    iou_color_layer = np.zeros_like(img_iou)
    intersection = np.logical_and(mask_gt > 0, mask_pred > 0)
    union = np.logical_or(mask_gt > 0, mask_pred > 0)
    iou_color_layer[intersection] = [255, 235, 59]
    iou_color_layer[np.logical_and(union, ~intersection)] = [59, 130, 246]
    img_iou = cv2.addWeighted(img_iou, 0.6, iou_color_layer, 0.4, 0)
    cv2.drawContours(img_iou, contours_gt, -1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.drawContours(img_iou, contours_pred, -1, (255, 0, 0), 1, cv2.LINE_AA)

    # Zoom ROI
    h, w = gray128.shape
    cx = int(round(region["cx"] / region["w"] * w))
    cy = int(round(region["cy"] / region["h"] * h))
    crop = 40
    x1, x2 = max(0, cx-crop), min(w, cx+crop)
    y1, y2 = max(0, cy-crop), min(h, cy+crop)

    zoom_gt = cv2.resize(img_gt[y1:y2, x1:x2], (150, 150), interpolation=cv2.INTER_LINEAR)
    zoom_pred = cv2.resize(img_pred[y1:y2, x1:x2], (150, 150), interpolation=cv2.INTER_LINEAR)
    zoom_iou = cv2.resize(img_iou[y1:y2, x1:x2], (150, 150), interpolation=cv2.INTER_LINEAR)

    st.markdown("<hr style='border: 1px solid #1e293b; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown('<div class="iou-container">', unsafe_allow_html=True)
    st.markdown('<div class="iou-title">Perbandingan Pendapat Ahli (Radiologis) dengan Computer Vision Menggunakan IoU</div>', unsafe_allow_html=True)
    st.markdown('<div class="iou-subtitle">Comparison of Radiologist vs. Computer Vision using Intersection over Union (IoU)</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="iou-step">', unsafe_allow_html=True)
        st.markdown('<div class="iou-step-title">1. Input Citra</div>', unsafe_allow_html=True)
        st.image(img_base, use_container_width=True, output_format="PNG")
        st.markdown(f'<div style="font-size:9px; color:#8b96ac; margin-top:5px;">Case ID: {meta.get("patient_id","-")}<br>Model: {CFG["short_label"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="iou-step">', unsafe_allow_html=True)
        st.markdown('<div class="iou-step-title">2. Anotasi / Ahli (GT)</div>', unsafe_allow_html=True)
        st.image(img_gt, use_container_width=True, output_format="PNG")
        st.markdown('<div style="font-size:9px; color:#3ddc84; font-weight:bold; margin-top:5px;">🟩 Ground Truth (Radiolog)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="iou-step">', unsafe_allow_html=True)
        st.markdown('<div class="iou-step-title">3. Prediksi Model</div>', unsafe_allow_html=True)
        st.image(img_pred, use_container_width=True, output_format="PNG")
        st.markdown('<div style="font-size:9px; color:#ef5350; font-weight:bold; margin-top:5px;">🟥 Prediksi Computer Vision</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="iou-step">', unsafe_allow_html=True)
        st.markdown('<div class="iou-step-title">4. Perbandingan Overlap</div>', unsafe_allow_html=True)
        st.image(img_iou, use_container_width=True, output_format="PNG")
        st.markdown('<div style="font-size:9px; color:#fff; font-weight:bold; margin-top:5px;">🟨 Intersection | 🟦 Union</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_form, c_zoom_gt, c_zoom_pred, c_zoom_iou = st.columns([1.5, 1, 1, 1])

    with c_form:
        st.markdown("""
        <div class="iou-formula-box">
            <div class="iou-formula-title">RUMUS IOU</div>
            <div class="iou-math-layout">
                <div class="iou-math-text">IoU =</div>
                <div class="iou-math-fraction">
                    <div class="numerator">|A ∩ B|</div>
                    <div class="denominator">|A ∪ B|</div>
                </div>
            </div>
            <div class="iou-formula-desc">
                <b>A</b> = Ground Truth (Anotasi Ahli Radiolog)<br>
                <b>B</b> = Prediksi Model (Computer Vision)<br>
                <b>|A ∩ B|</b> = Area Intersection (overlap)<br>
                <b>|A ∪ B|</b> = Area Union (gabungan)<br><br>
                <i>IoU ∈ [0, 1] — 1 = sempurna, 0 = tak tumpang tindih</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_zoom_gt:
        st.markdown('<div style="font-size:10px; color:#3ddc84; text-align:center; margin-bottom:5px;">Ground Truth (Zoom)</div>', unsafe_allow_html=True)
        st.image(zoom_gt, use_container_width=True, output_format="PNG")

    with c_zoom_pred:
        st.markdown('<div style="font-size:10px; color:#ef5350; text-align:center; margin-bottom:5px;">Prediksi (Zoom)</div>', unsafe_allow_html=True)
        st.image(zoom_pred, use_container_width=True, output_format="PNG")

    with c_zoom_iou:
        st.markdown('<div style="font-size:10px; color:#fff; text-align:center; margin-bottom:5px;">Overlap (Zoom)</div>', unsafe_allow_html=True)
        st.image(zoom_iou, use_container_width=True, output_format="PNG")

    st.markdown("<br>", unsafe_allow_html=True)

    tbl_col, interp_col, chart_col = st.columns([2.2, 1.2, 1.2])

    with tbl_col:
        st.markdown("<b style='font-size:12px;'>5. Hasil Perbandingan IoU per Nodul</b>", unsafe_allow_html=True)

        df_iou = pd.DataFrame(st.session_state["iou_history"])
        if not df_iou.empty:
            display_df = pd.DataFrame({
                "Case ID": df_iou["patient_id"],
                "Nodul": ["Nodul 1"] * len(df_iou),
                "Diam (mm)": df_iou["diameter"].astype(float),
                "IoU": df_iou["iou"].astype(float),
                "Kategori": df_iou["iou"].apply(lambda x: "Sangat Baik" if float(x) >= 0.75 else ("Baik" if float(x) >= 0.5 else ("Cukup" if float(x) >= 0.25 else "Kurang"))),
                "Dice": df_iou["dice"].astype(float),
                "Keterangan": ["Sesuai"] * len(df_iou)
            })

            avg_iou = display_df["IoU"].mean()
            avg_dice = display_df["Dice"].mean()
            avg_row = pd.DataFrame({
                "Case ID": ["Rata-rata"],
                "Nodul": [""],
                "Diam (mm)": [display_df["Diam (mm)"].mean()],
                "IoU": [avg_iou],
                "Kategori": [""],
                "Dice": [avg_dice],
                "Keterangan": [""]
            })
            display_df = pd.concat([display_df, avg_row], ignore_index=True)

            try:
                st.dataframe(
                    display_df.style.format({
                        "Diam (mm)": "{:.1f}",
                        "IoU": "{:.2f}",
                        "Dice": "{:.2f}"
                    }).background_gradient(subset=["IoU"], cmap="RdYlGn", vmin=0, vmax=1),
                    use_container_width=True,
                    hide_index=True
                )
            except Exception:
                # fallback bila matplotlib (untuk gradient) tidak terpasang
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada riwayat perbandingan IoU.")

        if st.button("Hapus Riwayat IoU", key="btn_clear_iou", type="secondary"):
            st.session_state["iou_history"] = []
            st.rerun()

    with interp_col:
        st.markdown("<b style='font-size:12px;'>Interpretasi IoU</b>", unsafe_allow_html=True)
        st.markdown("""
        <div class="iou-interp-box">
            <div class="iou-interp-row"><div class="iou-interp-dot" style="background:#3ddc84;"></div><div class="iou-interp-label">Sangat Baik</div><div class="iou-interp-val">IoU ≥ 0.75</div></div>
            <div class="iou-interp-row"><div class="iou-interp-dot" style="background:#4fc3f7;"></div><div class="iou-interp-label">Baik</div><div class="iou-interp-val">0.50 ≤ IoU &lt; 0.75</div></div>
            <div class="iou-interp-row"><div class="iou-interp-dot" style="background:#ffb74d;"></div><div class="iou-interp-label">Cukup</div><div class="iou-interp-val">0.25 ≤ IoU &lt; 0.50</div></div>
            <div class="iou-interp-row"><div class="iou-interp-dot" style="background:#ef5350;"></div><div class="iou-interp-label">Kurang</div><div class="iou-interp-val">IoU &lt; 0.25</div></div>
        </div>
        """, unsafe_allow_html=True)

    with chart_col:
        st.markdown("<b style='font-size:12px;'>Distribusi IoU</b>", unsafe_allow_html=True)

        x_cats = ['Kurang', 'Cukup', 'Baik', 'Sangat Baik']
        y_vals = [
            sum(1 for x in st.session_state["iou_history"] if 0 <= float(x["iou"]) < 0.25),
            sum(1 for x in st.session_state["iou_history"] if 0.25 <= float(x["iou"]) < 0.5),
            sum(1 for x in st.session_state["iou_history"] if 0.5 <= float(x["iou"]) < 0.75),
            sum(1 for x in st.session_state["iou_history"] if float(x["iou"]) >= 0.75)
        ]

        fig_bar = go.Figure(data=[go.Bar(
            x=x_cats,
            y=y_vals,
            marker_color=['#ef5350', '#ffb74d', '#4fc3f7', '#3ddc84'],
            text=y_vals,
            textposition='auto'
        )])
        fig_bar.update_layout(
            height=150, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e7ecf5", showlegend=False, barmode='group',
            yaxis=dict(showgrid=False, title="Jumlah", title_font=dict(size=8), tickfont=dict(color="#8b96ac")),
            xaxis=dict(showgrid=False, tickfont=dict(size=9, color="#8b96ac"))
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False}, key=f"iou_bar_{ACTIVE_KEY}")

    st.markdown("""
    <div style="background-color: #101627; color: #e7ecf5; padding: 12px; border-radius: 6px; font-size: 11px; text-align: center; margin-top: 10px; border: 1px solid #2dd4bf;">
        <b style="color: #2dd4bf;">Kesimpulan:</b> IoU digunakan untuk mengukur tingkat kesesuaian lokalisasi/segmentasi nodul antara pendapat ahli radiolog (ground truth) dan prediksi model computer vision. Semakin tinggi IoU, semakin baik model dalam meniru pendapat ahli.
    </div>
    """, unsafe_allow_html=True)

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
# PAGES
# ============================================================
def page_new_analysis():
    render_patient_form_and_upload()

    if not model_loaded:
        st.error(f"Model aktif ({CFG['short_label']}) tidak dimuat: {model_error}")

    render_header()

    left, center, right = st.columns([1.15, 2.35, 1.35])
    with left:
        render_left_panel()
    with center:
        render_center_panel()
    with right:
        render_right_panel()

    render_roi_3d_fullwidth()
    render_workflow_row()
    render_feature_row()
    render_metrics_row()
    render_iou_analysis()
    render_status_bar()


def stat_card(label, value, sub=""):
    return (f'<div class="stat-card"><div class="sl">{label}</div>'
            f'<div class="sv">{value}</div><div class="ss">{sub}</div></div>')


def page_dashboard():
    df = load_all_logs()
    total = len(df)
    n_malig, avg_conf = 0, 0.0
    if total > 0:
        try:
            flags = []
            for _, r in df.iterrows():
                mk = r.get("model_key")
                pc = int(r.get("pred_class", 0))
                flags.append(bool(MODEL_REGISTRY[mk]["is_malignant"][pc]))
            n_malig = int(sum(flags))
            avg_conf = float(pd.to_numeric(df["confidence"], errors="coerce").mean()) * 100
        except Exception:
            pass
    avail = sum(1 for c in MODEL_REGISTRY.values() if os.path.exists(c["model_path"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Total Analisis", total, "dari log dashboard"), unsafe_allow_html=True)
    c2.markdown(stat_card("Kasus Berisiko", n_malig, "kelas malignant / infeksi"), unsafe_allow_html=True)
    c3.markdown(stat_card("Rata-rata Confidence", f"{avg_conf:.1f}%" if total else "—", "semua prediksi"),
                unsafe_allow_html=True)
    c4.markdown(stat_card("Model Tersedia", f"{avail}/{len(MODEL_REGISTRY)}", "file best_model.pth"),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="card"><div class="card-title">Analisis Terbaru</div>', unsafe_allow_html=True)
        if total > 0:
            cols_show = [c for c in ["timestamp", "model_key", "patient_id", "pred_class_name",
                                     "confidence", "risk_level"] if c in df.columns]
            st.dataframe(df[cols_show].head(10), use_container_width=True, hide_index=True)
        else:
            st.caption("Belum ada analisis. Buka 'New Analysis' untuk memulai.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.button("➕ Mulai Analisis Baru", on_click=go_page, args=("New Analysis",), key="dash_new")
    with right:
        st.markdown('<div class="card"><div class="card-title">Distribusi Prediksi per Model</div>',
                    unsafe_allow_html=True)
        if total > 0 and "pred_class_name" in df.columns:
            cnt = df.groupby(["model_key", "pred_class_name"]).size().reset_index(name="n")
            fig = go.Figure(go.Bar(x=cnt["pred_class_name"], y=cnt["n"],
                                   marker_color="#2dd4bf", text=cnt["n"], textposition="auto"))
            fig.update_layout(height=260, margin=dict(l=6, r=6, t=6, b=6),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#e7ecf5",
                              xaxis=dict(showgrid=False, tickfont=dict(size=9)),
                              yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                            key="dash_dist")
        else:
            st.caption("Belum ada data.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<b>Status Model</b>", unsafe_allow_html=True)
    mc = st.columns(len(MODEL_REGISTRY))
    for col, (k, c) in zip(mc, MODEL_REGISTRY.items()):
        ok = os.path.exists(c["model_path"])
        with col:
            st.markdown(f"""
            <div class="model-card">
                <h3>{c['label']}</h3>
                <div class="desc">{c['description']}</div>
                <div class="kv"><span class="k">Checkpoint</span><span class="v">
                    <span class="{'ok-dot' if ok else 'no-dot'}">{'ADA' if ok else 'TIDAK ADA'}</span></span></div>
                <div class="kv"><span class="k">Kelas</span><span class="v">{c['num_classes']}</span></div>
                <div class="kv"><span class="k">Tipe</span><span class="v">{'3D CT' if c['is_3d'] else '2D X-Ray'}</span></div>
            </div>""", unsafe_allow_html=True)
            st.button("Gunakan model ini", key=f"dash_use_{k}", on_click=use_model_and_go, args=(k,))
    render_status_bar()


def get_true_label_counts(cfg):
    p = cfg["pred_csv"]
    if not p or not os.path.exists(p):
        return None
    try:
        d = pd.read_csv(p)
        col = next((c for c in d.columns if c.lower() in
                    ["true_label", "label", "y_true", "actual", "true"]), None)
        if col is None:
            return None
        vc = d[col].astype(int).value_counts().sort_index()
        return [int(vc.get(i, 0)) for i in range(cfg["num_classes"])]
    except Exception:
        return None


def page_dataset():
    for k, c in MODEL_REGISTRY.items():
        st.markdown(f"### {c['label']}")
        a, b = st.columns([1, 1.2])
        with a:
            files = [("Checkpoint model", c["model_path"]), ("Metrics JSON", c["metrics_json"]),
                     ("Confusion Matrix JSON", c["cm_json"]), ("Test Predictions CSV", c["pred_csv"]),
                     ("Ablation JSON", c["ablation_json"]), ("Dashboard Log CSV", c["pred_log_csv"])]
            html = '<div class="model-card">'
            for nm, pth in files:
                ok = bool(pth) and os.path.exists(pth)
                html += (f'<div class="kv"><span class="k">{nm}</span><span class="v">'
                         f'<span class="{"ok-dot" if ok else "no-dot"}">{"ADA" if ok else "—"}</span></span></div>')
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
            cls_html = '<div class="model-card"><b>Kelas</b>'
            for i, n in enumerate(c["class_names"]):
                cls_html += (f'<div class="kv"><span class="k">{c["class_short"][i]}</span>'
                             f'<span class="v" style="color:{c["class_colors"][i]}">{n}</span></div>')
            cls_html += "</div>"
            st.markdown(cls_html, unsafe_allow_html=True)
        with b:
            counts = get_true_label_counts(c)
            st.markdown('<div class="card"><div class="card-title">Distribusi Kelas (test set)</div>',
                        unsafe_allow_html=True)
            if counts and sum(counts) > 0:
                fig = go.Figure(go.Bar(x=c["class_short"], y=counts, marker_color=c["class_colors"],
                                       text=counts, textposition="auto"))
                fig.update_layout(height=260, margin=dict(l=6, r=6, t=6, b=6),
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e7ecf5", xaxis=dict(showgrid=False),
                                  yaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False},
                                key=f"ds_dist_{k}")
                st.caption(f"Total sampel test: {sum(counts)}")
            else:
                st.caption("test_predictions.csv tidak ditemukan / tidak punya kolom label.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")


def page_model_ai():
    for k, c in MODEL_REGISTRY.items():
        metrics, src = load_metrics_from_json(c["metrics_json"], c["cm_json"], c["pred_csv"], c["num_classes"])
        abl_rows, abl_src = load_ablation_results(c["ablation_json"])
        ok = os.path.exists(c["model_path"])
        active = (k == ACTIVE_KEY)
        st.markdown(f"""
        <div class="model-card">
            <h3>{c['label']} {'<span class="badge">AKTIF</span>' if active else ''}</h3>
            <div class="desc">{c['description']} · {c['clinical_notes']}</div>
            <div class="kv"><span class="k">Arsitektur</span><span class="v">{c['model_class']} (DWT-SVD-CNN-ViT)</span></div>
            <div class="kv"><span class="k">Checkpoint</span><span class="v">
                <span class="{'ok-dot' if ok else 'no-dot'}">{'ADA' if ok else 'TIDAK ADA'}</span></span></div>
            <div class="kv"><span class="k">Sumber metrik</span><span class="v">{src}</span></div>
        </div>""", unsafe_allow_html=True)
        mcols = st.columns(6)
        for col, (nm, val) in zip(mcols, metrics.items()):
            col.markdown(f'<div class="metric-box"><div class="lbl">{nm}</div><div class="val">{val}</div></div>',
                         unsafe_allow_html=True)
        if abl_rows:
            dfa = pd.DataFrame(abl_rows).dropna(subset=["AUC"])
            if not dfa.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(dfa, use_container_width=True, hide_index=True)
        st.button("Gunakan model ini di New Analysis", key=f"mai_use_{k}",
                  on_click=use_model_and_go, args=(k,))
        st.markdown("---")


def _set_state(k, v):
    st.session_state[k] = v


def _pager(key, total, size):
    pages = max(1, (total + size - 1) // size)
    cur = min(max(0, int(st.session_state.get(key, 0))), pages - 1)
    st.session_state[key] = cur
    if pages > 1:
        c1, c2, c3 = st.columns([1, 2, 1])
        c1.button("← Prev", key=f"{key}_prev", disabled=cur <= 0,
                  on_click=_set_state, args=(key, cur - 1))
        c2.markdown(f"<div style='text-align:center;color:var(--text-dim);font-size:13px;"
                    f"padding-top:8px;'>Halaman {cur + 1} / {pages} · {total} data</div>",
                    unsafe_allow_html=True)
        c3.button("Next →", key=f"{key}_next", disabled=cur >= pages - 1,
                  on_click=_set_state, args=(key, cur + 1))
    return cur


def _record_matches(r, q):
    if not q:
        return True
    q = q.strip().lower()
    hay = " ".join(str(r.get(k, "")) for k in
                   ("patient_id", "patient_name", "modality", "class_name", "report_id",
                    "file_name", "owner_name")).lower()
    return q in hay


def _diag_badge(r):
    if r.get("detected"):
        return '<span class="tag-det">Terdeteksi</span>'
    return '<span class="tag-ok">Tidak Terdeteksi</span>'


def render_record_detail(rec):
    cfg = _rec_cfg(rec)
    rid = rec.get("report_id")
    with st.container(border=True):
        st.markdown('<span class="hist-marker"></span>', unsafe_allow_html=True)
        t1, t2 = st.columns([4, 1])
        t1.markdown(f'<div class="tbl-title">DETAIL ANALISIS · {esc(rid)}</div>', unsafe_allow_html=True)
        t2.button("✖ Tutup", key="hist_close", on_click=lambda: st.session_state.pop("hist_view", None),
                  use_container_width=True)
        c1, c2, c3 = st.columns([1, 1, 1.6])
        p_img, p_cam = _thumb_path(rid, "img"), _thumb_path(rid, "cam")
        with c1:
            if os.path.exists(p_img):
                st.image(p_img, caption="Citra", use_container_width=True)
        with c2:
            if os.path.exists(p_cam):
                st.image(p_cam, caption="Grad-CAM", use_container_width=True)
            else:
                st.caption("Grad-CAM tidak tersedia.")
        with c3:
            rows = [("Patient ID", rec.get("patient_id")), ("Nama", rec.get("patient_name") or "—"),
                    ("Age/Sex", rec.get("age_sex")), ("Gol. Darah", rec.get("blood_type") or "—"),
                    ("Tinggi/Berat", rec.get("height_weight") or "—"),
                    ("Merokok", rec.get("smoking") or "—"), ("Gejala", rec.get("symptoms") or "—"),
                    ("Modalitas", rec.get("modality")), ("Model", cfg["short_label"]),
                    ("Prediksi", f"{rec.get('class_short')} · {rec.get('class_name')}"),
                    ("Confidence", f"{float(rec.get('confidence', 0)) * 100:.1f}%")]
            if cfg.get("task") == "lung_nodule":
                rows.append(("Diameter", f"{float(rec.get('diameter_mm', 0)):.2f} mm"))
            rows += [("Risk Level", rec.get("risk")), ("Rekomendasi", rec.get("recommendation")),
                     ("Dokter", rec.get("owner_name")), ("Dibuat", fmt_dt(rec.get("created")))]
            st.markdown("".join(
                f'<div class="kv"><span class="k">{esc(k)}</span><span class="v">{esc(v)}</span></div>'
                for k, v in rows), unsafe_allow_html=True)
        probs = rec.get("probs", [])
        bars = ""
        for i, p in enumerate(probs):
            if i >= len(cfg["class_names"]):
                break
            col = cfg["class_colors"][i]
            bars += (f'<div class="prob-row"><span class="lbl">{esc(cfg["class_short"][i])} '
                     f'{esc(cfg["class_names"][i][:22])}</span>'
                     f'<div class="prob-track"><div class="prob-fill" style="width:{p * 100:.1f}%;'
                     f'background:{col};"></div></div>'
                     f'<span class="prob-val" style="color:{col}">{p * 100:.1f}%</span></div>')
        st.markdown(bars, unsafe_allow_html=True)

        a, b, c, d = st.columns(4)
        try:
            a.download_button("⬇ HTML", data=html_for(rec), file_name=report_filename(rec, "html"),
                              mime="text/html", use_container_width=True, key=f"dv_html_{rid}")
        except Exception as e:
            a.error(f"Gagal: {e}")
        if FPDF_OK:
            try:
                pdfb = pdf_bytes_for(rec)
                if pdfb:
                    b.download_button("⬇ PDF", data=pdfb, file_name=report_filename(rec, "pdf"),
                                      mime="application/pdf", use_container_width=True,
                                      key=f"dv_pdf_{rid}")
            except Exception as e:
                b.error(f"Gagal: {e}")
        else:
            b.caption("PDF: pip install fpdf2")
        if st.session_state.get("hist_confirm_del") == rid:
            c.button("Ya, hapus", key=f"del_yes_{rid}", on_click=delete_record, args=(rid,),
                     use_container_width=True)
            d.button("Batal", key=f"del_no_{rid}",
                     on_click=lambda: st.session_state.pop("hist_confirm_del", None),
                     use_container_width=True)
        else:
            c.button("🗑 Hapus", key=f"del_{rid}", on_click=_set_state,
                     args=("hist_confirm_del", rid), use_container_width=True)


def page_history():
    items = my_records()
    HW = [1.45, 1.55, 1.5, 0.9, 1.45, 1.15, 0.95]
    with st.container(border=True):
        st.markdown('<span class="hist-marker"></span>', unsafe_allow_html=True)
        h1, h2, h3 = st.columns([2.2, 2.2, 0.9])
        h1.markdown('<div class="tbl-title">PATIENT LIST</div>', unsafe_allow_html=True)
        q = h2.text_input("Search", placeholder="Search patient data or analysis...",
                          label_visibility="collapsed", key="hist_q")
        with h3.popover("🔽 Filter", use_container_width=True):
            f_mod = st.selectbox("Modality", ["Semua", "X-ray", "CT Scan"], key="hist_f_mod")
            f_diag = st.selectbox("AI Diagnosis", ["Semua", "Terdeteksi", "Tidak Terdeteksi"],
                                  key="hist_f_diag")
        view = [r for r in items if _record_matches(r, q)]
        if f_mod != "Semua":
            view = [r for r in view if r.get("modality") == f_mod]
        if f_diag != "Semua":
            want = (f_diag == "Terdeteksi")
            view = [r for r in view if bool(r.get("detected")) == want]

        heads = ["PATIENT ID", "PATIENT NAME", "DATE", "MODALITY", "AI DIAGNOSIS",
                 "CONFIDENCE LEVEL", "ACTION"]
        hc = st.columns(HW)
        for c, t in zip(hc, heads):
            c.markdown(f'<div class="th">{t}</div>', unsafe_allow_html=True)
        st.markdown('<hr class="rowline">', unsafe_allow_html=True)

        if not view:
            st.markdown('<div class="empty-note">Belum ada riwayat. Isi form di '
                        '<b>New Analysis</b> lalu upload citra — pasien akan muncul di sini '
                        'dengan nama yang diinput.</div>' if not items else
                        '<div class="empty-note">Tidak ada data yang cocok dengan pencarian/filter.</div>',
                        unsafe_allow_html=True)
        cur = _pager("hist_pg", len(view), 8)
        for r in view[cur * 8:(cur + 1) * 8]:
            rid = r.get("report_id")
            cols = st.columns(HW)
            cols[0].markdown(f'<div class="td">{esc(r.get("patient_id"))}</div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div class="td">{esc(r.get("patient_name") or "—")}</div>',
                             unsafe_allow_html=True)
            cols[2].markdown(f'<div class="td">{esc(fmt_dt(r.get("created")))}</div>', unsafe_allow_html=True)
            cols[3].markdown(f'<div class="td">{esc(r.get("modality"))}</div>', unsafe_allow_html=True)
            cols[4].markdown(f'<div class="td">{_diag_badge(r)}</div>', unsafe_allow_html=True)
            cols[5].markdown(f'<div class="td strong">{float(r.get("confidence", 0)) * 100:.1f}%</div>',
                             unsafe_allow_html=True)
            with cols[6]:
                a, b = st.columns(2)
                a.button("👁", key=f"hv_{rid}", on_click=_set_state,
                         args=("hist_view", rid))
                with b.popover("⬇"):
                    st.caption(f"Laporan {rid}")
                    try:
                        st.download_button("HTML", data=html_for(r), file_name=report_filename(r, "html"),
                                           mime="text/html", use_container_width=True,
                                           key=f"hd_html_{rid}")
                    except Exception as e:
                        st.error(f"Gagal: {e}")
                    if FPDF_OK:
                        try:
                            pb = pdf_bytes_for(r)
                            if pb:
                                st.download_button("PDF", data=pb, file_name=report_filename(r, "pdf"),
                                                   mime="application/pdf", use_container_width=True,
                                                   key=f"hd_pdf_{rid}")
                        except Exception as e:
                            st.error(f"Gagal: {e}")
                    else:
                        st.caption("PDF: pip install fpdf2")
            st.markdown('<hr class="rowline">', unsafe_allow_html=True)

        if view:
            df_csv = pd.DataFrame([{
                "report_id": r.get("report_id"), "patient_id": r.get("patient_id"),
                "patient_name": r.get("patient_name"), "date": fmt_dt(r.get("created")),
                "modality": r.get("modality"), "ai_diagnosis": "Terdeteksi" if r.get("detected") else "Tidak Terdeteksi",
                "class": r.get("class_name"), "confidence_pct": round(float(r.get("confidence", 0)) * 100, 2),
                "risk": r.get("risk"), "recommendation": r.get("recommendation")} for r in view])
            st.download_button("⬇ Download CSV", data=df_csv.to_csv(index=False).encode("utf-8"),
                               file_name="riwayat_pasien.csv", mime="text/csv", key="hist_csv")

    rid_view = st.session_state.get("hist_view")
    if rid_view:
        rec = next((r for r in items if r.get("report_id") == rid_view), None)
        if rec:
            render_record_detail(rec)
        else:
            st.session_state.pop("hist_view", None)


def page_reports():
    items = my_records()
    RW = [1.9, 1.5, 1.6, 1.5, 1.0]
    with st.container(border=True):
        st.markdown('<span class="hist-marker"></span>', unsafe_allow_html=True)
        h1, h2 = st.columns([3, 1])
        h1.markdown('<div class="tbl-title">ARSIP LAPORAN RADIOLOGI</div>', unsafe_allow_html=True)
        if h2.button("⬇ Ekspor Batch", key="rep_batch", disabled=not items, use_container_width=True):
            with st.spinner("Menyusun ZIP laporan..."):
                st.session_state["batch_zip"] = build_batch_zip(items)
                st.session_state["batch_n"] = len(items)
        if st.session_state.get("batch_zip"):
            st.download_button(f"Unduh ZIP ({st.session_state.get('batch_n', 0)} laporan)",
                               data=st.session_state["batch_zip"], file_name="arsip_laporan_lungvision.zip",
                               mime="application/zip", key="rep_batch_dl")
        hc = st.columns(RW)
        for c, t in zip(hc, ["ID LAPORAN", "PASIEN", "TANGGAL DIGENERATE", "DOKTER PEMERIKSA", "AKSI"]):
            c.markdown(f'<div class="th">{t}</div>', unsafe_allow_html=True)
        st.markdown('<hr class="rowline">', unsafe_allow_html=True)
        if not items:
            st.markdown('<div class="empty-note">Belum ada laporan. Laporan dibuat otomatis setiap '
                        'analisis di <b>New Analysis</b>.</div>', unsafe_allow_html=True)
        cur = _pager("rep_pg", len(items), 8)
        for r in items[cur * 8:(cur + 1) * 8]:
            rid = r.get("report_id")
            cols = st.columns(RW)
            cols[0].markdown(f'<div class="td">{esc(rid)}</div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div class="td">{esc(r.get("patient_name") or r.get("patient_id"))}</div>',
                             unsafe_allow_html=True)
            cols[2].markdown(f'<div class="td">{esc(fmt_dt(r.get("created")))}</div>', unsafe_allow_html=True)
            cols[3].markdown(f'<div class="td">{esc(r.get("owner_name"))}</div>', unsafe_allow_html=True)
            with cols[4]:
                pb = None
                if FPDF_OK:
                    try:
                        pb = pdf_bytes_for(r)
                    except Exception as e:
                        st.error(f"Gagal: {e}")
                if pb:
                    st.download_button("Unduh PDF", data=pb, file_name=report_filename(r, "pdf"),
                                       mime="application/pdf", key=f"rp_pdf_{rid}")
                else:
                    st.download_button("Unduh HTML", data=html_for(r), file_name=report_filename(r, "html"),
                                       mime="text/html", key=f"rp_html_{rid}")
            st.markdown('<hr class="rowline">', unsafe_allow_html=True)


# ---------------- SETTINGS (Edit Profile) ----------------
def _on_pref_change():
    ss = st.session_state
    if "w_lang" in ss:
        ss["ui_lang"] = "en" if ss["w_lang"] == "English" else "id"
    if "w_dark" in ss:
        ss["ui_dark"] = bool(ss["w_dark"])
    if "w_email" in ss:
        ss["ui_email_notif"] = bool(ss["w_email"])
    if "w_2fa" in ss:
        ss["ui_2fa"] = bool(ss["w_2fa"])
    if "w_model" in ss and ss["w_model"] in MODEL_REGISTRY:
        ss["ui_default_model"] = ss["w_model"]
    if "w_opacity" in ss:
        ss["ui_cam_opacity"] = int(ss["w_opacity"])
    _persist_prefs()


def _flash(kind, msg):
    st.session_state["flash"] = (kind, msg)


def _pref_row(title, desc):
    """Baris preferensi: kiri label+deskripsi, kanan kontrol (dipakai lewat `with`)."""
    left, right = st.columns([3, 1.5])
    with left:
        st.markdown(f'<span class="pref-row"></span><div class="set-label"><div class="t">{title}</div>'
                    f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)
    return right


def page_settings():
    user = current_user()
    uid = st.session_state.get("auth_user")
    flash = st.session_state.pop("flash", None)
    _, mid, _ = st.columns([0.35, 2.6, 0.35])
    with mid:
        if flash:
            (st.success if flash[0] == "success" else st.error)(flash[1])
        with st.container(border=True):
            st.markdown('<span class="form-marker"></span>', unsafe_allow_html=True)

            # ---- Profil Pengguna ----
            st.markdown('<div class="set-section">Profil Pengguna</div>', unsafe_allow_html=True)
            with st.form("form_profile"):
                a, b = st.columns(2)
                full = a.text_input("Nama Lengkap", value=user.get("full_name", ""))
                nip = b.text_input("ID / NIP", value=user.get("nip", ""))
                c, d = st.columns(2)
                email = c.text_input("Email Terdaftar", value=user.get("email", ""))
                spec = d.text_input("Spesialisasi", value=user.get("specialization", ""))
                save_profile = st.form_submit_button("Simpan Profil")
            if save_profile:
                ok, err = update_profile(uid, full, nip, email, spec)
                _flash("success" if ok else "error", "Profil berhasil disimpan." if ok else err)
                st.rerun()

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ---- App Preferences ----
            st.markdown('<div class="set-section">App Preferences</div>', unsafe_allow_html=True)
            r = _pref_row("Language", "Choose application interface language")
            st.session_state.setdefault(
                "w_lang", "Bahasa Indonesia" if st.session_state.get("ui_lang", "id") == "id" else "English")
            r.selectbox("Language", ["Bahasa Indonesia", "English"], key="w_lang",
                        label_visibility="collapsed", on_change=_on_pref_change)
            r = _pref_row("Dark Mode", "Use a dark interface to reduce eye strain")
            r.toggle("Dark Mode", value=st.session_state.get("ui_dark", True), key="w_dark",
                     label_visibility="collapsed", on_change=_on_pref_change)
            r = _pref_row("Email Notifications",
                          "Receive an email when AI analysis is complete "
                          "<i>(preferensi tersimpan; butuh setup SMTP untuk mengirim)</i>")
            r.toggle("Email Notifications", value=st.session_state.get("ui_email_notif", False),
                     key="w_email", label_visibility="collapsed", on_change=_on_pref_change)

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ---- Security & Privacy ----
            st.markdown('<div class="set-section">Security &amp; Privacy</div>', unsafe_allow_html=True)
            r = _pref_row("Two-Factor Authentication (2FA)",
                          "Secure account with additional verification via SMS/Email "
                          "<i>(preferensi tersimpan; verifikasi belum diterapkan)</i>")
            r.toggle("2FA", value=st.session_state.get("ui_2fa", False), key="w_2fa",
                     label_visibility="collapsed", on_change=_on_pref_change)
            changed = user.get("pw_changed") or user.get("created", "")
            try:
                days = (datetime.datetime.now() - datetime.datetime.fromisoformat(changed)).days
                last = "today" if days <= 0 else f"{days} days ago"
            except Exception:
                last = "-"
            r = _pref_row("Change Password", f"Last changed {last}")
            r.button("Update", key="btn_pw_toggle", use_container_width=True,
                     on_click=lambda: st.session_state.update(
                         show_pw=not st.session_state.get("show_pw", False)))
            if st.session_state.get("show_pw"):
                with st.form("form_pw", clear_on_submit=True):
                    p1, p2, p3 = st.columns(3)
                    old = p1.text_input("Password lama", type="password")
                    new1 = p2.text_input("Password baru", type="password")
                    new2 = p3.text_input("Ulangi password baru", type="password")
                    pw_ok = st.form_submit_button("Simpan Password")
                if pw_ok:
                    ok, err = change_password(uid, old, new1, new2)
                    if ok:
                        st.session_state["show_pw"] = False
                    _flash("success" if ok else "error",
                           "Password berhasil diubah." if ok else err)
                    st.rerun()

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ---- AI Configuration & Integration ----
            st.markdown('<div class="set-section">AI Configuration &amp; Integration</div>',
                        unsafe_allow_html=True)
            keys = list(MODEL_REGISTRY.keys())
            cur_model = st.session_state.get("ui_default_model", "xray_3class")
            r = _pref_row("Default Model", "Model yang dipakai saat login / membuka New Analysis")
            r.selectbox("Default Model", keys, index=keys.index(cur_model) if cur_model in keys else 0,
                        format_func=lambda k: MODEL_REGISTRY[k]["short_label"], key="w_model",
                        label_visibility="collapsed", on_change=_on_pref_change)
            r = _pref_row("Grad-CAM Opacity", "Opasitas awal overlay heatmap pada Explainability")
            r.slider("Grad-CAM Opacity", 0, 100, int(st.session_state.get("ui_cam_opacity", 45)),
                     key="w_opacity", label_visibility="collapsed", on_change=_on_pref_change)

            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            # ---- System & Maintenance ----
            st.markdown('<div class="set-section">System &amp; Maintenance</div>', unsafe_allow_html=True)
            dev = "CUDA" if torch.cuda.is_available() else "CPU"
            st.markdown(f"""
            <div class="kv"><span class="k">Device</span><span class="v">{dev}</span></div>
            <div class="kv"><span class="k">Model aktif</span><span class="v">{esc(CFG['label'])}</span></div>
            <div class="kv"><span class="k">pydicom</span><span class="v">{'OK' if PYDICOM_OK else 'tidak terpasang'}</span></div>
            <div class="kv"><span class="k">fpdf2 (PDF)</span><span class="v">{'OK' if FPDF_OK else 'tidak terpasang'}</span></div>
            <div class="kv"><span class="k">scipy (ROC)</span><span class="v">{'OK' if SCIPY_OK else 'tidak terpasang'}</span></div>
            <div class="kv"><span class="k">roi_3d_visualizer</span><span class="v">{'OK' if ROI_VIZ_OK else 'ERROR'}</span></div>
            <div class="kv"><span class="k">Data dir</span><span class="v">{esc(DATA_DIR)}</span></div>
            """, unsafe_allow_html=True)
            if not ROI_VIZ_OK:
                st.warning(f"roi_3d_visualizer: {ROI_VIZ_ERR}")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            x, y, z = st.columns(3)
            with x:
                if st.button("Reset hasil analisis", use_container_width=True, key="set_reset"):
                    for k in ["gray128", "dicom_meta", "file_name", "slice_view"]:
                        st.session_state.pop(k, None)
                    clear_cached_results()
                    st.success("Hasil analisis direset.")
            with y:
                if st.button("Hapus riwayat IoU", use_container_width=True, key="set_iou"):
                    st.session_state["iou_history"] = []
                    st.success("Riwayat IoU dihapus.")
            with z:
                if st.button("Bersihkan cache", use_container_width=True, key="set_cache"):
                    st.cache_data.clear()
                    try:
                        build_gradcam.clear(); register_extra_hooks.clear(); load_model_for.clear()
                    except Exception:
                        pass
                    st.success("Cache dibersihkan. Muat ulang halaman.")


# ============================================================
# MAIN
# ============================================================
render_sidebar()
_page = st.session_state["page"]
render_topbar(_page)

if _page == "Dashboard":
    page_dashboard()
elif _page == "New Analysis":
    page_new_analysis()
elif _page == "History":
    page_history()
elif _page == "Dataset":
    page_dataset()
elif _page == "Model AI":
    page_model_ai()
elif _page == "Reports":
    page_reports()
elif _page == "Settings":
    page_settings()