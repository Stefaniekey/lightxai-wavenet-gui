"""
roi_3d_visualizer.py — 3D ROI Visualization untuk LightXAI-WaveNet
=====================================================================
Referensi: arahan dosen untuk visualisasi ROI 3D
- LIDC (CT): volumetric 3D dengan 3 orthogonal planes + nodule center
- X-Ray (2D): pseudo-3D surface plot dari intensitas

Install:
    pip install numpy matplotlib plotly scikit-image opencv-python

Fungsi:
    - build_lidc_3d_roi(gray128, region, ...)          → 3D volumetric + axial view
    - build_xray_pseudo_3d(gray128, ...)               → 2 panel (original + pseudo-3D)
    - build_xray_pseudo_3d_only(gray128, ...)          → 1 panel pseudo-3D full width
"""

import numpy as np
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from skimage.measure import marching_cubes
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False


# ============================================================
# 1. LIDC — 3D VOLUMETRIC ROI (CT SCAN)
# ============================================================

def build_lidc_3d_roi(gray128, region, vol_size=64, use_marching_cubes=False):
    """
    Bangun visualisasi 3D ROI dari crop CT + 3 orthogonal planes.
    FIX v2: opacity dinaikkan, surface_count diperbanyak, axial view proper.
    """
    if region is None:
        return _empty_figure("ROI 3D tidak tersedia")

    # ---- 1. Crop ROI dari gray128 ----
    cx_128 = int(round(region["cx"] / region["w"] * 128))
    cy_128 = int(round(region["cy"] / region["h"] * 128))
    crop_half = 32
    x1 = max(0, cx_128 - crop_half)
    x2 = min(128, cx_128 + crop_half)
    y1 = max(0, cy_128 - crop_half)
    y2 = min(128, cy_128 + crop_half)
    roi_crop = gray128[y1:y2, x1:x2]

    if roi_crop.size == 0:
        roi_crop = gray128

    # ---- 2. Bangun volume 3D dengan variasi axial yang bermakna ----
    h, w = roi_crop.shape
    depth = vol_size

    # Normalize crop ke [0, 1]
    crop_f = roi_crop.astype(np.float32)
    cmin, cmax = crop_f.min(), crop_f.max()
    if cmax > cmin:
        crop_norm = (crop_f - cmin) / (cmax - cmin)
    else:
        crop_norm = np.zeros_like(crop_f)

    # Buat volume: base = crop_norm, diperkuat gaussian blob di tengah
    zz, yy, xx = np.mgrid[0:depth, 0:h, 0:w]

    # Gaussian 3D blob di tengah (nodule-like)
    cz, cy_v, cx_v = depth // 2, h // 2, w // 2
    sigma = max(4, min(h, w) / 6)
    gaussian = np.exp(-((zz - cz) ** 2 + (yy - cy_v) ** 2 + (xx - cx_v) ** 2)
                       / (2 * sigma ** 2))

    # Fade axial: pinggir lebih gelap, tengah lebih terang (efek sphere)
    axial_fade = np.exp(-((zz - cz) ** 2) / (2 * (depth / 3) ** 2))

    # Volume: intensitas asli + nodule enhancement + fade
    base_volume = np.stack([crop_norm] * depth, axis=0)
    volume = base_volume * (0.4 + 0.6 * axial_fade) + 0.7 * gaussian
    volume = np.clip(volume, 0, 1)

    # ---- 3. Coordinate grid ----
    z, y, x = np.mgrid[0:depth, 0:h, 0:w]

    # ---- 4. Figure ----
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("", "Axial View (Z-center)"),
        horizontal_spacing=0.08
    )

    # ---- 5. Volumetric rendering ----
    fig.add_trace(
        go.Volume(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=volume.flatten(),
            isomin=0.15,
            isomax=1.0,
            opacity=0.25,
            surface_count=30,
            colorscale=[
                [0.0, "#000000"],
                [0.2, "#0d2a3a"],
                [0.4, "#1e4976"],
                [0.6, "#2dd4bf"],
                [0.8, "#ffb74d"],
                [1.0, "#ef5350"],
            ],
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,
            name="ROI Volume",
            hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z}<br>intensity: %{value:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # ---- 6. Nodule Center Marker ----
    cz_v, cy_v, cx_v = depth / 2, h / 2, w / 2
    fig.add_trace(
        go.Scatter3d(
            x=[cx_v], y=[cy_v], z=[cz_v],
            mode="markers",
            marker=dict(size=10, color="#ef5350", symbol="circle",
                         line=dict(color="white", width=2)),
            name="Nodule Center",
            hovertemplate="Nodule Center<br>x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>"
        ),
        row=1, col=1
    )

    # ---- 7. 3 Orthogonal Planes ----
    # Axial plane (Z=center) — BIRU
    ax_plane = volume[depth // 2, :, :]
    fig.add_trace(
        go.Surface(
            z=np.full_like(ax_plane, depth // 2),
            x=np.arange(w), y=np.arange(h),
            surfacecolor=ax_plane,
            colorscale="Blues",
            opacity=0.7,
            showscale=False,
            name="Axial (Z)",
            hovertemplate="Axial plane<extra></extra>"
        ),
        row=1, col=1
    )

    # Coronal plane (Y=center) — HIJAU
    cor_plane = volume[:, h // 2, :]
    fig.add_trace(
        go.Surface(
            z=np.arange(depth).reshape(-1, 1).repeat(w, axis=1),
            x=np.tile(np.arange(w), (depth, 1)),
            y=np.full((depth, w), h // 2),
            surfacecolor=cor_plane,
            colorscale="Greens",
            opacity=0.6,
            showscale=False,
            name="Coronal (Y)",
            hovertemplate="Coronal plane<extra></extra>"
        ),
        row=1, col=1
    )

    # Sagittal plane (X=center) — MERAH
    sag_plane = volume[:, :, w // 2]
    fig.add_trace(
        go.Surface(
            z=np.arange(depth).reshape(-1, 1).repeat(h, axis=1),
            x=np.full((depth, h), w // 2),
            y=np.tile(np.arange(h), (depth, 1)),
            surfacecolor=sag_plane,
            colorscale="Reds",
            opacity=0.6,
            showscale=False,
            name="Sagittal (X)",
            hovertemplate="Sagittal plane<extra></extra>"
        ),
        row=1, col=1
    )

    # ---- 8. Bounding cube cyan ----
    cube_edges_x, cube_edges_y, cube_edges_z = _cube_outline(0, w, 0, h, 0, depth)
    fig.add_trace(
        go.Scatter3d(
            x=cube_edges_x, y=cube_edges_y, z=cube_edges_z,
            mode="lines",
            line=dict(color="#2dd4bf", width=3),
            name="ROI Bounds",
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1, col=1
    )

    # ---- 9. Subplot kanan: Axial view 2D ----
    ax_display = np.flipud(ax_plane)
    fig.add_trace(
        go.Heatmap(
            z=ax_display,
            colorscale=[
                [0.0, "#000000"],
                [0.3, "#1e4976"],
                [0.6, "#2dd4bf"],
                [0.8, "#ffb74d"],
                [1.0, "#ffffff"],
            ],
            showscale=False,
            zsmooth="best",
            name="Axial View",
            hovertemplate="y: %{y}<br>x: %{x}<br>val: %{z:.2f}<extra></extra>"
        ),
        row=1, col=2
    )

    # Crosshair di axial view
    fig.add_trace(
        go.Scatter(
            x=[w / 2, w / 2], y=[0, h],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            showlegend=False, hoverinfo="skip"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[0, w], y=[h / 2, h / 2],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            showlegend=False, hoverinfo="skip"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[w / 2], y=[h / 2],
            mode="markers",
            marker=dict(size=14, color="yellow", symbol="x",
                         line=dict(color="red", width=2)),
            name="Nodule Center",
            showlegend=False, hoverinfo="skip"
        ),
        row=1, col=2
    )

    # ---- 10. Layout ----
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e7ecf5",
        showlegend=True,
        legend=dict(
            font=dict(size=9, color="#e7ecf5"),
            bgcolor="rgba(4,6,11,0.7)",
            bordercolor="#1e293b",
            borderwidth=1,
            x=0.02, y=0.98,
            xanchor="left", yanchor="top"
        ),
        scene=dict(
            xaxis=dict(
                title="X (Left-Right)",
                title_font=dict(size=10, color="#7dd3fc"),
                tickfont=dict(size=8, color="#8b96ac"),
                backgroundcolor="rgba(4,6,11,0.5)",
                gridcolor="#1e293b",
                showbackground=True,
            ),
            yaxis=dict(
                title="Y (Anterior-Posterior)",
                title_font=dict(size=10, color="#3ddc84"),
                tickfont=dict(size=8, color="#8b96ac"),
                backgroundcolor="rgba(4,6,11,0.5)",
                gridcolor="#1e293b",
                showbackground=True,
            ),
            zaxis=dict(
                title="Z (Superior-Inferior)",
                title_font=dict(size=10, color="#ef5350"),
                tickfont=dict(size=8, color="#8b96ac"),
                backgroundcolor="rgba(4,6,11,0.5)",
                gridcolor="#1e293b",
                showbackground=True,
            ),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            aspectmode="cube",
        ),
        uirevision="lidc_roi_3d",
    )

    return fig


def _cube_outline(x0, x1, y0, y1, z0, z1):
    """Generate edges untuk wireframe cube."""
    edges = [
        ([x0, x1], [y0, y0], [z0, z0]),
        ([x1, x1], [y0, y1], [z0, z0]),
        ([x1, x0], [y1, y1], [z0, z0]),
        ([x0, x0], [y1, y0], [z0, z0]),
        ([x0, x1], [y0, y0], [z1, z1]),
        ([x1, x1], [y0, y1], [z1, z1]),
        ([x1, x0], [y1, y1], [z1, z1]),
        ([x0, x0], [y1, y0], [z1, z1]),
        ([x0, x0], [y0, y0], [z0, z1]),
        ([x1, x1], [y0, y0], [z0, z1]),
        ([x1, x1], [y1, y1], [z0, z1]),
        ([x0, x0], [y1, y1], [z0, z1]),
    ]
    xs, ys, zs = [], [], []
    for (ex, ey, ez) in edges:
        xs += ex + [None]
        ys += ey + [None]
        zs += ez + [None]
    return xs, ys, zs


# ============================================================
# 2. X-RAY — PSEUDO-3D SURFACE (VERSI 2 PANEL)
# ============================================================

def build_xray_pseudo_3d(gray128, depth_scale=30, step=4, cmap_name="gray"):
    """
    Versi 2 panel: X-Ray original + pseudo-3D surface.
    (Disimpan sebagai backup — versi yang dipakai di GUI adalah
    build_xray_pseudo_3d_only dengan 1 panel full width)
    """
    if gray128 is None:
        return _empty_figure("X-Ray tidak tersedia")

    # ---- 1. Preprocess ----
    img = cv2.resize(gray128, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # ---- 2. Smoothing ----
    img_smooth = cv2.GaussianBlur(img, (7, 7), 0)

    # ---- 3. Downsample ----
    depth_small = img_smooth[::step, ::step]
    H, W = depth_small.shape

    # ---- 4. Grid ----
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_small * depth_scale

    # ---- 5. Figure ----
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.4, 0.6],
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("X-Ray Original", "Pseudo-3D Representation"),
        horizontal_spacing=0.05
    )

    # ---- 6. Original X-Ray ----
    fig.add_trace(
        go.Heatmap(
            z=img,
            colorscale="gray",
            showscale=False,
            name="Original",
            hovertemplate="Intensity: %{z:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # ---- 7. Pseudo-3D surface ----
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=depth_small,
            colorscale=[
                [0.0, "#000000"],
                [0.3, "#1e4976"],
                [0.6, "#2dd4bf"],
                [0.8, "#ffb74d"],
                [1.0, "#ef5350"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Depth", font=dict(size=10, color="#8b96ac")),
                tickfont=dict(size=8, color="#8b96ac"),
                x=1.02,
                len=0.7,
            ),
            name="Pseudo-3D",
            hovertemplate="x: %{x}<br>y: %{y}<br>depth: %{z:.2f}<extra></extra>"
        ),
        row=1, col=2
    )

    # ---- 8. Layout ----
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e7ecf5",
        showlegend=False,
        scene=dict(
            xaxis=dict(
                title="X (Left-Right)",
                title_font=dict(size=10, color="#7dd3fc"),
                tickfont=dict(size=8, color="#8b96ac"),
                backgroundcolor="rgba(4,6,11,0.4)",
                gridcolor="#1e293b",
            ),
            yaxis=dict(
                title="Y (Anterior-Posterior)",
                title_font=dict(size=10, color="#3ddc84"),
                tickfont=dict(size=8, color="#8b96ac"),
                backgroundcolor="rgba(4,6,11,0.4)",
                gridcolor="#1e293b",
            ),
            zaxis=dict(
                title="Pseudo Depth",
                title_font=dict(size=10, color="#ef5350"),
                tickfont=dict(size=8, color="#8b96ac"),
                backgroundcolor="rgba(4,6,11,0.4)",
                gridcolor="#1e293b",
            ),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        uirevision="xray_pseudo_3d",
    )

    return fig


# ============================================================
# 3. X-RAY — PSEUDO-3D SURFACE (VERSI 1 PANEL, FULL WIDTH)
# ============================================================

def build_xray_pseudo_3d_only(gray128, depth_scale=30, step=4):
    """
    Versi X-Ray pseudo-3D TANPA panel original (single plot full width).
    Dipakai di GUI karena original sudah ada di panel tengah.
    """
    if gray128 is None:
        return _empty_figure("X-Ray tidak tersedia")

    # Preprocess
    img = cv2.resize(gray128, (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_smooth = cv2.GaussianBlur(img, (7, 7), 0)
    depth_small = img_smooth[::step, ::step]
    H, W = depth_small.shape

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_small * depth_scale

    fig = go.Figure(data=go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=depth_small,
        colorscale=[
            [0.0, "#000000"],
            [0.3, "#1e4976"],
            [0.6, "#2dd4bf"],
            [0.8, "#ffb74d"],
            [1.0, "#ef5350"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Depth", font=dict(size=10, color="#8b96ac")),
            tickfont=dict(size=8, color="#8b96ac"),
            x=1.02, len=0.7,
        ),
        hovertemplate="x: %{x}<br>y: %{y}<br>depth: %{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e7ecf5",
        showlegend=False,
        scene=dict(
            xaxis=dict(
                title="X (Left-Right)",
                title_font=dict(size=10, color="#7dd3fc"),
                tickfont=dict(size=8, color="#8b96ac"),
                backgroundcolor="rgba(4,6,11,0.4)",
                gridcolor="#1e293b",
            ),
            yaxis=dict(
                title="Y (Anterior-Posterior)",
                title_font=dict(size=10, color="#3ddc84"),
                tickfont=dict(size=8, color="#8b96ac"),
                backgroundcolor="rgba(4,6,11,0.4)",
                gridcolor="#1e293b",
            ),
            zaxis=dict(
                title="Pseudo Depth",
                title_font=dict(size=10, color="#ef5350"),
                tickfont=dict(size=8, color="#8b96ac"),
                backgroundcolor="rgba(4,6,11,0.4)",
                gridcolor="#1e293b",
            ),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        uirevision="xray_pseudo_3d_only",
    )

    return fig


# ============================================================
# 4. HELPER
# ============================================================

def _empty_figure(message="No data"):
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False,
        xref="paper", yref="paper", x=0.5, y=0.5,
        font=dict(size=14, color="#8b96ac")
    )
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig