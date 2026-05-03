"""
=============================================================
Plant Disease Detection System
=============================================================
File    : app.py
Purpose : Streamlit web application for plant disease detection
Run     : streamlit run src/app.py
=============================================================
"""

import sys
import io
import time
from pathlib import Path

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image

# ── Path setup ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.predictor import (
    is_model_available,
    load_model_and_meta,
    predict,
    MODELS_DIR,
)

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #e2e8f0;
}

/* ── Background ── */
.stApp {
    background-color: #0b1120;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(34,197,94,0.08), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,185,129,0.05), transparent);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1929 !important;
    border-right: 1px solid #1e3a2f;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #4ade80 !important; }

/* ── Typography ── */
h1 {
    font-family: 'Space Mono', monospace !important;
    color: #f0fdf4 !important;
    letter-spacing: -0.04em;
    font-size: clamp(1.8rem, 4vw, 2.8rem) !important;
}
h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #d1fae5 !important;
    font-weight: 600 !important;
}
p, li { color: #94a3b8; line-height: 1.75; }

/* ── Cards ── */
.card {
    background: rgba(15, 25, 41, 0.8);
    border: 1px solid #1e3a2f;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
}
.card-green {
    background: rgba(20, 45, 30, 0.6);
    border: 1px solid rgba(74, 222, 128, 0.25);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-highlight {
    background: linear-gradient(135deg, rgba(20,45,30,0.8), rgba(15,25,41,0.9));
    border: 1px solid rgba(74, 222, 128, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.card-highlight::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4ade80, #22c55e, transparent);
}

/* ── Metric boxes ── */
.metric-box {
    background: rgba(15, 25, 41, 0.9);
    border: 1px solid #1e3a2f;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #4ade80;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Confidence bars ── */
.conf-wrap { margin-bottom: 14px; }
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.conf-label {
    font-size: 0.85rem;
    color: #94a3b8;
    max-width: 76%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.conf-label-top { color: #f0fdf4; font-weight: 600; }
.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
}
.conf-track {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 3px;
}

/* ── Severity badges ── */
.badge {
    display: inline-block;
    border-radius: 100px;
    padding: 3px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}
.badge-none   { background:#052e16; color:#4ade80; border:1px solid #166534; }
.badge-low    { background:#1c1917; color:#fde68a; border:1px solid #92400e; }
.badge-mod    { background:#1c0f05; color:#fb923c; border:1px solid #9a3412; }
.badge-high   { background:#1c0505; color:#f87171; border:1px solid #991b1b; }

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
}

/* ── Info rows ── */
.info-row {
    padding: 10px 0;
    border-bottom: 1px solid #1e293b;
}
.info-row:last-child { border-bottom: none; }
.info-title {
    font-size: 0.72rem;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
    font-family: 'Space Mono', monospace;
}
.info-text { font-size: 0.875rem; color: #94a3b8; line-height: 1.65; }

/* ── Upload zone ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15,25,41,0.6) !important;
    border: 2px dashed #1e3a2f !important;
    border-radius: 16px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #4ade80 !important;
    background: rgba(20,45,30,0.4) !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100% !important;
    background: #4ade80 !important;
    color: #052e16 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #22c55e !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(74,222,128,0.3) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e293b;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    padding: 0.75rem 1.5rem;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #4ade80 !important;
    border-bottom: 2px solid #4ade80 !important;
    background: transparent !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(15,25,41,0.6);
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0b1120; }
::-webkit-scrollbar-thumb { background: #1e3a2f; border-radius: 3px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the model (runs only once per session)."""
    return load_model_and_meta()


def severity_badge(severity: str) -> str:
    """Return HTML badge for a given severity level."""
    mapping = {
        "None"    : ("badge-none", "● HEALTHY"),
        "Low"     : ("badge-low",  "▲ LOW"),
        "Moderate": ("badge-mod",  "▲ MODERATE"),
        "High"    : ("badge-high", "▲ HIGH"),
    }
    cls, label = mapping.get(severity, ("badge-mod", f"▲ {severity.upper()}"))
    return f'<span class="badge {cls}">{label}</span>'


def render_confidence_bar(label: str, confidence: float, is_top: bool = False):
    """Render a styled confidence bar for a prediction."""
    pct        = confidence * 100
    color      = "#4ade80" if is_top else ("#166534" if pct > 10 else "#1e293b")
    label_cls  = "conf-label conf-label-top" if is_top else "conf-label"
    pct_color  = "#4ade80" if is_top else "#475569"
    star       = "★ " if is_top else f"<span style='color:#334155'>{['②','③','④','⑤'][min(int(pct<99),3)]}</span> "

    st.markdown(f"""
    <div class="conf-wrap">
        <div class="conf-row">
            <span class="{label_cls}">{star if is_top else ''}{label}</span>
            <span class="conf-pct" style="color:{pct_color};">{pct:.1f}%</span>
        </div>
        <div class="conf-track">
            <div class="conf-fill" style="width:{pct:.1f}%;background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_donut_gauge(confidence: float):
    """Render a circular confidence gauge using Matplotlib."""
    pct = confidence * 100
    fig, ax = plt.subplots(figsize=(2.8, 2.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#0f1929")
    ax.set_facecolor("#0f1929")

    sizes      = [pct, 100 - pct]
    colors     = ["#4ade80", "#1e293b"]
    wedgeprops = dict(width=0.35, edgecolor="#0f1929", linewidth=3)

    ax.pie(sizes, colors=colors, startangle=90,
           wedgeprops=wedgeprops, counterclock=False)

    ax.text(0,  0.12, f"{pct:.1f}%", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#4ade80",
            fontfamily="monospace")
    ax.text(0, -0.22, "confidence", ha="center", va="center",
            fontsize=6.5, color="#475569", fontfamily="monospace")

    plt.tight_layout(pad=0)
    return fig


def render_bar_chart(predictions: list):
    """Render a horizontal bar chart for top predictions."""
    names  = [p["display_name"][:32] for p in predictions]
    values = [p["confidence"] * 100   for p in predictions]
    colors = ["#4ade80" if i == 0 else "#166534" if i == 1 else "#1e3a2f"
              for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(names) * 0.55)))
    fig.patch.set_facecolor("#0f1929")
    ax.set_facecolor("#0f1929")

    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1],
                   height=0.5, edgecolor="none")

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.8,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color="#475569", fontsize=8, fontfamily="monospace")

    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("Confidence (%)", color="#334155", fontsize=8)
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#1e293b")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
                    font-weight:700;color:#4ade80;letter-spacing:-0.02em;">
            🌱 PDDS
        </div>
        <div style="font-size:0.75rem;color:#334155;margin-top:2px;
                    font-family:'Space Mono',monospace;">
            Plant Disease Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Model status ──
    if is_model_available():
        try:
            with st.spinner("Initialising model…"):
                _model, _meta = get_model()

            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
                <span style="width:8px;height:8px;background:#4ade80;
                             border-radius:50%;display:inline-block;
                             box-shadow:0 0 6px #4ade80;"></span>
                <span style="font-size:0.8rem;color:#4ade80;
                             font-family:'Space Mono',monospace;">MODEL READY</span>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{_meta.get('num_classes','—')}</div>
                    <div class="metric-label">Classes</div>
                </div>""", unsafe_allow_html=True)
            with cols[1]:
                acc = _meta.get('accuracy', _meta.get('accuracy_tta', 0))
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{acc*100:.1f}%</div>
                    <div class="metric-label">Accuracy</div>
                </div>""", unsafe_allow_html=True)

            backbone = _meta.get('backbone', 'EfficientNetB4')
            st.markdown(f"""
            <div class="metric-box" style="margin-top:8px;">
                <div style="font-size:0.75rem;color:#4ade80;
                            font-family:'Space Mono',monospace;">{backbone}</div>
                <div class="metric-label">Backbone Architecture</div>
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Model error: {e}")
    else:
        st.warning("⚠️ Model files not found")
        st.markdown("""
        **Setup required:**
        1. Train using the Colab notebook
        2. Place model files in `models/`
        3. Restart the app
        """)

    st.divider()

    # ── Inference settings ──
    st.markdown('<p class="section-label">Inference Settings</p>',
                unsafe_allow_html=True)

    tta_steps = st.slider(
        "TTA Passes",
        min_value=1, max_value=15, value=5,
        help="Test-Time Augmentation: more passes = higher accuracy, slower inference"
    )
    top_k = st.selectbox(
        "Predictions to Display",
        options=[3, 5, 10], index=1
    )
    show_chart = st.checkbox("Show confidence chart", value=True)

    st.divider()

    # ── Supported plants ──
    st.markdown('<p class="section-label">Supported Species</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem;color:#475569;line-height:2;">
    Apple &nbsp;·&nbsp; Blueberry &nbsp;·&nbsp; Cherry<br>
    Corn &nbsp;·&nbsp; Grape &nbsp;·&nbsp; Orange<br>
    Peach &nbsp;·&nbsp; Pepper &nbsp;·&nbsp; Potato<br>
    Raspberry &nbsp;·&nbsp; Soybean &nbsp;·&nbsp; Squash<br>
    Strawberry &nbsp;·&nbsp; Tomato
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom:0.5rem;">
    <h1>Plant Disease<br>Detection System</h1>
    <p style="color:#475569;font-size:0.95rem;margin-top:0.5rem;">
        Deep learning-based plant pathology classifier &nbsp;·&nbsp;
        EfficientNetB4 backbone &nbsp;·&nbsp; 38 disease classes
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────

tab_detect, tab_about, tab_guide = st.tabs([
    "🔬  Detect Disease",
    "📊  Model Info",
    "📖  Documentation",
])


# ═════════════════════════════════════════════════════════════
# TAB 1 — DETECT
# ═════════════════════════════════════════════════════════════

with tab_detect:

    if not is_model_available():
        st.error(
            "**Model not initialised.** "
            "Please refer to the Documentation tab for setup instructions."
        )
        st.stop()

    try:
        model, meta = get_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: Upload ──────────────────────────────────────────
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Upload Leaf Image")
        st.markdown(
            '<p style="font-size:0.82rem;color:#475569;margin-bottom:1rem;">'
            'Supported formats: JPG · PNG · WebP &nbsp;·&nbsp; Max size: 10 MB'
            '</p>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Drop an image here or click to browse",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            image_bytes = uploaded_file.read()
            pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            st.image(
                pil_image,
                use_container_width=True,
                caption=f"{uploaded_file.name}",
            )

            st.markdown(f"""
            <div style="display:flex;gap:20px;margin:8px 0 16px;flex-wrap:wrap;">
                <span style="font-size:0.78rem;color:#475569;">
                    📐 {pil_image.width} × {pil_image.height} px
                </span>
                <span style="font-size:0.78rem;color:#475569;">
                    💾 {len(image_bytes)/1024:.0f} KB
                </span>
                <span style="font-size:0.78rem;color:#475569;">
                    🎨 {pil_image.mode}
                </span>
            </div>
            """, unsafe_allow_html=True)

            run_inference = st.button("🔬  Analyse Plant", use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;">
                <div style="font-size:3.5rem;margin-bottom:1rem;opacity:0.2;">🍃</div>
                <p style="color:#1e3a2f;font-size:0.875rem;">
                    No image selected
                </p>
            </div>
            """, unsafe_allow_html=True)
            run_inference = False
            image_bytes   = None

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Right: Results ────────────────────────────────────────
    with col_right:

        if uploaded_file and run_inference and image_bytes:

            with st.spinner("Running inference…"):
                t_start     = time.perf_counter()
                predictions = predict(model, meta, image_bytes,
                                      tta_steps=tta_steps, top_k=top_k)
                elapsed_ms  = (time.perf_counter() - t_start) * 1000

            top        = predictions[0]
            info       = top["disease_info"]
            is_healthy = "healthy" in top["class_name"].lower()

            # ── Top prediction card ──
            st.markdown(f"""
            <div class="card-highlight">
                <div class="section-label">PRIMARY DIAGNOSIS</div>
                <div style="font-size:1.5rem;font-weight:700;color:#f0fdf4;
                            font-family:'Space Mono',monospace;
                            letter-spacing:-0.03em;margin:8px 0 10px;
                            line-height:1.2;">
                    {info['emoji']} {top['display_name']}
                </div>
                {severity_badge(info['severity'])}
            </div>
            """, unsafe_allow_html=True)

            # ── Gauge + metrics ──
            g_col, m1, m2 = st.columns([1.3, 1, 1])

            with g_col:
                fig_gauge = render_donut_gauge(top["confidence"])
                st.pyplot(fig_gauge, use_container_width=True)
                plt.close(fig_gauge)

            with m1:
                st.markdown(f"""
                <div class="metric-box" style="margin-bottom:8px;">
                    <div class="metric-value">{elapsed_ms:.0f}<span style="font-size:0.9rem;">ms</span></div>
                    <div class="metric-label">Inference Time</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{tta_steps}×</div>
                    <div class="metric-label">TTA Passes</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                acc = meta.get('accuracy', meta.get('accuracy_tta', 0))
                st.markdown(f"""
                <div class="metric-box" style="margin-bottom:8px;">
                    <div class="metric-value">{acc*100:.1f}%</div>
                    <div class="metric-label">Model Accuracy</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{meta.get('num_classes','—')}</div>
                    <div class="metric-label">Total Classes</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Disease details ──
            with st.expander("📋 Disease Details", expanded=True):
                st.markdown(f"""
                <div class="card" style="margin-bottom:0;">
                    <div class="info-row">
                        <div class="info-title">Description</div>
                        <div class="info-text">{info['description']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-title">💊 Recommended Treatment</div>
                        <div class="info-text">{info['treatment']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-title">🛡️ Prevention Measures</div>
                        <div class="info-text">{info['prevention']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Confidence bars ──
            st.markdown("#### All Predictions")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for p in predictions:
                render_confidence_bar(
                    p["display_name"],
                    p["confidence"],
                    is_top=(p["rank"] == 1)
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Bar chart ──
            if show_chart:
                with st.expander("📊 Confidence Chart", expanded=False):
                    fig_bar = render_bar_chart(predictions)
                    st.pyplot(fig_bar, use_container_width=True)
                    plt.close(fig_bar)

        elif uploaded_file and not run_inference:
            st.markdown("""
            <div class="card" style="text-align:center;padding:5rem 2rem;">
                <div style="font-size:3rem;margin-bottom:1rem;">👆</div>
                <p style="color:#4ade80;font-family:'Space Mono',monospace;font-size:0.9rem;">
                    READY TO ANALYSE
                </p>
                <p style="color:#334155;font-size:0.8rem;">
                    Click <strong style="color:#4ade80;">Analyse Plant</strong> to run the model
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:5rem 2rem;">
                <div style="font-size:3rem;margin-bottom:1rem;opacity:0.15;">🔬</div>
                <p style="color:#1e3a2f;font-family:'Space Mono',monospace;font-size:0.85rem;">
                    AWAITING INPUT
                </p>
                <p style="color:#1e293b;font-size:0.8rem;">
                    Upload a leaf image to begin analysis
                </p>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 2 — MODEL INFO
# ═════════════════════════════════════════════════════════════

with tab_about:

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### Model Architecture")
        st.markdown("""
        <div class="card">
        <div class="info-row">
            <div class="info-title">Backbone</div>
            <div class="info-text">EfficientNetB4 pretrained on ImageNet-1K.
            Selected for its superior accuracy-to-parameter ratio compared to
            ResNet, VGG, and standard EfficientNet variants.</div>
        </div>
        <div class="info-row">
            <div class="info-title">Training Strategy</div>
            <div class="info-text">Two-phase transfer learning:
            (1) Feature extraction — backbone frozen, classification head trained.
            (2) Fine-tuning — top layers unfrozen with a reduced learning rate.</div>
        </div>
        <div class="info-row">
            <div class="info-title">Regularisation</div>
            <div class="info-text">Dropout (0.4), Label Smoothing, Data Augmentation
            (flips, rotations, zoom, brightness, contrast adjustments).</div>
        </div>
        <div class="info-row">
            <div class="info-title">Optimiser</div>
            <div class="info-text">Adam with Cosine Decay learning rate schedule.
            Class-weighted loss to handle dataset imbalance.</div>
        </div>
        <div class="info-row">
            <div class="info-title">Inference Enhancement</div>
            <div class="info-text">Test-Time Augmentation (TTA) averages predictions
            across multiple augmented views, improving robustness by 1–3%.</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Dataset")
        st.markdown("""
        <div class="card">
        <div class="info-row">
            <div class="info-title">Source</div>
            <div class="info-text">PlantVillage Dataset — a publicly available
            benchmark dataset for plant pathology research.</div>
        </div>
        <div class="info-row">
            <div class="info-title">Scale</div>
            <div class="info-text">54,000+ labelled leaf images across 38 classes
            covering 14 plant species and 26 disease conditions.</div>
        </div>
        <div class="info-row">
            <div class="info-title">Split</div>
            <div class="info-text">80% training / 20% validation,
            stratified by class with fixed random seed.</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Performance")

        # Accuracy comparison chart
        fig_acc, ax = plt.subplots(figsize=(5, 3.5))
        fig_acc.patch.set_facecolor("#0f1929")
        ax.set_facecolor("#0f1929")

        phases = ["Baseline\n(Original)", "Phase 1\nHead Only",
                  "Phase 2\nFine-Tune", "With TTA\n(Final)"]
        values = [44, 85, 93, 95]
        colors = ["#1e293b", "#166534", "#22c55e", "#4ade80"]

        bars = ax.bar(phases, values, color=colors, edgecolor="none", width=0.55)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{val}%", ha="center", va="bottom",
                    color="#94a3b8", fontsize=9, fontfamily="monospace")

        ax.set_ylim(0, 108)
        ax.tick_params(colors="#475569", labelsize=8)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#1e293b")
        ax.yaxis.set_visible(False)
        ax.set_title("Validation Accuracy by Training Phase",
                     color="#94a3b8", fontsize=9, pad=10,
                     fontfamily="monospace")

        plt.tight_layout()
        st.pyplot(fig_acc, use_container_width=True)
        plt.close(fig_acc)

        st.caption("*Approximate values. Actual results depend on training run.")

        st.markdown("### Technology Stack")
        stack = [
            ("TensorFlow 2.15",    "Deep Learning Framework"),
            ("Keras 2.15",         "Model API"),
            ("EfficientNetB4",     "CNN Backbone"),
            ("Streamlit",          "Web Interface"),
            ("Google Colab T4",    "Training Hardware"),
            ("PlantVillage",       "Training Dataset"),
        ]
        rows = "".join(f"""
        <div class="info-row">
            <span style="font-family:'Space Mono',monospace;font-size:0.82rem;
                         color:#4ade80;">{name}</span>
            <span style="float:right;font-size:0.75rem;color:#334155;">{role}</span>
        </div>
        """ for name, role in stack)
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 3 — DOCUMENTATION
# ═════════════════════════════════════════════════════════════

with tab_guide:

    st.markdown("### Project Setup Guide")
    st.markdown(
        '<p style="color:#475569;">Follow these steps to set up the system from scratch.</p>',
        unsafe_allow_html=True
    )

    steps = [
        {
            "num"  : "01",
            "title": "Train the Model on Google Colab",
            "body" : """
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/train_colab.ipynb`
3. Enable GPU: **Runtime → Change runtime type → T4 GPU**
4. Run all cells sequentially (~30–60 minutes)
5. Model is saved automatically to your Google Drive
            """,
            "code" : None,
        },
        {
            "num"  : "02",
            "title": "Place Model Files",
            "body" : "Download from Google Drive and place in the `models/` directory:",
            "code" : """models/
├── plant_disease_savedmodel/    ← SavedModel folder (required)
├── model_metadata.json          ← Class names and config
└── class_indices.json           ← Class index mapping""",
        },
        {
            "num"  : "03",
            "title": "Create Virtual Environment",
            "body" : None,
            "code" : """python -m venv venv

# Windows
venv\\Scripts\\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt""",
        },
        {
            "num"  : "04",
            "title": "Run the Application",
            "body" : "The app opens automatically at http://localhost:8501",
            "code" : "streamlit run src/app.py",
        },
    ]

    for step in steps:
        with st.expander(f"Step {step['num']} — {step['title']}", expanded=(step['num'] == '01')):
            if step["body"]:
                st.markdown(step["body"])
            if step["code"]:
                st.code(step["code"], language="bash")

    st.divider()
    st.markdown("### Project Structure")
    st.code("""plant-disease-detection/
├── models/
│   ├── plant_disease_savedmodel/    ← Trained model (SavedModel format)
│   ├── model_metadata.json          ← Model configuration and class names
│   └── class_indices.json           ← Class index mapping
├── notebooks/
│   └── train_colab.ipynb            ← Google Colab training notebook
├── src/
│   ├── app.py                       ← Streamlit web application
│   └── utils/
│       ├── __init__.py
│       └── predictor.py             ← Inference engine
├── requirements.txt                 ← Python dependencies
├── .gitignore
└── README.md""", language="text")

    st.divider()
    st.markdown("### Troubleshooting")
    issues = [
        ("Model not found",          "Ensure `plant_disease_savedmodel/` folder is in `models/`"),
        ("Slow inference",           "Reduce TTA passes in the sidebar settings"),
        ("Wrong predictions",        "Use clear, well-lit photos with the leaf filling the frame"),
        ("pip install fails",        "Use Python 3.10 and activate virtual environment first"),
        ("Port already in use",      "Run `streamlit run src/app.py --server.port 8502`"),
    ]
    rows = "".join(f"""
    <div class="info-row">
        <span style="font-size:0.82rem;color:#94a3b8;">❓ {prob}</span><br>
        <span style="font-size:0.8rem;color:#475569;">→ {fix}</span>
    </div>
    """ for prob, fix in issues)
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)
