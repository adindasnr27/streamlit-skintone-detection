import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import io

# ─────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────
st.set_page_config(
    page_title="G2G Shade Finder",
    page_icon="💄",
    layout="wide",
)

# ─────────────────────────────────────────
# CUSTOM CSS – TEMA PINK FEMININ
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Root variables */
:root {
    --pink-blush:   #F9C6D0;
    --pink-hot:     #E8638C;
    --pink-soft:    #FDE8EF;
    --pink-deep:    #C2185B;
    --cream:        #FFF8FA;
    --text-dark:    #3D1A25;
    --text-mid:     #7D4455;
    --radius:       16px;
}

/* Background */
.stApp { background-color: var(--cream); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #F9C6D0 0%, #FADADD 60%, #FDE8EF 100%);
    border-right: 1px solid #F0A0B8;
}
[data-testid="stSidebar"] * { color: var(--text-dark) !important; }
[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 8px 12px;
    border-radius: 10px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(232,99,140,0.15);
}

/* Typography */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--pink-deep) !important;
}
p, li, label, div {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-dark);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--pink-blush);
    border-radius: var(--radius);
    padding: 18px 22px;
    box-shadow: 0 2px 12px rgba(232,99,140,0.08);
}
[data-testid="stMetricValue"] {
    color: var(--pink-deep) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #E8638C, #C2185B);
    color: white !important;
    border: none;
    border-radius: 50px;
    padding: 10px 28px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.5px;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* Uploader */
[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed var(--pink-blush);
    border-radius: var(--radius);
    padding: 16px;
}

/* Info / success / warning boxes */
.stAlert { border-radius: var(--radius) !important; }

/* Divider */
hr { border-color: var(--pink-blush); }

/* Badge helper classes */
.badge-ok {
    display:inline-block; background:#E8F5E9; color:#2E7D32;
    border-radius:50px; padding:3px 12px; font-size:0.82rem; font-weight:500;
}
.badge-warn {
    display:inline-block; background:#FFF3E0; color:#E65100;
    border-radius:50px; padding:3px 12px; font-size:0.82rem; font-weight:500;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #FDE8EF, #fff);
    border: 1.5px solid var(--pink-blush);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-top: 16px;
    box-shadow: 0 4px 20px rgba(232,99,140,0.12);
}
.result-card h2 { margin-bottom: 4px; }

/* Recommendation card */
.rec-card {
    background: white;
    border-left: 4px solid var(--pink-hot);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────
CLASS_NAMES = ["dark", "fair", "light"]          # sesuaikan urutan folder dataset
IMG_SIZE    = (224, 224)
MODEL_PATH  = "skin_tone_model.h5"

# G2G shade mapping per kelas skin tone
G2G_RECOMMENDATION = {
    "fair": {
        "shades":   "00 Allegato / 01 Buttercream",
        "skincare": "SPF 50+, Brightening Serum, Vitamin C",
        "tip":      "Kulit fair rentan sunburn – pastikan pakai sunscreen setiap hari!",
        "emoji":    "🌸",
    },
    "light": {
        "shades":   "01 Buttercream / 02 Praline",
        "skincare": "SPF 50+, Niacinamide, Hydrating Toner",
        "tip":      "Tone kulit light cocok dengan nuansa nude-pink untuk tampilan natural.",
        "emoji":    "✨",
    },
    "dark": {
        "shades":   "04 Ginger / 05 Cinnamon",
        "skincare": "Moisturizing Cream, SPF 30+, Shea Butter",
        "tip":      "Kulit dark terlihat glowing dengan foundation yang punya undertone warm.",
        "emoji":    "🍫",
    },
}

# ─────────────────────────────────────────
# HELPER: Load model (cache agar tidak reload tiap interaksi)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model .h5 dari repo."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ─────────────────────────────────────────
# HELPER: Preprocessing gambar
# ─────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Resize, normalize, expand dims → siap masuk model."""
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0   # normalisasi [0,1]
    return np.expand_dims(arr, axis=0)               # (1, 224, 224, 3)

# ─────────────────────────────────────────
# HELPER: Cek kualitas gambar
# ─────────────────────────────────────────
def check_image_quality(pil_img: Image.Image):
    """Return dict dengan status pencahayaan & deteksi wajah."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())

    # Cek pencahayaan
    if brightness < 80:
        light_status, light_msg = "warn", f"⚠️ Terlalu gelap (brightness {brightness:.0f})"
    elif brightness > 200:
        light_status, light_msg = "warn", f"⚠️ Terlalu terang (brightness {brightness:.0f})"
    else:
        light_status, light_msg = "ok", f"✅ Pencahayaan baik (brightness {brightness:.0f})"

    # Deteksi wajah dengan Haar Cascade
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces   = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        face_status, face_msg = "ok",   "✅ Wajah terdeteksi"
    else:
        face_status, face_msg = "warn", "⚠️ Wajah tidak terdeteksi – coba foto lebih dekat"

    return {
        "light_status": light_status, "light_msg": light_msg,
        "face_status":  face_status,  "face_msg":  face_msg,
    }

# ─────────────────────────────────────────
# SIDEBAR NAVIGASI
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💄 G2G Shade Finder")
    st.markdown("*AI-powered skin tone detection*")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["📷 Upload Image", "📊 Dataset Overview", "🔍 Model Insight", "💅 Beauty Recommendation"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Final Project · Machine Learning · 2025")

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "prediction" not in st.session_state:
    st.session_state.prediction    = None   # label string
    st.session_state.probabilities = None   # array float
    st.session_state.uploaded_img  = None   # PIL Image

model = load_model()

# ═══════════════════════════════════════════════════════════════
# HALAMAN 1 – UPLOAD IMAGE
# ═══════════════════════════════════════════════════════════════
if page == "📷 Upload Image":
    st.markdown("# Temukan Shade-mu 🌸")
    st.markdown("Upload foto wajah, dan AI kami akan mendeteksi skin tone kamu secara otomatis.")

    if model is None:
        st.error("⚠️ Model belum ditemukan. Pastikan `skin_tone_model.h5` ada di repo.")
        st.stop()

    uploaded = st.file_uploader("Upload foto wajah (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        pil_img = Image.open(uploaded)
        st.session_state.uploaded_img = pil_img

        col_img, col_info = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("#### Preview")
            st.image(pil_img, use_container_width=True)

        with col_info:
            # ── Cek kualitas gambar ──
            st.markdown("#### Cek Kualitas Foto")
            quality = check_image_quality(pil_img)
            badge_l = "badge-ok" if quality["light_status"] == "ok" else "badge-warn"
            badge_f = "badge-ok" if quality["face_status"]  == "ok" else "badge-warn"
            st.markdown(f"""
                <p><span class="{badge_l}">{quality["light_msg"]}</span></p>
                <p><span class="{badge_f}">{quality["face_msg"]}</span></p>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Prediksi ──
            st.markdown("#### Prediksi Skin Tone")
            with st.spinner("Menganalisis kulit kamu..."):
                tensor = preprocess_image(pil_img)
                probs  = model.predict(tensor, verbose=0)[0]          # array (n_classes,)
                idx    = int(np.argmax(probs))
                label  = CLASS_NAMES[idx]
                conf   = float(probs[idx]) * 100

            # Simpan ke session state
            st.session_state.prediction    = label
            st.session_state.probabilities = probs

            st.markdown(f"""
            <div class="result-card">
                <h2>{G2G_RECOMMENDATION[label]["emoji"]} {label.title()}</h2>
                <p style="font-size:0.95rem;color:#7D4455;">Skin Tone yang terdeteksi</p>
                <hr style="border-color:#F0A0B8;margin:12px 0;">
                <p style="font-size:1.6rem;font-weight:700;color:#C2185B;">{conf:.1f}%</p>
                <p style="font-size:0.85rem;color:#7D4455;">Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.info("Lihat **Beauty Recommendation** di sidebar untuk saran shade G2G kamu! 💄")

# ═══════════════════════════════════════════════════════════════
# HALAMAN 2 – DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Dataset Overview":
    st.markdown("# Dataset Overview 📊")
    st.markdown("Informasi distribusi dataset skin tone yang digunakan untuk melatih model.")

    DATASET_DIR = "dataset"   # ubah jika path berbeda

    # ── Statistik manual (fallback jika dataset tidak ada di cloud) ──
    # Ganti angka ini sesuai hasil eksplorasi notebook kamu
    class_counts = {
        "dark":  500,   # ← sesuaikan
        "fair":  500,
        "light": 500,
    }

    # Jika dataset ada di lokal / repo, hitung otomatis
    if os.path.exists(DATASET_DIR):
        class_counts = {}
        for cls in CLASS_NAMES:
            cls_path = os.path.join(DATASET_DIR, cls)
            if os.path.isdir(cls_path):
                class_counts[cls] = len([
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])

    total = sum(class_counts.values())

    # ── Metrics ──
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Kelas",  len(class_counts))
    col2.metric("Total Gambar", f"{total:,}")
    col3.metric("Model Input",  "224 × 224 px")

    st.markdown("---")

    # ── Distribusi kelas ──
    st.markdown("#### Distribusi Kelas")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    palette = ["#E8638C", "#F9A8C0", "#C2185B"]
    bars = ax.barh(
        list(class_counts.keys()),
        list(class_counts.values()),
        color=palette, height=0.5, edgecolor="white",
    )
    for bar, val in zip(bars, class_counts.values()):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=10, color="#3D1A25")
    ax.set_xlabel("Jumlah Gambar", fontsize=10)
    ax.set_facecolor("#FFF8FA")
    fig.patch.set_facecolor("#FFF8FA")
    ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Sample gambar per kelas (jika dataset ada) ──
    if os.path.exists(DATASET_DIR):
        st.markdown("#### Sample Gambar per Kelas")
        for cls in CLASS_NAMES:
            cls_path = os.path.join(DATASET_DIR, cls)
            if not os.path.isdir(cls_path):
                continue
            imgs = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg",".jpeg",".png"))][:2]
            if not imgs:
                continue
            st.markdown(f"**{cls.title()}**")
            cols = st.columns(2)
            for col, fname in zip(cols, imgs):
                col.image(os.path.join(cls_path, fname), use_container_width=True)
    else:
        st.info("💡 Folder `dataset/` tidak ditemukan di environment ini. Sample gambar hanya tersedia saat run lokal dengan dataset.")

# ═══════════════════════════════════════════════════════════════
# HALAMAN 3 – MODEL INSIGHT
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Model Insight":
    st.markdown("# Model Insight 🔍")

    if st.session_state.probabilities is None:
        st.warning("Upload gambar terlebih dahulu di halaman **📷 Upload Image**.")
        st.stop()

    probs = st.session_state.probabilities
    pred  = st.session_state.prediction

    # ── Confidence score predicted class ──
    conf = float(probs[CLASS_NAMES.index(pred)]) * 100
    st.metric("Confidence Score", f"{conf:.1f}%", help="Probabilitas kelas yang diprediksi")

    st.markdown("---")
    st.markdown("#### Probabilitas Semua Kelas")

    # ── Bar chart horizontal ──
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors  = ["#E8638C" if c == pred else "#F9C6D0" for c in CLASS_NAMES]
    h_bars  = ax.barh(
        [c.title() for c in CLASS_NAMES],
        [p * 100 for p in probs],
        color=colors, height=0.45, edgecolor="white",
    )
    for bar, p in zip(h_bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{p*100:.1f}%", va="center", fontsize=10, color="#3D1A25")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Probabilitas (%)", fontsize=10)
    ax.set_facecolor("#FFF8FA")
    fig.patch.set_facecolor("#FFF8FA")
    ax.spines[["top","right"]].set_visible(False)

    legend = [
        mpatches.Patch(color="#E8638C", label="Predicted class"),
        mpatches.Patch(color="#F9C6D0", label="Other classes"),
    ]
    ax.legend(handles=legend, fontsize=9, framealpha=0)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### Ringkasan Prediksi")
    rows = []
    for cls, p in zip(CLASS_NAMES, probs):
        rows.append({"Skin Tone": cls.title(), "Probabilitas": f"{p*100:.1f}%",
                     "Status": "✅ Predicted" if cls == pred else ""})
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# HALAMAN 4 – BEAUTY RECOMMENDATION
# ═══════════════════════════════════════════════════════════════
elif page == "💅 Beauty Recommendation":
    st.markdown("# Beauty Recommendation 💅")

    if st.session_state.prediction is None:
        st.warning("Upload gambar terlebih dahulu di halaman **📷 Upload Image**.")
        st.stop()

    pred = st.session_state.prediction
    rec  = G2G_RECOMMENDATION[pred]
    img  = st.session_state.uploaded_img

    col_photo, col_rec = st.columns([1, 1.4], gap="large")

    with col_photo:
        if img:
            st.image(img, caption="Foto kamu", use_container_width=True)

    with col_rec:
        st.markdown(f"### {rec['emoji']} Skin Tone: **{pred.title()}**")
        st.markdown("")

        # Card: G2G Shade
        st.markdown(f"""
        <div class="rec-card">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;margin-bottom:4px;">G2G Foundation Shade</p>
            <p style="font-size:1.2rem;font-weight:600;color:#C2185B;margin:0;">{rec['shades']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Card: Skincare
        st.markdown(f"""
        <div class="rec-card">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;margin-bottom:4px;">Skincare Recommendation</p>
            <p style="font-size:1.05rem;font-weight:500;color:#3D1A25;margin:0;">{rec['skincare']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Tip
        st.markdown(f"""
        <div class="rec-card" style="border-left-color:#F9C6D0;">
            <p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:1px;color:#7D4455;margin-bottom:4px;">Beauty Tip</p>
            <p style="font-size:0.95rem;color:#3D1A25;margin:0;">{rec['tip']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption("⚠️ Rekomendasi berdasarkan prediksi AI. Lakukan swatching sebelum membeli.")

    # ── G2G Shade Reference ──
    st.markdown("---")
    st.markdown("#### Semua Shade G2G")
    shades = [
        ("00", "Allegato",    "#F5D5B0"),
        ("01", "Buttercream", "#F2C89A"),
        ("02", "Praline",     "#D4A278"),
        ("03", "Cookies",     "#C49060"),
        ("04", "Ginger",      "#B07848"),
        ("05", "Cinnamon",    "#8B5E3C"),
    ]
    cols = st.columns(6)
    for col, (code, name, color) in zip(cols, shades):
        with col:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="width:52px;height:52px;border-radius:50%;background:{color};
                            margin:0 auto 6px;border:2px solid #F0A0B8;
                            box-shadow:0 2px 8px rgba(0,0,0,0.12);"></div>
                <p style="font-size:0.7rem;font-weight:600;color:#C2185B;margin:0;">{code}</p>
                <p style="font-size:0.72rem;color:#7D4455;margin:0;">{name}</p>
            </div>
            """, unsafe_allow_html=True)
