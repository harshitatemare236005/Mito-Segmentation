import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
from PIL import Image

from backend.model import load_unet_model
from utils.image_utils import load_tif_image, preprocess_image
from utils.patch_utils import extract_patches

# ──────────────────────────────────────────────────────────────────────────────
# 🧬  Streamlit UI CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mitochondria Segmentation • U‑Net",
    page_icon="🧬",
    layout="wide",
)

# Custom CSS for subtle typography tweaks
st.markdown(
    """
    <style>
        .title {font-size:3rem; font-weight:700; margin-bottom:0.25em;}
        .subheader {font-size:1.3rem; color:#4C4D64; margin-top:0;}
        .small {font-size:0.9rem; color:#6C6F7F;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 🏷️  HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    "<div class='title'>🧬 Mitochondria Segmentation with U‑Net</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subheader'>Interactive EM‑image analysis demo – part of my Master\'s thesis project</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 📚  SIDEBAR   (Project overview + glossary)
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔎 Project Overview")
    st.markdown(
        """
        **Goal**  
        Semantic segmentation of mitochondrial structures in electron‑microscope (EM) imagery using a deep **U‑Net** architecture.

        **Pipeline**
        1. Upload `.tif` microscopy slice  
        2. Pre‑process & normalise  
        3. **U‑Net** inference → probability mask  
        4. Threshold + post‑process  
        5. Patch extraction for downstream analysis
        """
    )

    st.header("📚 Glossary")
    st.markdown(
        """
        - **Mitochondria** – Cellular *powerhouses* visible as elongated organelles under EM.  
        - **U‑Net** – Encoder‑decoder CNN with skip connections, ideal for biomedical segmentation.  
        - **Segmentation mask** – Binary image where `1 = mitochondria`, `0 = background`.  
        - **Patch** – Cropped region around each detected mitochondrion.
        """
    )
    st.markdown("---")
    st.caption("© 2025 Your Name – Thesis project")

# ──────────────────────────────────────────────────────────────────────────────
# 📂  FILE UPLOAD
# ──────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "📂 Upload a 2‑D microscopy `.tif` slice", type=["tif", "tiff"]
)

# ──────────────────────────────────────────────────────────────────────────────
# 🧠  MODEL LOADING  (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Loading U‑Net weights…")
def get_model():
    """Load U‑Net model once and cache it for subsequent runs."""
    return load_unet_model()

# ──────────────────────────────────────────────────────────────────────────────
# 🔄  MAIN WORKFLOW
# ──────────────────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    # Read & display original image
    image = load_tif_image(uploaded_file)

    # Display original + predicted side‑by‑side
    col_orig, col_pred = st.columns(2, gap="large")

    with col_orig:
        st.subheader("🔬 Original Image")
        st.image(image, use_column_width=True)

    # Prepare model
    model = get_model()

    # Inference
    with st.spinner("🧠 Running inference…"):
        input_image = preprocess_image(image)
        prob_map = model.predict(input_image, verbose=0)[0, :, :, 0]

    with col_pred:
        st.subheader("🟢 Predicted Probability Mask")
        st.image(prob_map, use_column_width=True, clamp=True)

    # ──────────────────────────────────────────────────────────────────────
    # 📦  PATCH EXTRACTION SECTION
    # ──────────────────────────────────────────────────────────────────────

    st.markdown("## 📦 Patch Extraction")
    threshold = st.slider("Mask threshold", 0.0, 1.0, 0.5, 0.05)
    binary_mask = (prob_map > threshold).astype(np.uint8)
    patches = extract_patches(binary_mask)

    st.info(f"**{len(patches)} patches** detected above threshold {threshold:0.2f}.")

    if patches:
        patch_cols = st.columns(5)
        for i, patch in enumerate(patches[:10]):
            with patch_cols[i % 5]:
                st.image(
                    patch,
                    caption=f"Patch {i+1}",
                    width=120,
                    use_column_width=False,
                )

    # ──────────────────────────────────────────────────────────────────────
    # 🔧  ADVANCED DETAILS EXPANDER
    # ──────────────────────────────────────────────────────────────────────

    with st.expander("🔧 Advanced details – model JSON"):
        st.code(model.to_json(indent=2))

else:
    st.markdown(
        "<div class='small'>Awaiting upload…  Demo uses a ~31 M‑parameter U‑Net trained on the 2024 MITO‑EM challenge dataset.</div>",
        unsafe_allow_html=True,
    )
