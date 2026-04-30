import streamlit as st
from pathlib import Path
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"

st.title("📈 Model Performance")
st.markdown("---")

st.subheader("Overall Results")
col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy",  "96.5%")
col2.metric("Macro F1-Score", "0.959")
col3.metric("Total test images", "3,203")

st.markdown("---")
st.subheader("Per-Class Accuracy")
st.table({
    "Class": [
        "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
        "Septoria_leaf_spot", "Spider_mites", "Target_Spot",
        "YellowLeaf_Curl_Virus", "mosaic_virus", "healthy"
    ],
    "Accuracy": [
        "98.9%", "84.6% ⚠️", "95.5%", "97.8%",
        "96.5%", "98.4%", "91.2%",
        "98.6%", "94.8%", "99.7%"
    ]
})

st.info("""
**Early_blight** (84.6%) is the weakest class — it is visually similar to Late_blight and Target_Spot
(all produce brown spots). This is consistent with agronomic literature.
""")

st.markdown("---")
st.subheader("Confusion Matrix")
st.image(Image.open(ASSETS / "confusion_matrix.png"), caption="Confusion Matrix — TomatoCNN on test set", use_container_width=True)

st.markdown("---")
st.subheader("Precision / Recall / F1-Score")
st.image(Image.open(ASSETS / "f1_scores.png"), caption="Precision, Recall and F1-Score per class", use_container_width=True)

st.markdown("---")
st.subheader("Key Metric : Recall")
st.write("""
For plant disease detection, **Recall is the most critical metric** — it is better to over-detect
a disease than to miss it. Our model achieves high recall on all classes except Early_blight (84.6%),
which remains acceptable for a CNN trained from scratch.
""")
