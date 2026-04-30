import streamlit as st
from PIL import Image

st.set_page_config(page_title="Tomato Disease Classifier", page_icon="🍅", layout="wide")

st.title("🍅 Tomato Disease Classifier")
st.subheader("Deep Learning Project — aivancity School of AI & Data")
st.markdown("---")

st.markdown("""
### Project Overview
This application presents the full pipeline of a **CNN-based tomato disease classifier**
built from scratch using PyTorch, trained on the PlantVillage dataset.

### Navigate using the sidebar
| Page | Content |
|---|---|
| **Dataset** | Dataset exploration and class distribution |
| **Model Training** | Architecture, training configuration and curves |
| **Performance** | Confusion matrix, F1-score and per-class accuracy |
| **Demo** | Upload a tomato leaf image and get a prediction |

---
### Key Results
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Accuracy",   "96.5%")
col2.metric("Macro F1-Score",  "0.959")
col3.metric("Classes",         "10")
col4.metric("Training images", "12,808")

st.markdown("---")
st.markdown("""
**Dataset** : PlantVillage (Kaggle) — Tomato diseases only (10 classes, 16,011 images)

**Model** : Custom CNN (3 conv blocks + Dropout) trained with PyTorch and data augmentation
""")

st.sidebar.title("Scanner pour mobile")
qr_image = Image.open("qr_code.png")
st.sidebar.image(qr_image, caption="Scannez pour tester sur votre téléphone !")
