import streamlit as st
from pathlib import Path
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"

st.title("📊 Dataset Analysis")
st.markdown("---")

st.subheader("Dataset Source")
st.write("""
- **Source** : PlantVillage dataset (Kaggle — emmarex/plantdisease)
- **Scope** : Tomato plant diseases only
- **Total images** : 16,011
- **Number of classes** : 10
- **Split** : 80% training (12,808) / 20% test (3,203)
""")

st.markdown("---")
st.subheader("Distribution by Plant")
try:
    img = Image.open(ASSETS / "distribution.png")
    st.image(img, caption="Image distribution per plant — PlantVillage", width='stretch')
except Exception as e:
    st.error(f"Error loading image: {e}")

st.markdown("---")
st.subheader("Tomato Classes Distribution")
try:
    img = Image.open(ASSETS / "tomato_distribution.png")
    st.image(img, caption="Image count and percentage per tomato disease class", width='stretch')
except Exception as e:
    st.error(f"Error loading image: {e}")

st.markdown("---")
st.subheader("Sample Images per Class")
try:
    img = Image.open(ASSETS / "tomato_disease.png")
    st.image(img, caption="One example image per tomato disease class", width='stretch')
except Exception as e:
    st.error(f"Error loading image: {e}")

st.markdown("---")
st.subheader("Why Tomato?")
st.info("""
Tomato was selected because it has the highest number of images (16,011 — 77.6% of the dataset)
and the most disease classes (10), making it the most suitable plant for training a robust classifier.
""")
