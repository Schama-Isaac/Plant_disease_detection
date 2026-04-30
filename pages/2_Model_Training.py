import streamlit as st
from pathlib import Path
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"

st.title("🧠 Model Architecture & Training")
st.markdown("---")

st.subheader("Model : TomatoCNN (from scratch)")
st.write("A custom Convolutional Neural Network built with PyTorch.")

st.code("""
TomatoCNN(
  features:
    Conv2d(3 → 32)  + ReLU + MaxPool2d   # detects edges and basic textures
    Conv2d(32 → 64) + ReLU + MaxPool2d   # detects complex patterns
    Conv2d(64 → 128)+ ReLU + MaxPool2d   # detects disease-specific shapes

  classifier:
    Flatten
    Linear(32768 → 256) + ReLU + Dropout(0.4)
    Linear(256 → 10)                     # 10 disease classes
)
""", language="text")

st.markdown("---")
st.subheader("Architecture Choices")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Why CNN?**")
    st.write("CNNs detect local spatial patterns (spots, textures, shapes) — ideal for plant disease images.")
    st.markdown("**Why 3 conv blocks?**")
    st.write("Each block learns increasingly complex features: edges → textures → disease shapes.")

with col2:
    st.markdown("**Why Dropout(0.4)?**")
    st.write("Prevents overfitting by randomly disabling 40% of neurons during training.")
    st.markdown("**Why Adam optimizer?**")
    st.write("Adaptive learning rate — faster convergence than standard SGD.")

st.markdown("---")
st.subheader("Training Configuration")
st.table({
    "Parameter": ["Epochs", "Batch size", "Learning rate", "Optimizer", "Loss function", "Image size"],
    "Value":     ["10", "32", "0.001", "Adam", "CrossEntropyLoss", "128 × 128"]
})

st.markdown("---")
st.subheader("Data Augmentation (training only)")
st.write("""
To reduce overfitting, the following augmentations were applied during training:
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness ±0.2)
""")

st.markdown("---")
st.subheader("Training Curves")
st.image(Image.open(ASSETS / "training_curves.png"), caption="Loss and Accuracy over 10 epochs", use_container_width=True)
