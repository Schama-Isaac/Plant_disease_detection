import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

# ── Model definition ──────────────────────────────────────────────────────────
class TomatoCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(TomatoCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(APP_DIR / "class_mapping.json") as f:
        class_mapping = json.load(f)
    model = TomatoCNN(num_classes=10)
    model.load_state_dict(torch.load(APP_DIR / "tomato_cnn.pth", map_location="cpu"))
    model.eval()
    return model, class_mapping

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🍅 Live Demo — Tomato Disease Detector")
st.write("Upload a photo of a tomato leaf — the model will predict the disease.")
st.markdown("---")

model, class_mapping = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    # Predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output     = model(input_tensor)
        probs      = torch.softmax(output, dim=1)[0]
        pred_idx   = probs.argmax().item()
        confidence = probs[pred_idx].item() * 100

    pred_name = class_mapping[str(pred_idx)].replace("Tomato__", "").replace("Tomato_", "")

    with col2:
        st.subheader("Prediction")
        if "healthy" in pred_name:
            st.success(f"✅ **{pred_name}**")
        else:
            st.error(f"⚠️ **{pred_name}**")
        st.metric("Confidence", f"{confidence:.1f}%")

        st.markdown("**Top 3 predictions:**")
        top3 = probs.topk(3)
        for prob, idx in zip(top3.values, top3.indices):
            name = class_mapping[str(idx.item())].replace("Tomato__", "").replace("Tomato_", "")
            st.write(f"- {name} : {prob.item()*100:.1f}%")
