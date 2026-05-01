# 🍅 Tomato Disease Classifier

A deep learning-powered web application for detecting and classifying diseases in tomato plants using a custom CNN model trained with PyTorch. Built with Streamlit for easy interactive use.

## 📋 Overview

This project presents a complete machine learning pipeline for tomato disease classification using the PlantVillage dataset. The application identifies 10 different tomato plant conditions:

- **Bacterial spot**
- **Early blight**
- **Late blight**
- **Leaf Mold**
- **Septoria leaf spot**
- **Spider mites (Two-spotted)**
- **Target Spot**
- **Tomato Yellow Leaf Curl Virus**
- **Tomato Mosaic Virus**
- **Healthy**

### Key Performance Metrics
- **Test Accuracy**: 96.5%
- **Macro F1-Score**: 0.959
- **Training Images**: 12,808
- **Classes**: 10

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download the repository**
   ```bash
   cd Plant_disease_detection-dev
   ```
   > **Note**: All files and folders mentioned in this README should be located inside the `Plant_disease_detection-dev` folder. Ensure you are working within this directory for the app to run correctly.

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📱 Features & Navigation

The app includes four main pages accessible from the sidebar:

### 🎯 **Dataset** (`1_Dataset.py`)
- Explore the PlantVillage dataset
- Visualize class distribution
- Understand the training data composition

### 🔧 **Model Training** (`2_Model_Training.py`)
- Review the CNN architecture
- View training configuration and hyperparameters
- Inspect training curves and learning progress

### 📊 **Performance** (`3_Performance.py`)
- Analyze confusion matrix
- View per-class accuracy metrics
- Evaluate F1-scores and other performance indicators

### 🧪 **Demo** (`4_Demo.py`)
- Upload tomato leaf images
- Get real-time disease predictions
- View confidence scores for each classification

## 🏗️ Model Architecture

The classifier uses a custom **Convolutional Neural Network (CNN)** with:
- **3 Convolutional blocks** with ReLU activation
- **Max pooling** layers for dimension reduction
- **Dropout** (0.4) for regularization
- **256-unit** fully connected hidden layer
- **10-class output layer** (softmax)

Input image size: 256×256×3 (RGB)

## 📦 Project Structure

```
Plant_disease_detection-dev/
├── app.py                      # Main Streamlit app
├── pages/
│   ├── 1_Dataset.py            # Dataset exploration
│   ├── 2_Model_Training.py     # Training details
│   ├── 3_Performance.py        # Performance metrics
│   └── 4_Demo.py               # Interactive prediction demo
├── tomato_cnn.pth             # Pre-trained model weights
├── class_mapping.json         # Disease class definitions
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📚 Dataset

**Source**: PlantVillage (Kaggle)  
**Total Images**: 16,011  
**Classes**: 10 (9 diseases + 1 healthy)  
**Dataset Link**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

## 🛠️ Dependencies

All required packages are listed in `requirements.txt`:
- **streamlit** - Web application framework
- **torch** - Deep learning framework
- **torchvision** - Computer vision utilities
- **pillow** - Image processing

## 💻 System Requirements

- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB for model and dependencies
- **OS**: Windows, macOS, or Linux

## 🔄 Model Usage

The pre-trained model (`tomato_cnn.pth`) is loaded automatically by the Demo page. To make predictions:

1. Navigate to the **Demo** page
2. Upload a tomato leaf image (JPG, PNG, etc.)
3. The model will classify the diseased condition and confidence scores

## 📝 Training Details

The model was trained with:
- **Data Augmentation**: Rotation, flipping, color jittering
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy
- **Batch Size**: 32
- **Epochs**: Optimized for convergence

## 🐛 Troubleshooting

### Port already in use
If port 8501 is busy, run:
```bash
streamlit run app.py --server.port 8502
```

### Model not found
Ensure `tomato_cnn.pth` exists in the project root directory.

### Image upload issues
Try using standard image formats (JPG, PNG) and ensure the file size is reasonable (< 5MB).

## 📄 License

This project is part of the aivancity School of AI & Data curriculum.

## 👨‍🎓 About

**Institution**: aivancity School of AI & Data  
**Project Type**: Deep Learning / Computer Vision  
**Framework**: PyTorch + Streamlit

## 📧 Contact & Support

For issues or questions about the project, please refer to the repository's issues section.

---

**Happy Classifying! 🍅🤖**
