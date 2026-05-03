# 🌱 Plant Disease Detection System

A deep learning-based web application for automated plant disease classification from leaf images. Built with **EfficientNetB4** transfer learning and served through an interactive **Streamlit** interface.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25+-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

The Plant Disease Detection System enables farmers, researchers, and agronomists to diagnose plant diseases instantly by uploading a photo of a leaf. The system classifies the image into one of **38 disease categories** across **14 plant species** and provides actionable treatment and prevention recommendations.

---

## ✨ Features

- 🔬 **Real-time disease classification** from leaf photographs
- 🎯 **95%+ validation accuracy** using EfficientNetB4 transfer learning
- 📊 **Top-5 predictions** with confidence scores and visual charts
- 💊 **Treatment & prevention advice** for each detected disease
- ⚡ **Test-Time Augmentation (TTA)** for improved inference robustness
- 🌿 **38 disease classes** spanning 14 plant species
- 🖥️ **Clean, professional web UI** built with Streamlit

---

## 🏗️ System Architecture

```
plant-disease-detection/
├── models/
│   ├── plant_disease_savedmodel/    ← Trained model (SavedModel format)
│   ├── model_metadata.json          ← Model configuration and class names
│   └── class_indices.json           ← Class-to-index mapping
├── notebooks/
│   └── train_colab.ipynb            ← Google Colab training notebook
├── src/
│   ├── app.py                       ← Streamlit web application
│   └── utils/
│       ├── __init__.py
│       └── predictor.py             ← Inference engine
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Model Details

| Component         | Details                              |
|-------------------|--------------------------------------|
| Backbone          | EfficientNetB4 (ImageNet pretrained) |
| Framework         | TensorFlow 2.15 / Keras 2.15         |
| Input Size        | 160 × 160 × 3                        |
| Output Classes    | 38                                   |
| Training Strategy | Two-phase transfer learning          |
| Optimiser         | Adam + Cosine Decay LR schedule      |
| Regularisation    | Dropout, Label Smoothing, Augmentation |
| Dataset           | PlantVillage (54,000+ images)        |
| Training Hardware | Google Colab T4 GPU                  |

### Training Phases

| Phase             | Description                          | Accuracy  |
|-------------------|--------------------------------------|-----------|
| Baseline          | Original notebook (broken rescaling) | ~44%      |
| Phase 1           | Feature extraction (frozen backbone) | ~85%      |
| Phase 2           | Fine-tuning (top layers unfrozen)    | ~93%      |
| Final (with TTA)  | Test-Time Augmentation applied       | **~95%+** |

---

## 🌿 Supported Plant Species

| Plant       | Diseases Covered                                        |
|-------------|----------------------------------------------------------|
| Apple       | Apple Scab, Black Rot, Cedar Apple Rust, Healthy         |
| Blueberry   | Healthy                                                   |
| Cherry      | Powdery Mildew, Healthy                                   |
| Corn        | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape       | Black Rot, Esca, Leaf Blight, Healthy                    |
| Orange      | Citrus Greening (Huanglongbing)                          |
| Peach       | Bacterial Spot, Healthy                                   |
| Pepper      | Bacterial Spot, Healthy                                   |
| Potato      | Early Blight, Late Blight, Healthy                        |
| Raspberry   | Healthy                                                   |
| Soybean     | Healthy                                                   |
| Squash      | Powdery Mildew                                            |
| Strawberry  | Leaf Scorch, Healthy                                      |
| Tomato      | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10.11
- Git
- Google account (for Colab training)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection
```

### Step 2 — Train the Model (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `notebooks/train_colab.ipynb`
3. Enable GPU: **Runtime → Change runtime type → T4 GPU**
4. Run all cells (approx. 30–60 minutes)
5. Download the generated `plant_disease_savedmodel/` folder and JSON files
6. Place them in the `models/` directory

### Step 3 — Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run the Application

```bash
streamlit run src/app.py
```

The application opens automatically at **http://localhost:8501**

---

## 📸 Usage

1. Navigate to the **Detect Disease** tab
2. Upload a clear, well-lit photograph of a plant leaf
3. Click **Analyse Plant**
4. View the diagnosis, confidence scores, and treatment recommendations

**Tips for best results:**
- Ensure the leaf fills most of the frame
- Use natural lighting where possible
- Avoid blurry or heavily shadowed images
- Supported formats: JPG, PNG, WebP

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | Ensure `plant_disease_savedmodel/` is in `models/` |
| Slow inference | Reduce TTA passes in the sidebar |
| Wrong predictions | Use clear, well-lit leaf photos |
| `pip install` fails | Use Python 3.10 and activate venv first |
| Port already in use | Run with `--server.port 8502` flag |

---

## 📚 References

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)




