"""
=============================================================
Plant Disease Detection System
=============================================================
File    : predictor.py
Purpose : Model loading, image preprocessing, and inference
=============================================================
"""

import json
import io
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "plant_disease_savedmodel"
META_PATH  = MODELS_DIR / "model_metadata.json"


# ─────────────────────────────────────────────
# Disease Knowledge Base
# ─────────────────────────────────────────────

DISEASE_INFO = {
    "healthy": {
        "severity"   : "None",
        "emoji"      : "✅",
        "description": (
            "The plant appears completely healthy with no visible "
            "signs of infection, discoloration, or structural damage."
        ),
        "treatment"  : "No treatment required. Continue regular care and monitoring.",
        "prevention" : (
            "Maintain proper watering schedules, ensure good air circulation "
            "between plants, and perform routine inspections for early detection."
        ),
    },
    "blight": {
        "severity"   : "High",
        "emoji"      : "🔴",
        "description": (
            "Blight is an aggressive fungal or bacterial disease that causes "
            "rapid browning, wilting, and death of plant tissue. It spreads "
            "quickly under warm, humid conditions."
        ),
        "treatment"  : (
            "Immediately remove and destroy all infected plant material. "
            "Apply copper-based fungicide or chlorothalonil every 7–10 days. "
            "Avoid working with plants when wet."
        ),
        "prevention" : (
            "Use certified disease-free seeds and transplants. Practice crop "
            "rotation every season. Avoid overhead irrigation and ensure "
            "adequate plant spacing for airflow."
        ),
    },
    "rust": {
        "severity"   : "Moderate",
        "emoji"      : "🟠",
        "description": (
            "Rust disease appears as orange, yellow, or brown powdery pustules "
            "on the undersides of leaves. It weakens the plant over time by "
            "disrupting photosynthesis."
        ),
        "treatment"  : (
            "Apply sulfur-based or triazole fungicide at first sign of infection. "
            "Remove and dispose of heavily infected leaves. "
            "Repeat application every 10–14 days as needed."
        ),
        "prevention" : (
            "Choose rust-resistant plant varieties. Avoid wetting foliage "
            "during irrigation. Remove plant debris at end of growing season."
        ),
    },
    "scab": {
        "severity"   : "Moderate",
        "emoji"      : "🟠",
        "description": (
            "Scab causes dark, rough, corky lesions on leaves, stems, and fruit. "
            "It thrives in cool, wet weather and can significantly reduce yield quality."
        ),
        "treatment"  : (
            "Apply fungicide containing captan or myclobutanil starting at "
            "bud break. Remove fallen leaves and infected debris promptly."
        ),
        "prevention" : (
            "Plant scab-resistant varieties. Prune trees to improve airflow "
            "and light penetration. Rake and destroy fallen leaves each autumn."
        ),
    },
    "spot": {
        "severity"   : "Low",
        "emoji"      : "🟡",
        "description": (
            "Leaf spot diseases produce circular to irregular lesions on foliage, "
            "often surrounded by yellow halos. They can be caused by fungi or "
            "bacteria and spread through water splash."
        ),
        "treatment"  : (
            "Remove and destroy infected leaves. Apply copper-based bactericide "
            "or appropriate fungicide. Avoid overhead watering."
        ),
        "prevention" : (
            "Water at the base of plants. Maintain plant spacing. "
            "Clean up garden debris regularly to reduce pathogen populations."
        ),
    },
    "mold": {
        "severity"   : "Moderate",
        "emoji"      : "🟠",
        "description": (
            "Mold — including powdery and downy mildew — appears as white, "
            "grey, or purple dusty growth on leaf surfaces. It thrives in "
            "humid, poorly ventilated conditions."
        ),
        "treatment"  : (
            "Apply neem oil, potassium bicarbonate, or sulfur-based fungicide. "
            "Improve air circulation around plants. Remove severely affected leaves."
        ),
        "prevention" : (
            "Avoid excessive nitrogen fertilisation. Provide adequate plant spacing. "
            "Water in the morning so foliage dries before evening."
        ),
    },
    "virus": {
        "severity"   : "High",
        "emoji"      : "🔴",
        "description": (
            "Plant viruses cause mosaic patterns, chlorosis, leaf curl, "
            "stunted growth, and distorted fruit. They spread primarily "
            "through insect vectors such as aphids and whiteflies."
        ),
        "treatment"  : (
            "There is no cure for viral infections. Remove and destroy "
            "infected plants immediately. Control insect vector populations "
            "using appropriate insecticides or biological controls."
        ),
        "prevention" : (
            "Use certified virus-free planting material. Control aphid and "
            "whitefly populations. Disinfect pruning tools between plants. "
            "Use reflective mulches to deter insects."
        ),
    },
    "rot": {
        "severity"   : "High",
        "emoji"      : "🔴",
        "description": (
            "Rot diseases cause soft, water-soaked, discoloured tissue that "
            "deteriorates rapidly. They are caused by fungal or bacterial "
            "pathogens and are worsened by excess moisture and poor drainage."
        ),
        "treatment"  : (
            "Remove all infected plant material immediately. Improve soil "
            "drainage. Apply copper hydroxide or appropriate fungicide/bactericide "
            "to remaining healthy tissue."
        ),
        "prevention" : (
            "Ensure well-draining soil. Avoid overwatering and waterlogged "
            "conditions. Rotate crops and avoid planting in previously "
            "infected soil for at least two seasons."
        ),
    },
    "default": {
        "severity"   : "Moderate",
        "emoji"      : "⚠️",
        "description": (
            "A plant disease or abnormality has been detected. "
            "Carefully examine the affected plant for additional symptoms "
            "and consult an agricultural expert for a precise diagnosis."
        ),
        "treatment"  : (
            "Isolate the affected plant to prevent potential spread. "
            "Consult your local agricultural extension service or plant "
            "pathologist for targeted treatment recommendations."
        ),
        "prevention" : (
            "Maintain good garden hygiene, practice crop rotation, ensure "
            "proper watering techniques, and regularly scout for early signs "
            "of disease or pest activity."
        ),
    },
}


def get_disease_info(class_name: str) -> dict:
    """Return disease information based on keywords in the class name."""
    name_lower = class_name.lower()
    for keyword, info in DISEASE_INFO.items():
        if keyword != "default" and keyword in name_lower:
            return info
    return DISEASE_INFO["default"]


def format_class_name(raw: str) -> str:
    """
    Convert folder-style class name to a human-readable string.
    Example: 'Tomato___Early_blight' → 'Tomato — Early Blight'
    """
    if "___" in raw:
        plant, disease = raw.split("___", 1)
        plant   = plant.replace("_", " ").strip().title()
        disease = disease.replace("_", " ").strip().title()
        return f"{plant} — {disease}"
    return raw.replace("_", " ").title()


# ─────────────────────────────────────────────
# Model Management
# ─────────────────────────────────────────────

def is_model_available() -> bool:
    """Check whether the trained model and metadata files exist."""
    return MODEL_PATH.exists() and META_PATH.exists()


def load_model_and_meta() -> tuple:
    """
    Load the SavedModel and its associated metadata.

    Returns:
        tuple: (model, metadata_dict)

    Raises:
        FileNotFoundError: If model or metadata files are missing.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}\n"
            "Please train the model using the Colab notebook and "
            "place the 'plant_disease_savedmodel' folder in the models/ directory."
        )
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at: {META_PATH}"
        )

    # Load as TensorFlow SavedModel — version-agnostic format
    model = tf.saved_model.load(str(MODEL_PATH))

    with open(META_PATH, "r") as f:
        metadata = json.load(f)

    return model, metadata


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def preprocess_image(image_bytes: bytes, img_size: tuple) -> np.ndarray:
    """
    Load and preprocess an image for model inference.

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        img_size:    Target (width, height) tuple expected by the model.

    Returns:
        Float32 numpy array of shape (H, W, 3) with pixel values in [0, 255].
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(img_size, Image.LANCZOS)
    array = np.array(image, dtype=np.float32)
    return array


def predict(
    model,
    metadata: dict,
    image_bytes: bytes,
    tta_steps: int = 5,
    top_k: int = 5,
) -> list:
    """
    Run inference on an uploaded image using Test-Time Augmentation (TTA).

    TTA runs multiple augmented versions of the image through the model
    and averages the predictions, improving robustness and accuracy.

    Args:
        model:       Loaded TensorFlow SavedModel.
        metadata:    Dictionary loaded from model_metadata.json.
        image_bytes: Raw bytes of the image to classify.
        tta_steps:   Number of augmented forward passes (1 = no TTA).
        top_k:       Number of top predictions to return.

    Returns:
        List of prediction dicts sorted by confidence (highest first).
        Each dict contains: rank, class_name, display_name,
        confidence, and disease_info.
    """
    img_size    = tuple(metadata["img_size"])
    class_names = metadata["class_names"]
    num_classes = metadata["num_classes"]

    # Preprocess image
    array  = preprocess_image(image_bytes, img_size)
    batch  = np.expand_dims(array, axis=0)           # (1, H, W, 3)
    tensor = tf.constant(batch, dtype=tf.float32)

    # Define TTA augmentation pipeline
    tta_augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Run inference with TTA
    infer       = model.signatures["serving_default"]
    accumulated = np.zeros(num_classes, dtype=np.float32)

    for _ in range(tta_steps):
        augmented = tta_augment(tensor, training=True)
        output    = infer(augmented)
        output_key = list(output.keys())[-1]
        probs      = output[output_key].numpy()[0]
        accumulated += probs

    averaged_probs = accumulated / tta_steps

    # Build top-k results
    top_indices = np.argsort(averaged_probs)[::-1][:top_k]
    results = []

    for rank, idx in enumerate(top_indices, start=1):
        class_name = class_names[idx]
        results.append({
            "rank"        : rank,
            "class_name"  : class_name,
            "display_name": format_class_name(class_name),
            "confidence"  : float(averaged_probs[idx]),
            "disease_info": get_disease_info(class_name),
        })

    return results
