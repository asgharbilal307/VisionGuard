# flask_app/inference.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import io

IMG_SIZE = 224

# Model is loaded ONCE at startup -- not per request
# Loading per request would add ~2-5 seconds latency per call
# and exhaust memory under concurrent load
_model      = None
_model_name = None

def load_model():
    global _model, _model_name
    model_path = os.environ.get("MODEL_PATH", "models/champion_model.keras")

    if not os.path.isabs(model_path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"[Inference] Loading model from {model_path} ...")
    _model = tf.keras.models.load_model(model_path)
    _model_name = os.path.basename(model_path)
    print(f"[Inference] Model loaded: {_model_name}")
    return _model

def get_model():
    if _model is None:
        load_model()
    return _model

def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image bytes exactly as during training:
    - Decode to RGB
    - Resize to 224x224 (same as tf.image.resize in training)
    - Convert to float32 array
    - Add batch dimension
    Training and serving preprocessing MUST match -- any difference
    causes silent accuracy degradation (distribution shift).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype="float32")
    arr = np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)
    return arr, width, height

def predict(image_bytes: bytes) -> dict:
    model     = get_model()
    threshold = float(os.environ.get("THRESHOLD", "0.5"))

    arr, width, height = preprocess(image_bytes)
    prob  = float(model.predict(arr, verbose=0)[0][0])
    label = "AI-Generated" if prob >= threshold else "Real"

    return {
        "label":      label,
        "confidence": prob if label == "AI-Generated" else 1.0 - prob,
        "raw_prob":   prob,
        "threshold":  threshold,
        "model_name": _model_name or "champion_model.keras",
        "width":      width,
        "height":     height,
    }
