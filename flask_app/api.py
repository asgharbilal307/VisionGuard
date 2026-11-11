import os
import hmac

from flask import Flask, request, jsonify
from PIL import UnidentifiedImageError

from .inference import load_model, predict

app = Flask(__name__)

MAX_FILE_SIZE_MB = 10
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

# Load the model once when the API starts.
load_model()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model": os.path.basename(
            os.environ.get("MODEL_PATH", "models/champion_model.keras")
        )
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict_image():

    # Verify API secret.
    expected_secret = os.environ.get("API_SECRET")

    if not expected_secret:
        return jsonify({
            "error": "API authentication is not configured"
        }), 503

    supplied_secret = request.headers.get("X-API-Secret", "")

    if not hmac.compare_digest(supplied_secret, expected_secret):
        return jsonify({
            "error": "Unauthorized"
        }), 401

    # Check image upload.
    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({
            "error": "No file selected"
        }), 400

    extension = file.filename.rsplit(".", 1)[-1].lower()

    if "." not in file.filename or extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Unsupported image format"
        }), 400

    image_bytes = file.read()

    if not image_bytes:
        return jsonify({
            "error": "Uploaded image is empty"
        }), 400

    try:
        result = predict(image_bytes)

        return jsonify({
            "success": True,
            "prediction": result
        }), 200

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "Invalid or corrupted image"
        }), 400


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image exceeds the 10 MB limit"
    }), 413
