# flask_app/app.py
import os
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from .database import init_db, log_prediction, get_history


# App configuration
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

MAX_MB = float(os.environ.get("MAX_FILE_SIZE_MB", "10"))
MAX_BYTES = int(MAX_MB * 1024 * 1024)

ALLOWED_EXTS = {"jpg", "jpeg", "png", "webp", "bmp"}

# Render API configuration
RENDER_API_URL = os.environ.get("RENDER_API_URL", "").rstrip("/")
API_SECRET = os.environ.get("API_SECRET", "")

# Initialise database
Session = init_db(app)


# Helpers
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def is_valid_image(data: bytes) -> bool:
    try:
        from PIL import Image
        import io

        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict_route():

    # Check whether an image was uploaded
    if "image" not in request.files:
        return render_template(
            "error.html", error="No file part in the request."
        ), 400

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "error.html", error="No file selected."
        ), 400

    # Validate file extension
    if not allowed_file(file.filename):
        return render_template(
            "error.html",
            error=f"File type not allowed. Please upload: {', '.join(ALLOWED_EXTS)}",
        ), 415

    image_bytes = file.read()

    # Validate file size
    if len(image_bytes) > MAX_BYTES:
        return render_template(
            "error.html",
            error=f"File too large. Maximum size is {MAX_MB} MB.",
        ), 413

    # Validate image contents
    if not is_valid_image(image_bytes):
        return render_template(
            "error.html",
            error="File cannot be decoded as an image. It may be corrupt or mislabelled.",
        ), 422

    safe_name = secure_filename(file.filename)
    file_size_kb = round(len(image_bytes) / 1024, 2)

    # Check API configuration
    if not RENDER_API_URL or not API_SECRET:
        return render_template(
            "error.html",
            error="Prediction API is not configured.",
        ), 503

    # Send image to Render API
    try:
        response = requests.post(
            f"{RENDER_API_URL}/api/predict",
            headers={"X-API-Secret": API_SECRET},
            files={
                "image": (
                    safe_name,
                    image_bytes,
                    file.mimetype or "application/octet-stream",
                )
            },
            timeout=(15, 180),
        )

        if response.status_code != 200:
            return render_template(
                "error.html",
                error="Prediction service could not process the image. Please try again.",
            ), 502

        api_data = response.json()
        result = api_data["prediction"]

    except requests.RequestException:
        return render_template(
            "error.html",
            error="Could not connect to the prediction service. Please try again.",
        ), 503

    except (ValueError, KeyError, TypeError):
        return render_template(
            "error.html",
            error="Invalid response received from the prediction service.",
        ), 502

    # Save prediction in database
    log_prediction(
        Session,
        filename=safe_name,
        label=result["label"],
        confidence=result["confidence"],
        model_name=result["model_name"],
        threshold=result["threshold"],
        file_size_kb=file_size_kb,
        width=result["width"],
        height=result["height"],
    )

    # Display prediction result
    return render_template(
        "result.html",
        filename=safe_name,
        label=result["label"],
        confidence=round(result["confidence"] * 100, 1),
        raw_prob=round(result["raw_prob"], 4),
        threshold=result["threshold"],
        model_name=result["model_name"],
        width=result["width"],
        height=result["height"],
        file_size_kb=file_size_kb,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


@app.route("/history")
def history():
    records = get_history(Session, limit=50)
    return render_template("history.html", records=records)


@app.route("/health")
def health():
    database_url = os.environ.get("DATABASE_URL", "")

    return jsonify({
        "status": "ok",
        "prediction_api_configured": bool(RENDER_API_URL and API_SECRET),
        "database_configured": bool(database_url),
        "timestamp": datetime.utcnow().isoformat(),
    })


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return render_template(
        "error.html", error="Page not found."
    ), 404


@app.errorhandler(500)
def server_error(e):
    return render_template(
        "error.html",
        error="Internal server error. Please try again.",
    ), 500
