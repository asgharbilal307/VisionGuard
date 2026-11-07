# flask_app/app.py
import os
import io
import math
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from flask import (Flask, request, render_template,
                   redirect, url_for, jsonify)
from werkzeug.utils import secure_filename

from .inference import load_model, predict
from .database import init_db, log_prediction, get_history

# ── App factory ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

MAX_MB        = float(os.environ.get("MAX_FILE_SIZE_MB", "10"))
MAX_BYTES     = int(MAX_MB * 1024 * 1024)
ALLOWED_EXTS  = {"jpg", "jpeg", "png", "webp", "bmp"}

# Initialise DB session factory and load model at startup
Session = init_db(app)
load_model()

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return ("." in filename and
            filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS)

def is_valid_image(data: bytes) -> bool:
    """Try to open bytes as an image -- catches files that look like
    images by extension but are actually corrupt or malformed."""
    try:
        from PIL import Image
        import io
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    # Validation 1: file present
    if "image" not in request.files:
        return render_template("error.html",
            error="No file part in the request."), 400

    file = request.files["image"]

    if file.filename == "":
        return render_template("error.html",
            error="No file selected."), 400

    # Validation 2: file extension
    if not allowed_file(file.filename):
        return render_template("error.html",
            error=f"File type not allowed. "
                  f"Please upload: {', '.join(ALLOWED_EXTS)}"), 415

    # Read bytes -- do NOT save to disk
    image_bytes = file.read()

    # Validation 3: file size
    if len(image_bytes) > MAX_BYTES:
        return render_template("error.html",
            error=f"File too large. Maximum size is {MAX_MB} MB."), 413

    # Validation 4: actually a valid image
    if not is_valid_image(image_bytes):
        return render_template("error.html",
            error="File cannot be decoded as an image. "
                  "It may be corrupt or mislabelled."), 422

    # Safe filename for display only -- never written to disk
    safe_name = secure_filename(file.filename)
    file_size_kb = round(len(image_bytes) / 1024, 2)

    # Run inference
    result = predict(image_bytes)

    # Log to database
    log_prediction(
        Session,
        filename     = safe_name,
        label        = result["label"],
        confidence   = result["confidence"],
        model_name   = result["model_name"],
        threshold    = result["threshold"],
        file_size_kb = file_size_kb,
        width        = result["width"],
        height       = result["height"],
    )

    return render_template("result.html",
        filename     = safe_name,
        label        = result["label"],
        confidence   = round(result["confidence"] * 100, 1),
        raw_prob     = round(result["raw_prob"], 4),
        threshold    = result["threshold"],
        model_name   = result["model_name"],
        width        = result["width"],
        height       = result["height"],
        file_size_kb = file_size_kb,
        timestamp    = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

@app.route("/history")
def history():
    records = get_history(Session, limit=50)
    return render_template("history.html", records=records)

@app.route("/health")
def health():
    return jsonify({
        "status":    "ok",
        "model":     os.environ.get("MODEL_PATH", "unknown"),
        "threshold": os.environ.get("THRESHOLD", "0.5"),
        "db":        os.environ.get("DATABASE_URL", "sqlite"),
        "timestamp": datetime.utcnow().isoformat(),
    })

# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="Page not found."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html",
        error="Internal server error. Please try again."), 500
