import os
from unittest.mock import patch

# Configure environment BEFORE importing the Flask application
os.environ["MODEL_PATH"] = "models/champion_model.keras"
os.environ["DATABASE_URL"] = "sqlite:///test.db"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["THRESHOLD"] = "0.5"

# Prevent the actual trained model from loading during tests
with patch("flask_app.inference.load_model"):
    from flask_app.app import app


import pytest


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "ok"


def test_index(client):
    response = client.get("/")

    assert response.status_code == 200


def test_no_file(client):
    response = client.post("/predict", data={})

    assert response.status_code == 400


def test_wrong_extension(client):
    from io import BytesIO

    data = {
        "image": (BytesIO(b"fake content"), "test.txt")
    }

    response = client.post(
        "/predict",
        data=data,
        content_type="multipart/form-data"
    )

    assert response.status_code == 415
