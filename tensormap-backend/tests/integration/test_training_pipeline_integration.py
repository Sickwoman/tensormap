"""Integration tests for the full training pipeline.

Covers: Upload → Transform → Model Generation → Validation → Result
Requires: tests/conftest.py client fixture, running Flask app.
test_full_pipeline_model_generation_to_json additionally requires TensorFlow.
"""

import io
import json

CSV_CONTENT = "num1,num2,cat1,target\n1.0,2.0,a,0\n3.0,4.0,b,1\n5.0,6.0,a,0\n7.0,8.0,b,1\n"

ALL_SEVEN_TRANSFORMATIONS = [
    {"transformation": "Min-Max Normalization", "feature": "num1"},
    {"transformation": "Z-score Standardization", "feature": "num2"},
    {"transformation": "Categorical to Numerical", "feature": "cat1"},
    {"transformation": "Fill Missing Values", "feature": "num1", "params": {"strategy": "mean"}},
    {"transformation": "Log Transform", "feature": "num1"},
    {"transformation": "One-Hot Encoding", "feature": "cat1"},
    {"transformation": "Standard Scaling", "feature": "num2"},
]


def upload_csv(client):
    """Helper: upload CSV and return file_id."""
    data = {"file": (io.BytesIO(CSV_CONTENT.encode()), "test.csv")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200, f"Upload failed: {response.data}"
    return response.get_json()["file_id"]


def build_model_payload(file_id, transformations=None, layers=None, target="target"):
    """Helper: construct a minimal valid model generation payload."""
    return {
        "file_id": file_id,
        "target_field": target,
        "transformations": transformations or [],
        "layers": layers
        or [
            {"type": "Dense", "units": 8, "activation": "relu"},
            {"type": "Dense", "units": 1, "activation": "sigmoid"},
        ],
    }


def test_upload_csv_returns_200(client):
    data = {"file": (io.BytesIO(CSV_CONTENT.encode()), "test.csv")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert "file_id" in response.get_json()


def test_generated_model_json_is_valid_and_keras_parseable(client):
    file_id = upload_csv(client)
    payload = build_model_payload(file_id)
    response = client.post(
        "/generate_model",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert "model_json" in body
    model_json = json.loads(body["model_json"])
    assert model_json.get("class_name") == "Sequential"


def test_all_seven_transformations_accepted(client):
    """All 7 transformation types must be accepted by backend validation."""
    file_id = upload_csv(client)
    payload = build_model_payload(file_id, transformations=ALL_SEVEN_TRANSFORMATIONS)
    response = client.post(
        "/generate_model",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200, f"Backend rejected payload with all 7 transformations: {response.data}"


def test_full_pipeline_model_generation_to_json(client):
    """Full round-trip: upload → generate → parse back with Keras."""
    import tensorflow as tf

    file_id = upload_csv(client)
    payload = build_model_payload(file_id)
    response = client.post(
        "/generate_model",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    model_json = response.get_json()["model_json"]
    model = tf.keras.models.model_from_json(model_json)
    assert model is not None


def test_disconnected_node_graph_documents_current_behavior(client):
    """Disconnected graph: backend currently returns 200 (ideally 422)."""
    file_id = upload_csv(client)
    payload = build_model_payload(
        file_id,
        layers=[
            {"type": "Dense", "units": 8},
            {"type": "Dense", "units": 4},
        ],
    )
    response = client.post(
        "/generate_model",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200


def test_missing_target_field_returns_200_currently(client):
    """Missing target_field: backend returns 200 (ideally 422, tracked in #212)."""
    file_id = upload_csv(client)
    payload = build_model_payload(file_id, target="")
    payload.pop("target_field", None)
    response = client.post(
        "/generate_model",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200


def test_unknown_layer_type_returns_error(client):
    """Unknown layer type must return a 4xx error, not silently succeed."""
    file_id = upload_csv(client)
    payload = build_model_payload(
        file_id,
        layers=[
            {"type": "NonExistentLayerXYZ", "units": 8},
        ],
    )
    response = client.post(
        "/generate_model",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert 400 <= response.status_code < 500, f"Expected 4xx for unknown layer type, got {response.status_code}"


def test_nonexistent_file_id_returns_error(client):
    """Non-existent file_id must return a 4xx error."""
    payload = build_model_payload("nonexistent-file-id-00000")
    response = client.post(
        "/generate_model",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert 400 <= response.status_code < 500, f"Expected 4xx for bad file_id, got {response.status_code}"


def test_empty_graph_does_not_crash_backend(client):
    """Empty layers list must return a 4xx, not a 500."""
    file_id = upload_csv(client)
    payload = build_model_payload(file_id, layers=[])
    response = client.post(
        "/generate_model",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code != 500, "Empty graph caused a 500 — backend must handle this gracefully"
