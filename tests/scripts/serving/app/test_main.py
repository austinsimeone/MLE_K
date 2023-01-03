from scripts.serving.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == ["Model Server"]


def test_get_prediction_w_missing_features():
    x = [{"x1": 0.0, "x2": 0.0, "x3": "Fri", "x4": 0.0, "x6": "California", "x7": "mercedes"}]
    response = client.post(
        "/predict",
        json=x,
    )
    assert response.status_code == 422


def test_get_prediction():
    x = [{"x1": 0.0, "x2": 0.0, "x3": "Fri", "x4": 0.0, "x5": 0.0, "x6": "California", "x7": "mercedes"}]
    y = [1]
    response = client.post(
        "/predict",
        json=x,
    )
    assert response.status_code == 200
    pred = response.json()
    pred = [eval(i) for i in pred]
    assert len(pred) == 1
    assert pred == y
