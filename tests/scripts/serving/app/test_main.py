from scripts.serving.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == ["Model Server"]


def test_get_prediction_w_missing_features():
    x = [
        {
            "sepal_length_cm": 0,
            "sepal_width_cm": 0,
            "petal_length_cm": 0,
        }
    ]
    response = client.post(
        "/predict",
        json=x,
    )
    assert response.status_code == 422


def test_get_prediction():
    x = [
        {
            "sepal_length_cm": 0,
            "sepal_width_cm": 0,
            "petal_length_cm": 0,
            "petal_width_cm": 0,
        }
    ]
    y = ["setosa"]
    response = client.post(
        "/predict",
        json=x,
    )
    assert response.status_code == 200
    pred = response.json()
    assert len(pred) == 1
    assert pred == y
