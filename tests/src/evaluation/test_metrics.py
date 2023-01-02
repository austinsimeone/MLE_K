from src.evaluation.metrics import multi_class_metrics


def test_multi_class_metrics():
    y_true = [0, 0, 2, 3]
    y_pred = [0, 1, 2, 3]
    metrics = multi_class_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
