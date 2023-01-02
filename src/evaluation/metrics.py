from typing import Iterable, Dict
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def multi_class_metrics(y_true: Iterable[float], y_pred: Iterable[float], phase: str = "") -> Dict[str, float]:
    if phase != "":
        phase += "_"
    metrics = {
        f"{phase}accuracy": accuracy_score(y_true, y_pred),
        f"{phase}f1_score": f1_score(y_true, y_pred, average="micro"),
        f"{phase}mcc": matthews_corrcoef(y_true, y_pred),
    }
    return metrics
