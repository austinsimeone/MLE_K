from sklearn.base import is_classifier

from src.model.utils import locate_model


def test_locate_model():
    model = locate_model("sklearn.svm.SVC")
    assert is_classifier(model)
