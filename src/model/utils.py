from pydoc import locate
import mlflow


def fetch_logged_data(run_id: str):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


def locate_model(model_type: str):
    model = locate(model_type)
    return model
