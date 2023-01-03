# Kohls Take Home Exam

*Austin Simeone*

*1/3/2022*

## Setup

### Dependencies

- python >= 3.9
- pip >= 21.0
- docker >= 20.10

### Installation

1. Build module: `pip install .`
1. Install requirements: `pip install -r requirements.txt`

## Usage

### Options for running the ML pipeline

- Using the prefect flow: `python scripts/training/pipeline.py`

### Review Prefect Run 
- Launch UI: `prefect orion start`

### Review Run Metrics

1. Launch MLFlow UI: `mlflow ui`
2. Navigate to the UI: [mlflow](http://127.0.0.1:5000)

### Launch Model Server

1. Build and Run Container: `./scripts/serving/scripts/run.sh`

### Development

- Run tests: `nox`

## Technologies

### Machine Learning

- [sklearn](https://scikit-learn.org/0.21/documentation.html) - general machine learning
- [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/) - neural networks

### Model Server

- [docker](https://docs.docker.com/) - server container
- [fastapi](https://fastapi.tiangolo.com/) - web framework

### MLOps

- [mlflow](https://www.mlflow.org/docs/latest/index.html) - model management
- [prefect](https://docs.prefect.io/) - worfklow management

### Testing

- [nox](https://nox.thea.codes/en/stable/) - test automation
- [flake8](https://flake8.pycqa.org/en/latest/) - linter
- [black](https://black.readthedocs.io/en/stable/) - formatter
- [mypy](https://mypy.readthedocs.io/en/stable/getting_started.html) - static type checker
- [pytest](https://docs.pytest.org/en/stable/contents.html) - testing


### Theoretical Implementation
*The below arch diagram is cloud agnostic. I used azure because the icons were free!*

<img style="float: right;" src="imgs/ml flow architecture.svg">


