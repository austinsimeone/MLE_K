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

### Current File Structure
```bash
|   config.yaml
|   conftest.py
|   exam_readme.md
|   noxfile.py
|   README.md
|   requirements.txt
|   setup.cfg
|   setup.py
|
+---data
|   +---processed
|   |       train.csv
|   |       
|   +---processed_split
|   |       train_test.csv
|   |       train_train.csv
|   |       train_val.csv
|   |       
|   \---raw
|           score.csv
|           train.csv
|           
+---imgs
|       ml flow architecture.svg
|
+---mlruns
|   |   model.yml
|   |
|   +---0
|   |       meta.yaml
|   |       
|   +---1
|   |   |   meta.yaml
|   |   |   
|   |   \---9499a2ebe2934331adc6ce02fe80960c
|   |       |   meta.yaml
|   |       |   
|   |       +---artifacts
|   |       |   \---sklearn_model
|   |       |           conda.yaml
|   |       |           MLmodel
|   |       |           model.pkl
|   |       |           python_env.yaml
|   |       |           requirements.txt
|   |       |           
|   |       +---metrics
|   |       |       test_accuracy
|   |       |       test_f1_score
|   |       |       test_mcc
|   |       |       train_accuracy
|   |       |       train_f1_score
|   |       |       train_mcc
|   |       |       val_accuracy
|   |       |       val_f1_score
|   |       |       val_mcc
|   |       |       
|   |       +---params
|   |       |       seed
|   |       |       
|   |       \---tags
|   |               mlflow.log-model.history
|   |               mlflow.runName
|   |               mlflow.source.git.commit
|   |               mlflow.source.name
|   |               mlflow.source.type
|   |               mlflow.user
|   |               model_type
|   |
+---notebooks
|       Model_Training.ipynb
|       test.ipynb
|       
+---scripts
|   |   __init__.py
|   |   
|   +---orchestrating
|   |       deploy.py
|   |       
|   +---serving
|   |   |   docker-compose.yml
|   |   |   Dockerfile
|   |   |   __init__.py
|   |   |   
|   |   +---app
|   |   |   |   main.py
|   |   |   |   __init__.py
|   |   | 
|   |   |           
|   |   +---scripts
|   |   |       console.sh
|   |   |       rebuild.sh
|   |   |       run.sh
|   |   
|   +---training
|   |   |   pipeline.py
|   |   |   preprocess.py
|   |   |   register.py
|   |   |   split.py
|   |   |   train_val_test.py
|   |   |   __init__.py
|
|           
+---src
|   |   __init__.py
|   |   
|   +---data
|   |       __init__.py
|   |       
|   +---evaluation
|   |   |   metrics.py
|   |   |   __init__.py
|   |   
|   |           
|   +---model
|   |   |   pytorch.py
|   |   |   utils.py
|   |   |   __init__.py
|   |   
|   |           
|   +---processing
|       |   features.py
|       |   target.py
|       |   __init__.py
|           
+---tests
|   \---scripts
|       \---serving
|           \---app
|               |   test_main.py
```

### Theoretical Implementation
*The below arch diagram is cloud agnostic. I used azure because the icons were free!*

<img style="float: right;" src="imgs/ml flow architecture.svg">
