# Chapter 12 — Model Deployment

Companion code for Chapter 12 of *Machine Learning Mastery: Mathematics to MLOps*.

| Section | Notebook | Examples | Colab |
|---|---|:-:|:-:|
| 12.1 | [Saving ML Models](./12.01_saving-ml-models.ipynb) | 3 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-12-model-deployment/12.01_saving-ml-models.ipynb) |
| 12.2 | [Flask Deployment](./12.02_flask-deployment.ipynb) | 2 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-12-model-deployment/12.02_flask-deployment.ipynb) |
| 12.3 | [FastAPI Deployment](./12.03_fastapi-deployment.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-12-model-deployment/12.03_fastapi-deployment.ipynb) |
| 12.4 | [Streamlit Dashboards](./12.04_streamlit-dashboards.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-12-model-deployment/12.04_streamlit-dashboards.ipynb) |
| 12.5 | [Cloud Deployment](./12.05_cloud-deployment.ipynb) | 4 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-12-model-deployment/12.05_cloud-deployment.ipynb) |
| 12.6 | [API Integration](./12.06_api-integration.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-12-model-deployment/12.06_api-integration.ipynb) |
| 12.7 | [CI/CD for ML](./12.07_ci-cd-for-ml.ipynb) | 3 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-12-model-deployment/12.07_ci-cd-for-ml.ipynb) |

## Source files (`src/`)

These are the standalone files from the book (apps, scripts, configs) so you can run them outside a notebook.

| File | Section |
|---|---|
| [`app.py`](./src/app.py) | 12.2 |
| [`test_api.py`](./src/test_api.py) | 12.2 |
| [`main.py`](./src/main.py) | 12.3 |
| [`streamlit_app.py`](./src/streamlit_app.py) | 12.4 |
| [`Dockerfile`](./src/Dockerfile) | 12.5 |
| [`requirements.txt`](./src/requirements.txt) | 12.5 |
| [`docker-compose.yml`](./src/docker-compose.yml) | 12.5 |
| [`deploy_commands.sh`](./src/deploy_commands.sh) | 12.5 |
| [`ml_client.py`](./src/ml_client.py) | 12.6 |
| [`.github/workflows/ml_cicd.yml`](./src/.github/workflows/ml_cicd.yml) | 12.7 |
| [`train.py`](./src/train.py) | 12.7 |
| [`tests/test_api.py`](./src/tests/test_api.py) | 12.7 |

## Notes

Flask, FastAPI, Streamlit, Docker and a GitHub Actions workflow. Files in `src/` are the real app files from the book.

The CI/CD workflow is kept at `src/.github/workflows/ml_cicd.yml` so it does **not** run automatically on this repository. Copy it to the root `.github/workflows/` folder of your own project to use it.

[← Back to all chapters](../README.md)
