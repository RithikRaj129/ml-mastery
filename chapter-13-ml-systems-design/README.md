# Chapter 13 — ML Systems Design

Companion code for Chapter 13 of *Machine Learning Mastery: Mathematics to MLOps*.

| Section | Notebook | Examples | Colab |
|---|---|:-:|:-:|
| 13.1 | [ML System Architecture](./13.01_ml-system-architecture.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.01_ml-system-architecture.ipynb) |
| 13.2 | [Data Pipelines](./13.02_data-pipelines.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.02_data-pipelines.ipynb) |
| 13.3 | [Feature Stores](./13.03_feature-stores.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.03_feature-stores.ipynb) |
| 13.4 | [Batch vs Real-Time Inference](./13.04_batch-vs-real-time-inference.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.04_batch-vs-real-time-inference.ipynb) |
| 13.5 | [Apache Kafka for ML](./13.05_apache-kafka-for-ml.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.05_apache-kafka-for-ml.ipynb) |
| 13.6 | [Apache Airflow](./13.06_apache-airflow.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.06_apache-airflow.ipynb) |
| 13.7 | [Apache Spark for ML](./13.07_apache-spark-for-ml.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.07_apache-spark-for-ml.ipynb) |
| 13.8 | [Distributed Training](./13.08_distributed-training.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.08_distributed-training.ipynb) |
| 13.9 | [GPU vs TPU](./13.09_gpu-vs-tpu.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.09_gpu-vs-tpu.ipynb) |
| 13.10 | [Vector Databases](./13.10_vector-databases.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.10_vector-databases.ipynb) |
| 13.11 | [Scalable AI Systems](./13.11_scalable-ai-systems.ipynb) | 1 | [Open](https://colab.research.google.com/github/RithikRaj129/ml-mastery/blob/main/chapter-13-ml-systems-design/13.11_scalable-ai-systems.ipynb) |

## Source files (`src/`)

These are the standalone files from the book (apps, scripts, configs) so you can run them outside a notebook.

| File | Section |
|---|---|
| [`data_pipeline.py`](./src/data_pipeline.py) | 13.2 |
| [`feature_store_simulation.py`](./src/feature_store_simulation.py) | 13.3 |
| [`batch_inference.py`](./src/batch_inference.py) | 13.4 |
| [`kafka_ml_pipeline.py`](./src/kafka_ml_pipeline.py) | 13.5 |
| [`dags/ml_training_pipeline.py`](./src/dags/ml_training_pipeline.py) | 13.6 |
| [`pyspark_ml.py`](./src/pyspark_ml.py) | 13.7 |
| [`distributed_training.py`](./src/distributed_training.py) | 13.8 |
| [`gpu_usage.py`](./src/gpu_usage.py) | 13.9 |
| [`vector_db.py`](./src/vector_db.py) | 13.10 |
| [`reference_architecture.py`](./src/reference_architecture.py) | 13.11 |

## Notes

Kafka, Airflow, Spark, Ray, FAISS/ChromaDB examples need `requirements-advanced.txt` and, for Kafka/Airflow/Spark, a running service.

[← Back to all chapters](../README.md)
