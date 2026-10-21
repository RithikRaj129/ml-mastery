# reference_architecture.py
# Documents the complete ML system architecture with decision guidance

REFERENCE_ARCHITECTURE = {
    'Data Ingestion Layer': {
        'batch_ingestion': {
            'tools': ['Apache Spark', 'dbt', 'AWS Glue', 'Fivetran'],
            'patterns': ['Incremental load (append)', 'Full refresh (overwrite)', 'Merge/upsert'],
            'storage': ['S3/GCS (data lake)', 'BigQuery/Redshift (warehouse)', 'Delta Lake (lakehouse)'],
        },
        'stream_ingestion': {
            'tools': ['Apache Kafka', 'AWS Kinesis', 'Google Pub/Sub'],
            'patterns': ['Event sourcing', 'CDC (Change Data Capture)', 'Log aggregation'],
        },
    },

    'Feature Engineering Layer': {
        'batch_features': {
            'tools': ['Spark SQL', 'dbt', 'Pandas in Airflow tasks'],
            'schedule': 'Hourly/daily/weekly',
            'storage': 'Parquet in S3; BigQuery tables',
        },
        'streaming_features': {
            'tools': ['Kafka Streams', 'Flink', 'Spark Streaming'],
            'latency': '< 5 seconds',
            'storage': 'Redis, DynamoDB (online store)',
        },
        'feature_store': {
            'commercial': ['Tecton', 'Feast (OSS)', 'Hopsworks'],
            'responsibilities': ['Feature registry', 'Consistent serving (train=serve)', 'Monitoring'],
        },
    },

    'Training Layer': {
        'single_machine': ['sklearn', 'XGBoost', 'LightGBM', 'CatBoost'],
        'distributed': ['Ray Train', 'Horovod', 'PyTorch DDP', 'DeepSpeed'],
        'experiment_tracking': ['MLflow', 'Weights & Biases', 'Neptune'],
        'model_registry': ['MLflow Registry', 'SageMaker Registry', 'HuggingFace Hub'],
        'hyperparameter_tuning': ['Optuna', 'Ray Tune', 'SageMaker AMT'],
        'training_orchestration': ['Airflow', 'Kubeflow Pipelines', 'SageMaker Pipelines'],
    },

    'Serving Layer': {
        'batch_serving': {
            'tools': ['Spark batch scoring', 'Airflow jobs', 'AWS Batch'],
            'output': 'Predictions written to database or S3',
        },
        'real_time_serving': {
            'frameworks': ['FastAPI', 'TorchServe', 'Triton Inference Server', 'BentoML'],
            'infrastructure': ['Kubernetes', 'AWS ECS', 'Cloud Run'],
            'optimisations': ['Model caching', 'Request batching', 'GPU sharing', 'ONNX'],
        },
        'ab_testing': ['Feature flags', 'Traffic splitting', 'Multi-armed bandit'],
    },

    'Monitoring Layer': {
        'model_monitoring': {
            'tools': ['Evidently AI', 'WhyLogs', 'Fiddler', 'Arize'],
            'metrics': ['Prediction distribution', 'Data drift', 'Concept drift', 'Performance'],
        },
        'system_monitoring': {
            'tools': ['Prometheus + Grafana', 'Datadog', 'CloudWatch'],
            'metrics': ['Latency p50/p95/p99', 'Throughput', 'Error rate', 'GPU utilisation'],
        },
        'alerting': ['PagerDuty', 'Slack webhooks', 'Email', 'OpsGenie'],
    },
}

# ── Design principles for scalable ML systems ─────────────────────
DESIGN_PRINCIPLES = [
    ('Decouple training from serving',
     'Training and inference should be independent services with shared model registry'),
    ('Feature consistency',
     'Same feature computation in training and serving - use feature store'),
    ('Immutable artifacts',
     'Models, features, and datasets are versioned and immutable once created'),
    ('Fail safe',
     'Fall back to simpler model or cached prediction when primary model fails'),
    ('Monitor everything',
     'Alert on input data drift before model performance degrades - act proactively'),
    ('Shadow deployment',
     'Run new model in shadow mode (receives traffic but predictions not served) before cutover'),
    ('Gradual rollout',
     'Canary: serve 1% → 5% → 25% → 100%; roll back instantly if metrics degrade'),
    ('Cost optimisation',
     'Use spot/preemptible instances for training; CPU inference for simple models'),
    ('Infrastructure as code',
     'Every deployment is reproducible via Terraform, Pulumi, or CDK'),
    ('Observability',
     'Log every prediction with input features for debugging and retraining'),
]

print('=== Scalable AI System Design Principles ===')
for i, (name, desc) in enumerate(DESIGN_PRINCIPLES, 1):
    print(f'\n{i:2d}. {name}')
    print(f'    {desc}')

# ── ML Platform team sizing guide ─────────────────────────────────
print('\n=== ML Platform Sizing Guide ===')
stages = {
    'Stage 1 (0-10 models)': {
        'team':   '1-2 data scientists + 1 ML engineer',
        'serving':'Flask/FastAPI on a single server or Cloud Run',
        'infra':  'Managed cloud services; no Kubernetes yet',
        'key_tool':'MLflow for experiment tracking',
    },
    'Stage 2 (10-100 models)': {
        'team':   '5-10 data scientists + 3-5 ML engineers',
        'serving':'Kubernetes + Helm; A/B testing framework',
        'infra':  'Airflow for orchestration; feature store',
        'key_tool':'Feature store + model registry',
    },
    'Stage 3 (100+ models)': {
        'team':   'Dedicated ML Platform team (10+)',
        'serving':'Triton Inference Server; auto-scaling; multi-region',
        'infra':  'Full MLOps platform; data mesh; real-time features',
        'key_tool':'Custom ML platform or Kubeflow/Vertex AI',
    },
}
for stage, info in stages.items():
    print(f'\n{stage}:')
    for key, val in info.items():
        print(f'  {key:<12}: {val}')
