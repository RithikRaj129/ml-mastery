# mlflow_complete.py - Complete MLflow integration
# Install: pip install mlflow
# Start UI: mlflow ui --host 0.0.0.0 --port 5000

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                               precision_score, recall_score, confusion_matrix)
import matplotlib.pyplot as plt
import json, os
import warnings; warnings.filterwarnings('ignore')

# ── Setup ─────────────────────────────────────────────────────────
EXPERIMENT_NAME = 'cancer-classifier-v2'
MLFLOW_URI      = 'sqlite:///mlflow.db'  # Local SQLite; use PostgreSQL in production

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Data ─────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ── Helper: compute all metrics ───────────────────────────────────
def compute_metrics(model, X_tr, y_tr, X_te, y_te):
    y_pred      = model.predict(X_te)
    y_prob      = model.predict_proba(X_te)[:, 1]
    cv_auc      = cross_val_score(model, X_tr, y_tr, cv=5, scoring='roc_auc').mean()
    return {
        'test_accuracy':   float(accuracy_score(y_te, y_pred)),
        'test_auc':        float(roc_auc_score(y_te, y_prob)),
        'test_f1':         float(f1_score(y_te, y_pred)),
        'test_precision':  float(precision_score(y_te, y_pred)),
        'test_recall':     float(recall_score(y_te, y_pred)),
        'cv_auc_mean':     float(cv_auc),
        'train_accuracy':  float(model.score(X_tr, y_tr)),
    }

# ── Run experiments: multiple model types ─────────────────────────
experiments = [
    {
        'name':   'gradient-boosting-baseline',
        'model':  Pipeline([('sc',StandardScaler()),('gb',GradientBoostingClassifier(n_estimators=100,random_state=42))]),
        'params': {'model_type':'GradientBoosting','n_estimators':100,'learning_rate':0.1,'max_depth':3}
    },
    {
        'name':   'gradient-boosting-deep',
        'model': Pipeline([('sc',StandardScaler()),('gb',GradientBoostingClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,random_state=42))]),
        'params': {'model_type':'GradientBoosting','n_estimators':200,'learning_rate':0.05,'max_depth':5}
    },
    {
        'name':   'random-forest-200',
        'model':  Pipeline([('sc',StandardScaler()),('rf',RandomForestClassifier(n_estimators=200,random_state=42))]),
        'params': {'model_type':'RandomForest','n_estimators':200}
    },
    {
        'name':   'logistic-regression',
        'model':  Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(C=1.0,max_iter=1000))]),
        'params': {'model_type':'LogisticRegression','C':1.0}
    },
]

print(f'Running {len(experiments)} experiments in MLflow...')
run_ids = []

for exp in experiments:
    with mlflow.start_run(run_name=exp['name']) as run:
        # Log parameters
        mlflow.log_params(exp['params'])
        mlflow.log_param('data_version', 'breast-cancer-v1')
        mlflow.log_param('random_state', 42)
        mlflow.log_param('test_size', 0.2)

        # Train
        exp['model'].fit(X_tr, y_tr)

        # Log metrics
        metrics = compute_metrics(exp['model'], X_tr, y_tr, X_te, y_te)
        mlflow.log_metrics(metrics)

        # Log confusion matrix as artefact
        cm = confusion_matrix(y_te, exp['model'].predict(X_te))
        cm_df = pd.DataFrame(cm, index=['Malignant','Benign'], columns=['Pred Mal','Pred Ben'])
        cm_df.to_csv('/tmp/confusion_matrix.csv')
        mlflow.log_artifact('/tmp/confusion_matrix.csv', 'metrics')

        # Log feature names as tag
        mlflow.set_tag('features', ','.join(data.feature_names[:5]) + '...')
        mlflow.set_tag('data_source', 'sklearn-breast-cancer')

        # Log the model
        signature = mlflow.models.infer_signature(X_tr, exp['model'].predict_proba(X_tr))
        mlflow.sklearn.log_model(
            exp['model'],
            artifact_path='model',
            signature=signature,
            registered_model_name=EXPERIMENT_NAME,
            input_example=X_tr[:3]
        )

        run_ids.append(run.info.run_id)
        print(f'  {exp["name"]:35s}: AUC={metrics["test_auc"]:.4f}  run_id={run.info.run_id[:8]}')

# ── Model Registry workflow ───────────────────────────────────────
print('\n=== MLflow Model Registry ===')
client = MlflowClient(MLFLOW_URI)

# Find best run by test_auc
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['metrics.test_auc DESC'],
    max_results=1
)
best_run = runs[0]
best_auc = best_run.data.metrics['test_auc']
print(f'Best run: {best_run.info.run_id[:8]}  AUC={best_auc:.4f}')

# Get latest model version
try:
    latest_versions = client.get_latest_versions(EXPERIMENT_NAME)
    if latest_versions:
        latest_v = latest_versions[-1]
        print(f'Latest model: version={latest_v.version}  stage={latest_v.current_stage}')

        # Transition to Staging
        client.transition_model_version_stage(
            name=EXPERIMENT_NAME,
            version=latest_v.version,
            stage='Staging',
            archive_existing_versions=False
        )
        print(f'Transitioned v{latest_v.version} → Staging')

        # Load from staging and verify
        model_uri = f'models:/{EXPERIMENT_NAME}/Staging'
        loaded_model = mlflow.sklearn.load_model(model_uri)
        staging_auc  = roc_auc_score(y_te, loaded_model.predict_proba(X_te)[:,1])
        print(f'Loaded from staging: AUC={staging_auc:.4f} ✓')

        # Promote to Production if meets threshold
        if staging_auc >= 0.95:
            client.transition_model_version_stage(
                name=EXPERIMENT_NAME,
                version=latest_v.version,
                stage='Production',
            )
            print(f'Promoted to PRODUCTION: v{latest_v.version}  AUC={staging_auc:.4f}')
        else:
            print(f'AUC {staging_auc:.4f} below threshold - NOT promoted to Production')

except Exception as e:
    print(f'Registry note: {e}')

# ── Compare all runs ─────────────────────────────────────────────
print('\nAll experiment results (sorted by AUC):')
all_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['metrics.test_auc DESC']
)
print(f'{'Run Name':35s} {'AUC':>8} {'Accuracy':>10} {'F1':>8}')
print('='*65)
for run in all_runs:
    name = run.data.tags.get('mlflow.runName', run.info.run_id[:8])
    auc  = run.data.metrics.get('test_auc', 0)
    acc  = run.data.metrics.get('test_accuracy', 0)
    f1   = run.data.metrics.get('test_f1', 0)
    print(f'{name:35s} {auc:>8.4f} {acc:>10.4f} {f1:>8.4f}')
