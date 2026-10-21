# dags/ml_training_pipeline.py - Production Airflow ML DAG
# Deploy by placing in Airflow's dags/ directory

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ── Default arguments applied to all tasks ────────────────────────
DEFAULT_ARGS = {
    'owner':            'ml-team',
    'depends_on_past':  False,
    'email':            ['ml-alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry':   False,
    'retries':          2,
    'retry_delay':      timedelta(minutes=5),
    'execution_timeout':timedelta(hours=2),
}

# ── Task functions ────────────────────────────────────────────────
def extract_features(**context):
    '''Extract and compute features for training.'''
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_breast_cancer

    logger.info(f'Extracting features for execution_date={context["ds"]}')

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Save to shared location (S3, GCS, or local in dev)
    output_path = f'/tmp/features_{context["ds"]}.parquet'
    df.to_parquet(output_path, index=False)

    logger.info(f'Features saved: {output_path} ({len(df)} rows)')

    # Push path to XCom for next task
    return output_path


def train_model(**context):
    '''Train the ML model.'''
    import pandas as pd
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    # Pull feature path from XCom
    ti = context['task_instance']
    feature_path = ti.xcom_pull(task_ids='extract_features')

    df = pd.read_parquet(feature_path)
    feature_cols = [c for c in df.columns if c != 'target']

    X_tr, X_te, y_tr, y_te = train_test_split(
        df[feature_cols], df['target'], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])
    pipeline.fit(X_tr, y_tr)

    model_path = f'/tmp/model_{context["ds"]}.joblib'
    joblib.dump(pipeline, model_path)

    logger.info(f'Model trained and saved: {model_path}')
    return model_path


def evaluate_model(**context):
    '''Evaluate model and return metrics.'''
    import pandas as pd
    import joblib
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score

    ti = context['task_instance']
    model_path   = ti.xcom_pull(task_ids='train_model')
    feature_path = ti.xcom_pull(task_ids='extract_features')

    pipeline = joblib.load(model_path)
    df = pd.read_parquet(feature_path)
    feature_cols = [c for c in df.columns if c != 'target']

    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, df[feature_cols], df['target'],
                                  cv=cv, scoring='roc_auc')

    metrics = {
        'auc_mean': float(cv_scores.mean()),
        'auc_std':  float(cv_scores.std()),
        'model_path': model_path,
    }
    logger.info(f'Evaluation: AUC={metrics["auc_mean"]:.4f} ± {metrics["auc_std"]:.4f}')
    return metrics


def quality_gate(**context):
    '''Branch: deploy if AUC >= threshold, else alert and stop.'''
    ti = context['task_instance']
    metrics = ti.xcom_pull(task_ids='evaluate_model')

    MIN_AUC = 0.95
    if metrics['auc_mean'] >= MIN_AUC:
        logger.info(f'Quality gate PASSED: AUC={metrics["auc_mean"]:.4f} >= {MIN_AUC}')
        return 'deploy_model'
    else:
        logger.warning(f'Quality gate FAILED: AUC={metrics["auc_mean"]:.4f} < {MIN_AUC}')
        return 'alert_team'


def deploy_model(**context):
    '''Deploy approved model to production.'''
    import shutil
    ti = context['task_instance']
    metrics = ti.xcom_pull(task_ids='evaluate_model')
    model_path = metrics['model_path']
    shutil.copy(model_path, '/app/models/model_latest.joblib')
    logger.info(f'Model deployed: {model_path} → /app/models/model_latest.joblib')


# ── DAG definition ────────────────────────────────────────────────
with DAG(
    dag_id            = 'ml_training_pipeline',
    default_args      = DEFAULT_ARGS,
    description       = 'Weekly ML model retraining pipeline',
    schedule_interval = '0 2 * * 1',   # Every Monday at 2 AM
    start_date        = datetime(2024, 1, 1),
    catchup           = False,           # Don't backfill missed runs
    tags              = ['ml', 'training', 'production'],
    max_active_runs   = 1,               # Prevent concurrent runs
) as dag:

    start = DummyOperator(task_id='start')
    end   = DummyOperator(task_id='end', trigger_rule=TriggerRule.NONE_FAILED)

    t_extract  = PythonOperator(task_id='extract_features',   python_callable=extract_features)
    t_train    = PythonOperator(task_id='train_model',        python_callable=train_model)
    t_evaluate = PythonOperator(task_id='evaluate_model',     python_callable=evaluate_model)
    t_gate     = BranchPythonOperator(task_id='quality_gate', python_callable=quality_gate)
    t_deploy   = PythonOperator(task_id='deploy_model',       python_callable=deploy_model)
    t_alert    = BashOperator(task_id='alert_team',
                               bash_command='echo "MODEL QUALITY GATE FAILED - manual review required"')

    # DAG structure: start → extract → train → evaluate → gate → deploy or alert → end
    start >> t_extract >> t_train >> t_evaluate >> t_gate
    t_gate >> [t_deploy, t_alert]
    [t_deploy, t_alert] >> end

print('DAG defined: ml_training_pipeline (Monday 2 AM)')
print('Tasks: start → extract → train → evaluate → gate → [deploy|alert] → end')
