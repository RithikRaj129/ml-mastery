# churn_mlops_pipeline.py - Complete production MLOps case study
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import json
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import ks_2samp
import warnings; warnings.filterwarnings('ignore')

print('='*70)
print('PRODUCTION ML CASE STUDY: CUSTOMER CHURN PREDICTION')
print('='*70)

# ── Step 1: Data Simulation ───────────────────────────────────────
print('\n[STEP 1] Generating customer dataset...')
np.random.seed(42)
n_customers = 50_000

df = pd.DataFrame({
    'customer_id':          np.arange(n_customers),
    'tenure_months':        np.random.randint(1, 72, n_customers),
    'monthly_spend':        np.random.exponential(85, n_customers),
    'support_tickets_90d':  np.random.poisson(1.2, n_customers),
    'login_frequency_30d':  np.random.poisson(8, n_customers),
    'products_used':        np.random.randint(1, 6, n_customers),
    'last_activity_days':   np.random.exponential(15, n_customers),
    'nps_score':            np.random.randint(1, 11, n_customers),
    'contract_type':        np.random.choice(['monthly','annual','2year'], n_customers, p=[0.5,0.35,0.15]),
    'payment_method':       np.random.choice(['credit','debit','bank'], n_customers, p=[0.55,0.30,0.15]),
    'region':               np.random.choice(['North','South','East','West'], n_customers),
})

# Create realistic churn signal
churn_prob = (
    0.05
    + 0.15 * (df['tenure_months'] < 3)
    + 0.20 * (df['support_tickets_90d'] > 3)
    + 0.15 * (df['nps_score'] < 5)
    + 0.10 * (df['last_activity_days'] > 30)
    - 0.10 * (df['contract_type'] == 'annual')
    - 0.15 * (df['contract_type'] == '2year')
    + 0.05 * np.random.normal(0, 1, n_customers)
).clip(0, 0.95)
df['churned'] = (np.random.random(n_customers) < churn_prob).astype(int)

print(f'  Dataset: {len(df):,} customers')
print(f'  Churn rate: {df.churned.mean():.1%}')
print(f'  Features: {len([c for c in df.columns if c not in ["customer_id","churned"]])}')

# ── Step 2: Feature Engineering ───────────────────────────────────
print('\n[STEP 2] Feature engineering...')
df['spend_per_product']      = df['monthly_spend'] / (df['products_used'] + 0.001)
df['ticket_rate']            = df['support_tickets_90d'] / (df['tenure_months'] + 1)
df['engagement_score']       = df['login_frequency_30d'] / (df['last_activity_days'] + 1)
df['is_low_nps']             = (df['nps_score'] <= 5).astype(int)
df['is_new_customer']        = (df['tenure_months'] <= 3).astype(int)
df['contract_annual']        = (df['contract_type'] == 'annual').astype(int)
df['contract_2year']         = (df['contract_type'] == '2year').astype(int)
df['region_north']           = (df['region'] == 'North').astype(int)

FEATURE_COLS = [
    'tenure_months','monthly_spend','support_tickets_90d','login_frequency_30d',
    'products_used','last_activity_days','nps_score',
    'spend_per_product','ticket_rate','engagement_score',
    'is_low_nps','is_new_customer','contract_annual','contract_2year','region_north',
]

X = df[FEATURE_COLS].values
y = df['churned'].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_te, y_te, test_size=0.5, random_state=42)

print(f'  Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_te):,}')

# ── Step 3: MLflow Experiment Training ───────────────────────────
print('\n[STEP 3] Training with MLflow tracking...')

mlflow.set_tracking_uri('sqlite:///churn_mlops.db')
mlflow.set_experiment('churn-prediction-v1')

model_configs = [
    {'n_estimators':100,'learning_rate':0.1,'max_depth':3,'subsample':1.0},
    {'n_estimators':200,'learning_rate':0.05,'max_depth':4,'subsample':0.8},
    {'n_estimators':300,'learning_rate':0.03,'max_depth':5,'subsample':0.7},
]

best_val_auc = 0
best_model   = None
best_run_id  = None

for cfg in model_configs:
    with mlflow.start_run() as run:
        mlflow.log_params(cfg)
        mlflow.log_param('feature_count', len(FEATURE_COLS))
        mlflow.log_param('train_size', len(X_tr))

        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('gb', GradientBoostingClassifier(**cfg, random_state=42))
        ])
        pipe.fit(X_tr, y_tr)

        # Metrics
        val_auc   = roc_auc_score(y_val, pipe.predict_proba(X_val)[:,1])
        train_auc = roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:,1])
        cv_auc    = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring='roc_auc').mean()

        mlflow.log_metric('val_auc', val_auc)
        mlflow.log_metric('train_auc', train_auc)
        mlflow.log_metric('cv_auc', cv_auc)
        mlflow.log_metric('overfit_gap', train_auc - val_auc)
        mlflow.sklearn.log_model(pipe, 'model')

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model   = pipe
            best_run_id  = run.info.run_id

        print(f'  n={cfg["n_estimators"]:>3} lr={cfg["learning_rate"]} depth={cfg["max_depth"]}: val_auc={val_auc:.4f} cv={cv_auc:.4f}')

# ── Step 4: Final Evaluation on Hold-out Test ─────────────────────
print(f'\n[STEP 4] Final evaluation on hold-out test set...')
test_auc  = roc_auc_score(y_te, best_model.predict_proba(X_te)[:,1])
test_f1   = f1_score(y_te, best_model.predict(X_te))
test_acc  = accuracy_score(y_te, best_model.predict(X_te))

print(f'  Best model: run_id={best_run_id[:8]}')
print(f'  Test AUC:  {test_auc:.4f}')
print(f'  Test F1:   {test_f1:.4f}')
print(f'  Test Acc:  {test_acc:.4f}')

# ── Step 5: Simulate Production Monitoring ────────────────────────
print(f'\n[STEP 5] Simulating 6 months of production monitoring...')

monitoring_log = []
for month in range(1, 7):
    # Simulate increasing data drift and eventual model degradation
    drift = month * 0.08
    X_prod = X_te + np.random.normal(0, drift, X_te.shape)
    y_prod = y_te.copy()
    if month >= 5:
        flip_n = int(len(y_prod)*0.07*(month-4))
        flip_i = np.random.choice(len(y_prod), flip_n, replace=False)
        y_prod[flip_i] = 1 - y_prod[flip_i]

    prod_auc    = roc_auc_score(y_prod, best_model.predict_proba(X_prod)[:,1])
    pred_proba  = best_model.predict_proba(X_prod)[:,1]

    # PSI on first feature (simplified)
    ref_hist, bins = np.histogram(X_te[:,0], bins=10, density=True)
    cur_hist, _    = np.histogram(X_prod[:,0], bins=bins, density=True)
    ref_pct = ref_hist / (ref_hist.sum() + 1e-9)
    cur_pct = cur_hist / (cur_hist.sum() + 1e-9)
    eps     = 1e-8
    psi     = float(np.sum((cur_pct+eps - ref_pct+eps) * np.log((cur_pct+eps)/(ref_pct+eps))))

    status = 'RETRAIN' if (prod_auc < 0.80 or psi > 0.25) else 'OK'
    monitoring_log.append({
        'month':    month,
        'auc':      round(prod_auc, 4),
        'psi':      round(max(0, psi), 4),
        'pred_mean':round(pred_proba.mean(), 4),
        'status':   status
    })

# ── Step 6: Print monitoring summary ─────────────────────────────
print(f'\n{'Month':>8} {'AUC':>8} {'PSI':>8} {'Pred Mean':>10} {'Status':>10}')
print('='*50)
for row in monitoring_log:
    icon = ' ' if row['status']=='RETRAIN' else ' '
    print(f'{row["month"]:>8} {row["auc"]:>8.4f} {row["psi"]:>8.4f} {row["pred_mean"]:>10.4f} {icon}{row["status"]:>9}')

retrain_months = [r['month'] for r in monitoring_log if r['status']=='RETRAIN']
if retrain_months:
    print(f'\n   Retraining triggered in months: {retrain_months}')
    print('   → Automated retraining pipeline would execute:')
    print('   → 1. Pull latest 90 days of labelled data')
    print('   → 2. Run training pipeline with same config')
    print('   → 3. Champion-challenger evaluation on holdout')
    print('   → 4. Promote if AUC improves; else alert team')
else:
    print('\n  All months within acceptable bounds - no retraining needed')

print(f'\n{'='*70}')
print('CASE STUDY COMPLETE')
print(f'MLflow runs logged to: churn_mlops.db (view with: mlflow ui)')
print(f'Model saved to MLflow registry as: churn-classifier')
print(f'{'='*70}')
