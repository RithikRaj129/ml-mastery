# automl_demo.py - AutoML with multiple frameworks
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
import warnings; warnings.filterwarnings('ignore')

data = load_breast_cancer()
X, y = data.data, data.target
X_sc = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, stratify=y, random_state=42)

print('=== AutoML Comparison ===')

# ── 1. Manual Baseline ────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier
t0 = time.time()
manual = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
print(f'Manual (GradBoost defaults): AUC={roc_auc_score(y_te,manual.predict_proba(X_te)[:,1]):.4f}  Time={time.time()-
t0:.1f}s')

# ── 2. Optuna AutoML (Bayesian optimization over algorithm + HPs) ─
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC

    def automl_objective(trial):
        algorithm = trial.suggest_categorical('algorithm',
                       ['gradient_boosting','random_forest','extra_trees','logistic'])

        if algorithm == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators = trial.suggest_int('n_estimators', 50, 400),
                learning_rate= trial.suggest_float('lr', 0.01, 0.3, log=True),
                max_depth    = trial.suggest_int('max_depth', 2, 7),
                random_state = 42
            )
        elif algorithm == 'random_forest':
            model = RandomForestClassifier(
                n_estimators = trial.suggest_int('n_estimators', 50, 400),
                max_features = trial.suggest_float('max_features', 0.1, 1.0),
                n_jobs=-1, random_state=42
            )
        elif algorithm == 'extra_trees':
            model = ExtraTreesClassifier(
                n_estimators = trial.suggest_int('n_estimators', 50, 400),
                n_jobs=-1, random_state=42
            )
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                C = trial.suggest_float('C', 0.001, 100, log=True),
                max_iter=1000
            )

        from sklearn.model_selection import cross_val_score
        return cross_val_score(model, X_tr, y_tr, cv=3, scoring='roc_auc').mean()

    t0 = time.time()
    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(automl_objective, n_trials=30, n_jobs=-1)
    elapsed = time.time()-t0

    print(f'Optuna AutoML (30 trials): best_AUC={study.best_value:.4f}  Time={elapsed:.1f}s')
    print(f'  Best algorithm: {study.best_params.get("algorithm","N/A")}')

except ImportError:
    print('Optuna not available')

# ── 3. PyCaret AutoML ─────────────────────────────────────────────
try:
    from pycaret.classification import setup, compare_models, predict_model
    import pandas as pd

    df = pd.DataFrame(X_sc, columns=data.feature_names)
    df['target'] = y

    t0 = time.time()
    setup(data=df, target='target', verbose=False, session_id=42)
    best_pycaret = compare_models(n_select=1, verbose=False)
    elapsed = time.time()-t0

    preds = predict_model(best_pycaret)
    auc_pc = roc_auc_score(preds['target'], preds['prediction_score'])
    print(f'PyCaret AutoML:            AUC={auc_pc:.4f}  Time={elapsed:.1f}s')

except ImportError:
    print('PyCaret not installed (pip install pycaret)')

# ── 4. AutoGluon ─────────────────────────────────────────────────
try:
    from autogluon.tabular import TabularDataset, TabularPredictor
    import pandas as pd

    df_tr = pd.DataFrame(np.hstack([X_tr, y_tr.reshape(-1,1)]),
                          columns=list(data.feature_names)+['label'])
    df_te = pd.DataFrame(X_te, columns=list(data.feature_names))

    t0 = time.time()
    predictor = TabularPredictor(label='label', eval_metric='roc_auc',
                                   verbosity=0).fit(df_tr, time_limit=60)
    elapsed   = time.time() - t0
    ag_auc    = predictor.evaluate(pd.DataFrame(np.hstack([X_te,y_te.reshape(-1,1)]),
                                    columns=list(data.feature_names)+['label']))['roc_auc']
    print(f'AutoGluon (60s budget):    AUC={ag_auc:.4f}  Time={elapsed:.1f}s')

except ImportError:
    print('AutoGluon not installed (pip install autogluon)')
    print('AutoGluon would typically achieve: AUC ≈ 0.996 in 60s (ensembles many models)')
