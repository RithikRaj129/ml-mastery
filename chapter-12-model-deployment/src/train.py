# train.py - Reproducible training script for CI/CD
import argparse
import json
import os
import datetime
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

def train(output_dir: str, data_version: str, random_state: int = 42):
    print(f'Training model... data_version={data_version}')

    # Load data (replace with your data loading logic)
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Train
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(n_estimators=200, random_state=random_state))
    ])
    pipeline.fit(X_tr, y_tr)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'{output_dir}/model_{ts}.joblib'
    joblib.dump(pipeline, model_path)

    # Save metadata
    meta = {
        'model_path': model_path,
        'data_version': data_version,
        'created_at': datetime.datetime.now().isoformat(),
        'random_state': random_state,
        'n_train': len(X_tr),
        'n_test': len(X_te),
    }
    with open(f'{output_dir}/model_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Model saved: {model_path}')
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='models/')
    parser.add_argument('--data-version', default='local')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    train(args.output_dir, args.data_version, args.random_state)
