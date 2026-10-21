# batch_inference.py - Batch scoring pipeline
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def run_batch_scoring(
    model_path: str,
    input_df: pd.DataFrame,
    feature_cols: list,
    entity_col: str,
    output_path: str,
    chunk_size: int = 10000
) -> pd.DataFrame:
    '''
    Run batch inference on a large DataFrame in chunks.
    Writes results to output_path with entity_id + score + timestamp.
    '''
    model = joblib.load(model_path)
    run_ts = datetime.utcnow().isoformat()

    all_scores = []
    n_chunks   = (len(input_df) + chunk_size - 1) // chunk_size

    logger.info(f'Batch scoring: {len(input_df):,} samples in {n_chunks} chunks')

    for i in range(n_chunks):
        chunk = input_df.iloc[i*chunk_size:(i+1)*chunk_size]
        X_chunk = chunk[feature_cols].fillna(0)

        probs  = model.predict_proba(X_chunk)[:, 1]
        labels = model.predict(X_chunk)

        chunk_scores = pd.DataFrame({
            entity_col:  chunk[entity_col].values,
            'score':     probs,
            'label':     labels,
            'scored_at': run_ts,
            'model':     model_path,
        })
        all_scores.append(chunk_scores)

        if (i+1) % 10 == 0:
            logger.info(f'  Chunk {i+1}/{n_chunks} complete')

    results = pd.concat(all_scores, ignore_index=True)
    results.to_parquet(output_path, index=False)
    logger.info(f'Saved {len(results):,} scores to {output_path}')
    return results


# Simulate batch scoring
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
feature_names = list(data.feature_names)

# Train and save model
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
pipe = Pipeline([('sc',StandardScaler()),('gb',GradientBoostingClassifier(n_estimators=100,random_state=42))])
pipe.fit(X_tr, y_tr)
joblib.dump(pipe, 'batch_model.joblib')

# Large scoring dataset
n_score = 100_000
np.random.seed(42)
X_score = np.random.randn(n_score, 30) * data.data.std(0) + data.data.mean(0)
df_score = pd.DataFrame(X_score, columns=feature_names)
df_score['customer_id'] = np.arange(n_score)

import time
t0 = time.time()
results = run_batch_scoring(
    'batch_model.joblib', df_score, feature_names,
    'customer_id', '/tmp/scores.parquet'
)
elapsed = time.time() - t0

print(f'\nBatch scoring complete:')
print(f'  Samples:   {len(results):,}')
print(f'  Time:      {elapsed:.2f}s')
print(f'  Throughput:{len(results)/elapsed:,.0f} samples/sec')
print(f'  Avg score: {results.score.mean():.4f}')
print(f'  Score dist:\n{results.score.describe().round(4)}')
