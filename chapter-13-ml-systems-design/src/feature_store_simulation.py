# feature_store_simulation.py
# Simulates a simple feature store with online and offline components

import pandas as pd
import numpy as np
import redis
import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

# ── Feature definition ────────────────────────────────────────────
@dataclass
class FeatureDefinition:
    name: str
    description: str
    entity: str           # e.g. 'customer_id', 'product_id'
    value_type: str       # 'float', 'int', 'string'
    ttl_seconds: int = 3600  # Time-to-live for online store
    transformation: str = ''  # SQL or Python expression


# ── Simple in-memory feature store (production: use Feast/Tecton) ─
class SimpleFeatureStore:
    '''
    Simulated feature store with offline (DataFrame) and online (dict) layers.
    Production systems use Feast, Tecton, or Hopsworks.
    '''

    def __init__(self):
        self._offline_store: Dict[str, pd.DataFrame] = {}
        self._online_store:  Dict[str, Dict[str, Any]] = {}  # entity_id -> features
        self._feature_defs:  Dict[str, FeatureDefinition] = {}
        self._lineage:       List[Dict] = []

    def register_feature(self, feature_def: FeatureDefinition) -> None:
        '''Register a feature definition.'''
        self._feature_defs[feature_def.name] = feature_def
        print(f'Registered feature: {feature_def.name} (entity={feature_def.entity})')

    def ingest_offline(self, feature_group: str, df: pd.DataFrame,
                         entity_col: str, timestamp_col: Optional[str] = None) -> None:
        '''Ingest features into offline store (for training).'''
        self._offline_store[feature_group] = df.copy()
        self._lineage.append({
            'action': 'offline_ingest',
            'feature_group': feature_group,
            'n_rows': len(df),
            'timestamp': datetime.utcnow().isoformat()
        })
        print(f'Offline ingested: {feature_group} ({len(df)} rows)')

    def materialize_online(self, feature_group: str, entity_col: str) -> None:
        '''Copy latest offline features to online store.'''
        if feature_group not in self._offline_store:
            raise KeyError(f'Feature group {feature_group} not in offline store')

        df = self._offline_store[feature_group]
        feature_cols = [c for c in df.columns if c != entity_col]

        count = 0
        for _, row in df.iterrows():
            entity_id = str(row[entity_col])
            if entity_id not in self._online_store:
                self._online_store[entity_id] = {}
            for col in feature_cols:
                self._online_store[entity_id][col] = row[col]
            count += 1

        print(f'Materialized {count} entities from {feature_group} to online store')

    def get_online_features(self, entity_ids: List[str],
                              feature_names: List[str]) -> pd.DataFrame:
        '''Fetch features for entities from online store (< 5ms target).'''
        t0 = time.perf_counter()
        rows = []
        missing_entities = []

        for entity_id in entity_ids:
            if entity_id in self._online_store:
                row = {'entity_id': entity_id}
                for feat in feature_names:
                    row[feat] = self._online_store[entity_id].get(feat, np.nan)
                rows.append(row)
            else:
                missing_entities.append(entity_id)
                rows.append({'entity_id': entity_id, **{f: np.nan for f in feature_names}})

        latency_ms = (time.perf_counter()-t0)*1000

        if missing_entities:
            print(f'Warning: {len(missing_entities)} entities not found in online store')

        df = pd.DataFrame(rows)
        df._metadata = {'latency_ms': latency_ms}
        return df

    def get_training_dataset(self, feature_group: str,
                               feature_names: List[str],
                               label_col: str,
                               entity_col: str) -> pd.DataFrame:
        '''Create training dataset from offline store.'''
        df = self._offline_store[feature_group]
        cols = [entity_col] + feature_names + [label_col]
        return df[[c for c in cols if c in df.columns]].copy()


# ── Demonstrate feature store ─────────────────────────────────────
fs = SimpleFeatureStore()

# Register features
fs.register_feature(FeatureDefinition(
    name='account_age_days',
    description='Days since account was created',
    entity='customer_id', value_type='int', ttl_seconds=86400
))
fs.register_feature(FeatureDefinition(
    name='transaction_count_30d',
    description='Number of transactions in last 30 days',
    entity='customer_id', value_type='int', ttl_seconds=3600
))
fs.register_feature(FeatureDefinition(
    name='avg_transaction_value_30d',
    description='Average transaction value in last 30 days',
    entity='customer_id', value_type='float', ttl_seconds=3600
))

# Simulate customer features
np.random.seed(42)
n_customers = 10000
customer_features = pd.DataFrame({
    'customer_id':             np.arange(n_customers),
    'account_age_days':        np.random.randint(1, 3650, n_customers),
    'transaction_count_30d':   np.random.poisson(12, n_customers),
    'avg_transaction_value_30d': np.random.exponential(150, n_customers),
    'total_spend_lifetime':    np.random.exponential(2000, n_customers),
    'churn_label':             np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
})

# Ingest into feature store
fs.ingest_offline('customer_features', customer_features,
                    entity_col='customer_id')

# Materialize to online store
fs.materialize_online('customer_features', entity_col='customer_id')

# Training dataset retrieval
feature_names = ['account_age_days','transaction_count_30d',
                  'avg_transaction_value_30d','total_spend_lifetime']
train_df = fs.get_training_dataset('customer_features', feature_names,
                                     'churn_label', 'customer_id')
print(f'\nTraining dataset: {train_df.shape}')

# Online feature retrieval (inference time)
t0 = time.perf_counter()
predict_customers = [str(i) for i in np.random.choice(n_customers, 100, replace=False)]
online_features = fs.get_online_features(predict_customers, feature_names)
latency = (time.perf_counter()-t0)*1000
print(f'Online feature fetch: 100 customers in {latency:.2f}ms')
print(f'Missing values: {online_features.isnull().sum().sum()}')
print('\nSample features:')
print(online_features.head(3).round(2))

# Train model using features from store
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X = train_df[feature_names].fillna(0)
y = train_df['churn_label']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
print(f'\nModel AUC: {roc_auc_score(y_te, model.predict_proba(X_te)[:,1]):.4f}')
