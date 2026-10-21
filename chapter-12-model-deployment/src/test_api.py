# test_api.py - Test the Flask API with requests
import requests
import json
import numpy as np
from sklearn.datasets import load_breast_cancer

BASE_URL = 'http://localhost:5000'
API_KEY  = 'dev-secret-key-123'
HEADERS  = {'X-API-Key': API_KEY, 'Content-Type': 'application/json'}

data = load_breast_cancer()
sample = data.data[0].tolist()  # First sample

# ── 1. Health check ───────────────────────────────────────────────
r = requests.get(f'{BASE_URL}/health')
print('Health:', json.dumps(r.json(), indent=2))

# ── 2. Single prediction (dict format) ───────────────────────────
payload = {'features': dict(zip(data.feature_names, sample))}
r = requests.post(f'{BASE_URL}/predict', json=payload, headers=HEADERS)
print('\nSingle prediction:', json.dumps(r.json(), indent=2))

# ── 3. Single prediction (list format) ───────────────────────────
payload = {'features': [sample]}
r = requests.post(f'{BASE_URL}/predict', json=payload, headers=HEADERS)
print('\nList format:', r.json()['predictions'])

# ── 4. Batch prediction ───────────────────────────────────────────
batch_samples = data.data[:50].tolist()
payload = {'features': batch_samples}
r = requests.post(f'{BASE_URL}/predict/batch', json=payload, headers=HEADERS)
resp = r.json()
print(f'\nBatch: {resp["n_samples"]} samples, {resp["latency_ms"]}ms, {resp["throughput_qps"]} QPS')

# ── 5. Error handling test ────────────────────────────────────────
r = requests.post(f'{BASE_URL}/predict', json={'features': [1, 2, 3]}, headers=HEADERS)
print(f'\nError response (wrong features): {r.status_code} - {r.json()["error"]}')

r = requests.post(f'{BASE_URL}/predict', json={'features': [sample]})  # No API key
print(f'Auth error (no key): {r.status_code} - {r.json()["error"]}')
