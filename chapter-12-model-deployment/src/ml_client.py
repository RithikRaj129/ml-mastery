# ml_client.py - Python SDK for the ML API
# Usage: from ml_client import CancerClassifierClient

import requests
import numpy as np
from typing import Union, List, Dict, Optional
from dataclasses import dataclass
import time
import json

@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float

    @property
    def is_malignant(self) -> bool:
        return self.prediction == 'malignant'

    def __repr__(self):
        return (f'PredictionResult(diagnosis={self.prediction}, '
                f'confidence={self.confidence:.1%}, latency={self.latency_ms:.1f}ms)')


class CancerClassifierClient:
    '''Python client for the Cancer Classifier API.'''

    def __init__(self, base_url: str, api_key: str, timeout: int = 10,
                 max_retries: int = 3):
        self.base_url    = base_url.rstrip('/')
        self.api_key     = api_key
        self.timeout     = timeout
        self.max_retries = max_retries
        self.session     = requests.Session()
        self.session.headers.update({'X-API-Key': api_key,
                                      'Content-Type': 'application/json'})

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        '''Make HTTP request with retry logic.'''
        url = f'{self.base_url}{endpoint}'
        last_exc = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if response.status_code in (400, 401, 404):  # Client errors: don't retry
                    raise
                last_exc = e
            except requests.exceptions.ConnectionError as e:
                last_exc = e
            except requests.exceptions.Timeout as e:
                last_exc = e

            if attempt < self.max_retries - 1:
                wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                time.sleep(wait)

        raise last_exc

    def health(self) -> dict:
        '''Check API health.'''
        return self._request('GET', '/health')

    def predict(self, features: Union[List[float], np.ndarray]) -> PredictionResult:
        '''Predict for a single sample.'''
        if isinstance(features, np.ndarray):
            features = features.tolist()

        result = self._request('POST', '/predict', json={'features': features})
        return PredictionResult(
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            latency_ms=result['latency_ms']
        )

    def predict_batch(self, samples: Union[List[List[float]], np.ndarray],
                       chunk_size: int = 500) -> List[PredictionResult]:
        '''Predict for multiple samples with automatic chunking.'''
        if isinstance(samples, np.ndarray):
            samples = samples.tolist()

        all_results = []
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i+chunk_size]
            result = self._request('POST', '/predict/batch', json={'samples': chunk})
            for pred, probs in zip(result['predictions'], result['probabilities']):
                all_results.append(PredictionResult(
                    prediction=pred,
                    confidence=max(probs.values()),
                    probabilities=probs,
                    latency_ms=result['latency_ms']/len(chunk)
                ))
        return all_results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.session.close()


# ── Usage example ─────────────────────────────────────────────────
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()

    with CancerClassifierClient('http://localhost:8000', 'dev-secret-key') as client:
        # Health check
        health = client.health()
        print(f'API status: {health["status"]}')

        # Single prediction
        result = client.predict(data.data[0])
        print(f'Single: {result}')
        print(f'  Is malignant: {result.is_malignant}')
        print(f'  P(malignant): {result.probabilities["malignant"]:.1%}')

        # Batch prediction
        results = client.predict_batch(data.data[:50])
        malignant_count = sum(1 for r in results if r.is_malignant)
        print(f'\nBatch (50 samples):')
        print(f'  Malignant: {malignant_count}  Benign: {len(results)-malignant_count}')
        print(f'  Avg confidence: {np.mean([r.confidence for r in results]):.1%}')
