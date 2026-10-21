# tests/test_api.py - Automated API tests for CI
import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer
from fastapi.testclient import TestClient

# Import the app (adjust import path as needed)
from main import app

client = TestClient(app)
data   = load_breast_cancer()
VALID_FEATURES = data.data[0].tolist()
API_KEY = 'fastapi-secret-key'
HEADERS = {'X-API-Key': API_KEY}

class TestHealth:
    def test_health_returns_200(self):
        r = client.get('/health')
        assert r.status_code == 200

    def test_health_contains_status(self):
        r = client.get('/health')
        assert 'status' in r.json()
        assert r.json()['status'] in ['healthy', 'degraded']

class TestPredict:
    def test_valid_prediction(self):
        r = client.post('/predict',
                        json={'features': VALID_FEATURES},
                        headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert 'prediction' in body
        assert body['prediction'] in ['malignant', 'benign']
        assert 0 <= body['confidence'] <= 1
        assert body['latency_ms'] > 0

    def test_unauthorised_returns_401(self):
        r = client.post('/predict', json={'features': VALID_FEATURES})
        assert r.status_code == 401

    def test_wrong_feature_count_returns_400(self):
        r = client.post('/predict',
                        json={'features': [1.0, 2.0, 3.0]},
                        headers=HEADERS)
        assert r.status_code == 422  # Pydantic validation error

    def test_predictions_are_deterministic(self):
        r1 = client.post('/predict', json={'features': VALID_FEATURES}, headers=HEADERS)
        r2 = client.post('/predict', json={'features': VALID_FEATURES}, headers=HEADERS)
        assert r1.json()['prediction'] == r2.json()['prediction']
        assert r1.json()['confidence'] == r2.json()['confidence']

    def test_batch_prediction(self):
        samples = data.data[:10].tolist()
        r = client.post('/predict/batch',
                        json={'samples': samples},
                        headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        assert body['n_samples'] == 10
        assert len(body['predictions']) == 10

    def test_model_accuracy_meets_threshold(self):
        '''Integration test: model should exceed 95% accuracy on test set.'''
        samples   = data.data.tolist()
        r = client.post('/predict/batch', json={'samples': samples}, headers=HEADERS)
        predictions = r.json()['predictions']
        pred_labels = [0 if p=='malignant' else 1 for p in predictions]
        accuracy    = np.mean(np.array(pred_labels) == data.target)
        assert accuracy >= 0.95, f'Accuracy {accuracy:.4f} below 0.95 threshold'

    def test_latency_under_100ms(self):
        r = client.post('/predict', json={'features': VALID_FEATURES}, headers=HEADERS)
        assert r.json()['latency_ms'] < 100, 'Prediction too slow'
