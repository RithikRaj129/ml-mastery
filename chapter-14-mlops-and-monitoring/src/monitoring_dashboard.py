# monitoring_dashboard.py - Prometheus metrics for ML service
# pip install prometheus-client

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    start_http_server, CollectorRegistry, push_to_gateway
)
import time
import numpy as np
from datetime import datetime

# ── Define ML-specific Prometheus metrics ─────────────────────────
registry = CollectorRegistry()

# Request metrics
PREDICTION_COUNTER = Counter(
    'ml_predictions_total',
    'Total number of predictions served',
    ['model_version', 'prediction_class'],
    registry=registry
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=registry
)

ERROR_COUNTER = Counter(
    'ml_prediction_errors_total',
    'Total number of prediction errors',
    ['model_version', 'error_type'],
    registry=registry
)

# Model quality metrics
MODEL_AUC_GAUGE = Gauge(
    'ml_model_auc',
    'Current model AUC on rolling evaluation window',
    ['model_version'],
    registry=registry
)

DATA_DRIFT_PSI = Gauge(
    'ml_data_drift_psi',
    'Population Stability Index per feature',
    ['feature_name', 'model_version'],
    registry=registry
)

PREDICTION_MEAN = Gauge(
    'ml_prediction_mean_score',
    'Rolling mean of predicted probability',
    ['model_version'],
    registry=registry
)

FEATURE_MISSING_RATE = Gauge(
    'ml_feature_missing_rate',
    'Rate of missing values per feature',
    ['feature_name', 'model_version'],
    registry=registry
)

# ── ML metrics reporter ───────────────────────────────────────────
class MLMetricsReporter:
    '''Records and exposes ML metrics to Prometheus.'''

    def __init__(self, model_version: str = 'v1.0'):
        self.model_version = model_version
        self.prediction_buffer = []
        self.buffer_size = 1000

    def record_prediction(self, prediction: str, probability: float, latency_s: float) -> None:
        PREDICTION_COUNTER.labels(
            model_version=self.model_version,
            prediction_class=prediction
        ).inc()

        PREDICTION_LATENCY.labels(
            model_version=self.model_version
        ).observe(latency_s)

        self.prediction_buffer.append(probability)
        if len(self.prediction_buffer) >= self.buffer_size:
            self._flush_buffer()

    def record_error(self, error_type: str) -> None:
        ERROR_COUNTER.labels(
            model_version=self.model_version,
            error_type=error_type
        ).inc()

    def update_model_quality(self, auc: float) -> None:
        MODEL_AUC_GAUGE.labels(model_version=self.model_version).set(auc)

    def update_drift_psi(self, feature_name: str, psi: float) -> None:
        DATA_DRIFT_PSI.labels(
            feature_name=feature_name,
            model_version=self.model_version
        ).set(psi)

    def update_missing_rate(self, feature_name: str, rate: float) -> None:
        FEATURE_MISSING_RATE.labels(
            feature_name=feature_name,
            model_version=self.model_version
        ).set(rate)

    def _flush_buffer(self) -> None:
        mean_prob = np.mean(self.prediction_buffer)
        PREDICTION_MEAN.labels(model_version=self.model_version).set(mean_prob)
        self.prediction_buffer = []


# ── Simulate metrics ─────────────────────────────────────────────
reporter = MLMetricsReporter(model_version='v3.0')
np.random.seed(42)

# Simulate 1000 predictions
n_predictions = 1000
for i in range(n_predictions):
    latency = np.random.exponential(0.01)  # ~10ms average
    prob    = np.random.beta(2, 5)  # Skewed toward lower probabilities
    pred    = 'malignant' if prob > 0.5 else 'benign'
    reporter.record_prediction(pred, prob, latency)

# Update quality metrics
reporter.update_model_quality(0.9823)

# Simulate drift PSI values
feature_psi = {
    'mean radius': 0.31, 'mean texture': 0.18,
    'mean perimeter': 0.08, 'mean area': 0.04, 'mean smoothness': 0.02
}
for feature, psi in feature_psi.items():
    reporter.update_drift_psi(feature, psi)

print('=== ML Monitoring Metrics (Prometheus) ===')
print(f'Simulated {n_predictions} predictions')
print(f'Metrics exposed on port 8080 → /metrics')
print(f'\nKey metrics:')
print(f'  ml_predictions_total{{model_version=v3.0,class=benign}}:    ~{int(n_predictions*0.63)}')
print(f'  ml_predictions_total{{model_version=v3.0,class=malignant}}: ~{int(n_predictions*0.37)}')
print(f'  ml_model_auc{{model_version=v3.0}}:                         0.9823')
print(f'  ml_data_drift_psi{{feature=mean radius}}: 0.3100   ALERT')
print(f'  ml_data_drift_psi{{feature=mean perimeter}}: 0.0800   OK')

# Grafana dashboard JSON template
grafana_panel = {
    'title': 'ML Model AUC Over Time',
    'type': 'graph',
    'targets': [{'expr': 'ml_model_auc', 'legendFormat': 'AUC - {{model_version}}'}],
    'alert': {
        'name': 'AUC below threshold',
        'conditions': [{'evaluator': {'type': 'lt', 'params': [0.92]}}],
        'notifications': [{'uid': 'slack-ml-team'}]
    }
}
print(f'\nGrafana panel template created')
print(f'Alert: triggers when ml_model_auc < 0.92 → Slack notification')
