# structured_logging.py - Production ML logging and alerting
import logging
import json
import time
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import deque
import threading

# ── Structured prediction log entry ──────────────────────────────
@dataclass
class PredictionLog:
    request_id:       str
    timestamp:        str
    model_version:    str
    features:         Dict[str, float]
    prediction:       str
    probability:      float
    confidence:       float
    latency_ms:       float
    error:            Optional[str] = None
    user_id:          Optional[str] = None
    environment:      str = 'production'

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


# ── ML-specific structured logger ────────────────────────────────
class MLStructuredLogger:
    '''
    Logs predictions in structured JSON format.
    Enables: debugging, retraining data collection, audit trails.
    '''

    def __init__(self, model_version: str, log_file: str = 'predictions.jsonl',
                  sample_rate: float = 1.0,  # Log 100% of predictions
                  pii_fields: List[str] = None):
        self.model_version = model_version
        self.log_file      = log_file
        self.sample_rate   = sample_rate
        self.pii_fields    = pii_fields or []

        # Set up structured logger
        self.logger = logging.getLogger(f'ml.{model_version}')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_prediction(self, features: Dict, prediction: str,
                         probability: float, latency_ms: float,
                         user_id: str = None) -> str:
        # Sample rate (log fraction of requests to reduce storage)
        if np.random.random() > self.sample_rate:
            return ''

        request_id = str(uuid.uuid4())[:8]

        # Redact PII fields
        safe_features = {
            k: '[REDACTED]' if k in self.pii_fields else v
            for k, v in features.items()
        }

        log_entry = PredictionLog(
            request_id    = request_id,
            timestamp     = datetime.utcnow().isoformat(),
            model_version = self.model_version,
            features      = safe_features,
            prediction    = prediction,
            probability   = round(probability, 4),
            confidence    = round(max(probability, 1-probability), 4),
            latency_ms    = round(latency_ms, 3),
            user_id       = user_id,
        )
        self.logger.info(log_entry.to_json())
        return request_id

    def log_error(self, error: str, features: Dict = None,
                   request_id: str = None) -> None:
        entry = {
            'type':        'prediction_error',
            'timestamp':   datetime.utcnow().isoformat(),
            'model':       self.model_version,
            'request_id':  request_id or str(uuid.uuid4())[:8],
            'error':       error,
            'has_features':features is not None,
        }
        self.logger.error(json.dumps(entry))


# ── Alert manager ─────────────────────────────────────────────────
class AlertManager:
    '''
    Evaluates alert conditions and routes notifications.
    Includes cooldown to prevent alert fatigue.
    '''

    def __init__(self):
        self.rules          = []
        self.alert_history  = []
        self.last_alert_ts  = {}   # rule_name → last alert timestamp
        self.cooldown_secs  = 300  # 5 minute cooldown between same alert

    def add_rule(self, name: str, condition, severity: str,
                  message_template: str, channels: List[str]) -> None:
        self.rules.append({
            'name': name, 'condition': condition,
            'severity': severity, 'message': message_template,
            'channels': channels
        })

    def evaluate(self, metrics: Dict[str, float]) -> List[Dict]:
        fired_alerts = []
        now = time.time()

        for rule in self.rules:
            try:
                if rule['condition'](metrics):
                    last = self.last_alert_ts.get(rule['name'], 0)
                    if now - last > self.cooldown_secs:
                        alert = {
                            'rule':      rule['name'],
                            'severity':  rule['severity'],
                            'message':   rule['message'].format(**metrics),
                            'channels':  rule['channels'],
                            'timestamp': datetime.utcnow().isoformat(),
                            'metrics':   metrics,
                        }
                        fired_alerts.append(alert)
                        self.alert_history.append(alert)
                        self.last_alert_ts[rule['name']] = now
                        self._dispatch(alert)
            except Exception as e:
                pass

        return fired_alerts

    def _dispatch(self, alert: Dict) -> None:
        for channel in alert['channels']:
            if channel == 'slack':
                print(f'[SLACK] [{alert["severity"]}] {alert["message"]}')
            elif channel == 'pagerduty':
                print(f'[PAGERDUTY]   {alert["message"]}')
            elif channel == 'email':
                print(f'[EMAIL] To: ml-team@company.com  Subject: {alert["rule"]}')


# ── Set up alert rules ────────────────────────────────────────────
alert_mgr = AlertManager()

alert_mgr.add_rule(
    name='model_auc_degradation',
    condition=lambda m: m.get('current_auc', 1.0) < 0.92,
    severity='CRITICAL',
    message_template='Model AUC dropped to {current_auc:.4f} (threshold: 0.92). Immediate retraining required.',
    channels=['pagerduty', 'slack', 'email']
)
alert_mgr.add_rule(
    name='data_drift_warning',
    condition=lambda m: m.get('max_psi', 0) > 0.25,
    severity='WARNING',
    message_template='Data drift detected. Max PSI={max_psi:.4f}. Review feature distributions.',
    channels=['slack']
)
alert_mgr.add_rule(
    name='high_latency',
    condition=lambda m: m.get('latency_p99_ms', 0) > 500,
    severity='WARNING',
    message_template='p99 latency is {latency_p99_ms:.1f}ms (threshold: 500ms).',
    channels=['slack']
)
alert_mgr.add_rule(
    name='prediction_distribution_shift',
    condition=lambda m: abs(m.get('pred_mean', 0.5) - 0.5) > 0.15,
    severity='INFO',
    message_template='Prediction mean={pred_mean:.4f} deviating from baseline. Monitor carefully.',
    channels=['slack']
)

# ── Simulate monitoring cycle ─────────────────────────────────────
print('=== Alert Manager - Monitoring Cycle ===')
monitoring_scenarios = [
    {'current_auc':0.9812,'max_psi':0.08,'latency_p99_ms':120,'pred_mean':0.48, 'label':'Normal'},
    {'current_auc':0.9634,'max_psi':0.18,'latency_p99_ms':180,'pred_mean':0.52, 'label':'Slight drift'},
    {'current_auc':0.9012,'max_psi':0.34,'latency_p99_ms':620,'pred_mean':0.71, 'label':'Crisis!'},
]

for scenario in monitoring_scenarios:
    label = scenario.pop('label')
    print(f'\n--- Scenario: {label} ---')
    fired = alert_mgr.evaluate(scenario)
    if not fired:
        print('    All clear - no alerts fired')
    else:
        print(f'  {len(fired)} alert(s) fired')
