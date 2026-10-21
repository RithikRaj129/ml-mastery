# app.py - Complete Flask ML prediction API
# Run with: python app.py
# Or production: gunicorn -w 4 -b 0.0.0.0:5000 app:app

from flask import Flask, request, jsonify, abort
import joblib
import numpy as np
import json
import os
import logging
import time
from functools import wraps
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ── Load model at startup (not on every request!) ─────────────────
app = Flask(__name__)

MODEL_PATH   = os.getenv('MODEL_PATH', 'model_pipeline.joblib')
API_KEY      = os.getenv('API_KEY', 'dev-secret-key-123')  # Use secrets in production

try:
    pipeline = joblib.load(MODEL_PATH)
    logger.info(f'Model loaded from {MODEL_PATH}')
except FileNotFoundError:
    logger.error(f'Model file not found: {MODEL_PATH}')
    pipeline = None

# ── Feature configuration (load from model metadata in production) ─
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]
CLASS_NAMES = ['malignant', 'benign']

# ── Middleware: API key authentication ────────────────────────────
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if key != API_KEY:
            logger.warning(f'Unauthorised request from {request.remote_addr}')
            abort(401, description='Invalid or missing API key')
        return f(*args, **kwargs)
    return decorated

# ── Request validation ────────────────────────────────────────────
def validate_features(data):
    '''Validate and parse feature input.'''
    if 'features' not in data:
        raise ValueError('Request must contain "features" key')

    features = data['features']

    if isinstance(features, dict):
        # Named features dict: {'mean radius': 17.99, ...}
        missing = [f for f in FEATURE_NAMES if f not in features]
        if missing:
            raise ValueError(f'Missing features: {missing[:3]}...')
        return np.array([[features[f] for f in FEATURE_NAMES]])

    elif isinstance(features, list):
        # List of lists: [[v1, v2, ..., v30], ...]
        arr = np.array(features)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        if arr.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f'Expected {len(FEATURE_NAMES)} features, got {arr.shape[1]}')
        return arr

    else:
        raise ValueError('features must be a dict or list')

# ── Routes ────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    '''Health check endpoint - used by load balancers and monitoring.'''
    return jsonify({
        'status':    'healthy' if pipeline else 'degraded',
        'model':     MODEL_PATH,
        'timestamp': datetime.utcnow().isoformat(),
        'version':   '1.0.0'
    })


@app.route('/info', methods=['GET'])
def info():
    '''Model information endpoint.'''
    return jsonify({
        'model_type':    type(pipeline.named_steps.get('model', pipeline)).__name__,
        'features':      FEATURE_NAMES,
        'classes':       CLASS_NAMES,
        'n_features':    len(FEATURE_NAMES),
    })


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    '''
    Main prediction endpoint.

    Request (JSON):
      {
        "features": {"mean radius": 17.99, "mean texture": 10.38, ...}
        or
        "features": [[17.99, 10.38, ..., 0.1189]]  # list of lists
      }

    Response (JSON):
      {
        "predictions": ["malignant"],
        "probabilities": {"malignant": 0.23, "benign": 0.77},
        "latency_ms": 1.2
      }
    '''
    if pipeline is None:
        abort(503, description='Model not loaded')

    # Parse request
    if not request.is_json:
        abort(400, description='Content-Type must be application/json')

    data = request.get_json(force=True)

    # Validate
    try:
        X = validate_features(data)
    except ValueError as e:
        abort(400, description=str(e))

    # Predict
    t0 = time.time()
    try:
        predictions = pipeline.predict(X)
        probabilities= pipeline.predict_proba(X)
        latency_ms   = (time.time() - t0) * 1000
    except Exception as e:
        logger.error(f'Prediction error: {e}')
        abort(500, description=f'Prediction failed: {str(e)}')

    # Format response
    response = {
        'predictions':   [CLASS_NAMES[p] for p in predictions],
        'probabilities': [
            dict(zip(CLASS_NAMES, probs.tolist()))
            for probs in probabilities
        ],
        'latency_ms':    round(latency_ms, 3),
        'model_version': '1.0.0',
    }

    logger.info(f'Prediction: {predictions[0]}  latency={latency_ms:.1f}ms')
    return jsonify(response)


@app.route('/predict/batch', methods=['POST'])
@require_api_key
def predict_batch():
    '''Batch prediction endpoint - process multiple samples at once.'''
    if pipeline is None:
        abort(503, description='Model not loaded')

    data = request.get_json(force=True)
    max_batch = 1000

    if isinstance(data.get('features'), list) and len(data['features']) > max_batch:
        abort(400, description=f'Batch size exceeds maximum ({max_batch})')

    try:
        X = validate_features(data)
    except ValueError as e:
        abort(400, description=str(e))

    t0 = time.time()
    predictions   = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    latency_ms    = (time.time()-t0)*1000

    return jsonify({
        'n_samples':      len(predictions),
        'predictions':    [CLASS_NAMES[p] for p in predictions],
        'probabilities':  [dict(zip(CLASS_NAMES,p.tolist())) for p in probabilities],
        'latency_ms':     round(latency_ms, 3),
        'throughput_qps': round(len(predictions)/(latency_ms/1000), 1),
    })


# ── Error handlers ─────────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):    return jsonify({'error': str(e.description), 'code': 400}), 400

@app.errorhandler(401)
def unauthorised(e):   return jsonify({'error': str(e.description), 'code': 401}), 401

@app.errorhandler(500)
def server_error(e):   return jsonify({'error': str(e.description), 'code': 500}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
