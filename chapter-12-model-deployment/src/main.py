# main.py - Production FastAPI ML service
# Install: pip install fastapi uvicorn pydantic
# Run: uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
# Docs: http://localhost:8000/docs

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import joblib
import numpy as np
import logging
import time
import os
from datetime import datetime
from collections import defaultdict

# ── App configuration ─────────────────────────────────────────────
logger = logging.getLogger(__name__)

app = FastAPI(
    title='Cancer Classifier API',
    description='Breast cancer malignancy prediction using ML',
    version='2.0.0',
    docs_url='/docs',        # Swagger UI at /docs
    redoc_url='/redoc',      # ReDoc at /redoc
)

# CORS middleware (for browser-based clients)
app.add_middleware(CORSMiddleware,
    allow_origins=['*'],     # Restrict in production
    allow_methods=['*'],
    allow_headers=['*'],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Global state ──────────────────────────────────────────────────
model = None
request_counts = defaultdict(int)
start_time = datetime.utcnow()

# ── Pydantic models (automatic validation + documentation) ────────
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_items=30, max_items=30,
        description='30 cell nucleus features in standard order',
        example=[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                 0.2654, 0.4601, 0.1189]
    )
    return_probabilities: bool = Field(True, description='Include class probabilities in response')

    @validator('features')
    def validate_no_nan(cls, v):
        if any(np.isnan(f) for f in v):
            raise ValueError('Features cannot contain NaN values')
        return v


class BatchPredictRequest(BaseModel):
    samples: List[List[float]] = Field(
        ..., min_items=1, max_items=1000,
        description='List of feature vectors (max 1000 per batch)'
    )


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Optional[Dict[str, float]]
    latency_ms: float
    model_version: str = '2.0.0'


class BatchPredictResponse(BaseModel):
    n_samples: int
    predictions: List[str]
    probabilities: Optional[List[Dict[str, float]]]
    latency_ms: float
    throughput_qps: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_requests: int
    model_loaded: bool

# ── Authentication dependency ─────────────────────────────────────
API_KEY = os.getenv('API_KEY', 'fastapi-secret-key')

async def verify_api_key(x_api_key: str = Header(..., alias='X-API-Key')):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='Invalid API key')
    return x_api_key

# ── Startup: load model ───────────────────────────────────────────
@app.on_event('startup')
async def startup_event():
    global model
    model_path = os.getenv('MODEL_PATH', 'model_pipeline.joblib')
    try:
        model = joblib.load(model_path)
        logger.info(f'Model loaded: {model_path}')
    except FileNotFoundError:
        logger.error(f'Model not found: {model_path}')

# ── Routes ────────────────────────────────────────────────────────
@app.get('/health', response_model=HealthResponse, tags=['Status'])
async def health_check():
    uptime = (datetime.utcnow() - start_time).total_seconds()
    return HealthResponse(
        status='healthy' if model else 'degraded',
        uptime_seconds=round(uptime, 1),
        total_requests=sum(request_counts.values()),
        model_loaded=model is not None
    )


@app.post('/predict', response_model=PredictResponse, tags=['Prediction'],
          summary='Single sample prediction',
          description='Predict malignancy for a single cell nucleus sample')
async def predict(
    request: PredictRequest,
    api_key: str = Depends(verify_api_key)
):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not available')

    request_counts['predict'] += 1

    X = np.array(request.features).reshape(1, -1)

    t0 = time.perf_counter()
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    class_names = ['malignant', 'benign']
    pred_label  = class_names[prediction]
    confidence  = float(max(probabilities))
    probs_dict  = dict(zip(class_names, probabilities.tolist()))

    return PredictResponse(
        prediction=pred_label,
        confidence=round(confidence, 4),
        probabilities=probs_dict if request.return_probabilities else None,
        latency_ms=round(latency_ms, 3)
    )


@app.post('/predict/batch', response_model=BatchPredictResponse, tags=['Prediction'])
async def predict_batch(
    request: BatchPredictRequest,
    api_key: str = Depends(verify_api_key)
):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not available')

    X = np.array(request.samples)
    if X.shape[1] != 30:
        raise HTTPException(status_code=400, detail=f'Expected 30 features, got {X.shape[1]}')

    t0 = time.perf_counter()
    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)
    latency_ms    = (time.perf_counter() - t0) * 1000

    class_names = ['malignant', 'benign']
    n = len(predictions)

    return BatchPredictResponse(
        n_samples=n,
        predictions=[class_names[p] for p in predictions],
        probabilities=[dict(zip(class_names,p.tolist())) for p in probabilities],
        latency_ms=round(latency_ms, 3),
        throughput_qps=round(n/(latency_ms/1000), 1)
    )


# ── Run locally ────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True, workers=1)
