import io
import json
import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("models/fraud_detection_model.pkl", os.path.join(os.path.dirname(__file__), "models/fraud_detection_model.pkl"))

_model = None
_feature_names: List[str] = []


def get_model():
    """Lazy-load the model once and cache it."""
    global _model, _feature_names
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. "
                "Copy fraud_detection_model.pkl next to app.py or set the MODEL_PATH env var."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        _feature_names = list(_model.feature_names_in_)
    return _model


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Allow the deployed frontend to call this API. Tighten allow_origins in production
# to your real domain instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Risk banding used to translate a raw probability into a human label
RISK_BANDS = [
    (0.75, "high"),
    (0.4, "medium"),
    (0.0, "low"),
]


def band_for(score: float) -> str:
    for threshold, label in RISK_BANDS:
        if score >= threshold:
            return label
    return "low"


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    # Extra fields (like a ground-truth "label") are tolerated and simply ignored
    # by the model — they don't break prediction, and we echo "label" back if present.
    model_config = ConfigDict(extra="allow")

    amount: float
    hour: int
    day_of_week: int
    month: int
    is_night: int
    client_mean_amount: float
    amount_to_credit_ratio: float
    tx_count_same_day: int
    client_merchant_freq: int
    is_online: int
    is_chip: int
    has_error: int


class PredictRequest(BaseModel):
    transactions: List[Transaction]


class PredictionResult(BaseModel):
    index: int
    fraud_probability: float
    prediction: int
    risk_level: str
    true_label: Optional[int] = None


class PredictResponse(BaseModel):
    count: int
    fraud_flagged: int
    average_probability: float
    accuracy: Optional[float] = None
    results: List[PredictionResult]


# ---------------------------------------------------------------------------
# Core scoring logic (shared by both endpoints)
# ---------------------------------------------------------------------------

def score_dataframe(df: pd.DataFrame) -> PredictResponse:
    model = get_model()

    # Pad missing features with 0
    for col in _feature_names:
        if col not in df.columns:
            df[col] = 0

    true_labels = df["label"].tolist() if "label" in df.columns else None

    X = df[_feature_names].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isnull().any().any():
        bad_cols = X.columns[X.isnull().any()].tolist()
        raise HTTPException(status_code=422, detail=f"Non-numeric or missing values found in columns: {bad_cols}")

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = []
    for i, (p, pred) in enumerate(zip(probs, preds)):
        results.append(
            PredictionResult(
                index=i,
                fraud_probability=round(float(p), 6),
                prediction=int(pred),
                risk_level=band_for(float(p)),
                true_label=int(true_labels[i]) if true_labels is not None else None,
            )
        )

    accuracy = None
    if true_labels is not None:
        correct = sum(1 for t, p in zip(true_labels, preds) if int(t) == int(p))
        accuracy = round(correct / len(true_labels), 4)

    return PredictResponse(
        count=len(results),
        fraud_flagged=int(preds.sum()),
        average_probability=round(float(np.mean(probs)), 6),
        accuracy=accuracy,
        results=results,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        model = get_model()
        return {"status": "ok", "n_features": model.n_features_in_, "features": _feature_names}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if not payload.transactions:
        raise HTTPException(status_code=422, detail="No transactions provided.")
    df = pd.DataFrame([t.model_dump() for t in payload.transactions])
    return score_dataframe(df)


@app.post("/predict/file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(".json"):
            data = json.loads(contents.decode("utf-8"))
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        else:
            raise HTTPException(status_code=422, detail="Only .json and .csv files are supported.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded file contained no rows.")

    return score_dataframe(df)