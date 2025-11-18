import pickle
from typing import List, Dict, Union, Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import preprocessing  # ensure IDSPreprocessor class is importable for pickle

# ==============================
# Load trained model + metadata
# ==============================

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

preprocessor = bundle["preprocessor"]
model = bundle["model"]
metrics = bundle.get("metrics", {})
performance = bundle.get("performance", {})

ENGINEERED_COLS = ["byte_ratio", "packet_rate"]
RAW_FEATURE_COLS = [c for c in preprocessor.feature_cols if c not in ENGINEERED_COLS]

# ==============================
# FastAPI setup
# ==============================

app_api = FastAPI(
    title="IDS Inference API",
    description="FastAPI backend for ML-based Intrusion Detection System (NSL-KDD).",
    version="1.0.0",
)


class Sample(BaseModel):
    features: Dict[str, Union[str, int, float]]


class PredictRequest(BaseModel):
    samples: List[Sample]


class PredictResult(BaseModel):
    prediction: str
    attack_probability: Union[float, None]
    raw_features: Dict[str, Any]


class PredictResponse(BaseModel):
    results: List[PredictResult]


def build_dataframe(samples: List[Sample]) -> pd.DataFrame:
    rows = []
    for s in samples:
        row = {}
        for col in RAW_FEATURE_COLS:
            if col in s.features:
                row[col] = s.features[col]
            else:
                if col in ["protocol_type", "service", "flag"]:
                    row[col] = "unknown"
                else:
                    row[col] = 0
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def predict_internal(samples: List[Sample]) -> List[PredictResult]:
    df_raw = build_dataframe(samples)
    X_processed, _, _ = preprocessor.transform(df_raw)

    preds = model.predict(X_processed)

    probs = [None] * len(preds)
    if hasattr(model, "predict_proba") and len(getattr(model, "classes_", [])) > 1:
        classes = list(model.classes_)
        if "Attack" in classes:
            attack_idx = classes.index("Attack")
            prob_array = model.predict_proba(X_processed)[:, attack_idx]
            probs = prob_array.tolist()

    results: List[PredictResult] = []
    for sample, pred, prob in zip(samples, preds, probs):
        results.append(
            PredictResult(
                prediction=pred,
                attack_probability=prob,
                raw_features=sample.features,
            )
        )
    return results


@app_api.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": getattr(model, "classes_", []).tolist()
        if hasattr(model, "classes_")
        else [],
    }


@app_api.get("/model-info")
def model_info():
    return {
        "metrics": metrics,
        "performance": performance,
        "expected_raw_features": RAW_FEATURE_COLS,
        "engineered_features": ENGINEERED_COLS,
    }


@app_api.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    results = predict_internal(req.samples)
    return PredictResponse(results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app_api", host="0.0.0.0", port=8000, reload=True)
