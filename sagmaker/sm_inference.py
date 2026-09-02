"""Inference functions for the SageMaker SKLearn serving / batch-transform container.

The container loads this module and calls:
  model_fn  -> load model from /opt/ml/model
  input_fn  -> parse request body (CSV lines)
  predict_fn -> run model.predict
  output_fn -> serialize predictions
"""
from __future__ import annotations

import io
import os

import joblib
import pandas as pd


FEATURE_COLUMNS = ["battery_power", "ram", "px_height", "px_width"]


def model_fn(model_dir: str):
    path = os.path.join(model_dir, "model.joblib")
    return joblib.load(path)


def input_fn(request_body, request_content_type: str = "text/csv"):
    if request_content_type not in ("text/csv", "application/x-csv"):
        raise ValueError(f"Unsupported content type: {request_content_type}")
    # Batch Transform often sends CSV without a header.
    df = pd.read_csv(io.StringIO(request_body), header=None)
    if df.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} columns {FEATURE_COLUMNS}, got {df.shape[1]}"
        )
    df.columns = FEATURE_COLUMNS
    return df


def predict_fn(input_data: pd.DataFrame, model):
    return model.predict(input_data)


def output_fn(prediction, accept: str = "text/csv"):
    if accept not in ("text/csv", "application/x-csv", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")
    out = pd.DataFrame(prediction)
    return out.to_csv(header=False, index=False), "text/csv"
