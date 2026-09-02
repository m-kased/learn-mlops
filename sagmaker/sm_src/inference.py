"""Inference worker for CreateModel / Batch Transform.

WHERE THIS RUNS: inside the SageMaker SKLearn inference container on AWS
during the TransformStep (NOT on your laptop).

The container loads this module and calls the four functions below in order
for each batch of input lines from S3:

  model_fn  -> load model once
  input_fn  -> parse raw CSV bytes into a DataFrame
  predict_fn -> model.predict(...)
  output_fn -> serialize predictions back to CSV bytes -> S3 *.out
"""
from __future__ import annotations

import io
import os

import joblib
import pandas as pd


# Must match the feature order written by sm_pipeline.prepare_batch_input().
FEATURE_COLUMNS = ["battery_power", "ram", "px_height", "px_width"]


def model_fn(model_dir: str):
    """Load the artifact produced by train.py (inside model.tar.gz)."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, request_content_type: str = "text/csv"):
    """Parse one Transform request body.

    Batch Transform often sends CSV without a header. We assign FEATURE_COLUMNS
    ourselves so column names match what the model was trained on.
    """
    if request_content_type not in ("text/csv", "application/x-csv"):
        raise ValueError(f"Unsupported content type: {request_content_type}")
    df = pd.read_csv(io.StringIO(request_body), header=None)
    if df.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Expected {len(FEATURE_COLUMNS)} columns, got {df.shape[1]}"
        )
    df.columns = FEATURE_COLUMNS
    return df


def predict_fn(input_data: pd.DataFrame, model):
    """Run the sklearn model. Return numpy array / list of labels."""
    return model.predict(input_data)


def output_fn(prediction, accept: str = "text/csv"):
    """Turn predictions into CSV text that Transform writes to S3."""
    if accept not in ("text/csv", "application/x-csv", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")
    return pd.DataFrame(prediction).to_csv(header=False, index=False), "text/csv"
