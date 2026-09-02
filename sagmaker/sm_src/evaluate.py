"""Evaluation worker for the Pipeline ProcessingStep.

WHERE THIS RUNS: inside a SageMaker Processing container on AWS
(NOT on your laptop).

SageMaker mounts inputs under /opt/ml/processing/..., runs this script,
then uploads anything under the evaluation output path to S3.

Mental model (you already did this on GKE CT):
  train produces model  ->  evaluate scores holdout  ->  gate / next steps
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import tarfile

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Soft gate: fail the Processing Job (and thus block Transform) if too low.
    parser.add_argument("--min-accuracy", type=float, default=0.55)
    return parser.parse_args()


def load_model(model_dir: str):
    """Training output is model.tar.gz; Processing downloads it as a file."""
    tar_candidates = sorted(glob.glob(os.path.join(model_dir, "*.tar.gz")))
    if tar_candidates:
        extract_dir = os.path.join(model_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_candidates[0], "r:gz") as tar:
            tar.extractall(extract_dir)
        model_path = os.path.join(extract_dir, "model.joblib")
    else:
        model_path = os.path.join(model_dir, "model.joblib")

    if not os.path.isfile(model_path):
        raise SystemExit(f"model.joblib not found under {model_dir}")
    return joblib.load(model_path)


def main() -> None:
    args = parse_args()

    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    out_dir = "/opt/ml/processing/evaluation"
    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    if not csv_files:
        raise SystemExit(f"No CSV files in {test_dir}")

    print(f"test_dir={test_dir} files={csv_files}", flush=True)
    df = pd.read_csv(csv_files[0])
    y = df["price_range"]
    x = df.drop(columns=["price_range"])

    model = load_model(model_dir)
    accuracy = float(model.score(x, y))
    metrics = {
        "accuracy": accuracy,
        "n_test": int(len(df)),
        "min_accuracy": args.min_accuracy,
        "pass": accuracy >= args.min_accuracy,
    }

    out_path = os.path.join(out_dir, "evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics), flush=True)
    print(f"wrote {out_path}", flush=True)

    if accuracy < args.min_accuracy:
        raise SystemExit(
            f"FAIL: accuracy={accuracy:.4f} < min_accuracy={args.min_accuracy:.4f}"
        )
    print("PASS: quality gate ok", flush=True)


if __name__ == "__main__":
    main()
