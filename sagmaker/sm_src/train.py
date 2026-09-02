"""Training worker for the Pipeline TrainingStep.

WHERE THIS RUNS: inside the SageMaker SKLearn training container on AWS
(NOT on your laptop).

SageMaker starts a machine, downloads this script + S3 training data, sets
env vars, runs `python train.py`, packs /opt/ml/model into model.tar.gz,
uploads it to S3, then stops the machine.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    # Hyperparameters from ModelTrainer become CLI args, e.g.:
    #   python train.py --n-estimators 50 --max-depth 8
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # SageMaker convention:
    #   channel "train"  ->  SM_CHANNEL_TRAIN=/opt/ml/input/data/train
    #   model output     ->  SM_MODEL_DIR=/opt/ml/model  (later packed as model.tar.gz)
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    csv_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
    if not csv_files:
        raise SystemExit(f"No CSV files in {train_dir}")

    print(f"train_dir={train_dir} files={csv_files}", flush=True)
    df = pd.read_csv(csv_files[0])
    y = df["price_range"]  # label
    x = df.drop(columns=["price_range"])  # features

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    print(f"train_accuracy={float(model.score(x, y)):.4f} n={len(df)}", flush=True)

    # Anything left under SM_MODEL_DIR is what SageMaker ships as model.tar.gz
    # for the next pipeline step (CreateModel / Transform / Evaluate).
    os.makedirs(model_dir, exist_ok=True)
    out = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, out)
    print(f"saved {out}", flush=True)

    # Batch Transform looks for /opt/ml/model/code/inference.py inside the
    # unpacked artifact. Pack it here so model.tar.gz is self-contained.
    code_dir = os.path.join(model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    inference_src = os.path.join(os.path.dirname(__file__), "inference.py")
    inference_dst = os.path.join(code_dir, "inference.py")
    shutil.copy(inference_src, inference_dst)
    print(f"packed {inference_dst}", flush=True)


if __name__ == "__main__":
    main()
