"""Training script that runs *inside* the SageMaker SKLearn container.

SageMaker maps:
  SM_CHANNEL_TRAIN  ->  /opt/ml/input/data/train   (copied from s3://.../data/train/)
  SM_MODEL_DIR      ->  /opt/ml/model              (packed into model.tar.gz on exit)

This is the same idea as your GKE Job command, except AWS chooses the node.
"""
from __future__ import annotations

import argparse
import glob
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    csv_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
    if not csv_files:
        raise SystemExit(f"No CSV files in {train_dir}")

    print(f"train_dir={train_dir} files={csv_files}", flush=True)
    df = pd.read_csv(csv_files[0])
    y = df["price_range"]
    x = df.drop(columns=["price_range"])

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    acc = float(model.score(x, y))
    print(f"train_accuracy={acc:.4f} n={len(df)}", flush=True)

    os.makedirs(model_dir, exist_ok=True)
    out = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
