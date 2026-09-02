"""SageMaker Batch Transform (step 4) — score a whole S3 file, no live endpoint.

Flow:
  1. Train a tiny sklearn model locally (so we do not need a Training Job)
  2. Package model.joblib + code/inference.py into model.tar.gz
  3. Upload model + batch input CSV to S3
  4. CreateModel (SKLearn inference image)
  5. CreateTransformJob (starts machine, scores file, stops machine)
  6. Print prediction S3 path

This is the DevOps-friendly alternative to a 24/7 endpoint.
"""
from __future__ import annotations

import os
import tarfile
import tempfile
import time
from pathlib import Path

import boto3
import joblib
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS = ["battery_power", "ram", "px_height", "px_width"]
LABEL = "price_range"
# Official SKLearn 1.2-1 inference image for us-east-1
SKLEARN_IMAGE = (
    "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
    "sagemaker-scikit-learn:1.2-1-cpu-py3"
)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing env var {name}")
    return value


def train_local(train_csv: Path, model_path: Path) -> float:
    df = pd.read_csv(train_csv)
    y = df[LABEL]
    x = df[FEATURE_COLUMNS]
    model = RandomForestClassifier(
        n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
    )
    model.fit(x, y)
    acc = float(model.score(x, y))
    joblib.dump(model, model_path)
    return acc


def make_batch_input(test_csv: Path, out_csv: Path) -> int:
    df = pd.read_csv(test_csv)
    # Batch Transform CSV for our inference.py: no header, features only.
    features = df[FEATURE_COLUMNS]
    features.to_csv(out_csv, header=False, index=False)
    return len(features)


def package_model(model_joblib: Path, inference_py: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_joblib, arcname="model.joblib")
        tar.add(inference_py, arcname="code/inference.py")


def wait_transform(sm, job_name: str, timeout_s: int = 1800) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        desc = sm.describe_transform_job(TransformJobName=job_name)
        status = desc["TransformJobStatus"]
        print(f"transform status={status}", flush=True)
        if status == "Completed":
            return desc
        if status in {"Failed", "Stopped"}:
            reason = desc.get("FailureReason", "unknown")
            raise SystemExit(f"Transform job {status}: {reason}")
        time.sleep(30)
    raise SystemExit(f"Timed out waiting for transform job {job_name}")


def main() -> None:
    profile = required_env("AWS_PROFILE")
    region = required_env("AWS_REGION")
    bucket = required_env("S3_BUCKET")
    role = required_env("SAGEMAKER_ROLE")
    instance_type = os.environ.get("SM_INSTANCE", "ml.m5.large")

    root = Path(__file__).resolve().parent
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    inference_py = root / "sm_inference.py"
    if not train_csv.is_file() or not test_csv.is_file():
        raise SystemExit("train.csv / test.csv missing — run make sm-upload first")
    if not inference_py.is_file():
        raise SystemExit("sm_inference.py missing")

    session = boto3.Session(profile_name=profile, region_name=region)
    s3 = session.client("s3")
    sm = session.client("sagemaker")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"mob-rf-batch-{stamp}"
    job_name = f"mob-batch-transform-{stamp}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        model_joblib = tmp_dir / "model.joblib"
        batch_csv = tmp_dir / "batch_input.csv"
        tar_path = tmp_dir / "model.tar.gz"

        acc = train_local(train_csv, model_joblib)
        n_rows = make_batch_input(test_csv, batch_csv)
        package_model(model_joblib, inference_py, tar_path)
        print(f"local train_accuracy={acc:.4f} batch_rows={n_rows}", flush=True)

        model_key = f"models/batch-demo/{stamp}/model.tar.gz"
        input_key = f"data/batch/{stamp}/input.csv"
        output_prefix = f"output/batch/{stamp}"

        s3.upload_file(str(tar_path), bucket, model_key)
        s3.upload_file(str(batch_csv), bucket, input_key)
        model_data = f"s3://{bucket}/{model_key}"
        input_s3 = f"s3://{bucket}/{input_key}"
        output_s3 = f"s3://{bucket}/{output_prefix}"
        print(f"uploaded model={model_data}", flush=True)
        print(f"uploaded input={input_s3}", flush=True)

    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={
                "Image": SKLEARN_IMAGE,
                "ModelDataUrl": model_data,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": region,
                },
            },
            ExecutionRoleArn=role,
        )
    except ClientError as exc:
        raise SystemExit(f"create_model failed: {exc}") from exc
    print(f"created model={model_name}", flush=True)

    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        MaxConcurrentTransforms=1,
        MaxPayloadInMB=6,
        BatchStrategy="MultiRecord",
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_s3,
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": output_s3,
            "AssembleWith": "Line",
            "Accept": "text/csv",
        },
        TransformResources={
            "InstanceType": instance_type,
            "InstanceCount": 1,
        },
    )
    print(f"started transform job={job_name} instance={instance_type}", flush=True)
    desc = wait_transform(sm, job_name)

    print("=== Batch Transform completed ===", flush=True)
    print(f"job={job_name}", flush=True)
    print(f"model={model_name}", flush=True)
    print(f"predictions={desc['TransformOutput']['S3OutputPath']}", flush=True)
    print("Inspect with: make sm-batch-results", flush=True)
    print(
        "Cleanup model (optional): "
        f"aws sagemaker delete-model --model-name {model_name} "
        f"--profile {profile} --region {region}",
        flush=True,
    )


if __name__ == "__main__":
    main()
