"""Launch a SageMaker SKLearn training job (local laptop -> AWS API).

This script does *not* train locally. It asks SageMaker to:
  1. start a short-lived ml.m5.xlarge (training free tier)
  2. download s3://.../data/train
  3. run sm_script.py in the SKLearn container
  4. upload model.tar.gz to s3://.../output
  5. stop the machine
"""
from __future__ import annotations

import os

import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing env var {name}")
    return value


def main() -> None:
    profile = required_env("AWS_PROFILE")
    region = required_env("AWS_REGION")
    bucket = required_env("S3_BUCKET")
    role = required_env("SAGEMAKER_ROLE")
    instance_type = os.environ.get("SM_INSTANCE", "ml.m5.xlarge")

    boto_session = boto3.Session(profile_name=profile, region_name=region)
    sm_session = Session(boto_session=boto_session)

    estimator = SKLearn(
        entry_point="sm_script.py",
        role=role,
        instance_count=1,
        instance_type=instance_type,
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sm_session,
        output_path=f"s3://{bucket}/output",
        hyperparameters={"n-estimators": 50, "max-depth": 8},
        disable_profiler=True,
    )

    train_uri = f"s3://{bucket}/data/train"
    print(f"starting training job  train={train_uri}  instance={instance_type}", flush=True)
    estimator.fit({"train": train_uri}, wait=True, logs=True)

    job_name = estimator.latest_training_job.name
    model_s3 = estimator.model_data
    print(f"job={job_name}", flush=True)
    print(f"model_artifact={model_s3}", flush=True)
    print("Next: aws s3 ls --recursive, then step 3 = real-time endpoint", flush=True)


if __name__ == "__main__":
    main()
