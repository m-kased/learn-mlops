"""Quality gate for Continuous Training.

Reads metrics.json from the PVC and exits:
  0 if accuracy >= MIN_ACCURACY
  1 otherwise

This is the CT idea: model quality becomes a deployment gate.
"""
import json
import os
import sys

DATA_DIR = os.environ.get("DATA_DIR", "/data")
RUN_NAME = os.environ.get("RUN_NAME", "run-a")
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.98"))
METRICS_PATH = os.path.join(DATA_DIR, "checkpoints", RUN_NAME, "metrics.json")


def main():
    if not os.path.isfile(METRICS_PATH):
        raise SystemExit(f"Metrics not found: {METRICS_PATH} (run: make -C ct evaluate)")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    accuracy = float(metrics["accuracy"])
    print(json.dumps(metrics), flush=True)
    print(
        f"quality gate: accuracy={accuracy:.4f} min_accuracy={MIN_ACCURACY:.4f}",
        flush=True,
    )

    if accuracy < MIN_ACCURACY:
        print("FAIL: candidate model is below threshold", flush=True)
        sys.exit(1)

    print("PASS: candidate model is good enough to export/deploy", flush=True)


if __name__ == "__main__":
    main()
