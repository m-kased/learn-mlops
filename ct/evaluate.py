"""Evaluate a trained MNIST checkpoint on the test set → metrics.json on PVC.

This is the missing piece for Continuous Training: it turns a trained model
into a NUMBER (accuracy) that a quality gate can allow/deny a deploy on.

Reads:  /data/checkpoints/<RUN_NAME>/mnist.pt   (written by labs/train.py)
Writes: /data/checkpoints/<RUN_NAME>/metrics.json
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = os.environ.get("DATA_DIR", "/data")
RUN_NAME = os.environ.get("RUN_NAME", "run-a")
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints", RUN_NAME)
CKPT_PATH = os.path.join(CKPT_DIR, "mnist.pt")
METRICS_PATH = os.path.join(CKPT_DIR, "metrics.json")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))


class Net(nn.Module):
    """Must match labs/train.py exactly or load_state_dict fails."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    if not os.path.isfile(CKPT_PATH):
        raise SystemExit(
            f"Checkpoint not found: {CKPT_PATH}  (run training first)"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating run={RUN_NAME} device={device}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = Net().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    avg_loss = loss_sum / total
    metrics = {
        "run": RUN_NAME,
        "accuracy": round(accuracy, 4),
        "avg_loss": round(avg_loss, 4),
        "correct": correct,
        "total": total,
        "trained_epochs": ckpt.get("epoch"),
    }

    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics), flush=True)
    print(f"accuracy={accuracy:.4f} -> {METRICS_PATH}", flush=True)


if __name__ == "__main__":
    main()
