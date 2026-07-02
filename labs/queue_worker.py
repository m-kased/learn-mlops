"""GPU queue worker — pulls jobs from Redis and runs MNIST inference.

Job format: a number = MNIST test image index (0..9999).
Anything else (e.g. "job-1") picks a random test image.
"""
import os
import random
import time

import redis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Use QUEUE_REDIS_* — not REDIS_HOST/REDIS_PORT (K8s injects REDIS_PORT=tcp://... for the Service)
REDIS_HOST = os.environ.get("QUEUE_REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("QUEUE_REDIS_PORT", "6379"))
QUEUE_NAME = os.environ.get("QUEUE_NAME", "inference:queue")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
RUN_NAME = os.environ.get("RUN_NAME", "run-a")
CKPT_PATH = os.environ.get(
    "CKPT_PATH", os.path.join(DATA_DIR, "checkpoints", RUN_NAME, "mnist.pt")
)
PROCESS_DELAY_SEC = float(os.environ.get("PROCESS_DELAY_SEC", "0"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
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


print(f"Loading {CKPT_PATH} on {DEVICE}", flush=True)
model = Net().to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

_normalize = transforms.Normalize((0.1307,), (0.3081,))
_test = datasets.MNIST(DATA_DIR, train=False, download=True,
                       transform=transforms.ToTensor())
print(f"Worker ready. Waiting on queue '{QUEUE_NAME}'...", flush=True)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


@torch.no_grad()
def classify(img_2d):
    x = _normalize(img_2d.unsqueeze(0)).unsqueeze(0).to(DEVICE)
    pred = int(F.softmax(model(x), dim=1)[0].argmax())
    return pred


while True:
    item = r.brpop(QUEUE_NAME, timeout=0)  # blocks until a job arrives
    job_id = item[1]
    try:
        idx = int(job_id)
    except ValueError:
        idx = random.randrange(len(_test))
    img, label = _test[idx]
    pred = classify(img.squeeze(0))
    ok = "OK" if pred == label else "MISS"
    print(f"job={job_id} idx={idx} actual={label} predicted={pred} {ok}", flush=True)
    if PROCESS_DELAY_SEC > 0:
        time.sleep(PROCESS_DELAY_SEC)  # demo: slow drain so KEDA can add workers
