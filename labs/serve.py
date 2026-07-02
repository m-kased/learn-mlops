"""MNIST inference server — loads a training checkpoint from the PVC and serves HTTP.

Endpoints:
  GET  /healthz         -> readiness probe
  GET  /metrics         -> Prometheus text metrics (http_requests_total counter)
  GET  /predict/sample  -> classify a random MNIST test image
  POST /predict         -> {"pixels": [...784 floats or 28x28...]}
"""
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DATA_DIR = os.environ.get("DATA_DIR", "/data")
RUN_NAME = os.environ.get("RUN_NAME", "run-a")
CKPT_PATH = os.path.join(DATA_DIR, "checkpoints", RUN_NAME, "mnist.pt")
PORT = int(os.environ.get("PORT", "8080"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_requests_total = 0
_metrics_lock = threading.Lock()


def _inc_requests():
    global _requests_total
    with _metrics_lock:
        _requests_total += 1


def _prometheus_metrics():
    with _metrics_lock:
        n = _requests_total
    return (
        "# HELP http_requests_total Total inference HTTP requests\n"
        "# TYPE http_requests_total counter\n"
        f"http_requests_total{{app=\"mnist-serve\"}} {n}\n"
    )


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
print("Model ready.", flush=True)

_normalize = transforms.Normalize((0.1307,), (0.3081,))
_test = datasets.MNIST(DATA_DIR, train=False, download=True,
                       transform=transforms.ToTensor())


@torch.no_grad()
def classify(img_2d):
    x = _normalize(img_2d.unsqueeze(0)).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    pred = int(probs.argmax())
    return pred, [round(float(p), 4) for p in probs]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"status": "ok", "device": DEVICE, "run": RUN_NAME})
        elif self.path == "/metrics":
            body = _prometheus_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/predict/sample":
            _inc_requests()
            idx = random.randrange(len(_test))
            img, label = _test[idx]
            pred, probs = classify(img.squeeze(0))
            self._send(200, {"actual": int(label), "predicted": pred, "probs": probs})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            _inc_requests()
            data = json.loads(self.rfile.read(n))
            pixels = torch.tensor(data["pixels"], dtype=torch.float32).reshape(28, 28)
            pred, probs = classify(pixels)
            self._send(200, {"predicted": pred, "probs": probs})
        except Exception as e:
            self._send(400, {"error": str(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"Serving on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
