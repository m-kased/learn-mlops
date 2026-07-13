"""Send one MNIST test image to Triton HTTP infer (optional model version)."""
import json
import os
import urllib.request

import numpy as np
from torchvision import datasets, transforms

TRITON_URL = os.environ.get("TRITON_URL", "http://triton:8000")
MODEL = os.environ.get("TRITON_MODEL", "mnist")
# Empty = default URL (highest version). Set "1" or "2" to pin a version.
VERSION = os.environ.get("TRITON_MODEL_VERSION", "").strip()

test = datasets.MNIST("/tmp/mnist", train=False, download=True,
                      transform=transforms.ToTensor())
img, label = test[0]
pixels = img.numpy().astype(np.float32)

payload = {
    "inputs": [{
        "name": "INPUT__0",
        "shape": [1, 1, 28, 28],
        "datatype": "FP32",
        "data": pixels.reshape(-1).tolist(),
    }],
    "outputs": [{"name": "OUTPUT__0"}],
}

if VERSION:
    url = f"{TRITON_URL}/v2/models/{MODEL}/versions/{VERSION}/infer"
else:
    url = f"{TRITON_URL}/v2/models/{MODEL}/infer"

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.loads(resp.read())

logits = np.array(result["outputs"][0]["data"], dtype=np.float32)
pred = int(logits.argmax())
ok = "OK" if pred == label else "MISS"
ver = VERSION or "default"
print(f"version={ver} actual={label} predicted={pred} {ok}", flush=True)
