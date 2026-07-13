"""Export MNIST checkpoint to Triton model repo (TorchScript and/or ONNX).

Runs in Job triton-export (make export). Scales Triton down first — PVC is RWO.
"""
import os
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.environ.get("DATA_DIR", "/data")
RUN_NAME = os.environ.get("RUN_NAME", "run-a")
MODEL_REPO = os.environ.get("MODEL_REPO", os.path.join(DATA_DIR, "triton_models"))
CKPT_PATH = os.path.join(DATA_DIR, "checkpoints", RUN_NAME, "mnist.pt")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1")
# torch | onnx | both
FORMAT = os.environ.get("FORMAT", "both").lower()
# How many copies of the model on the GPU (instance_group count)
INSTANCES = int(os.environ.get("INSTANCES", "2"))


class Net(nn.Module):
    """Same CNN as labs/train.py — must match or weights won't load."""

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


class InferenceWrapper(nn.Module):
    """Bake MNIST normalize into the graph (clients send pixels 0–1)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = (x - 0.1307) / 0.3081
        return self.net(x)


def config_pbtxt(name: str, platform: str) -> str:
    # count > 1 = multiple model copies on the same GPU (more parallel requests,
    # more VRAM). For a tiny MNIST net, 2 is safe on a T4.
    return textwrap.dedent(f"""\
        name: "{name}"
        platform: "{platform}"
        max_batch_size: 16
        version_policy {{
          all {{ }}
        }}
        input [
          {{
            name: "INPUT__0"
            data_type: TYPE_FP32
            dims: [ 1, 28, 28 ]
          }}
        ]
        output [
          {{
            name: "OUTPUT__0"
            data_type: TYPE_FP32
            dims: [ 10 ]
          }}
        ]
        dynamic_batching {{
          preferred_batch_size: [ 8, 16 ]
          max_queue_delay_microseconds: 50000
        }}
        instance_group [
          {{
            count: {INSTANCES}
            kind: KIND_GPU
          }}
        ]
    """)


def load_wrapped() -> nn.Module:
    net = Net()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return InferenceWrapper(net)


def export_torch(wrapped: nn.Module, version: str) -> None:
    """pytorch_libtorch backend → model.pt (TorchScript)."""
    model_dir = os.path.join(MODEL_REPO, "mnist")
    version_dir = os.path.join(model_dir, version)
    os.makedirs(version_dir, exist_ok=True)
    example = torch.randn(1, 1, 28, 28)
    traced = torch.jit.trace(wrapped, example)
    out_path = os.path.join(version_dir, "model.pt")
    traced.save(out_path)
    with open(os.path.join(model_dir, "config.pbtxt"), "w", encoding="utf-8") as f:
        f.write(config_pbtxt("mnist", "pytorch_libtorch"))
    print(f"  torch  {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


def export_onnx(wrapped: nn.Module, version: str) -> None:
    """onnxruntime_onnx backend → model.onnx (same weights, different format)."""
    model_dir = os.path.join(MODEL_REPO, "mnist_onnx")
    version_dir = os.path.join(model_dir, version)
    os.makedirs(version_dir, exist_ok=True)
    example = torch.randn(1, 1, 28, 28)
    out_path = os.path.join(version_dir, "model.onnx")
    torch.onnx.export(
        wrapped,
        example,
        out_path,
        input_names=["INPUT__0"],
        output_names=["OUTPUT__0"],
        dynamic_axes={
            "INPUT__0": {0: "batch"},
            "OUTPUT__0": {0: "batch"},
        },
        opset_version=17,
    )
    with open(os.path.join(model_dir, "config.pbtxt"), "w", encoding="utf-8") as f:
        f.write(config_pbtxt("mnist_onnx", "onnxruntime_onnx"))
    print(f"  onnx   {out_path} ({os.path.getsize(out_path)} bytes)", flush=True)


def main():
    if not os.path.isfile(CKPT_PATH):
        raise SystemExit(
            f"Checkpoint not found: {CKPT_PATH}  (run: make -C labs workloads)"
        )
    if not MODEL_VERSION.isdigit():
        raise SystemExit(f"MODEL_VERSION must be an integer, got: {MODEL_VERSION!r}")
    if FORMAT not in ("torch", "onnx", "both"):
        raise SystemExit(f"FORMAT must be torch|onnx|both, got: {FORMAT!r}")
    if INSTANCES < 1:
        raise SystemExit(f"INSTANCES must be >= 1, got: {INSTANCES}")

    wrapped = load_wrapped()
    print(
        f"Exporting version={MODEL_VERSION} format={FORMAT} "
        f"instances={INSTANCES} -> {MODEL_REPO}/",
        flush=True,
    )
    if FORMAT in ("torch", "both"):
        export_torch(wrapped, MODEL_VERSION)
    if FORMAT in ("onnx", "both"):
        export_onnx(wrapped, MODEL_VERSION)
    print("Done. Next: make deploy", flush=True)


if __name__ == "__main__":
    main()
