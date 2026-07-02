"""Minimal MNIST CNN trainer for the GPU-in-Kubernetes learning path.

Reads/writes everything under /data (a PersistentVolumeClaim) so the dataset
download and model checkpoints survive pod restarts and spot-node reclaims.
Resumes from the last checkpoint if one exists.
"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = os.environ.get("DATA_DIR", "/data")
RUN_NAME = os.environ.get("RUN_NAME", "default")
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints", RUN_NAME)
CKPT_PATH = os.path.join(CKPT_DIR, "mnist.pt")
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
LR = float(os.environ.get("LR", "0.01"))


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Run: {RUN_NAME} | device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    start_epoch = 0
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from checkpoint at epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i % 100 == 0:
                print(f"epoch {epoch} step {i:4d}/{len(train_loader)} "
                      f"loss {loss.item():.4f}", flush=True)
        avg = running / len(train_loader)
        dt = time.time() - t0
        print(f"== epoch {epoch} done | avg loss {avg:.4f} | {dt:.1f}s ==", flush=True)

        torch.save(
            {"model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "epoch": epoch + 1},
            CKPT_PATH,
        )
        print(f"Saved checkpoint -> {CKPT_PATH}", flush=True)

    print("--- Training complete! ---", flush=True)


if __name__ == "__main__":
    main()
