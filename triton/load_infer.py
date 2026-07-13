"""Concurrent requests to Triton — print throughput + latency."""
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

TRITON_URL = os.environ.get("TRITON_URL", "http://triton:8000")
MODEL = os.environ.get("TRITON_MODEL", "mnist")
REQUESTS = int(os.environ.get("LOAD_REQUESTS", "128"))
WORKERS = int(os.environ.get("LOAD_WORKERS", "32"))


def one_infer():
    pixels = [random.random() for _ in range(1 * 1 * 28 * 28)]
    payload = {
        "inputs": [{
            "name": "INPUT__0",
            "shape": [1, 1, 28, 28],
            "datatype": "FP32",
            "data": pixels,
        }],
        "outputs": [{"name": "OUTPUT__0"}],
    }
    req = urllib.request.Request(
        f"{TRITON_URL}/v2/models/{MODEL}/infer",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def main():
    print(f"Load: {REQUESTS} req, {WORKERS} workers, model={MODEL}", flush=True)
    try:
        one_infer()
    except urllib.error.URLError as e:
        raise SystemExit(f"Triton not reachable: {e}") from e

    latencies = []
    errors = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(one_infer) for _ in range(REQUESTS)]
        for fut in as_completed(futs):
            try:
                latencies.append(fut.result())
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  error: {e}", flush=True)
    wall = time.perf_counter() - t0
    ok = len(latencies)
    print(f"ok={ok} errors={errors} wall={wall:.2f}s", flush=True)
    if not latencies:
        raise SystemExit("all failed")
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    print(
        f"throughput={ok / wall:.1f} req/s  "
        f"latency_ms avg={statistics.mean(latencies):.1f} p50={p50:.1f} p95={p95:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
