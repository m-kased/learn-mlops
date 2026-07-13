# Learn MLOps (`learn-mlops`)

DevOps → MLOps learning path — from Python/ML fundamentals through tooling, Kubernetes, monitoring, and LLMOps.

Phases 1–2 done; current focus is Phase 3 (K8s for ML). The GPU / GKE labs in this repo are hands-on practice for that phase.

```
✅  Foundations · Phase 1 · Phase 2 · GPU labs · Triton
🚧  KServe (next)
⬜  CI/CD/CT · drift · LLMOps
```

---

## Quick start

```bash
make cluster-help           # GKE lifecycle
make labs-help              # completed GPU labs
make triton-help            # Triton (done — under triton/)
make -C labs workloads      # train → checkpoint on PVC
make -C triton up           # export + deploy Triton
make -C triton test
```


| Path                 | Purpose                      |
| -------------------- | ---------------------------- |
| [cluster/](cluster/) | GKE, GPU pools, cost safety  |
| [labs/](labs/)       | Train, serve, queue, KEDA    |
| [triton/](triton/)   | Triton export / serve / tune |
| Root                 | Next: KServe                 |


---



## Roadmap


| Phase | Focus                                      | Status         |
| ----- | ------------------------------------------ | -------------- |
| —     | DevOps foundations                         | ✅              |
| 1     | Python + ML fundamentals                   | ✅              |
| 2     | ML tooling (DVC, MLflow, Airflow, serving) | ✅              |
| **3** | **K8s for ML + GPU + CI/CD/CT**            | **🚧 current** |
| 4     | Drift, monitoring, model registry          | ⬜              |
| 5+    | LLMOps                                     | ⬜              |




### Phase 1 — Python & ML ✅


| Topic                                          | Status | Evidence / notes               |
| ---------------------------------------------- | ------ | ------------------------------ |
| Python for ML (numpy, pandas, scikit-learn)    | ✅      | Data stack + PyTorch           |
| ML fundamentals (splits, overfitting, metrics) | ✅      | MNIST train/eval               |
| End-to-end model + publish                     | ✅      | [labs/train.py](labs/train.py) |


Resources: [Real Python](https://realpython.com) · [Kaggle Learn](https://www.kaggle.com/learn/python) · [StatQuest](https://www.youtube.com/@statquest) · [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) · [Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction)

### Phase 2 — ML tooling ✅


| Topic                                                             | Status | Evidence / notes  |
| ----------------------------------------------------------------- | ------ | ----------------- |
| Data versioning (DVC)                                             | ✅      | —                 |
| Experiment tracking (MLflow)                                      | ✅      | —                 |
| Orchestration (Airflow)                                           | ✅      | —                 |
| Model serving (FastAPI / BentoML)                                 | ✅      | —                 |
| [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) | ✅      | Structured course |


Resources: [DVC](https://dvc.org) · [MLflow](https://mlflow.org/docs/latest/index.html) · [Airflow](https://airflow.apache.org/docs/) · [FastAPI](https://fastapi.tiangolo.com/tutorial/) · [BentoML](https://docs.bentoml.com)

### Phase 3 — Kubernetes & GPU ML 🚧


| Topic                             | Status | Evidence / notes                             |
| --------------------------------- | ------ | -------------------------------------------- |
| GKE GPU cluster + cost safety     | ✅      | [cluster/](cluster/) · scale-to-zero         |
| GPU / PyTorch smoke tests         | ✅      | labs                                         |
| GPU time-sharing                  | ✅      | labs                                         |
| MNIST training Job + PVC          | ✅      | [labs/train.py](labs/train.py)               |
| HTTP serve + CPU HPA              | ✅      | [labs/serve.py](labs/serve.py)               |
| Redis queue + GPU workers         | ✅      | [labs/queue_worker.py](labs/queue_worker.py) |
| KEDA (queue depth + RPS)          | ✅      | labs                                         |
| NVIDIA GPU nodes / scheduling     | ✅      | GPU pools on GKE                             |
| Triton export (checkpoint → repo) | ✅      | [triton/](triton/)                           |
| Triton deploy + infer + tune      | ✅      | versions, ONNX, metrics, instances           |
| KServe                            | 🚧     | next                                         |
| CI/CD/CT (Continuous Training)    | ⬜      | [CML](https://cml.dev)                       |




### Phase 4 — Drift & monitoring ⬜


| Topic                                  | Status | Evidence / notes                                      |
| -------------------------------------- | ------ | ----------------------------------------------------- |
| Data / concept / prediction drift      | ⬜      | —                                                     |
| Drift metrics (KS, PSI, JS divergence) | ⬜      | —                                                     |
| Evidently AI                           | ⬜      | [evidently](https://github.com/evidentlyai/evidently) |
| Model registry (MLflow / BentoML)      | ⬜      | —                                                     |
| Feature stores (Feast) — later         | ⬜      | [Feast](https://feast.dev)                            |
| Drift alerts + runbook                 | ⬜      | [labs/prometheus.yaml](labs/prometheus.yaml)          |




### Phase 5 — LLMOps ⬜


| Topic                                    | Status | Evidence / notes                                                                                                                 |
| ---------------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| LLM serving (vLLM)                       | ⬜      | [docs.vllm.ai](https://docs.vllm.ai)                                                                                             |
| LLM observability (Langfuse / LangSmith) | ⬜      | [Langfuse](https://langfuse.com) · [LangSmith](https://www.langchain.com/langsmith)                                              |
| LLM concepts                             | ⬜      | [Karpathy Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g) · [HF LLM Course](https://huggingface.co/learn/llm-course) |


---



## Background

**Already solid:** Linux · Git · Docker · Kubernetes · CI/CD · Cloud · Prometheus/Grafana · IaC · production ops


| Strength      | Applied to MLOps                  |
| ------------- | --------------------------------- |
| Cost control  | GPU scale-to-zero, right-sizing   |
| Automation    | Pipelines around training + infra |
| Observability | Metrics/alerts for models         |
| Incidents     | Runbooks, on-call, post-mortems   |
| Security      | IAM, networking, secrets          |


**Approach:** infra over research · concepts before tools · lean on DevOps · ship runnable labs

---

✅ done · 🚧 in progress · ⬜ not started