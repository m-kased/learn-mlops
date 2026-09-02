# Learn MLOps (`learn-mlops`)

DevOps → MLOps learning path — from Python/ML fundamentals through tooling, Kubernetes, monitoring, and LLMOps.

Phases 1–3 and SageMaker are complete. Current focus is Phase 4: drift, monitoring, and model lifecycle.

```
✅  Foundations · Phase 1 · Phase 2 · Phase 3 · SageMaker
🚧  Phase 4: Drift · Monitoring · Model lifecycle
⬜  LLMOps
```

---

## Quick start

```bash
make cluster-help           # GKE lifecycle
make labs-help              # GPU labs
make triton-help            # Triton (done)
make kserve-help            # KServe (done)
make ct-help                # Continuous Training (done)
make dvc-help               # DVC helpers (done)
make sm-help                # SageMaker lab (done) → make -C sm <target>
```


| Path                 | Purpose                        |
| -------------------- | ------------------------------ |
| [cluster/](cluster/) | GKE, GPU pools, cost safety    |
| [labs/](labs/)       | Train, serve, queue, KEDA      |
| [triton/](triton/)   | Triton export / serve / tune   |
| [kserve/](kserve/)   | InferenceService, graphs, xform|
| [ct/](ct/)           | Continuous Training pipeline   |
| [dvc/](dvc/)         | DVC + GCS helper commands      |
| [sagmaker/](sagmaker/)           | SageMaker (done)               |
| Root                 | Module shortcuts               |


---



## Roadmap


| Phase | Focus                                      | Status         |
| ----- | ------------------------------------------ | -------------- |
| —     | DevOps foundations                         | ✅              |
| 1     | Python + ML fundamentals                   | ✅              |
| 2     | ML tooling (DVC, MLflow, Airflow, serving) | ✅              |
| 3     | K8s for ML + GPU + CI/CD/CT                | ✅              |
| —     | SageMaker (AWS train / serve)              | ✅              |
| 4     | Drift, monitoring, model lifecycle         | 🚧              |
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

### Phase 3 — Kubernetes & GPU ML ✅


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
| KServe                            | ✅      | [kserve/](kserve/) — Standard, graphs, xform |
| CI/CD/CT (Continuous Training)    | ✅      | [ct/](ct/) · Actions · [CML](https://cml.dev) |
| Data versioning on GCS            | ✅      | [dvc/](dvc/) · DVC remote + demo pointer      |


### SageMaker (AWS) ✅


| Topic                                      | Status | Evidence / notes                                      |
| ------------------------------------------ | ------ | ----------------------------------------------------- |
| IAM / S3 / profile                         | ✅      | [sm/](sm/)                                            |
| Training + Batch Transform                 | ✅      | `sm_batch.py`, `sm_src/`                              |
| Pipelines (Train → Eval → Model → Transform) | ✅    | `sm_pipeline.py`                                      |
| Processing Job (evaluate / gate)           | ✅      | `sm_src/evaluate.py`                                  |
| Inference options (concepts)               | ✅      | real-time / serverless / async / MME / IC             |
| Model Registry / endpoint polish           | ⬜ optional | skip for now; revisit if needed                    |


Resources: `make sm-help` · `make -C sm pipeline-status`


### Phase 4 — Drift, Monitoring & Model Lifecycle 🚧 (fresh)


| Topic                                  | Status | Evidence / notes                                      |
| -------------------------------------- | ------ | ----------------------------------------------------- |
| Data / concept / prediction drift      | ⬜      | start here                                            |
| Drift metrics (KS, PSI, JS divergence) | ⬜      | —                                                     |
| Evidently AI                           | ⬜      | [evidently](https://github.com/evidentlyai/evidently) |
| Model registry (MLflow / BentoML)      | ⬜      | light pass — you already know MLflow                  |
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
