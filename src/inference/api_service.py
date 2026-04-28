from __future__ import annotations

import json
import os
import threading
import time
import pandas as pd
import base64
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
import torch

from src.inference.clip_helpers import (
    DEFAULT_CLASS_NAMES,
    clip_predict_classes,
    resolve_image_input,
)
from src.inference.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    ScoredLabel,
)
from src.inference.wandb_helpers import (
    download_artifact,
    find_first_pt_file,
    get_history,
    get_metadata,
    list_logged_artifacts,
    list_runs,
    load_run,
)

# ----------------------
# CONFIG
# ----------------------
WANDB_API_KEY = os.getenv("WANDB_API_KEY") or ""
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or ""
WANDB_PROJECT = os.getenv("WANDB_PROJECT") or ""
WANDB_RUN_ID = os.getenv("WANDB_RUN_ID") or ""
WANDB_RUNS_LIMIT = int(os.getenv("WANDB_RUNS_LIMIT", "50"))
ARTIFACT_CACHE_DIR = os.getenv("WANDB_ARTIFACT_CACHE_DIR", "/tmp/clip_sat_wandb_artifacts")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
WANDB_CACHE_TTL_SECONDS = float(os.getenv("WANDB_CACHE_TTL_SECONDS", "60"))

# ----------------------
# FASTAPI + METRICS
# ----------------------
app = FastAPI(title="CLIP-Sat Inference API")
app.mount("/metrics", make_asgi_app())

REQUEST_COUNTER = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint", "model", "status"],
)
ERROR_COUNTER = Counter(
    "inference_errors_total",
    "Total inference errors",
    ["endpoint", "model"],
)
INFERENCE_TIME = Histogram(
    "inference_latency_seconds",
    "Inference latency",
    ["endpoint", "model"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)
MODEL_LOAD_TIME = Histogram(
    "model_load_latency_seconds",
    "Model download/load latency",
    ["model", "artifact"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60],
)
PREPROCESSING_TIME = Histogram(
    "preprocessing_latency_seconds",
    "Preprocessing latency",
    ["endpoint", "model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
)
REQUEST_PAYLOAD_BYTES = Histogram(
    "inference_request_bytes",
    "Inference request payload size in bytes",
    ["endpoint"],
)
RESPONSE_PAYLOAD_BYTES = Histogram(
    "inference_response_bytes",
    "Inference response payload size in bytes",
    ["endpoint"],
)
INFERENCE_INFLIGHT = Gauge(
    "inference_inflight",
    "In-flight inference requests",
    ["endpoint"],
)
CACHE_HIT = Counter("model_cache_hit_total", "Model cache hits", ["model"])
CACHE_MISS = Counter("model_cache_miss_total", "Model cache misses", ["model"])
MODEL_LOAD_ERRORS = Counter("model_load_errors_total", "Model load errors", ["model"])
PAGE_VIEWS = Counter("page_views_total", "Total page hits", ["endpoint"])

# ----------------------
# MODEL ARTIFACT CACHE (per-run download paths)
# ----------------------
_artifact_lock = threading.Lock()
_artifact_cache: Dict[str, Dict[str, Any]] = {}

_wandb_cache_lock = threading.Lock()
_wandb_cache: Dict[str, Dict[str, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    if WANDB_CACHE_TTL_SECONDS <= 0:
        return None
    now = time.time()
    with _wandb_cache_lock:
        item = _wandb_cache.get(key)
        if not item:
            return None
        if now - item["ts"] > WANDB_CACHE_TTL_SECONDS:
            _wandb_cache.pop(key, None)
            return None
        return item["value"]

def _cache_set(key: str, value: Any) -> None:
    if WANDB_CACHE_TTL_SECONDS <= 0:
        return
    with _wandb_cache_lock:
        _wandb_cache[key] = {"ts": time.time(), "value": value}

def _can_use_cuda() -> bool:
    if os.getenv("FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.tensor([0.0], device="cuda") + 1.0
        torch.cuda.synchronize()
    except Exception:
        return False
    return True


# ----------------------
# W&B RUN PROCESSING
# ----------------------
def _require_wandb_config():
    if not WANDB_ENTITY or not WANDB_PROJECT:
        raise HTTPException(
            status_code=500,
            detail="W&B config missing. Set WANDB_ENTITY and WANDB_PROJECT (and WANDB_API_KEY if private).",
        )

def _artifact_root_for_run(run_id: str) -> str:
    safe = "".join(c for c in run_id if c.isalnum() or c in "-_").strip() or "run"
    return str(Path(ARTIFACT_CACHE_DIR) / safe)

def _get_or_sync_model_artifact(run_id: str) -> Dict[str, Any]:
    with _artifact_lock:
        if run_id in _artifact_cache:
            CACHE_HIT.labels(run_id).inc()
            return _artifact_cache[run_id]
        CACHE_MISS.labels(run_id).inc()

    _require_wandb_config()

    start = time.time()
    try:
        run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)
        artifacts = list(run.logged_artifacts() or [])
        artifacts = sorted(
            artifacts,
            key=lambda a: getattr(a, "created_at", None) or 0,
            reverse=True,
        )
        if not artifacts:
            raise HTTPException(status_code=404, detail=f"No logged artifacts found for run_id={run_id}")

        artifact = None
        artifact_dir = None
        pt_path = None
        for candidate in artifacts:
            candidate_dir = download_artifact(candidate, root_dir=_artifact_root_for_run(run_id))
            candidate_pt = find_first_pt_file(candidate_dir)
            if candidate_pt:
                artifact = candidate
                artifact_dir = candidate_dir
                pt_path = candidate_pt
                break

        if not artifact or not artifact_dir or not pt_path:
            raise HTTPException(
                status_code=404,
                detail=f"No .pt file found in logged artifacts for run_id={run_id}",
            )

        entry = {
            "run_id": run_id,
            "artifact_name": getattr(artifact, "name", None) or getattr(artifact, "qualified_name", None),
            "artifact_dir": artifact_dir,
            "pt_path": pt_path,
            "synced_at": time.time(),
        }
        with _artifact_lock:
            _artifact_cache[run_id] = entry

        MODEL_LOAD_TIME.labels(run_id, entry["artifact_name"] or "").observe(time.time() - start)
        return entry

    except HTTPException:
        MODEL_LOAD_ERRORS.labels(run_id).inc()
        raise

    except Exception as e:
        MODEL_LOAD_ERRORS.labels(run_id).inc()
        raise HTTPException(status_code=500, detail=f"W&B model sync failed: {e}") from e


def _effective_run_id(body_run_id: Optional[str], query_model: Optional[str]) -> Optional[str]:
    return body_run_id or query_model or WANDB_RUN_ID


def _make_response(*, run_id: Optional[str], artifact: Optional[str], probs: list[float], top_k: int, class_names: list[str]) -> PredictResponse:

    if len(probs) != len(class_names):
        raise HTTPException(status_code=500, detail="Model output size does not match class list.")

    pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))
    scored = sorted([ScoredLabel(label=class_names[i], score=float(probs[i])) for i in range(len(probs))], key=lambda x: x.score, reverse=True)[:top_k]

    return PredictResponse(
        run_id=run_id,
        artifact=artifact,
        predicted_label=class_names[pred_id],
        predicted_id=pred_id,
        labels=class_names,
        probs=[float(p) for p in probs],
        top_k=scored,
    )

# ----------------------
# ROUTES
# ----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ui/page_view")
def ui_page_view(page: str = Query(..., min_length=1)):
    PAGE_VIEWS.labels(f"/ui/{page}").inc()
    return {"status": "ok", "page": page}

@app.get("/models")
def models(refresh: bool = Query(default=False, description="Bypass in-memory cache")):
    _require_wandb_config()
    try:
        cache_key = f"models:{WANDB_ENTITY}:{WANDB_PROJECT}:{WANDB_RUNS_LIMIT}"
        if not refresh:
            cached = _cache_get(cache_key)
            if cached is not None:
                return cached

        runs = list_runs(WANDB_ENTITY, WANDB_PROJECT, per_page=WANDB_RUNS_LIMIT)
        payload = {
            "default": WANDB_RUN_ID,
            "runs": [
                {
                    "id": r.id,
                    "name": r.name,
                    "created_at": getattr(r, "created_at", None),
                }
                for r in runs
            ],
        }
        _cache_set(cache_key, payload)
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/wandb/history")
def wandb_history(run_id: str, refresh: bool = Query(default=False, description="Bypass in-memory cache")):
    _require_wandb_config()
    cache_key = f"history:{WANDB_ENTITY}:{WANDB_PROJECT}:{run_id}"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)
    payload = get_history(run)
    _cache_set(cache_key, payload)
    return payload

@app.get("/wandb/metadata")
def wandb_metadata(run_id: str, refresh: bool = Query(default=False, description="Bypass in-memory cache")):
    _require_wandb_config()
    cache_key = f"metadata:{WANDB_ENTITY}:{WANDB_PROJECT}:{run_id}"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)
    meta = get_metadata(run)
    payload = {
        "creation": meta.get("created_at"),
        "runtime": meta.get("summary", {}).get("_runtime"),
        "config": meta.get("config"),
    }
    _cache_set(cache_key, payload)
    return payload

@app.get("/wandb/clf_report")
def wandb_clf_report(run_id: str, refresh: bool = Query(default=False, description="Bypass in-memory cache")):
    _require_wandb_config()
    cache_key = f"clf_report:{WANDB_ENTITY}:{WANDB_PROJECT}:{run_id}"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)

    for f in run.files():
        if "clf_report" in f.name:
            path = f.download(exist_ok=True).name

            with open(path) as fp:
                html = fp.read()

            payload = {"html": html}
            _cache_set(cache_key, payload)
            return payload

    return {"error": "not found"}

@app.get("/wandb/confusion_matrix")
def wandb_confusion_matrix(run_id: str, refresh: bool = Query(default=False, description="Bypass in-memory cache")):
    _require_wandb_config()
    cache_key = f"confusion_matrix:{WANDB_ENTITY}:{WANDB_PROJECT}:{run_id}"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)

    candidates = []
    for f in run.files():
        name = f.name.lower()
        if ("confusion" in name or "confusion_matrix" in name) and name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            candidates.append(f)

    if not candidates:
        return {"error": "not found"}

    # Prefer most recently updated file when available.
    candidates = sorted(
        candidates,
        key=lambda f: getattr(f, "updated_at", None) or getattr(f, "created_at", None) or 0,
        reverse=True,
    )
    f = candidates[0]
    path = f.download(exist_ok=True).name
    with open(path, "rb") as fp:
        data_b64 = base64.b64encode(fp.read()).decode("utf-8")
    payload = {"filename": f.name, "image_base64": data_b64}
    _cache_set(cache_key, payload)
    return payload

@app.get("/wandb/artifacts")
def wandb_artifacts(run_id: str, refresh: bool = Query(default=False, description="Bypass in-memory cache")):
    _require_wandb_config()
    cache_key = f"artifacts:{WANDB_ENTITY}:{WANDB_PROJECT}:{run_id}"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)
    payload = {"artifacts": list_logged_artifacts(run)}
    _cache_set(cache_key, payload)
    return payload

@app.post("/wandb/sync_model")
def wandb_sync_model(run_id: str):
    """Download latest model artifact (.pt) for a run and warm the cache."""
    entry = _get_or_sync_model_artifact(run_id)
    return {
        "run_id": run_id,
        "artifact": entry.get("artifact_name"),
        "pt_path": entry.get("pt_path"),
        "synced_at": entry.get("synced_at"),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    model: Optional[str] = Query(default=None, description="Alias for run_id"),
    device: Optional[str] = Query(default=None, description="Inference device: cpu (default) or cuda"),
):
    endpoint = "/predict"
    INFERENCE_INFLIGHT.labels(endpoint).inc()
    start = time.time()

    run_id = _effective_run_id(req.run_id, model)
    model_label = run_id or "zero-shot"

    try:
        raw_size = len(json.dumps(req.model_dump()).encode("utf-8"))
        REQUEST_PAYLOAD_BYTES.labels(endpoint).observe(raw_size)

        preprocess_start = time.time()
        image_input = resolve_image_input(req.image_url, req.image_base64)
        PREPROCESSING_TIME.labels(endpoint, model_label).observe(time.time() - preprocess_start)

        device_to_use = "cpu"
        if device is not None:
            device_norm = device.strip().lower()
            if device_norm not in {"cpu", "cuda"}:
                raise HTTPException(status_code=400, detail="Invalid device. Use 'cpu' or 'cuda'.")
            if device_norm == "cuda":
                if not _can_use_cuda():
                    raise HTTPException(
                        status_code=400,
                        detail="CUDA is not usable in this environment. Retry with device=cpu (default).",
                    )
                device_to_use = "cuda"

        artifact_name = None
        state_dict_path = None
        if run_id:
            entry = _get_or_sync_model_artifact(run_id)
            artifact_name = entry.get("artifact_name")
            state_dict_path = entry.get("pt_path")

        probs = clip_predict_classes(
            state_dict_path=state_dict_path,
            image_input=image_input,
            class_names=DEFAULT_CLASS_NAMES,
            model_name=CLIP_MODEL_NAME,
            device=device_to_use,
        )

        resp = _make_response(
            run_id=run_id,
            artifact=artifact_name,
            probs=probs,
            top_k=req.top_k,
            class_names=DEFAULT_CLASS_NAMES,
        )

        RESPONSE_PAYLOAD_BYTES.labels(endpoint).observe(len(resp.model_dump_json().encode("utf-8")))
        REQUEST_COUNTER.labels(endpoint, model_label, "ok").inc()
        INFERENCE_TIME.labels(endpoint, model_label).observe(time.time() - start)
        return resp
    except HTTPException:
        REQUEST_COUNTER.labels(endpoint, model_label, "error").inc()
        ERROR_COUNTER.labels(endpoint, model_label).inc()
        raise
    except Exception as e:
        REQUEST_COUNTER.labels(endpoint, model_label, "error").inc()
        ERROR_COUNTER.labels(endpoint, model_label).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        INFERENCE_INFLIGHT.labels(endpoint).dec()


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    req: BatchPredictRequest,
    model: Optional[str] = Query(default=None, description="Alias for run_id"),
    device: Optional[str] = Query(default=None, description="Inference device: cpu (default) or cuda (opt-in)"),
):
    endpoint = "/predict/batch"
    INFERENCE_INFLIGHT.labels(endpoint).inc()
    start = time.time()

    run_id = _effective_run_id(req.run_id, model)
    model_label = run_id or "zero-shot"

    try:
        raw_size = len(json.dumps(req.model_dump()).encode("utf-8"))
        REQUEST_PAYLOAD_BYTES.labels(endpoint).observe(raw_size)

        device_to_use = "cpu"
        if device is not None:
            device_norm = device.strip().lower()
            if device_norm not in {"cpu", "cuda"}:
                raise HTTPException(status_code=400, detail="Invalid device. Use 'cpu' or 'cuda'.")
            if device_norm == "cuda":
                if not _can_use_cuda():
                    raise HTTPException(
                        status_code=400,
                        detail="CUDA is not usable in this environment. Retry with device=cpu (default).",
                    )
                device_to_use = "cuda"

        artifact_name = None
        state_dict_path = None
        if run_id:
            entry = _get_or_sync_model_artifact(run_id)
            artifact_name = entry.get("artifact_name")
            state_dict_path = entry.get("pt_path")

        results = []
        for item in req.items:
            preprocess_start = time.time()
            image_input = resolve_image_input(item.image_url, item.image_base64)
            PREPROCESSING_TIME.labels(endpoint, model_label).observe(time.time() - preprocess_start)

            probs = clip_predict_classes(
                state_dict_path=state_dict_path,
                image_input=image_input,
                class_names=DEFAULT_CLASS_NAMES,
                model_name=CLIP_MODEL_NAME,
                device=device_to_use,
            )
            results.append(
                _make_response(
                    run_id=run_id,
                    artifact=artifact_name,
                    probs=probs,
                    top_k=item.top_k,
                    class_names=DEFAULT_CLASS_NAMES,
                )
            )

        resp = BatchPredictResponse(results=results)
        RESPONSE_PAYLOAD_BYTES.labels(endpoint).observe(len(resp.model_dump_json().encode("utf-8")))
        REQUEST_COUNTER.labels(endpoint, model_label, "ok").inc()
        INFERENCE_TIME.labels(endpoint, model_label).observe(time.time() - start)
        return resp
    except HTTPException:
        REQUEST_COUNTER.labels(endpoint, model_label, "error").inc()
        ERROR_COUNTER.labels(endpoint, model_label).inc()
        raise
    except Exception as e:
        REQUEST_COUNTER.labels(endpoint, model_label, "error").inc()
        ERROR_COUNTER.labels(endpoint, model_label).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        INFERENCE_INFLIGHT.labels(endpoint).dec()
