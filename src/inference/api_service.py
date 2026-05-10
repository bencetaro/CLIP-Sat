from __future__ import annotations

import json
import os
import threading
import time
import logging
import pandas as pd
import base64
import torch
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from src.inference.db import check_db, init_db, log_prediction
from src.inference.helpers.clip_helpers import (
    DEFAULT_CLASS_NAMES,
    clip_predict_classes,
    resolve_image_input,
)
from src.inference.helpers.wandb_helpers import (
    download_artifact,
    find_first_pt_file,
    get_history,
    get_metadata,
    list_logged_artifacts,
    list_runs,
    load_run,
)
from src.inference.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    ScoredLabel,
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

logger = logging.getLogger("clip_sat.api")

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
INVALID_PAYLOAD = Counter(
    "invalid_payload_total",
    "Invalid request payloads rejected by validation or preprocessing",
    ["endpoint"],
)
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Prediction failures after payload preprocessing",
    ["endpoint", "model", "stage"],
)
PREDICTION_VALUE = Histogram(
    "prediction_value",
    "Prediction probability distribution values",
    ["endpoint", "model", "label"],
    buckets=[0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],
)
PREDICTED_CLASS = Counter(
    "predicted_class_total",
    "Top predicted class counts",
    ["endpoint", "model", "label"],
)
CACHE_HIT = Counter("model_cache_hit_total", "Model cache hits", ["model"])
CACHE_MISS = Counter("model_cache_miss_total", "Model cache misses", ["model"])
WANDB_CACHE_LOOKUPS = Counter(
    "wandb_cache_lookups_total",
    "W&B response cache lookups",
    ["cache", "result"],
)
WANDB_CACHE_ENTRY_AGE = Histogram(
    "wandb_cache_entry_age_seconds",
    "Age of W&B cache entries when returned",
    ["cache"],
    buckets=[1, 5, 15, 30, 60, 120, 300, 600],
)
WANDB_CACHE_ENTRIES = Gauge(
    "wandb_cache_entries",
    "Current number of W&B response cache entries",
)
WANDB_FETCH_TIME = Histogram(
    "wandb_fetch_latency_seconds",
    "Latency of W&B API fetches after cache misses or refreshes",
    ["cache"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)
MODEL_LOAD_ERRORS = Counter("model_load_errors_total", "Model load errors", ["model"])
PAGE_VIEWS = Counter("page_views_total", "Total page hits", ["endpoint"])
DB_ERRORS = Counter("db_errors_total", "DB write/init errors", ["stage"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    INVALID_PAYLOAD.labels(request.url.path).inc()
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ----------------------
# MODEL ARTIFACT CACHE (per-run download paths)
# ----------------------
_artifact_lock = threading.Lock()
_artifact_cache: Dict[str, Dict[str, Any]] = {}

_wandb_cache_lock = threading.Lock()
_wandb_cache: Dict[str, Dict[str, Any]] = {}


def _cache_name(key: str) -> str:
    return key.split(":", 1)[0]


def _cache_get(key: str) -> Optional[Any]:
    cache = _cache_name(key)
    if WANDB_CACHE_TTL_SECONDS <= 0:
        WANDB_CACHE_LOOKUPS.labels(cache, "disabled").inc()
        return None
    now = time.time()
    with _wandb_cache_lock:
        item = _wandb_cache.get(key)
        if not item:
            WANDB_CACHE_LOOKUPS.labels(cache, "miss").inc()
            return None
        if now - item["ts"] > WANDB_CACHE_TTL_SECONDS:
            _wandb_cache.pop(key, None)
            WANDB_CACHE_ENTRIES.set(len(_wandb_cache))
            WANDB_CACHE_LOOKUPS.labels(cache, "expired").inc()
            return None
        WANDB_CACHE_LOOKUPS.labels(cache, "hit").inc()
        WANDB_CACHE_ENTRY_AGE.labels(cache).observe(now - item["ts"])
        return item["value"]

def _cache_set(key: str, value: Any) -> None:
    if WANDB_CACHE_TTL_SECONDS <= 0:
        return
    with _wandb_cache_lock:
        _wandb_cache[key] = {"ts": time.time(), "value": value}
        WANDB_CACHE_ENTRIES.set(len(_wandb_cache))

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

def _observe_prediction_metrics(endpoint: str, model_label: str, probs: list[float], class_names: list[str]) -> None:
    if len(probs) != len(class_names):
        return
    for label, prob in zip(class_names, probs):
        PREDICTION_VALUE.labels(endpoint, model_label, label).observe(float(prob))
    pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))
    PREDICTED_CLASS.labels(endpoint, model_label, class_names[pred_id]).inc()

# ----------------------
# ROUTES
# ----------------------
@app.get("/health")
def health():
    db_enabled = os.getenv("DB_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
    payload: Dict[str, Any] = {"status": "ok", "db": "disabled" if not db_enabled else "unknown"}
    if db_enabled:
        try:
            check_db()
            payload["db"] = "ok"
        except Exception as e:
            payload["db"] = "error"
            payload["db_error"] = str(e)
    return payload

@app.on_event("startup")
def startup_event():
    if os.getenv("DB_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        logger.warning("DB is disabled via DB_ENABLED=0; skipping init_db()")
        return
    try:
        init_db()
    except Exception as e:
        DB_ERRORS.labels("init").inc()
        logger.exception("DB init failed; continuing without DB. Error: %s", e)

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

        fetch_start = time.time()
        runs = list_runs(WANDB_ENTITY, WANDB_PROJECT, per_page=WANDB_RUNS_LIMIT)
        WANDB_FETCH_TIME.labels(_cache_name(cache_key)).observe(time.time() - fetch_start)
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
    fetch_start = time.time()
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)
    payload = get_history(run)
    WANDB_FETCH_TIME.labels(_cache_name(cache_key)).observe(time.time() - fetch_start)
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
    fetch_start = time.time()
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)
    meta = get_metadata(run)
    WANDB_FETCH_TIME.labels(_cache_name(cache_key)).observe(time.time() - fetch_start)
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
    fetch_start = time.time()
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)

    for f in run.files():
        if "clf_report" in f.name:
            path = f.download(exist_ok=True).name

            with open(path) as fp:
                html = fp.read()

            payload = {"html": html}
            WANDB_FETCH_TIME.labels(_cache_name(cache_key)).observe(time.time() - fetch_start)
            _cache_set(cache_key, payload)
            return payload

    WANDB_FETCH_TIME.labels(_cache_name(cache_key)).observe(time.time() - fetch_start)
    return {"error": "not found"}

@app.get("/wandb/confusion_matrix")
def wandb_confusion_matrix(run_id: str, refresh: bool = Query(default=False, description="Bypass in-memory cache")):
    _require_wandb_config()
    cache_key = f"confusion_matrix:{WANDB_ENTITY}:{WANDB_PROJECT}:{run_id}"
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
    fetch_start = time.time()
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)

    candidates = []
    for f in run.files():
        name = f.name.lower()
        if ("confusion" in name or "confusion_matrix" in name) and name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            candidates.append(f)

    if not candidates:
        WANDB_FETCH_TIME.labels(_cache_name(cache_key)).observe(time.time() - fetch_start)
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
    WANDB_FETCH_TIME.labels(_cache_name(cache_key)).observe(time.time() - fetch_start)
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
    fetch_start = time.time()
    run = load_run(WANDB_ENTITY, WANDB_PROJECT, run_id)
    payload = {"artifacts": list_logged_artifacts(run)}
    WANDB_FETCH_TIME.labels(_cache_name(cache_key)).observe(time.time() - fetch_start)
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
                INVALID_PAYLOAD.labels(endpoint).inc()
                raise HTTPException(status_code=400, detail="Invalid device. Use 'cpu' or 'cuda'.")
            if device_norm == "cuda":
                if not _can_use_cuda():
                    PREDICTION_ERRORS.labels(endpoint, model_label, "device").inc()
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

        try:
            probs = clip_predict_classes(
                state_dict_path=state_dict_path,
                image_input=image_input,
                class_names=DEFAULT_CLASS_NAMES,
                model_name=CLIP_MODEL_NAME,
                device=device_to_use,
            )
        except Exception:
            PREDICTION_ERRORS.labels(endpoint, model_label, "predict").inc()
            raise

        resp = _make_response(
            run_id=run_id,
            artifact=artifact_name,
            probs=probs,
            top_k=req.top_k,
            class_names=DEFAULT_CLASS_NAMES,
        )
        _observe_prediction_metrics(endpoint, model_label, probs, DEFAULT_CLASS_NAMES)

        if os.getenv("DB_ENABLED", "1").strip().lower() not in {"0", "false", "no"}:
            try:
                db_preds = {DEFAULT_CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
                log_prediction(
                    image_url=req.image_url,
                    model_name=CLIP_MODEL_NAME,
                    device_type=device_to_use,
                    predictions=db_preds,
                    chatbot_ans=None,
                    user_rating=None,
                    user_feedback=None,
                )
            except Exception as e:
                DB_ERRORS.labels("write").inc()
                logger.exception("DB write failed; returning prediction anyway. Error: %s", e)

        RESPONSE_PAYLOAD_BYTES.labels(endpoint).observe(len(resp.model_dump_json().encode("utf-8")))
        REQUEST_COUNTER.labels(endpoint, model_label, "ok").inc()
        INFERENCE_TIME.labels(endpoint, model_label).observe(time.time() - start)
        return resp
    except ValueError as e:
        INVALID_PAYLOAD.labels(endpoint).inc()
        REQUEST_COUNTER.labels(endpoint, model_label, "error").inc()
        ERROR_COUNTER.labels(endpoint, model_label).inc()
        raise HTTPException(status_code=400, detail=str(e)) from e
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
