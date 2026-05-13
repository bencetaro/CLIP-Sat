from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
import torch
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from dotenv import load_dotenv
load_dotenv()

from src.inference.utils.db import check_db, init_db, log_app_feedback, log_prediction
from src.inference.utils.clip_helpers import (
    DEFAULT_CLASS_NAMES,
    clip_predict_classes,
    resolve_image_input,
)
from src.inference.utils.llm_helpers import (
    build_llm_review_prompt,
    compute_distribution_warnings,
    extract_json,
    get_llm_client,
    llm_enabled,
)
from src.inference.utils.wandb_helpers import (
    download_artifact,
    find_first_pt_file,
    get_history,
    get_metadata,
    list_logged_artifacts,
    list_runs,
    load_run,
)
from src.inference.utils.prom_metrics import (
    CACHE_HIT,
    CACHE_MISS,
    DB_ERRORS,
    ERROR_COUNTER,
    INFERENCE_INFLIGHT,
    INFERENCE_TIME,
    INVALID_PAYLOAD,
    LLM_LATENCY,
    LLM_REQUESTS,
    MODEL_LOAD_ERRORS,
    MODEL_LOAD_TIME,
    PAGE_VIEWS,
    PREDICTED_CLASS,
    PREDICTION_ERRORS,
    PREDICTION_VALUE,
    PREPROCESSING_TIME,
    REQUEST_COUNTER,
    REQUEST_PAYLOAD_BYTES,
    RESPONSE_PAYLOAD_BYTES,
    WANDB_CACHE_ENTRIES,
    WANDB_CACHE_ENTRY_AGE,
    WANDB_CACHE_LOOKUPS,
    WANDB_FETCH_TIME,
)
from src.inference.utils.schemas import (
    FeedbackRequest,
    LLMReviewRequest,
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
WANDB_CACHE_TTL_SECONDS = float(os.getenv("WANDB_CACHE_TTL_SECONDS", "60"))
ARTIFACT_CACHE_DIR = os.getenv("WANDB_ARTIFACT_CACHE_DIR", "/tmp/clip_sat_wandb_artifacts")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# ----------------------
# FASTAPI + METRICS
# ----------------------
app = FastAPI(title="CLIP-Sat Inference API")
app.mount("/metrics", make_asgi_app())

logger = logging.getLogger("clip_sat.api")


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
            candidate_dir = download_artifact(
                candidate, root_dir=_artifact_root_for_run(run_id)
            )
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
            "artifact_name": getattr(artifact, "name", None)
            or getattr(artifact, "qualified_name", None),
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


def _make_response(
    *,
    run_id: Optional[str],
    artifact: Optional[str],
    probs: list[float],
    top_k: int,
    class_names: list[str],
) -> PredictResponse:

    if len(probs) != len(class_names):
        raise HTTPException(status_code=500, detail="Model output size does not match class list.")

    pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))
    scored = sorted(
        [
            ScoredLabel(label=class_names[i], score=float(probs[i]))
            for i in range(len(probs))
        ],
        key=lambda x: x.score,
        reverse=True,
    )[:top_k]

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
    db_enabled = os.getenv("DB_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    payload: Dict[str, Any] = {
        "status": "ok",
        "db": "disabled" if not db_enabled else "unknown",
    }
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


@app.post("/ui/feedback")
def ui_feedback(req: FeedbackRequest):
    if os.getenv("DB_ENABLED", "1").strip().lower() in {"0", "false", "no"}:
        raise HTTPException(status_code=503, detail="DB is disabled (DB_ENABLED=0).")
    try:
        log_app_feedback(app_rating=req.user_rating, app_comment=req.user_feedback)
    except Exception as e:
        DB_ERRORS.labels("write_feedback").inc()
        logger.exception("Feedback write failed. Error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save feedback.") from e
    return {"status": "ok"}

@app.post("/llm/review")
def llm_review(req: LLMReviewRequest):
    if not llm_enabled():
        raise HTTPException(status_code=503, detail="LLM is disabled (set LLM_ENABLED=1).")
    if len(req.labels) != len(req.probs):
        raise HTTPException(status_code=400, detail="labels and probs length mismatch.")

    pairs = [(req.labels[i], float(req.probs[i])) for i in range(len(req.labels))]
    top = sorted(pairs, key=lambda x: x[1], reverse=True)[:req.top_k]
    warnings = compute_distribution_warnings(req.probs)
    prompt = build_llm_review_prompt(top, req.top_k)

    endpoint = "/llm/review"
    start = time.time()
    try:
        backend = req.backend
        use_gpu = req.use_gpu
        client = get_llm_client(backend=backend, use_gpu=use_gpu)
        raw = client.llm_response(prompt)
    except Exception as e:
        LLM_REQUESTS.labels(endpoint, "error").inc()
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}") from e

    parsed = extract_json(raw)
    LLM_REQUESTS.labels(endpoint, "ok").inc()
    LLM_LATENCY.labels(endpoint).observe(time.time() - start)
    return {
        "status": "ok",
        "took_s": time.time() - start,
        "backend": backend,
        "use_gpu": use_gpu,
        "top_k": top,
        "warnings": warnings,
        "parsed": parsed,
        "raw": raw if parsed is None else None,
    }


@app.post("/llm/warmup")
def llm_warmup(
    backend: str = Query(default="llama_cpp"),
    use_gpu: bool = Query(default=False),
):
    """Warm up the LLM: download + load model weights into memory."""
    if not llm_enabled():
        raise HTTPException(status_code=503, detail="LLM is disabled (set LLM_ENABLED=1).")
    endpoint = "/llm/warmup"
    start = time.time()
    try:
        client = get_llm_client(backend=backend, use_gpu=use_gpu)
    except Exception as e:
        LLM_REQUESTS.labels(endpoint, "error").inc()
        raise HTTPException(status_code=500, detail=f"LLM warmup failed: {e}") from e
    LLM_REQUESTS.labels(endpoint, "ok").inc()
    LLM_LATENCY.labels(endpoint).observe(time.time() - start)
    return {
        "status": "ok",
        "backend": backend,
        "use_gpu": use_gpu,
        "model_id": getattr(client, "model_id", None),
        "model_path": getattr(client, "model_path", None),
        "device": getattr(client, "device", None),
        "took_s": time.time() - start,
    }


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
def wandb_history(
    run_id: str,
    refresh: bool = Query(default=False, description="Bypass in-memory cache"),
):
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
def wandb_metadata(
    run_id: str,
    refresh: bool = Query(default=False, description="Bypass in-memory cache"),
):
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
def wandb_clf_report(
    run_id: str,
    refresh: bool = Query(default=False, description="Bypass in-memory cache"),
):
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
def wandb_confusion_matrix(
    run_id: str,
    refresh: bool = Query(default=False, description="Bypass in-memory cache"),
):
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
        if ("confusion" in name or "confusion_matrix" in name) and name.endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        ):
            candidates.append(f)

    if not candidates:
        WANDB_FETCH_TIME.labels(_cache_name(cache_key)).observe(time.time() - fetch_start)
        return {"error": "not found"}

    # Prefer most recently updated file when available.
    candidates = sorted(
        candidates,
        key=lambda f: (getattr(f, "updated_at", None) or getattr(f, "created_at", None) or 0),
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
def wandb_artifacts(
    run_id: str,
    refresh: bool = Query(default=False, description="Bypass in-memory cache"),
):
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
                raise HTTPException(
                    status_code=400, detail="Invalid device. Use 'cpu' or 'cuda'."
                )
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
