from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

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

# LLM metrics
LLM_REQUESTS = Counter("llm_requests_total", "Total LLM requests", ["endpoint", "status"])
LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "LLM latency seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600],
)
LLM_LOADED = Gauge("llm_loaded", "Whether LLM is loaded in memory (1/0)")
