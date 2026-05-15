from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from src.inference.model.llm import SimpleLLM as LlamaCppLLM
from src.inference.model.llm_heavy import SimpleLLM as HeavyLLM
from src.inference.utils.prom_metrics import LLM_LOADED

_llm_lock = threading.Lock()
_llm_clients: Dict[str, Any] = {}

LLM_BACKENDS = {"llama_cpp", "hf_heavy"}


def llm_enabled() -> bool:
    return os.getenv("LLM_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


def normalize_backend(backend: Optional[str]) -> str:
    value = (backend or "llama_cpp").strip().lower()
    if value not in LLM_BACKENDS:
        return "llama_cpp"
    return value


def _client_key(backend: str, use_gpu: bool) -> str:
    return f"{backend}|gpu={int(use_gpu)}"


def get_llm_client(backend: str = "llama_cpp", use_gpu: bool = False):
    normalized_backend = normalize_backend(backend)
    key = _client_key(normalized_backend, use_gpu)
    if key in _llm_clients:
        return _llm_clients[key]

    with _llm_lock:
        if key in _llm_clients:
            return _llm_clients[key]

        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "500"))

        if normalized_backend == "hf_heavy":
            model_id = os.getenv("LLM_HEAVY_MODEL_ID", os.getenv("LLM_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
            device = "cuda" if use_gpu else os.getenv("LLM_DEVICE", "cpu")
            client = HeavyLLM(
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                device=device,
            )
        else:
            model_path = os.getenv("LLM_CPP_MODEL_PATH", os.getenv("LLM_MODEL_ID", "/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"))
            n_ctx = int(os.getenv("LLM_CTX", "2048"))
            n_threads = int(os.getenv("LLM_THREADS", "8"))
            n_gpu_layers = int(os.getenv("LLM_GPU_LAYERS", "35")) if use_gpu else 0
            client = LlamaCppLLM(
                model_path=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
            )

        _llm_clients[key] = client
        LLM_LOADED.set(len(_llm_clients))
        return client


def compute_distribution_warnings(probs: List[float]) -> List[str]:
    probs_sorted = sorted([float(p) for p in probs], reverse=True)
    warnings: List[str] = []
    if probs_sorted and probs_sorted[0] > 0.90:
        warnings.append("Very high top probability (>0.90).")
    if probs_sorted and probs_sorted[0] < 0.10:
        warnings.append("Very low top probability (<0.10).")
    if sum(1 for p in probs_sorted if p > 0.50) >= max(3, len(probs_sorted) // 2):
        warnings.append("Many classes have high probability (>0.50); distribution looks unusual.")
    if probs_sorted and all(p < 0.10 for p in probs_sorted):
        warnings.append("All probabilities are low (<0.10); model may be uncertain or input out-of-domain.")
    return warnings


def build_llm_review_prompt(top_pairs: List[Tuple[str, float]], top_k:int=5) -> str:
    return f"""
You are analyzing CLIP model prediction probabilities for a satellite image (you do NOT see the image).

Top {top_k} predictions in format: (label, prob):
{top_pairs}

Tasks:
1) Make a best-guess of what might be in the image based only on the score distribution.
2) Describe typical land-use / landscape patterns for the likely classes (human activity, habitat, climate cues).
3) If the distribution looks unusual, warn about it (e.g., too many high scores or all very low).

Output rules (strict):
- Output MUST be a single valid JSON object.
- Do NOT include any explanation, markdown, code fences, or extra text.
- Do NOT output placeholder strings like "...".

Required JSON schema:
{{
  "guesses": [string, ...],
  "confidence_analysis": [string, ...],
  "summary": string
}}
"""


def extract_json(data: Any) -> Optional[dict]:
    if data is None:
        return None

    if isinstance(data, dict):
        return data

    text = str(data).strip()
    if not text:
        return None

    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    if not matches:
        return None

    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None
