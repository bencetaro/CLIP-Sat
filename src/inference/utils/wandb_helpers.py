from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import os
import threading
import wandb
import pandas as pd

_api_lock = threading.Lock()
_api_client: Optional[wandb.Api] = None


def _api() -> wandb.Api:
    """
    Return a singleton W&B API client.
    W&B public API calls can time out; configure via WANDB_API_TIMEOUT seconds.
    """
    global _api_client
    if _api_client is not None:
        return _api_client
    with _api_lock:
        if _api_client is not None:
            return _api_client
        timeout_s = int(os.getenv("WANDB_API_TIMEOUT", "60"))
        _api_client = wandb.Api(timeout=timeout_s)
        return _api_client


def load_run(entity: str, project: str, run_id: str):
    return _api().run(f"{entity}/{project}/{run_id}")

def list_runs(entity: str, project: str, per_page: int = 50):
    return _api().runs(f"{entity}/{project}", per_page=per_page)

def get_history(run, keys: Optional[List[str]] = None, samples: int = 1000):
    if keys is None:
        keys = ["epoch", "train_loss", "val_loss"]
    df = run.history(keys=keys, samples=samples)
    return df.to_dict(orient="records")

def get_metadata(run) -> Dict[str, Any]:
    return {
        "id": run.id,
        "name": run.name,
        "created_at": getattr(run, "created_at", None),
        "url": getattr(run, "url", None),
        "config": dict(run.config),
        "summary": dict(run.summary),
        "tags": list(getattr(run, "tags", []) or []),
    }

def _iter_logged_artifacts(run) -> Iterable:
    try:
        yield from run.logged_artifacts()
    except Exception:
        return None

def list_logged_artifacts(run) -> List:
    infos: List = []
    for a in _iter_logged_artifacts(run):
        infos.append({
            "name": a.name,
            "version": a.version,
            "aliases": a.aliases,
            "type": a.type,
            "created_at": a.created_at,
        })
    return infos

def download_artifact(artifact, root_dir: str) -> str:
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    return artifact.download(root=root_dir)

def find_first_pt_file(artifact_dir: str) -> Optional[str]:
    base = Path(artifact_dir)
    for p in base.rglob("*.pt"):
        return str(p)
    return None

def parse_clf_report(text: str) -> pd.DataFrame:
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    rows = []
    for line in lines:
        parts = line.split()

        # skip header
        if parts[0] == "precision":
            continue

        # accuracy row
        if parts[0] == "accuracy":
            rows.append({
                "label": "accuracy",
                "precision": None,
                "recall": None,
                "f1-score": float(parts[-2]),
                "support": int(parts[-1])
            })
            continue

        # normal rows (including "macro avg", "weighted avg")
        if len(parts) >= 5:
            label = " ".join(parts[:-4])
            precision, recall, f1, support = parts[-4:]

            rows.append({
                "label": label,
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(f1),
                "support": int(support)
            })

    return pd.DataFrame(rows)


def parse_clf_report_split(text: str):
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    class_rows = []
    summary_rows = []

    for line in lines:
        parts = line.split()

        # skip header
        if parts[0] == "precision":
            continue

        label = parts[0]

        # accuracy row
        if label == "accuracy":
            summary_rows.append({
                "label": "accuracy",
                "precision": None,
                "recall": None,
                "f1-score": float(parts[-2]),
                "support": int(parts[-1])
            })
            continue

        # macro avg / weighted avg
        if label in ["macro", "weighted"]:
            label_full = " ".join(parts[:-4])
            precision, recall, f1, support = parts[-4:]

            summary_rows.append({
                "label": label_full,
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(f1),
                "support": int(support)
            })
            continue

        # normal class rows
        if len(parts) >= 5:
            label_full = " ".join(parts[:-4])
            precision, recall, f1, support = parts[-4:]

            class_rows.append({
                "label": label_full,
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(f1),
                "support": int(support)
            })

    df_classes = pd.DataFrame(class_rows)
    df_summary = pd.DataFrame(summary_rows)

    return df_classes, df_summary
