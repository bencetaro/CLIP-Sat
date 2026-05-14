# CLIP-Sat 🛰️

CLIP-Sat is a satellite image classification demo built around a finetuned CLIP model, with:
- FastAPI inference API
- Streamlit UI
- Postgres persistence
- W&B run/artifact integration
- Optional LLM explanation layer (`llama.cpp` or Hugging Face backend)

Dataset source:
- https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset

Original training experiment:
- https://www.kaggle.com/code/bencetar/clip-hard-example-mining-finetuning

Training notebook for scheduled runs:
- `src/training/notebook/satellite-imagery-training.ipynb`

## Training & Inference

They are managed separately:
- Training was mainly based on the training notebook via GitHub Actions CICD
- Inference was tested/can be initialized through docker-compose

## Architecture

Services (`docker-compose.yml`):
- `fastapi`: main API (`src/inference/api_service.py`)
- `streamlit`: UI (`src/inference/client.py`)
- `postgres`: persistence
- `prometheus` + `grafana` + `node-exporter`: monitoring

Core Python modules:
- `src/inference/api_service.py`: routes, orchestration, metrics hooks
- `src/inference/utils/clip_helpers.py`: CLIP preprocessing/inference helpers
- `src/inference/utils/wandb_helpers.py`: W&B API wrappers + caching
- `src/inference/utils/db.py`: DB schema/init, inserts, updates
- `src/inference/utils/llm_helpers.py`: LLM backend selection + client cache
- `src/inference/model/llm.py`: `llama.cpp` GGUF backend
- `src/inference/model/llm_heavy.py`: Hugging Face backend
- `src/inference/ui/*.py`: Streamlit pages/components

## Storage layout

Mounted host folders:
- `./models -> /app/models`: local GGUF model files (for `llama.cpp`)
- `./media -> /app/media`: persisted app assets and W&B artifact cache

Named Docker volume:
- `postgres_data`: Postgres data directory

W&B model artifacts are downloaded to:
- `WANDB_ARTIFACT_CACHE_DIR` (default `/app/media/wandb_artifacts`)

## Environment variables

Use `.env.example` as reference.

Important keys:
- `INFERENCE_API_BASE_URL`: Streamlit -> FastAPI URL (`http://fastapi:8000` in Docker)
- `WANDB_*`: W&B project/run credentials and settings
- `POSTGRES_*`: DB connection values
- `DB_ENABLED`: enable/disable DB writes

LLM keys:
- `LLM_ENABLED`: enables LLM endpoints
- `LLM_CPP_MODEL_PATH`: GGUF file path inside container (for `llama.cpp`)
- `LLM_CPP_MODEL_URL`: startup download URL for GGUF
- `LLM_CTX`, `LLM_THREADS`, `LLM_GPU_LAYERS`: `llama.cpp` runtime knobs
- `LLM_HEAVY_MODEL_ID`: HF model id (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
- `LLM_DEVICE`: default HF device (`cpu` / `cuda`)
- `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`
- `HF_TOKEN`: optional token for private/gated downloads

## Run

1. Create `.env` from `.env.example` and fill real values.
2. Start stack:

```bash
docker compose up -d --build
```

3. Open apps:
- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## LLM flow

Streamlit "Generate explanation" does:
1. Warmup selected backend (`/llm/warmup`)
2. Generate answer (`/llm/review`)
3. Persist LLM output to matching prediction row (if DB enabled and `prediction_id` exists)

Backends:
- `llama_cpp`: lighter memory profile, requires local GGUF
- `hf_heavy`: heavier memory profile, model loaded via Transformers

## Database tables

`clip_predict`:
- stores prediction metadata, score distribution, and optional LLM answer fields

`app_feedback`:
- stores user app rating and comment

## Common commands

Start/rebuild:
```bash
docker compose up -d --build
```

Follow logs:
```bash
docker compose logs -f fastapi
docker compose logs -f streamlit
```

Inspect DB quickly:
```bash
docker exec -it postgres sh
psql -U <POSTGRES_USER> -d <POSTGRES_DB>
```

Run DB inspect helper (inside `fastapi` container):
```bash
python scripts/db_inspect.py
```

## Notes and troubleshooting

- `psql` is not installed in the `fastapi` image by default; use `postgres` container for interactive SQL.
- `llama_context: n_ctx_seq (2048) < n_ctx_train (32768)` is a warning, not an error.
- First model warmup can be slow; subsequent calls use in-process cache.
- If W&B calls timeout, increase `WANDB_API_TIMEOUT`.
