# CLIP-Sat 🛰️

CLIP-Sat is a project where I explore satellite image classification with a finetuned CLIP model. It currently includes:
- FastAPI inference API
- Streamlit UI
- Postgres persistence
- W&B run/artifact integration
- Optional LLM explanation layer (`llama.cpp` or Hugging Face backend)

Dataset:
- https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset

Original training experiment:
- https://www.kaggle.com/code/bencetar/clip-hard-example-mining-finetuning

Training notebook used for scheduled runs:
- `src/training/notebook/satellite-imagery-training.ipynb`

## Process Model

![Workflow](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/clipsat-process.png)

## Features

- CLIP-based satellite image inference with top-k class scores
- W&B-backed model/run discovery and artifact synchronization
- Streamlit UI for image upload/URL inference and prediction visualization
- Optional LLM explanations from prediction score distributions
- Two LLM backends: `llama.cpp` (GGUF) and Hugging Face Transformers
- Postgres persistence for predictions, LLM outputs, and app feedback
- Prometheus/Grafana monitoring for service and inference metrics

## Training & Inference

I keep training and inference separate on purpose.

### Training

- Training experiments are notebook-based and tracked with W&B.
- The main reference experiment is published on Kaggle:
  - https://www.kaggle.com/code/bencetar/clip-hard-example-mining-finetuning
- The repository also includes a notebook for scheduled/automated runs:
  - `src/training/notebook/satellite-imagery-training.ipynb`
- Every `feature/**` branch push will trigger a GitHub Actions workflow to:
  - 1. Update a new dataset version.
  - 2. Run a new Kaggle notebook and W&B experiment.

### Inference

- Inference is served through FastAPI, with Streamlit as the frontend.
- You can provide an image either by uploading it or by URL.
- The API loads the selected W&B artifact, runs CLIP classification, and returns:
  - predicted class
  - top-k results
  - full score distribution
- If needed, the app can also generate an LLM-based explanation from the score distribution and store it in Postgres.

#### LLM flow

Streamlit "Generate explanation" does:
1. Warmup selected backend (`/llm/warmup`)
2. Generate answer (`/llm/review`)
3. Persist LLM output to matching prediction row (if DB enabled and `prediction_id` exists)

Backends:
- `llama_cpp`: lighter memory profile, requires local GGUF
- `hf_heavy`: heavier memory profile, model loaded via Transformers

> **Note:** Inference currently only supports CPU.

> *Side note: Ideally, if the CLIP model were very accurate in the multiclass sense, then the LLM model would better define the content of the image, for example in relation to climate zone or the presence of human activity. This is of course also true for the LLM model, as only simpler lightweight models are currently implemented.*

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

## Database tables

`clip_predict`:
- stores prediction metadata, score distributions, and optional LLM answer fields

| Column          | Description                            |
| --------------- | -------------------------------------- |
| `id`            | Primary key                            |
| `timestamp`     | Prediction timestamp                   |
| `image_url`     | Input image location                   |
| `run_id`        | W&B training run                       |
| `device_type`   | Inference device (cpu or cuda)         |
| `predictions`   | Serialized prediction probabilities    |
| `top_k`         | Top-k predicted classes                |
| `llm_status`    | LLM generation status                  |
| `llm_backend`   | LLM backend used                       |
| `llm_answer`    | Generated explanation                  |
| `llm_timestamp` | LLM response timestamp                 |

`app_feedback`:
- stores user app rating and comment

| Column        | Description               |
| ------------- | ------------------------- |
| `id`          | Primary key               |
| `timestamp`   | Submission timestamp      |
| `app_rating`  | Numeric app rating        |
| `app_comment` | Optional written feedback |

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

## Common commands

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
- `llama_context: n_ctx_seq (2048) < n_ctx_train (32768)` is a warning, not an error (refers to per-sequence context size).
- First model warmup can be slow; subsequent calls use in-process cache.
- If W&B calls timeout, increase `WANDB_API_TIMEOUT`.

---

## Screenshots of the project

### Home page
![Home](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/home1.png)
![Home](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/home2.png)

### Inference page
![Inference](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/inf1.png)
![Inference](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/inf2.png)
![Inference](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/inf3.png)
![Inference](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/inf4.png)

### Training summary page
![Training](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/train1.png)
![Training](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/train2.png)
![Training](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/train3.png)

### Grafana dashboard
![Grafana](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/graf1.png)
![Grafana](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/graf2.png)
![Grafana](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/graf3.png)

### Postgres db
![Postgres](https://github.com/bencetaro/CLIP-Sat/blob/feature/inference/screenshots/pg1.png)
