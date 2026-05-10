# CLIP-Sat 🛰️
 
CLIP-Sat is a portfolio project, that I plan to develop further in the coming weeks. The data comes from a public Kaggle dataset:

https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset


## Demonstrate CLIP finetuning

CLIP is a very well-trained and informative model that effectively classifies common objects. However, it may require more fine-grained domain knowledge in areas such as satellite imagery. So I started looking to see if there was anything new to add and eventually did some finetuning on the above mentioned dataset. This experimental work is published in: 

https://www.kaggle.com/code/bencetar/clip-hard-example-mining-finetuning


**Note:** Kaggle CI runs do not guarantee a specific GPU type and may fall back to P100 or CPU. The default pytorch version is incompatible with P100, and installing a compatible version would require a runtime restart, which is not supported in CI. That is why I ended up using CPU during training.

**NOTE:** This is a work in progress, so bugs and errors are very much expected.

## Running (Docker Compose)

1) Configure environment variables in `.env` (see `.env.example`).
2) Start the stack:

`docker compose up --build`

### Networking/IPs (quick rules)

- `0.0.0.0` is used *inside containers* so the process listens on all container interfaces (required for port publishing).
- `127.0.0.1:<port>:<port>` in `docker-compose.yml` publishes a container port **only to the local machine** (not your LAN).
- Between containers, use **service names** (Docker DNS), e.g. `http://fastapi:8000` and `DB_HOST=postgres`.
- Use a LAN IP like `192.168.x.x` only when you intentionally want other devices on your network to reach a published port (and you bind ports to that interface / all interfaces).

### Postgres

This project runs Postgres as a separate `postgres` service in `docker-compose.yml`.
FastAPI initializes the table at startup and writes predictions, but DB failures will not block inference responses (see `/health` for DB status).
