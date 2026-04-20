import argparse
import os
import subprocess
import zipfile
from pathlib import Path
import wandb


def unzip(zip_path: Path, dest: Path):
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)


def find_run_files(root: Path):
    return list(root.rglob("run-*.wandb"))


def sync_runs(run_files):
    if not run_files:
        raise RuntimeError("No W&B runs found")

    run_ids = []
    for f in run_files:
        run_id = f.stem.replace("run-", "")
        run_ids.append(run_id)
        subprocess.run(["wandb", "sync", str(f), "--include-offline"], check=True)

    return list(set(run_ids)) # dedup


def log_artifacts(output_dir: Path, run_ids):
    files = [f for f in output_dir.rglob("*") if f.is_file()]

    model_files = [f for f in files if f.suffix == ".pt"]
    other_files = [
        f for f in files
        if f.suffix != ".pt" and "wandb" not in f.as_posix()
    ]

    if not model_files and not other_files:
        return

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "CLIP-Sat"),
        entity=os.getenv("WANDB_ENTITY"),
        job_type="artifact-sync",
        config={"source_run_ids": run_ids},
    )

    for f in model_files:
        art = wandb.Artifact(f.stem, type="model")
        art.add_file(str(f))
        run.log_artifact(art)

    if other_files:
        art = wandb.Artifact("kaggle-output", type="dataset")
        for f in other_files:
            art.add_file(str(f))
        run.log_artifact(art)

    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    out = Path(args.output_dir)

    # unzip artifacts
    unzip(out / "wandb_run.zip", out)
    unzip(out / "wandb_sync_bundle.zip", out)

    # sync runs
    run_files = find_run_files(out)
    run_ids = sync_runs(run_files)

    # log artifacts
    log_artifacts(out, run_ids)


if __name__ == "__main__":
    main()