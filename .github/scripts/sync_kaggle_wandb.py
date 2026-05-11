import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path
import wandb

def unzip_if_exists(zip_path: Path, destination: Path) -> bool:
    if not zip_path.exists():
        return False
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(destination)
    return True


def collect_run_files(output_root: Path) -> list[Path]:
    return sorted(output_root.rglob("run-*.wandb"))


def sync_offline_runs(run_files: list[Path], entity: str | None) -> list[str]:
    if not run_files:
        raise RuntimeError("No offline W&B run files found.")

    source_run_ids: list[str] = []
    sync_args = ["--include-offline", "--quiet"]
    if entity:
        sync_args.extend(["--entity", entity])

    print("Discovered W&B offline runs:")
    for run_file in run_files:
        run_id = run_file.stem.removeprefix("run-")
        source_run_ids.append(run_id)
        print(run_file)
        subprocess.run(["wandb", "sync", str(run_file), *sync_args], check=True)

    return source_run_ids


def load_manifest(bundle_root: Path) -> dict:
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def collect_downloadable_files(output_root: Path, bundle_root: Path | None) -> tuple[list[Path], list[Path]]:
    if bundle_root and (bundle_root / "files").exists():
        candidates = [path for path in sorted((bundle_root / "files").rglob("*")) if path.is_file()]
    else:
        candidates = [
            path for path in sorted(output_root.rglob("*"))
            if path.is_file()
            and "output/wandb/" not in path.as_posix()
            and path.name not in {"wandb_run.zip", "wandb_sync_bundle.zip"}
        ]

    model_files = [path for path in candidates if path.suffix == ".pt"]
    other_files = [path for path in candidates if path.suffix != ".pt"]
    return model_files, other_files


def log_downloaded_artifacts(
    output_root: Path,
    bundle_root: Path | None,
    source_run_ids: list[str],
    entity: str | None,
    project: str,
) -> None:
    model_files, other_files = collect_downloadable_files(output_root, bundle_root)
    if not model_files and not other_files:
        print("No downloadable Kaggle output files found outside the W&B bundle.")
        return

    init_kwargs = {
        "project": project,
        "job_type": "artifact-sync",
        "config": {"source_run_ids": source_run_ids},
    }
    if entity:
        init_kwargs["entity"] = entity

    # Always try to attach artifacts to an existing synced run to avoid creating a separate
    # "artifact-only" run in the W&B UI.
    if source_run_ids:
        init_kwargs["id"] = source_run_ids[0]
        init_kwargs["resume"] = "allow"
    else:
        init_kwargs["name"] = "kaggle-output-sync-manual"

    run = wandb.init(**init_kwargs)

    for model_file in model_files:
        artifact = wandb.Artifact(
            name=model_file.stem,
            type="model",
            metadata={"source": "kaggle-output", "source_run_ids": source_run_ids},
        )
        artifact_name = model_file.name
        if bundle_root and bundle_root in model_file.parents:
            artifact_name = model_file.relative_to(bundle_root / "files").as_posix()
        artifact.add_file(str(model_file), name=artifact_name)
        run.log_artifact(artifact, aliases=["latest"])
        print(f"Logged model artifact from {model_file}")

    if other_files:
        suffix = source_run_ids[0] if len(source_run_ids) == 1 else "batch"
        artifact = wandb.Artifact(
            name=f"kaggle-output-{suffix}",
            type="dataset",
            metadata={"source": "kaggle-output", "source_run_ids": source_run_ids},
        )
        for path in other_files:
            if bundle_root and bundle_root in path.parents:
                artifact_name = path.relative_to(bundle_root / "files").as_posix()
            else:
                artifact_name = path.relative_to(output_root).as_posix()
            artifact.add_file(str(path), name=artifact_name)
        run.log_artifact(artifact, aliases=["latest"])
        print(f"Logged bundled output artifact with {len(other_files)} files")

    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--project", default=os.getenv("WANDB_PROJECT", "CLIP-Sat"))
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    entity = os.getenv("WANDB_ENTITY")

    unzip_if_exists(output_root / "wandb_run.zip", output_root)

    bundle_root = output_root / "wandb_sync_bundle"
    if not bundle_root.exists():
        unzip_if_exists(output_root / "wandb_sync_bundle.zip", bundle_root)

    print("Downloaded Kaggle output files:")
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            print(path)

    manifest = load_manifest(bundle_root)
    if manifest:
        print(f"Loaded W&B sync manifest from {bundle_root / 'manifest.json'}")

    run_files = collect_run_files(output_root)
    source_run_ids = sync_offline_runs(run_files, entity)
    if manifest.get("source_run_ids"):
        source_run_ids = manifest["source_run_ids"]

    log_downloaded_artifacts(output_root, bundle_root if bundle_root.exists() else None, source_run_ids, entity, args.project)


if __name__ == "__main__":
    main()
