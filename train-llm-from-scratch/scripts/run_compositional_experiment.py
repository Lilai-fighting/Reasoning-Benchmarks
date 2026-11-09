"""Run every training seed from one compositional experiment configuration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def torchrun_executable() -> str:
    candidate = shutil.which("torchrun")
    if candidate:
        return candidate
    local = Path(sys.executable).with_name(
        "torchrun.exe" if os.name == "nt" else "torchrun"
    )
    if local.exists():
        return str(local)
    raise FileNotFoundError("torchrun was not found next to Python or on PATH")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a seed when its final checkpoint already exists",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    seeds = [int(value) for value in config["training"]["seeds"]]
    experiment = config["experiment"]
    output_root = (REPO_ROOT / experiment["output_root"]).resolve()
    nproc = int(experiment["nproc_per_node"])
    output_root.mkdir(parents=True, exist_ok=True)

    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for seed in seeds:
        run_dir = output_root / f"seed_{seed}"
        final_checkpoint = run_dir / "checkpoints" / "final.pt"
        if args.skip_completed and final_checkpoint.exists():
            print(f"Skipping completed seed {seed}: {final_checkpoint}")
            continue

        base = [
            "scripts/train_compositional.py",
            "--config",
            str(config_path),
            "--seed",
            str(seed),
            "--run-dir",
            str(run_dir),
        ]
        if nproc > 1:
            command = [
                torchrun_executable(),
                "--standalone",
                f"--nproc_per_node={nproc}",
                *base,
            ]
        else:
            command = [sys.executable, *base]

        print(f"Starting training seed {seed}")
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )

    if bool(experiment.get("plot_after_training", True)):
        subprocess.run(
            [
                sys.executable,
                experiment.get("plot_script", "scripts/plot_compositional_learning.py"),
                "--config",
                str(config_path),
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    main()
