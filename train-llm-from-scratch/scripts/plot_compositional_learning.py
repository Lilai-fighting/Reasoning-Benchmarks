"""Plot mean edit-aligned action accuracy across compositional training seeds."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from src.compositional.metrics import TYPE_NAMES


REPO_ROOT = Path(__file__).resolve().parents[1]
T_CRITICAL_95_DF4 = 2.776


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    seeds = [int(value) for value in config["training"]["seeds"]]
    experiment = config["experiment"]
    output_root = (REPO_ROOT / experiment["output_root"]).resolve()

    # values[(type, step)][seed] = metric; using a seed map also detects duplicate rows.
    values: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    epochs: dict[int, int] = {}
    for seed in seeds:
        path = output_root / f"seed_{seed}" / "eval_metrics.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation metrics for seed {seed}: {path}")
        for record in read_jsonl(path):
            key = (str(record["type"]), int(record["step"]))
            if seed in values[key]:
                raise ValueError(f"Duplicate metric row for seed={seed}, key={key}")
            values[key][seed] = float(record["edit_aligned_action_accuracy"])
            epochs[int(record["step"])] = int(record["epoch"])

    figure, axis = plt.subplots(figsize=(10, 6))
    for type_name in TYPE_NAMES:
        steps = sorted(step for kind, step in values if kind == type_name)
        means, lower, upper = [], [], []
        seed_point_steps: list[int] = []
        seed_point_values: list[float] = []
        for step in steps:
            by_seed = values[(type_name, step)]
            missing = set(seeds).difference(by_seed)
            if missing:
                raise ValueError(
                    f"Missing seeds {sorted(missing)} for type={type_name}, step={step}"
                )
            observations = [by_seed[seed] for seed in seeds]
            center = mean(observations)
            half_width = T_CRITICAL_95_DF4 * sample_sd(observations) / math.sqrt(len(seeds))
            means.append(center)
            lower.append(max(0.0, center - half_width))
            upper.append(min(1.0, center + half_width))
            seed_point_steps.extend([step] * len(observations))
            seed_point_values.extend(observations)

        (line,) = axis.plot(steps, means, marker="o", label=type_name)
        axis.fill_between(steps, lower, upper, color=line.get_color(), alpha=0.16)
        # Faint seed points make instability visible rather than hiding it in a band.
        axis.scatter(
            seed_point_steps,
            seed_point_values,
            color=line.get_color(),
            alpha=0.12,
            s=10,
        )

    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Edit-aligned action accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(
        "Compositional learning evolution\n"
        "mean across five training seeds; shared seeded top-k decoding"
    )
    axis.grid(alpha=0.25)
    axis.legend(title="Sentence type")
    figure.tight_layout()

    output_path = output_root / experiment["plot_filename"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved plot -> {output_path}")


if __name__ == "__main__":
    main()