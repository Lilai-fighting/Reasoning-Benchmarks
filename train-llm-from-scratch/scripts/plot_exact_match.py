"""Plot exact-match accuracy against epochs, overall or for selected sample IDs.

python scripts/plot_exact_match.py --config configs/experiment_length9_12.json
python scripts/plot_exact_match.py --config configs/experiment_length9_12.json --sample-id sample_1
Each curve is for one training run; sample scores average its 20 decoding trials.
A base ID expands into all eight suffixes 000 through 111.
"""
import argparse
from collections import defaultdict
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = {
    "000": "straightforward",
    "001": "retrospective",
    "010": "segment",
    "011": "retrospective & segment",
    "100": "conditional",
    "101": "conditional & retrospective",
    "110": "conditional & segment",
    "111": "conditional & retrospective & segment",
}


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_seeds(seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be nonempty and distinct")


def expand_sample_ids(base_ids):
    if not base_ids:
        return None
    for base in base_ids:
        if not re.fullmatch(r"sample_[1-9][0-9]*", base):
            raise ValueError("Use a base ID such as sample_1, without a class suffix")
    return [f"{base}_{suffix:03b}" for base in dict.fromkeys(base_ids) for suffix in range(8)]


def sample_legend_label(sample_id):
    suffix = sample_id.rsplit("_", 1)[-1]
    if suffix not in CLASS_NAMES:
        raise ValueError(f"Unknown three-digit class suffix: {suffix}")
    return f"{suffix} ({CLASS_NAMES[suffix]})"


def collect(output_root, seeds, sample_ids=None, all_types=False):
    validate_seeds(seeds)
    sample_ids = expand_sample_ids(sample_ids)
    points = defaultdict(dict)
    labels = {}
    found = set()
    for seed in seeds:
        run = output_root / f"seed_{seed}"
        if sample_ids:
            paths = sorted((run / "evaluation_samples").glob("*/epoch_*.jsonl"))
            unified = {path for path in paths if "_rank_" not in path.stem}
            paths = [path for path in paths if "_rank_" not in path.stem or
                     path.with_name(path.stem.split("_rank_")[0] + ".jsonl") not in unified]
            if not paths:
                raise ValueError(f"No per-sample evaluation files in {run}")
            records = (row for path in paths for row in read_rows(path) if row["id"] in sample_ids)
        else:
            records = (row for row in read_rows(run / "eval_metrics.jsonl") if all_types or row["type"] == "all")
        for row in records:
            series = row["id"] if sample_ids else (row["type"] if all_types else row["split"])
            key = (row["split"], series, row["epoch"])
            if seed in points[key]:
                raise ValueError(f"Duplicate result for seed {seed}, {key}")
            points[key][seed] = float(row["exact_match_accuracy"])
            labels[(row["split"], series)] = (
                sample_legend_label(series) if sample_ids else row["split"]
            )
            found.add(series)
    if sample_ids and set(sample_ids) - found:
        raise ValueError(f"IDs not found: {sorted(set(sample_ids)-found)}")
    if not points:
        raise ValueError("No evaluation results found")
    # Require each class/split at every evaluated epoch, even if absent in all seeds.
    epochs = {key[2] for key in points}
    expected_series = ({("train" if sid[-3:] in {"000", "001", "010", "100"} else "test", sid)
                        for sid in sample_ids} if sample_ids else
                       ({(key[0], key[1]) for key in points} if all_types else {("train", "train"), ("test", "test")}))
    for split, series in expected_series:
        for epoch in epochs:
            if (split, series, epoch) not in points:
                raise ValueError(f"Missing class/split {split}/{series} at epoch {epoch}")
    result = []
    for (split, series, epoch), values in sorted(points.items()):
        if set(values) != set(seeds):
            raise ValueError(f"Incomplete training runs for {split}/{series} at epoch {epoch}")
        for seed, accuracy in values.items():
            result.append(dict(split=split, series=series,
                label=(labels[(split, series)] if sample_ids else
                       f"{labels[(split, series)]} [run {seed}]"), epoch=epoch,
                accuracy=accuracy, training_seed=seed))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment_length9_12.json")
    parser.add_argument("--sample-id", nargs="+")
    parser.add_argument("--seed", type=int, help="Select one training run; never averages training runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = [args.seed] if args.seed is not None else config["training"]["seeds"]
    output_root = ROOT / config["experiment"]["output_root"]
    points = collect(output_root, seeds, args.sample_id)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    grouped = defaultdict(list)
    for point in points:
        grouped[(point["label"], point["training_seed"])].append(point)
    for (label, _training_seed), values in grouped.items():
        values.sort(key=lambda p: p["epoch"])
        ax.plot([p["epoch"] for p in values], [p["accuracy"] for p in values], marker="o", label=label)
    ax.set(xlabel="Epoch", ylabel="Exact-match accuracy", ylim=(-0.03, 1.03),
           )
    ax.set_xticks(sorted({p["epoch"] for p in points}))
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        borderaxespad=0,
    )
    fig.tight_layout()
    output = args.output or output_root / (("_".join(args.sample_id) + "_eight_classes.png") if args.sample_id else config["experiment"]["plot_filename"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(points, indent=2)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()