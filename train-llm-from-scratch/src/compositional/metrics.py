"""Action-program metrics for compositional evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist


ACTIONS = (
    "I_TURN_LEFT",
    "I_TURN_RIGHT",
    "I_WALK",
    "I_RUN",
    "I_JUMP",
    "I_LOOK",
    "I_NOOP",
)
ACTION_SET = set(ACTIONS)
TYPE_NAMES = (
    "straightforward",
    "conditional",
    "exception",
    "retrospective",
)


def parse_actions(text: str) -> tuple[list[str], bool]:
    """Parse a generated action string without silently dropping invalid tokens."""
    pieces = text.strip().split()
    valid = bool(pieces) and all(piece in ACTION_SET for piece in pieces)
    if "I_NOOP" in pieces and pieces != ["I_NOOP"]:
        valid = False
    return pieces, valid


def levenshtein_distance(left: list[str], right: list[str]) -> int:
    """Unit-cost edit distance with O(min(m, n)) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, start=1):
        current = [row]
        for column, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def edit_aligned_action_accuracy(gold: list[str], predicted: list[str]) -> float:
    """One minus normalized action-level Levenshtein distance."""
    denominator = max(len(gold), len(predicted), 1)
    return 1.0 - levenshtein_distance(gold, predicted) / denominator


@dataclass
class MetricAccumulator:
    """Accumulate sample-macro metrics for each of the four sentence types."""

    # Columns: count, edit sum, exact sum, valid sum, length error sum,
    # no-op decision correct sum.
    values: torch.Tensor = field(
        default_factory=lambda: torch.zeros((len(TYPE_NAMES), 6), dtype=torch.float64)
    )

    type_names: tuple[str, ...] = TYPE_NAMES

    def __post_init__(self):
        if self.values.shape[0] != len(self.type_names):
            self.values = torch.zeros((len(self.type_names), 6), dtype=torch.float64)

    def update(
        self,
        type_id: int,
        gold: list[str],
        predicted: list[str],
        valid: bool,
    ) -> None:
        score = edit_aligned_action_accuracy(gold, predicted) if valid else 0.0
        gold_noop = gold == ["I_NOOP"]
        predicted_noop = predicted == ["I_NOOP"]
        row = self.values[type_id]
        row[0] += 1
        row[1] += score
        row[2] += float(valid and predicted == gold)
        row[3] += float(valid)
        row[4] += abs(len(predicted) - len(gold))
        row[5] += float(valid and gold_noop == predicted_noop)

    def distributed_reduce(self, device: str) -> None:
        """Sum disjoint DDP evaluation shards on every rank."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        reduced = self.values.to(device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        self.values = reduced.cpu()

    def results(self) -> list[dict[str, float | int | str]]:
        records = []
        for type_id, type_name in enumerate(self.type_names):
            count = int(self.values[type_id, 0].item())
            if count == 0:
                continue
            row = self.values[type_id]
            records.append(
                {
                    "type": type_name,
                    "num_examples": count,
                    "edit_aligned_action_accuracy": float(row[1] / count),
                    "exact_match_accuracy": float(row[2] / count),
                    "valid_sequence_rate": float(row[3] / count),
                    "mean_absolute_length_error": float(row[4] / count),
                    "noop_decision_accuracy": float(row[5] / count),
                }
            )
        return records
