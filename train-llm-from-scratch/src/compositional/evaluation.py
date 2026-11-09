"""Full-corpus compositional evaluation with seeded top-k action sampling."""

from __future__ import annotations

from typing import Any
import json
from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
from src.compositional.evaluation_output import merge_rank_outputs

from data_loader.compositional_dataset import CompositionalDataset, EvaluationExample
from src.compositional.metrics import MetricAccumulator, parse_actions
from src.compositional.sampling import ConstrainedTopKSampler


def _chunks(values: list[EvaluationExample], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.no_grad()
def evaluate_compositional(
    model,
    dataset: CompositionalDataset,
    evaluation_config: dict[str, Any],
    ctx,
    *,
    results_path: Path | None = None,
    result_metadata: dict[str, Any] | None = None,
) -> list[dict[str, float | int | str]]:
    """Shard documents without padding and generate every configured decoding seed per document."""
    was_training = model.training
    model.eval()

    try:
        tokenizer = tiktoken.get_encoding(dataset.tokenizer_name)
        decoding_seeds = evaluation_config.get("decoding_seeds", [evaluation_config.get("decoding_seed", 0)])
        if not decoding_seeds or any(type(seed) is not int for seed in decoding_seeds) or len(set(decoding_seeds)) != len(decoding_seeds):
            raise ValueError("decoding_seeds must be distinct integers")
        samplers = [ConstrainedTopKSampler(
            tokenizer, eot_id=dataset.eot_id, decoding_seed=seed,
            temperature=float(evaluation_config["temperature"]), top_k=int(evaluation_config["top_k"]),
            max_actions=int(evaluation_config["max_actions"]), max_bpe_tokens=int(evaluation_config["max_bpe_tokens"]),
        ) for seed in decoding_seeds]

        # Strided sharding has no padding, so the union of ranks is exactly the corpus.
        local_examples = [
            dataset.evaluation_example(index)
            for index in range(ctx.rank, len(dataset), ctx.world_size)
        ]
        accumulator = MetricAccumulator(type_names=dataset.type_names)
        if results_path is not None:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            shard_paths = [results_path.with_name(f".{results_path.stem}.rank_{rank}.tmp") for rank in range(ctx.world_size)]
            shard_path = shard_paths[ctx.rank]
            shard_path.write_text("", encoding="utf-8")
        batch_size = int(evaluation_config["batch_size"])

        for batch in _chunks(local_examples, batch_size):
            trials = [[] for _ in batch]
            for decoding_seed, sampler in zip(decoding_seeds, samplers):
                sampled = sampler.sample_batch(
                    model, [example.prompt_ids for example in batch],
                    [example.document_id for example in batch], device=ctx.device,
                )
                if len(sampled) != len(batch):
                    raise ValueError("Sampler returned an incorrect batch size")
                for index, (example, result) in enumerate(zip(batch, sampled)):
                    gold_actions, gold_valid = parse_actions(tokenizer.decode(example.gold_ids))
                    if not gold_valid:
                        raise ValueError(f"Invalid gold actions for {example.document_id}")
                    trials[index].append({
                        "decoding_seed": decoding_seed, "predicted_actions": result.actions,
                        "valid": result.finished, "generated_token_ids": result.generated_token_ids,
                        "model_output": tokenizer.decode(result.generated_token_ids),
                        "exact_match": int(result.finished and result.actions == gold_actions),
                    })
                    accumulator.update(example.type_id, gold_actions, result.actions, valid=result.finished)
            if results_path is not None:
                with shard_path.open("a", encoding="utf-8") as handle:
                    for example, generations in zip(batch, trials):
                        gold_actions, _ = parse_actions(tokenizer.decode(example.gold_ids))
                        successes = sum(g["exact_match"] for g in generations)
                        handle.write(json.dumps({**(result_metadata or {}),
                            "id": example.document_id, "type": dataset.type_names[example.type_id],
                            "gold_actions": gold_actions, "num_generations": len(generations),
                            "num_exact_matches": successes,
                            "exact_match_accuracy": successes / len(generations),
                            "generations": generations}) + "\n")

        if results_path is not None:
            if ctx.world_size > 1:
                dist.barrier()
            error = [None]
            if ctx.rank == 0:
                try:
                    expected_ids = [dataset.evaluation_example(i).document_id for i in range(len(dataset))]
                    merge_rank_outputs(shard_paths, results_path, expected_ids)
                except Exception as exc:
                    error[0] = f"Could not publish evaluation results: {exc}"
            if ctx.world_size > 1:
                dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise RuntimeError(error[0])
        accumulator.distributed_reduce(ctx.device)
        records = accumulator.results()
        for record in records:
            record["num_generations"] = record["num_examples"]
            record["num_examples"] //= len(decoding_seeds)
            record["generations_per_sample"] = len(decoding_seeds)
        return records
    finally:
        model.train(was_training)
