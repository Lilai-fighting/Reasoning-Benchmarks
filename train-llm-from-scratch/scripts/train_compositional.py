"""Train one compositional model from scratch with pure output-token loss.

Run one GPU/process directly::

    python scripts/train_compositional.py --config configs/experiment.json \
        --seed 42 --run-dir ephemeral/compositional_runs/seed_42

Run with DDP::

    torchrun --standalone --nproc_per_node=2 scripts/train_compositional.py \
        --config configs/experiment.json --seed 42 \
        --run-dir ephemeral/compositional_runs/seed_42
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_loader.compositional_dataset import (
    CompositionalDataset,
    make_collate_fn,
)
from src.compositional.evaluation import evaluate_compositional
from src.post_training.distributed import (
    barrier,
    cleanup,
    ddp_setup,
    ddp_wrap,
    reduce_scalar,
)
from src.post_training.optim import configure_optimizer, cosine_lr
from src.post_training.utils import (
    amp_autocast,
    build_model_from_config,
    save_stage_ckpt,
    set_seed,
    unwrap,
)


def load_experiment(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "model",
        "data",
        "training",
        "evaluation",
        "checkpointing",
        "experiment",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Experiment config is missing sections: {sorted(missing)}")
    return config


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def output_only_loss_sum(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_mask: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Summed causal loss on gold output actions and their final EOT.

    The caller normalizes across the complete gradient-accumulation group, including
    all DDP ranks. This gives every output token equal weight even though documents
    contain different numbers of actions.
    """
    model_inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:] & output_mask[:, 1:]
    if not target_mask.any():
        raise ValueError("Training batch contains no output target tokens")

    logits, _ = model(model_inputs)
    per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return per_token[target_mask].sum()


def checkpoint_path(run_dir: Path, epoch: int, keep_periodic: bool) -> Path:
    if keep_periodic:
        return run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
    return run_dir / "checkpoints" / "latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_experiment(args.config)
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    eval_config = config["evaluation"]
    checkpoint_config = config["checkpointing"]

    if args.seed not in [int(value) for value in train_config["seeds"]]:
        raise ValueError(f"Seed {args.seed} is not listed in training.seeds")

    ctx = ddp_setup(str(model_config["device"]))
    # Rank-specific seeds give DDP workers independent incidental RNG state, while the
    # DistributedSampler still uses the shared run seed for one reproducible shuffle.
    set_seed(args.seed + ctx.rank)

    run_dir = args.run_dir
    existing_outputs = [
        run_dir / "train_metrics.jsonl",
        run_dir / "eval_metrics.jsonl",
        run_dir / "checkpoints" / "final.pt",
    ]
    if any(path.exists() for path in existing_outputs):
        raise FileExistsError(
            f"Run directory already contains experiment outputs: {run_dir}. "
            "Choose a new directory or use --skip-completed in the seed runner."
        )
    if ctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump({**config, "active_training_seed": args.seed}, handle, indent=2)
    barrier(ctx)

    dataset = CompositionalDataset(
        data_config["path"], context_length=int(model_config["context_length"])
    )
    evaluation_datasets = {"train": dataset}
    if eval_config.get("test_path"):
        evaluation_datasets["test"] = CompositionalDataset(
            eval_config["test_path"], context_length=int(model_config["context_length"])
        )
    if int(eval_config["every_epochs"]) <= 0:
        raise ValueError("evaluation.every_epochs must be positive")
    distributed_sampler = DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        sampler=distributed_sampler,
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]),
        collate_fn=make_collate_fn(dataset.eot_id),
        persistent_workers=int(data_config["num_workers"]) > 0,
    )

    model = build_model_from_config(SimpleNamespace(**model_config)).to(ctx.device)
    if bool(model_config["compile"]):
        model = torch.compile(model)
    model = ddp_wrap(model, ctx)
    optimizer = configure_optimizer(
        unwrap(model),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    grad_accum = int(train_config["grad_accum"])
    updates_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = int(train_config["epochs"]) * updates_per_epoch
    warmup_steps = round(float(train_config["warmup_ratio"]) * total_steps)
    train_metrics_path = run_dir / "train_metrics.jsonl"
    eval_metrics_path = run_dir / "eval_metrics.jsonl"
    global_step = 0

    def run_evaluation(epoch: int) -> None:
        # Both splits use the current in-memory weights after the epoch's last update.
        for split, evaluation_dataset in evaluation_datasets.items():
            started = time.perf_counter()
            records = evaluate_compositional(
                model, evaluation_dataset, eval_config, ctx,
                results_path=run_dir / "evaluation_samples" / split / f"epoch_{epoch:03d}.jsonl",
                result_metadata={"seed": args.seed, "epoch": epoch, "step": global_step, "split": split},
            )
            count = sum(r["num_examples"] for r in records)
            overall = sum(r["exact_match_accuracy"] * r["num_examples"] for r in records) / count
            records.append({"type": "all", "num_examples": count, "exact_match_accuracy": overall})
            if ctx.is_main:
                for record in records:
                    append_jsonl(eval_metrics_path, {
                        "seed": args.seed, "step": global_step, "epoch": epoch, "split": split,
                        "evaluation_seconds": time.perf_counter() - started, **record,
                    })
                print(f"[eval] epoch={epoch} split={split} exact_match_accuracy={overall:.4f}")

    try:
        if bool(eval_config["at_start"]):
            run_evaluation(epoch=0)

        epochs = int(train_config["epochs"])
        log_every = int(train_config["log_every_steps"])
        save_every = int(checkpoint_config["save_every_epochs"])
        eval_every = int(eval_config["every_epochs"])
        keep_periodic = bool(checkpoint_config["keep_periodic"])

        for epoch in range(1, epochs + 1):
            distributed_sampler.set_epoch(epoch)
            model.train()
            epoch_started = time.perf_counter()
            loader_iterator = iter(loader)

            for _ in range(updates_per_epoch):
                micro_batches = []
                for _micro in range(grad_accum):
                    try:
                        micro_batches.append(next(loader_iterator))
                    except StopIteration:
                        break
                if not micro_batches:
                    break

                learning_rate = cosine_lr(
                    global_step,
                    warmup_steps=warmup_steps,
                    max_steps=total_steps,
                    lr=float(train_config["lr"]),
                    min_lr=float(train_config["min_lr"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

                # Count targets before forward passes so every micro-batch and DDP rank
                # can be normalized by the same effective-batch token count.
                local_target_count = sum(
                    int(
                        (
                            batch["attention_mask"][:, 1:]
                            & batch["output_mask"][:, 1:]
                        ).sum()
                    )
                    for batch in micro_batches
                )
                global_target_count_tensor = torch.tensor(
                    float(local_target_count), device=ctx.device
                )
                if ctx.enabled:
                    dist.all_reduce(global_target_count_tensor, op=dist.ReduceOp.SUM)
                global_target_count = float(global_target_count_tensor.item())
                if global_target_count <= 0:
                    raise ValueError("Gradient-accumulation group has no output targets")

                optimizer.zero_grad(set_to_none=True)
                local_loss_sum = 0.0
                for micro_index, batch in enumerate(micro_batches):
                    input_ids = batch["input_ids"].to(ctx.device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(
                        ctx.device, non_blocking=True
                    )
                    output_mask = batch["output_mask"].to(
                        ctx.device, non_blocking=True
                    )

                    is_group_end = micro_index + 1 == len(micro_batches)
                    sync_context = contextlib.nullcontext()
                    if ctx.enabled and not is_group_end:
                        sync_context = model.no_sync()
                    with sync_context, amp_autocast(
                        model_config["amp_dtype"], ctx.device
                    ):
                        loss_sum = output_only_loss_sum(
                            model,
                            input_ids,
                            attention_mask,
                            output_mask,
                            vocab_size=int(model_config["vocab_size"]),
                        )
                        # DDP averages gradients across ranks, so multiplying by
                        # world_size recovers a true global token mean.
                        scaled_loss = (
                            loss_sum * ctx.world_size / global_target_count
                        )
                    scaled_loss.backward()
                    local_loss_sum += loss_sum.detach().float().item()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config["grad_clip"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                global_loss_sum = reduce_scalar(local_loss_sum, ctx, average=False)
                mean_loss = global_loss_sum / global_target_count
                if ctx.is_main and (global_step % log_every == 0 or global_step == 1):
                    record = {
                        "seed": args.seed,
                        "step": global_step,
                        "epoch": epoch,
                        "output_loss": mean_loss,
                        "lr": learning_rate,
                    }
                    append_jsonl(train_metrics_path, record)
                    print(
                        f"epoch={epoch}/{epochs} step={global_step}/{total_steps} "
                        f"output_loss={mean_loss:.4f} lr={learning_rate:.2e}"
                    )

            if ctx.is_main:
                print(
                    f"Finished epoch {epoch}/{epochs} in "
                    f"{time.perf_counter() - epoch_started:.1f}s"
                )

            should_evaluate = epoch % eval_every == 0
            if epoch == epochs and bool(eval_config["at_end"]):
                should_evaluate = True
            if should_evaluate:
                run_evaluation(epoch)

            if epoch % save_every == 0 or epoch == epochs:
                if ctx.is_main:
                    path = checkpoint_path(run_dir, epoch, keep_periodic)
                    save_stage_ckpt(
                        str(path),
                        model,
                        optimizer,
                        stage="compositional_from_scratch",
                        cfg=config,
                        step=global_step,
                        metrics={"epoch": epoch},
                        extra={"training_seed": args.seed, "epoch": epoch},
                    )
                    print(f"Saved checkpoint -> {path}")
                barrier(ctx)

        if ctx.is_main:
            final_path = run_dir / "checkpoints" / "final.pt"
            save_stage_ckpt(
                str(final_path),
                model,
                optimizer,
                stage="compositional_from_scratch",
                cfg=config,
                step=global_step,
                metrics={"epoch": epochs},
                extra={"training_seed": args.seed, "epoch": epochs},
            )
            print(f"Training complete -> {final_path}")
        barrier(ctx)
    finally:
        for evaluation_dataset in evaluation_datasets.values():
            evaluation_dataset.close()
        cleanup(ctx)


if __name__ == "__main__":
    main()
