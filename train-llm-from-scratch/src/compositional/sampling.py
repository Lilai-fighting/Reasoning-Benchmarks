"""Seeded, grammar-constrained top-k sampling for symbolic SCAN actions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from src.compositional.metrics import ACTIONS
from src.post_training.utils import unwrap


_TERMINAL = "_action"


def stable_example_seed(base_seed: int, document_id: str) -> int:
    """Derive the same portable seed for a document on every model/checkpoint."""
    digest = hashlib.sha256(f"{base_seed}:{document_id}".encode("utf-8")).digest()
    # torch.Generator.manual_seed accepts a signed 64-bit-safe positive integer.
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _build_trie(tokenizer, leading_space: bool) -> dict[Any, Any]:
    root: dict[Any, Any] = {}
    for action in ACTIONS:
        text = (" " if leading_space else "") + action
        token_ids = tokenizer.encode_ordinary(text)
        if not token_ids:
            raise ValueError(f"Tokenizer produced no tokens for action {action!r}")
        node = root
        for token_id in token_ids:
            node = node.setdefault(token_id, {})
        if _TERMINAL in node:
            raise ValueError(f"Duplicate action tokenization for {action!r}")
        node[_TERMINAL] = action
    return root


@dataclass
class SampleResult:
    actions: list[str]
    finished: bool
    generated_token_ids: list[int]


@dataclass
class _State:
    document_id: str
    actions: list[str] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    node: dict[Any, Any] | None = None
    at_boundary: bool = True
    finished: bool = False


class ConstrainedTopKSampler:
    """Sample valid action programs while applying top-k at each BPE decision.

    The grammar guarantees that completed samples contain only known actions, contain
    no action after ``I_NOOP``, and stop after at most ``max_actions`` actions.
    """

    def __init__(
        self,
        tokenizer,
        *,
        eot_id: int,
        decoding_seed: int,
        temperature: float,
        top_k: int,
        max_actions: int,
        max_bpe_tokens: int,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_actions <= 0 or max_bpe_tokens <= 0:
            raise ValueError("generation limits must be positive")
        self.eot_id = eot_id
        self.decoding_seed = decoding_seed
        self.temperature = temperature
        self.top_k = top_k
        self.max_actions = max_actions
        self.max_bpe_tokens = max_bpe_tokens
        self.first_action_trie = _build_trie(tokenizer, leading_space=False)
        self.next_action_trie = _build_trie(tokenizer, leading_space=True)

    def _allowed_tokens(self, state: _State) -> list[int]:
        if state.at_boundary:
            if state.actions == ["I_NOOP"] or len(state.actions) >= self.max_actions:
                return [self.eot_id]
            root = (
                self.first_action_trie
                if not state.actions
                else self.next_action_trie
            )
            allowed = [key for key in root if isinstance(key, int)]
            if state.actions:
                allowed.append(self.eot_id)
            return sorted(allowed)
        assert state.node is not None
        return sorted(key for key in state.node if isinstance(key, int))

    def _consume(self, state: _State, token_id: int) -> None:
        if token_id == self.eot_id and state.at_boundary and state.actions:
            state.finished = True
            state.generated.append(token_id)
            return

        if state.at_boundary:
            root = (
                self.first_action_trie
                if not state.actions
                else self.next_action_trie
            )
            state.node = root[token_id]
            state.at_boundary = False
        else:
            assert state.node is not None
            state.node = state.node[token_id]
        state.generated.append(token_id)

        assert state.node is not None
        if _TERMINAL in state.node:
            state.actions.append(str(state.node[_TERMINAL]))
            state.node = None
            state.at_boundary = True

    @torch.no_grad()
    def sample_batch(
        self,
        model,
        prompt_ids: list[list[int]],
        document_ids: list[str],
        *,
        device: str,
    ) -> list[SampleResult]:
        """Generate one independently seeded sample for each prompt in a batch."""
        if len(prompt_ids) != len(document_ids):
            raise ValueError("prompt_ids and document_ids must have equal length")
        if not prompt_ids:
            return []

        base_model = unwrap(model)
        context_length = int(base_model.context_length)
        batch_size = len(prompt_ids)
        sequences = torch.full(
            (batch_size, context_length), self.eot_id, dtype=torch.long, device=device
        )
        lengths = []
        for row, prompt in enumerate(prompt_ids):
            if not prompt or len(prompt) >= context_length:
                raise ValueError(
                    f"Prompt {document_ids[row]!r} has invalid length {len(prompt)} "
                    f"for context_length={context_length}"
                )
            sequences[row, : len(prompt)] = torch.tensor(prompt, device=device)
            lengths.append(len(prompt))

        states = [_State(document_id=value) for value in document_ids]
        generators = []
        generator_device = torch.device(device)
        for document_id in document_ids:
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(
                stable_example_seed(self.decoding_seed, document_id)
            )
            generators.append(generator)

        for _ in range(self.max_bpe_tokens):
            active = [
                row
                for row, state in enumerate(states)
                if not state.finished and lengths[row] < context_length
            ]
            if not active:
                break

            active_lengths = torch.tensor(
                [lengths[row] for row in active], device=device, dtype=torch.long
            )
            maximum_length = int(active_lengths.max().item())
            active_sequences = sequences[active, :maximum_length]

            # Project only the final real hidden state. Calling model(...) would create
            # logits for every prompt position and waste substantial evaluation memory.
            hidden = base_model.forward_hidden(active_sequences)
            last_hidden = hidden[
                torch.arange(len(active), device=device), active_lengths - 1
            ]
            logits = base_model.lm_head(last_hidden).float()

            for local_row, global_row in enumerate(active):
                state = states[global_row]
                allowed = self._allowed_tokens(state)
                allowed_tensor = torch.tensor(allowed, device=device, dtype=torch.long)
                allowed_logits = logits[local_row, allowed_tensor] / self.temperature
                keep = min(self.top_k, len(allowed))
                top_values, top_positions = torch.topk(allowed_logits, keep)
                probabilities = torch.softmax(top_values, dim=-1)
                sampled_position = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generators[global_row],
                    ).item()
                )
                token_id = allowed[int(top_positions[sampled_position].item())]
                sequences[global_row, lengths[global_row]] = token_id
                lengths[global_row] += 1
                self._consume(state, token_id)

        return [
            SampleResult(
                actions=state.actions,
                finished=state.finished,
                generated_token_ids=state.generated,
            )
            for state in states
        ]