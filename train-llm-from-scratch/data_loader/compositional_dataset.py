"""Document-aware loader for the metadata-rich compositional HDF5 corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EvaluationExample:
    """One complete prompt and its gold symbolic action program."""

    document_id: str
    type_id: int
    prompt_ids: list[int]
    gold_ids: list[int]


class CompositionalDataset(Dataset):
    """Expose complete documents instead of arbitrary flat-token windows.

    The HDF5 handle is opened lazily so every DataLoader worker owns its own handle;
    sharing an h5py handle across forked workers is unsafe.
    """

    REQUIRED_DATASETS = {
        "tokens",
        "output_mask",
        "document_offsets",
        "document_type_ids",
        "document_ids",
    }

    def __init__(self, path: str | Path, context_length: int) -> None:
        self.path = str(Path(path))
        self.context_length = context_length
        self._h5: h5py.File | None = None

        with h5py.File(self.path, "r") as handle:
            missing = self.REQUIRED_DATASETS.difference(handle.keys())
            if missing:
                raise ValueError(
                    f"{self.path} is not the metadata-rich HDF5 schema; "
                    f"missing {sorted(missing)}"
                )
            self.offsets = np.asarray(handle["document_offsets"], dtype=np.int64)
            self.type_ids = np.asarray(handle["document_type_ids"], dtype=np.uint8)
            self.document_ids = list(handle["document_ids"].asstr()[:])
            self.eot_id = int(handle.attrs["eot_id"])
            self.tokenizer_name = str(handle.attrs["tokenizer"])
            self.type_names = tuple(json.loads(handle.attrs["type_names"]))

        if len(self.offsets) != len(self.type_ids) + 1:
            raise ValueError("document_offsets must have one more item than documents")
        if len(self.document_ids) != len(self.type_ids):
            raise ValueError("document_ids and document_type_ids are not aligned")
        maximum = int(np.diff(self.offsets).max(initial=0))
        if maximum > context_length:
            raise ValueError(
                f"Longest document has {maximum} tokens, exceeding "
                f"context_length={context_length}"
            )

    def __len__(self) -> int:
        return len(self.type_ids)

    def _handle(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, Any]:
        start, end = map(int, self.offsets[index : index + 2])
        handle = self._handle()
        return {
            "tokens": np.asarray(handle["tokens"][start:end], dtype=np.int64),
            "output_mask": np.asarray(
                handle["output_mask"][start:end], dtype=np.bool_
            ),
            "type_id": int(self.type_ids[index]),
            "document_id": self.document_ids[index],
        }

    def evaluation_example(self, index: int) -> EvaluationExample:
        """Split a document at the first output target token.

        ``output_mask`` includes EOT. EOT is removed from the gold action text but is
        retained as the stopping target during training.
        """
        item = self[index]
        tokens = item["tokens"]
        mask = item["output_mask"]
        output_positions = np.flatnonzero(mask)
        if output_positions.size == 0:
            raise ValueError(f"Document {item['document_id']} has no output tokens")
        output_start = int(output_positions[0])
        gold = tokens[output_start:].tolist()
        if gold and gold[-1] == self.eot_id:
            gold.pop()
        return EvaluationExample(
            document_id=item["document_id"],
            type_id=item["type_id"],
            prompt_ids=tokens[:output_start].tolist(),
            gold_ids=gold,
        )

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()


def make_collate_fn(pad_id: int):
    """Pad a document batch; padded positions never contribute to output loss."""

    def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(item["tokens"]) for item in items)
        batch_size = len(items)
        tokens = torch.full((batch_size, max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        output_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

        for row, item in enumerate(items):
            length = len(item["tokens"])
            tokens[row, :length] = torch.from_numpy(item["tokens"])
            attention_mask[row, :length] = True
            output_mask[row, :length] = torch.from_numpy(item["output_mask"])

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "output_mask": output_mask,
            "type_ids": torch.tensor(
                [item["type_id"] for item in items], dtype=torch.long
            ),
            "document_ids": [item["document_id"] for item in items],
        }

    return collate