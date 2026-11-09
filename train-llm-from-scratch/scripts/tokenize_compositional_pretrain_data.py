"""Tokenize serialized compositional-pretraining JSONL into a metadata-rich HDF5 stream.

Input rows must contain ``text`` and may also contain ``id`` and ``type``. For legacy
serialized files that contain only ``text``, pass ``--metadata-input`` pointing to the
aligned source JSONL containing the original ``id`` and ``type`` fields.

The text in every row is tokenized with ``tiktoken``'s ``r50k_base`` encoding,
followed by the end-of-text token. The output keeps the original flat ``tokens``
dataset used by the pretraining loader and adds aligned category/output metadata plus
document boundaries for category-aware and sample-level evaluation.

Example:
    python scripts/tokenize_compositional_pretrain_data.py \
        --input ephemeral/data/compositional_pretrain.jsonl \
        --metadata-input ../pretraining_merged_42.jsonl \
        --output ephemeral/data/compositional_pretrain.h5
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import tiktoken


TOKENIZER_NAME = "r50k_base"
EOT_ID = 50256
ENCODE_BATCH_SIZE = 1_024
WRITE_CHUNK_TOKENS = 1_000_000
OUTPUT_MARKER = "\n\nOutput:\n"
TYPE_NAMES = (
    "straightforward",
    "conditional",
    "exception",
    "retrospective",
)
TYPE_TO_ID = {name: index for index, name in enumerate(TYPE_NAMES)}


def append_aligned(
    tokens_dataset: h5py.Dataset,
    type_dataset: h5py.Dataset,
    output_mask_dataset: h5py.Dataset,
    token_values: list[int],
    type_values: list[int],
    output_mask_values: list[int],
) -> int:
    """Append three aligned token-level arrays and return their shared length."""
    size = len(token_values)
    if len(type_values) != size or len(output_mask_values) != size:
        raise ValueError("Token, category, and output-mask buffers are not aligned")
    if not size:
        return 0

    start = tokens_dataset.shape[0]
    end = start + size
    for dataset in (tokens_dataset, type_dataset, output_mask_dataset):
        dataset.resize(end, axis=0)
    tokens_dataset[start:end] = np.asarray(token_values, dtype=np.int32)
    type_dataset[start:end] = np.asarray(type_values, dtype=np.uint8)
    output_mask_dataset[start:end] = np.asarray(output_mask_values, dtype=np.uint8)
    return size


def read_jsonl(path: Path, *, label: str) -> list[dict]:
    """Read nonblank JSONL objects with precise source errors."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} JSONL does not exist: {path}")

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{label} line {line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{label} line {line_number}: expected a JSON object"
                )
            records.append(record)
    return records


def serialize_source_record(record: dict, *, record_number: int) -> str | None:
    """Recreate serialized text when an original source record is supplied."""
    fields = ("instruction", "context", "output")
    if not all(field in record for field in fields):
        return None

    instruction = record["instruction"]
    context = record["context"]
    output = record["output"]
    if not isinstance(instruction, str) or not isinstance(context, str):
        raise ValueError(
            f"Metadata record {record_number}: instruction/context must be strings"
        )
    if not isinstance(output, list) or not all(isinstance(x, str) for x in output):
        raise ValueError(
            f"Metadata record {record_number}: output must be a list of strings"
        )
    return (
        f"Instruction:\n{instruction.strip()}\n\n"
        f"Context:\n{context.strip()}\n\n"
        f"Output:\n{' '.join(action.strip() for action in output)}"
    )


def resolve_metadata(
    serialized_record: dict,
    source_record: dict | None,
    *,
    record_number: int,
) -> tuple[str, str, str]:
    """Return validated ``(text, document_id, type_name)`` for one record."""
    text = serialized_record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"Serialized record {record_number}: text must be a nonempty string"
        )

    document_id = serialized_record.get("id")
    type_name = serialized_record.get("type")
    if source_record is not None:
        source_id = source_record.get("id")
        source_type = source_record.get("type")
        if document_id is not None and document_id != source_id:
            raise ValueError(
                f"Record {record_number}: embedded id {document_id!r} does not match "
                f"metadata id {source_id!r}"
            )
        if type_name is not None and type_name != source_type:
            raise ValueError(
                f"Record {record_number}: embedded type {type_name!r} does not match "
                f"metadata type {source_type!r}"
            )
        document_id = document_id if document_id is not None else source_id
        type_name = type_name if type_name is not None else source_type

        expected_text = source_record.get("text")
        if expected_text is None:
            expected_text = serialize_source_record(
                source_record, record_number=record_number
            )
        if expected_text is not None and text != expected_text:
            raise ValueError(
                f"Record {record_number}: serialized text is not aligned with metadata"
            )

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError(
            f"Record {record_number}: missing nonempty id; add id/type to the input "
            "or pass --metadata-input"
        )
    if type_name not in TYPE_TO_ID:
        raise ValueError(
            f"Record {record_number}: unknown or missing type {type_name!r}; "
            f"expected one of {list(TYPE_NAMES)}"
        )
    return text, document_id, type_name


def tokenize_jsonl_to_hdf5(
    input_path: Path,
    output_path: Path,
    *,
    batch_size: int,
    expected_count: int | None,
    overwrite: bool,
    metadata_input: Path | None = None,
) -> tuple[int, int]:
    """Tokenize all JSONL records and return ``(record_count, token_count)``."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    serialized_records = read_jsonl(input_path, label="Serialized input")
    metadata_records = (
        read_jsonl(metadata_input, label="Metadata input")
        if metadata_input is not None
        else None
    )
    if metadata_records is not None and len(metadata_records) != len(serialized_records):
        raise ValueError(
            "Serialized and metadata inputs contain different record counts: "
            f"{len(serialized_records):,} != {len(metadata_records):,}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    record_count = 0
    token_count = 0
    documents: list[tuple[str, str, str]] = []
    token_buffer: list[int] = []
    type_buffer: list[int] = []
    output_mask_buffer: list[int] = []
    document_offsets = [0]
    document_type_ids: list[int] = []
    document_ids: list[str] = []
    seen_document_ids: set[str] = set()
    type_record_counts = np.zeros(len(TYPE_NAMES), dtype=np.int64)
    type_token_counts = np.zeros(len(TYPE_NAMES), dtype=np.int64)

    try:
        with h5py.File(temporary_path, "w") as hdf5_file:
            tokens = hdf5_file.create_dataset(
                "tokens",
                shape=(0,),
                maxshape=(None,),
                dtype="i4",
                chunks=(WRITE_CHUNK_TOKENS,),
            )
            token_type_ids = hdf5_file.create_dataset(
                "token_type_ids",
                shape=(0,),
                maxshape=(None,),
                dtype="u1",
                chunks=(WRITE_CHUNK_TOKENS,),
            )
            output_mask = hdf5_file.create_dataset(
                "output_mask",
                shape=(0,),
                maxshape=(None,),
                dtype="u1",
                chunks=(WRITE_CHUNK_TOKENS,),
            )

            def encode_documents() -> None:
                nonlocal documents, token_buffer, type_buffer
                nonlocal output_mask_buffer, token_count
                if not documents:
                    return

                texts = [document[0] for document in documents]
                prefixes: list[str] = []
                for text in texts:
                    marker_position = text.rfind(OUTPUT_MARKER)
                    if marker_position < 0:
                        raise ValueError(
                            f"Record {record_count - len(documents) + 1}: "
                            f"missing output marker {OUTPUT_MARKER!r}"
                        )
                    prefixes.append(
                        text[: marker_position + len(OUTPUT_MARKER)]
                    )

                encoded_texts = tokenizer.encode_ordinary_batch(texts)
                encoded_prefixes = tokenizer.encode_ordinary_batch(prefixes)
                for (text, document_id, type_name), token_ids, prefix_ids in zip(
                    documents, encoded_texts, encoded_prefixes
                ):
                    del text  # encoded above; keep tuple structure explicit
                    if token_ids[: len(prefix_ids)] != prefix_ids:
                        raise ValueError(
                            f"Document {document_id!r}: output marker is not on a "
                            "token boundary"
                        )

                    type_id = TYPE_TO_ID[type_name]
                    document_tokens = [*token_ids, EOT_ID]
                    output_start = len(prefix_ids)
                    document_mask = (
                        [0] * output_start
                        + [1] * (len(document_tokens) - output_start)
                    )
                    token_buffer.extend(document_tokens)
                    type_buffer.extend([type_id] * len(document_tokens))
                    output_mask_buffer.extend(document_mask)

                    document_offsets.append(
                        document_offsets[-1] + len(document_tokens)
                    )
                    document_type_ids.append(type_id)
                    document_ids.append(document_id)
                    type_record_counts[type_id] += 1
                    type_token_counts[type_id] += len(document_tokens)

                documents = []
                if len(token_buffer) >= WRITE_CHUNK_TOKENS:
                    token_count += append_aligned(
                        tokens,
                        token_type_ids,
                        output_mask,
                        token_buffer,
                        type_buffer,
                        output_mask_buffer,
                    )
                    token_buffer = []
                    type_buffer = []
                    output_mask_buffer = []

            for index, serialized_record in enumerate(serialized_records):
                source_record = (
                    metadata_records[index]
                    if metadata_records is not None
                    else None
                )
                text, document_id, type_name = resolve_metadata(
                    serialized_record,
                    source_record,
                    record_number=index + 1,
                )
                if document_id in seen_document_ids:
                    raise ValueError(
                        f"Record {index + 1}: duplicate document id {document_id!r}"
                    )
                seen_document_ids.add(document_id)
                documents.append((text, document_id, type_name))
                record_count += 1
                if len(documents) >= batch_size:
                    encode_documents()

            encode_documents()
            token_count += append_aligned(
                tokens,
                token_type_ids,
                output_mask,
                token_buffer,
                type_buffer,
                output_mask_buffer,
            )

            if expected_count is not None and record_count != expected_count:
                raise ValueError(
                    f"Expected {expected_count:,} records but tokenized "
                    f"{record_count:,}"
                )

            if document_offsets[-1] != token_count:
                raise ValueError(
                    "Document offsets do not end at the final token: "
                    f"{document_offsets[-1]:,} != {token_count:,}"
                )

            hdf5_file.create_dataset(
                "document_offsets",
                data=np.asarray(document_offsets, dtype=np.int64),
            )
            hdf5_file.create_dataset(
                "document_type_ids",
                data=np.asarray(document_type_ids, dtype=np.uint8),
            )
            string_dtype = h5py.string_dtype(encoding="utf-8")
            hdf5_file.create_dataset(
                "document_ids",
                data=np.asarray(document_ids, dtype=object),
                dtype=string_dtype,
            )

            hdf5_file.attrs["tokenizer"] = TOKENIZER_NAME
            hdf5_file.attrs["eot_id"] = EOT_ID
            hdf5_file.attrs["record_count"] = record_count
            hdf5_file.attrs["token_count"] = token_count
            hdf5_file.attrs["schema_version"] = 2
            hdf5_file.attrs["type_names"] = json.dumps(TYPE_NAMES)
            hdf5_file.attrs["type_to_id"] = json.dumps(TYPE_TO_ID)
            hdf5_file.attrs["output_marker"] = OUTPUT_MARKER
            hdf5_file.attrs["output_mask_includes_eot"] = True
            for type_id, type_name in enumerate(TYPE_NAMES):
                hdf5_file.attrs[f"{type_name}_record_count"] = int(
                    type_record_counts[type_id]
                )
                hdf5_file.attrs[f"{type_name}_token_count"] = int(
                    type_token_counts[type_id]
                )

        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return record_count, token_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Serialized JSONL")
    parser.add_argument(
        "--metadata-input",
        type=Path,
        default=None,
        help=(
            "Aligned source JSONL providing id/type when they are absent from "
            "the serialized input"
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 file")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ENCODE_BATCH_SIZE,
        help="Number of documents passed to the tokenizer at once",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=20_000,
        help="Required number of records; use 0 to disable the check",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_count < 0:
        raise ValueError("--expected-count must be nonnegative")

    record_count, token_count = tokenize_jsonl_to_hdf5(
        args.input,
        args.output,
        batch_size=args.batch_size,
        expected_count=args.expected_count or None,
        overwrite=args.overwrite,
        metadata_input=args.metadata_input,
    )
    print(f"Tokenized {record_count:,} records into {token_count:,} tokens")
    print(f"Tokenizer: {TOKENIZER_NAME}")
    print(f"EOT ID:    {EOT_ID}")
    print(f"Schema:    metadata-rich v2 ({', '.join(TYPE_NAMES)})")
    print(f"Output:    {args.output}")


if __name__ == "__main__":
    main()