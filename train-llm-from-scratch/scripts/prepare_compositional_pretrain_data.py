"""Serialize the compositional-reasoning JSONL as plain-text pretraining documents.

This script performs no tokenization and creates no train/dev/test split. Every valid
input record becomes one output JSONL object with a single ``text`` field, ready for a
separate tokenizer/pretraining-data conversion step.

Example:
    python scripts/prepare_compositional_pretrain_data.py \
        --input ../pretraining_merged_42.jsonl \
        --output /ephemeral/data/compositional_pretrain.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


EXPECTED_FIELDS = {"context", "instruction", "output"}


def serialize_record(record: dict, *, line_number: int) -> str:
    """Validate and render one source record as a plain-text document."""
    missing = EXPECTED_FIELDS.difference(record)
    if missing:
        raise ValueError(
            f"Line {line_number}: missing required field(s): {sorted(missing)}"
        )

    instruction = record["instruction"]
    context = record["context"]
    output = record["output"]

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Line {line_number}: instruction must be a nonempty string")
    # Straightforward examples intentionally have no separate context. Keep the
    # fixed Context section in the serialized text, but allow its value to be empty.
    if not isinstance(context, str):
        raise ValueError(f"Line {line_number}: context must be a string")
    if (
        not isinstance(output, list)
        or not output
        or not all(isinstance(action, str) and action.strip() for action in output)
    ):
        raise ValueError(
            f"Line {line_number}: output must be a nonempty list of nonempty strings"
        )

    actions = " ".join(action.strip() for action in output)
    return (
        f"Instruction:\n{instruction.strip()}\n\n"
        f"Context:\n{context.strip()}\n\n"
        f"Output:\n{actions}"
    )


def serialize_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    expected_count: int | None,
    overwrite: bool,
) -> int:
    """Serialize every nonblank input line and atomically write the output JSONL."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    count = 0

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)

            with input_path.open("r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Line {line_number}: invalid JSON: {error}"
                        ) from error
                    if not isinstance(record, dict):
                        raise ValueError(f"Line {line_number}: expected a JSON object")

                    text = serialize_record(record, line_number=line_number)
                    output_file.write(
                        json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    )
                    count += 1

        if expected_count is not None and count != expected_count:
            raise ValueError(
                f"Expected {expected_count:,} records but serialized {count:,}"
            )

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL containing one {'text': ...} object per record",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=20_000,
        help="Required number of serialized records; use 0 to disable the check",
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
    expected_count = args.expected_count or None

    count = serialize_jsonl(
        args.input,
        args.output,
        expected_count=expected_count,
        overwrite=args.overwrite,
    )
    print(f"Serialized {count:,} records")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("Tokenization: not performed")


if __name__ == "__main__":
    main()