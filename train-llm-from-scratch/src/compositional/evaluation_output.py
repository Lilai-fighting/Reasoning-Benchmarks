"""Publish rank-local evaluation streams as one validated JSONL per epoch."""
import json
import os
from pathlib import Path
import tempfile
from contextlib import ExitStack
from itertools import zip_longest


def merge_rank_outputs(paths, output, expected_ids):
    """Undo strided rank sharding, verify IDs, and atomically publish the result."""
    output = Path(output)
    staged = None
    try:
        with ExitStack() as stack:
            handles = [stack.enter_context(Path(path).open(encoding="utf-8")) for path in paths]
            target = stack.enter_context(tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="\n", dir=output.parent, delete=False))
            staged = Path(target.name)
            count = 0
            for group in zip_longest(*handles):
                for line in group:
                    if line is None:
                        continue
                    row = json.loads(line)
                    if count >= len(expected_ids) or row["id"] != expected_ids[count]:
                        raise ValueError(f"Unexpected/missing/duplicate evaluation ID at index {count}")
                    target.write(line.rstrip("\r\n") + "\n")
                    count += 1
            if count != len(expected_ids):
                raise ValueError(f"Expected {len(expected_ids)} results, found {count}")
        os.replace(staged, output)
        for path in paths:
            Path(path).unlink()
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
