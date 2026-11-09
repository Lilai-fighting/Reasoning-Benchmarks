"""Convert raw JSONL to model tokens and separate, aligned evaluation metadata.

Run from train-llm-from-scratch:
  python scripts/convert_length9_12_hdf5.py
  python scripts/convert_length9_12_hdf5.py --verify-only
Only the five MODEL_FIELDS appear in tokenized text. Metadata is never tokenized.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from sample_length9_12_straightforward import write_new

MODEL_FIELDS = ("context", "instruction", "intervention", "proportion", "output")
MARKER = "\n\nOutput:\n"


def serialize(row):
    for name in MODEL_FIELDS[:3]:
        if not isinstance(row[name], str):
            raise ValueError(f"{name} must be a string")
    if type(row["proportion"]) not in (int, float) or not 0 < row["proportion"] <= 1:
        raise ValueError("Invalid proportion")
    if not isinstance(row["output"], list) or not row["output"] or not all(isinstance(t, str) for t in row["output"]):
        raise ValueError("Invalid output")
    return (f"Context:\n{row['context']}\n\nInstruction:\n{row['instruction']}"
            f"\n\nIntervention:\n{row['intervention']}\n\nProportion:\n{row['proportion']}"
            + MARKER + " ".join(row["output"]))


def verify(path, metadata_path, rows):
    enc = tiktoken.get_encoding("r50k_base")
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    with h5py.File(path, "r") as h:
        names = json.loads(h.attrs["type_names"])
        offsets = h["document_offsets"][:]
        ids = h["document_ids"].asstr()[:]
        types = h["document_type_ids"][:]
        tokens = h["tokens"][:]
        masks = h["output_mask"][:]
        if len(ids) != len(rows) or len(metadata) != len(rows) or len(offsets) != len(rows)+1:
            raise ValueError("Incorrect document count")
        assert offsets[0] == 0 and offsets[-1] == len(tokens) == len(masks)
        for i, row in enumerate(rows):
            start, end = map(int, offsets[i:i+2])
            expected_map = dict(document_index=i, id=row["id"], type=row["type"],
                                type_id=int(types[i]), token_start=start, token_end=end)
            assert metadata[i] == expected_map
            assert ids[i] == row["id"] and names[int(types[i])] == row["type"]
            assert tokens[end-1] == enc.eot_token
            assert enc.decode(tokens[start:end-1].tolist()) == serialize(row)
            prefix = serialize(row).rsplit(MARKER, 1)[0] + MARKER
            boundary = start + len(enc.encode_ordinary(prefix))
            assert np.all(masks[start:boundary] == 0) and np.all(masks[boundary:end] == 1)
            assert enc.decode(tokens[boundary:end-1].tolist()) == " ".join(row["output"])
        return dict(passed=True, samples=len(rows), tokens=len(tokens),
                    maximum_document_tokens=int(np.diff(offsets).max()),
                    model_fields=list(MODEL_FIELDS), metadata_separate=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT.parent / "length9_12_train_42.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "ephemeral/data/length9_12_train_42.h5")
    parser.add_argument("--expected-count", type=int, default=9600)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != args.expected_count or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Wrong sample count or duplicate IDs")
    metadata_path = args.output.with_suffix(".metadata.jsonl")
    report = args.output.with_suffix(".report.json")
    if not args.verify_only:
        if any(p.exists() for p in (args.output, metadata_path, report)):
            raise FileExistsError("Output or metadata/report exists; use --verify-only or a new output path")
        enc = tiktoken.get_encoding("r50k_base")
        names = sorted({r["type"] for r in rows})
        if len(names) > 256:
            raise ValueError("Too many types")
        tokens, masks, type_ids, offsets, mapping = [], [], [], [0], []
        for i, row in enumerate(rows):
            text = serialize(row)
            prefix = text.rsplit(MARKER, 1)[0] + MARKER
            encoded, prefix_ids = enc.encode_ordinary(text), enc.encode_ordinary(prefix)
            assert encoded[:len(prefix_ids)] == prefix_ids
            encoded.append(enc.eot_token)
            if len(encoded) > args.context_length:
                raise ValueError(f"Sample {row['id']} exceeds context length")
            type_id = names.index(row["type"])
            tokens.extend(encoded)
            masks.extend([0]*len(prefix_ids) + [1]*(len(encoded)-len(prefix_ids)))
            type_ids.append(type_id)
            offsets.append(len(tokens))
            mapping.append(dict(document_index=i, id=row["id"], type=row["type"],
                                type_id=type_id, token_start=offsets[-2], token_end=offsets[-1]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "x") as h:
            for name, data, dtype in (("tokens", tokens, "i4"), ("output_mask", masks, "u1"),
                    ("document_offsets", offsets, "i8"), ("document_type_ids", type_ids, "u1")):
                h.create_dataset(name, data=np.asarray(data, dtype=dtype))
            h.create_dataset("document_ids", data=[r["id"] for r in rows], dtype=h5py.string_dtype("utf-8"))
            h.attrs.update(tokenizer="r50k_base", eot_id=enc.eot_token, schema_version=2,
                           record_count=len(rows), token_count=len(tokens), type_names=json.dumps(names),
                           model_fields=json.dumps(MODEL_FIELDS), output_mask_includes_eot=True)
        write_new(metadata_path, "".join(json.dumps(m)+"\n" for m in mapping))
    result = verify(args.output, metadata_path, rows)
    if not args.verify_only:
        write_new(report, json.dumps(dict(source=str(args.input.resolve()),
                  source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  hdf5_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  verification=result), indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
