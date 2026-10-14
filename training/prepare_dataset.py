"""Validate source-grounded JSONL and create leakage-aware train/validation/test splits.

Input rows need: question, context, answer, source_url or source_document.
For answers that must abstain, include a clear answer such as "Insufficient evidence" and
record why in notes. This script does not invent legal examples.
"""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


REQUIRED = {"question", "context", "answer"}


def _normalize(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", folded).strip()


def _stable_id(row: dict) -> str:
    material = "|".join(_normalize(str(row.get(key, ""))) for key in ("question", "context"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def read_dataset(path: Path) -> list[dict]:
    rows = []
    seen: dict[str, dict] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc.msg}") from exc
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"Line {line_no} misses fields: {', '.join(sorted(missing))}")
        if not all(isinstance(row[key], str) and row[key].strip() for key in REQUIRED):
            raise ValueError(f"Line {line_no} fields question/context/answer must be non-empty strings.")
        if not (row.get("source_url") or row.get("source_document") or row.get("abstention_reason")):
            raise ValueError(f"Line {line_no} needs source_url, source_document, or an abstention_reason.")
        fingerprint = _stable_id(row)
        if fingerprint in seen:
            if _normalize(seen[fingerprint]["answer"]) != _normalize(row["answer"]):
                raise ValueError(f"Line {line_no} has a conflicting answer for a repeated question/context pair.")
            continue
        seen[fingerprint] = row
        row["_group"] = _normalize(str(row.get("source_document") or row.get("source_url") or row["context"]))
        row["_fingerprint"] = fingerprint
        rows.append(row)
    if len(rows) < 3:
        raise ValueError("At least 3 unique, source-grounded examples are required for a three-way split.")
    return rows


def split_dataset(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["_group"], []).append(row)
    if len(groups) < 3:
        raise ValueError("Need at least 3 distinct source-document/context groups to avoid train/test leakage.")
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    train: list[dict] = []
    validation: list[dict] = []
    test: list[dict] = []
    train_target = max(1, int(len(rows) * 0.8))
    validation_target = max(1, int(len(rows) * 0.1))
    for _group, values in group_items:
        if len(train) < train_target:
            train.extend(values)
        elif len(validation) < validation_target:
            validation.extend(values)
        else:
            test.extend(values)
    if not validation or not test:
        # Keep at least one entire source group for each evaluation split.
        train = []
        validation = []
        test = []
        group_count = len(group_items)
        for index, (_group, values) in enumerate(group_items):
            target = "train" if index < group_count - 2 else ("validation" if index == group_count - 2 else "test")
            {"train": train, "validation": validation, "test": test}[target].extend(values)
    output = {"train": train, "validation": validation, "test": test}
    for values in output.values():
        for row in values:
            row.pop("_fingerprint", None)
            row.pop("_group", None)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Source-checked JSONL dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("training/data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_dataset(args.input)
    splits = split_dataset(rows, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
        print(f"{name}: {len(values)} rows -> {path}")
    print("No model was trained. Review the split and source provenance before using train_qlora.py.")


if __name__ == "__main__":
    main()
