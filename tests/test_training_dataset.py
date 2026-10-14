import pytest

from training.prepare_dataset import read_dataset, split_dataset


def test_training_split_groups_rows_by_source_document(tmp_path):
    path = tmp_path / "train_input.jsonl"
    rows = []
    for source in ("source-a", "source-b", "source-c"):
        for index in range(2):
            rows.append({"question": f"q {source} {index}", "context": f"context {source}", "answer": f"answer {index}", "source_document": source})
    import json

    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    splits = split_dataset(read_dataset(path), seed=2)
    source_splits = {}
    for split_name, values in splits.items():
        for row in values:
            source_splits.setdefault(row["source_document"], set()).add(split_name)
    assert all(len(names) == 1 for names in source_splits.values())
    assert set(source_splits) == {"source-a", "source-b", "source-c"}


def test_conflicting_training_answers_for_same_question_context_are_rejected(tmp_path):
    import json

    path = tmp_path / "conflicting.jsonl"
    rows = [
        {"question": "same", "context": "same", "answer": "one", "source_document": "source-a"},
        {"question": "same", "context": "same", "answer": "two", "source_document": "source-b"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting answer"):
        read_dataset(path)
