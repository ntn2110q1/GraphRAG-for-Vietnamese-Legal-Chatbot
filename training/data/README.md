# QLoRA data

No training examples are shipped. This prevents sample legal answers from being mistaken for verified law. Prepare a source-checked JSONL file with `question`, `context`, `answer`, and `source_url` or `source_document`; include an `evidence` description and cite the exact checked text. If the correct behavior is abstention, include `abstention_reason`.

Run `python training/prepare_dataset.py <source-checked.jsonl>`. It deduplicates question/context pairs and keeps examples from one source document in the same split. Review all three output splits before training.
