# Evaluation data

`eval.jsonl` contains three source-grounded questions based on the bundled official consolidated Labor Code (125/VBHN-VPQH). Labels use document number plus article number, which the evaluator resolves to the indexed chunks at runtime. Each expected answer is a short human-review reference; it is not a generated gold answer or a claim that the 2025 consolidation reflects every later amendment.

Schema:

```json
{"id":"case-001","question":"...","relevant_document_number":"125/VBHN-VPQH","relevant_article_numbers":["113"],"expected_answer":"...","source_document":"Official document title and number","source_url":"https://...","notes":"Version and verification scope"}
```

For an unanswerable question, set `expected_behavior` to `abstain` and include a `source_document` or `source_url` describing the checked corpus/scope. `relevant_chunk_ids` may be used for directly chunk-labeled cases. The evaluator reports retrieval recall@K, MRR, citation precision when a generation model returns citations, and latency. Answer correctness and faithfulness require manual review in the app.
