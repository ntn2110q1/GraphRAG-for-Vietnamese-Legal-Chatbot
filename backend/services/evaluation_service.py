import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.models.tables import DocumentChunk, EvaluationResult, LegalDocument
from backend.services.llm_service import generate_answer
from backend.services.retrieval_service import retrieve

logger = logging.getLogger(__name__)


def load_cases() -> list[dict]:
    path = Path(settings.legal_eval_dataset)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists():
        return []
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSONL at line {line_no}: {exc.msg}") from exc
        if not isinstance(case, dict) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise HTTPException(status_code=422, detail=f"Evaluation case at line {line_no} needs a non-empty question.")
        if "relevant_chunk_ids" in case and not isinstance(case["relevant_chunk_ids"], list):
            raise HTTPException(status_code=422, detail=f"relevant_chunk_ids must be a list at line {line_no}.")
        if "relevant_article_numbers" in case and not isinstance(case["relevant_article_numbers"], list):
            raise HTTPException(status_code=422, detail=f"relevant_article_numbers must be a list at line {line_no}.")
        if case.get("relevant_chunk_ids") and case.get("relevant_article_numbers"):
            raise HTTPException(status_code=422, detail=f"Use either relevant_chunk_ids or relevant_article_numbers, not both, at line {line_no}.")
        if not (case.get("source_url") or case.get("source_document") or case.get("expected_behavior") == "abstain"):
            raise HTTPException(status_code=422, detail=f"Evaluation case at line {line_no} needs source provenance or expected_behavior=abstain.")
        if not (case.get("relevant_chunk_ids") or case.get("relevant_article_numbers") or case.get("expected_behavior") == "abstain"):
            raise HTTPException(status_code=422, detail=f"Evaluation case at line {line_no} needs labeled chunk/article relevance or expected_behavior=abstain.")
        cases.append(case)
    return cases


def run_evaluation(db: Session, mode: str, top_k: int) -> dict:
    cases = load_cases()
    if not cases:
        raise HTTPException(status_code=409, detail="Evaluation dataset has no labeled questions. Add source-checked JSONL cases first.")
    modes = ["vector", "graphrag"] if mode == "both" else [mode]
    run_id = str(uuid4())
    rows = []
    for case in cases:
        relevant = set(case.get("relevant_chunk_ids", []))
        article_numbers = [str(value) for value in case.get("relevant_article_numbers", [])]
        if article_numbers:
            source_document_number = case.get("relevant_document_number")
            query = db.query(DocumentChunk.id).join(LegalDocument, LegalDocument.id == DocumentChunk.document_id).filter(DocumentChunk.article_number.in_(article_numbers))
            if source_document_number:
                query = query.filter(LegalDocument.document_number == source_document_number)
            resolved_ids = {value for (value,) in query.all()}
            if not resolved_ids:
                raise HTTPException(status_code=409, detail=f"Evaluation labels for case {case.get('id', case['question'][:40])} do not resolve to indexed article chunks.")
            relevant.update(resolved_ids)
        for retrieval_mode in modes:
            started = time.perf_counter()
            result = retrieve(db, case["question"], top_k, retrieval_mode)
            latency_ms = (time.perf_counter() - started) * 1000
            retrieved = [item["chunk_id"] for item in result["evidence"]]
            answer_result = generate_answer(case["question"], result["evidence"])
            cited = {item["citation_id"] for item in answer_result["citations"]} if "citations" in answer_result else set(answer_result.get("citation_ids", []))
            evidence_by_label = {item["citation_id"]: item for item in result["evidence"]}
            cited_items = [evidence_by_label[label] for label in cited if label in evidence_by_label]
            cited_chunk_ids = {item["chunk_id"] for item in cited_items}
            if article_numbers:
                retrieved_articles = {str(item.get("article_number")) for item in result["evidence"] if item.get("article_number")}
                recall = len(set(article_numbers).intersection(retrieved_articles)) / len(set(article_numbers))
                rank = next((idx for idx, item in enumerate(result["evidence"], 1) if str(item.get("article_number")) in article_numbers), None)
                cited_relevant_count = sum(1 for item in cited_items if str(item.get("article_number")) in article_numbers)
                citation_accuracy = cited_relevant_count / len(cited_items) if cited_items else None
            else:
                recall = len(relevant.intersection(retrieved)) / len(relevant) if relevant else None
                rank = next((idx for idx, chunk_id in enumerate(retrieved, 1) if chunk_id in relevant), None)
                citation_accuracy = (len(cited_chunk_ids.intersection(relevant)) / len(cited_chunk_ids)) if relevant and cited_chunk_ids else None
            reciprocal_rank = (1 / rank) if rank and (relevant or article_numbers) else None
            row = EvaluationResult(
                run_id=run_id,
                question=case["question"],
                retrieval_mode=retrieval_mode,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                citation_accuracy=citation_accuracy,
                answer_text=answer_result.get("answer"),
                latency_ms=latency_ms,
                details_json=json.dumps({
                    "retrieved_chunk_ids": retrieved,
                    "cited_chunk_ids": sorted(cited_chunk_ids),
                    "relevant_chunk_ids": sorted(relevant),
                    "relevant_article_numbers": article_numbers,
                    "expected_answer": case.get("expected_answer"),
                    "generation_mode": answer_result.get("generation_mode"),
                    "warnings": result["warnings"] + answer_result.get("warnings", []),
                    "sources": [{key: item.get(key) for key in ("citation_id", "chunk_id", "document_title", "document_number", "location_label", "source_url", "validity_status", "text")} for item in result["evidence"]],
                }, ensure_ascii=False),
            )
            db.add(row)
            rows.append({
                "case_id": case.get("id"),
                "question": case["question"],
                "retrieval_mode": retrieval_mode,
                "retrieved_chunk_ids": retrieved,
                "recall_at_k": recall,
                "reciprocal_rank": reciprocal_rank,
                "citation_accuracy": citation_accuracy,
                "cited_chunk_ids": sorted(cited_chunk_ids),
                "answer_text": answer_result.get("answer"),
                "expected_answer": case.get("expected_answer"),
                "sources": [{key: item.get(key) for key in ("citation_id", "chunk_id", "document_title", "document_number", "location_label", "source_url", "validity_status", "text")} for item in result["evidence"]],
                "latency_ms": round(latency_ms, 2),
                "warnings": result["warnings"],
            })
    db.commit()
    aggregates = {}
    for retrieval_mode in modes:
        subset = [row for row in rows if row["retrieval_mode"] == retrieval_mode]
        aggregates[retrieval_mode] = {
            "questions": len(subset),
            "mean_recall_at_k": _mean(row["recall_at_k"] for row in subset),
            "mean_mrr": _mean(row["reciprocal_rank"] for row in subset),
            "mean_citation_accuracy": _mean(row["citation_accuracy"] for row in subset),
            "mean_latency_ms": _mean(row["latency_ms"] for row in subset),
        }
    return {"run_id": run_id, "dataset_cases": len(cases), "top_k": top_k, "aggregates": aggregates, "results": rows}


def _mean(values):
    materialized = [value for value in values if value is not None]
    return round(sum(materialized) / len(materialized), 4) if materialized else None
