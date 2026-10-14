import re
import unicodedata
import math
import time
from collections import Counter
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.models.tables import DocumentChunk, LegalDocument
from backend.services.embedding_service import vector_scores
from backend.services.graph_service import expand_chunk_ids


STOP_WORDS = {
    "va", "la", "cua", "cho", "toi", "theo", "trong", "nhu", "duoc", "co", "nhung", "cac", "mot", "nao", "bao",
    "nhieu", "the", "thi", "tu", "den", "voi", "hay", "gi", "khong", "nguoi", "lao", "dong", "luat", "quy", "dinh",
    "theo", "tai", "ve", "khi", "doi", "voi", "phap", "ly", "nam", "ngay", "thang",
}


def fold(text: str) -> str:
    text = text.casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def tokens(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", fold(text)) if len(term) > 1 and term not in STOP_WORDS]


def is_historical_query(question: str) -> bool:
    folded = fold(question)
    if re.search(r"(?i)\b(truoc day|lich su|trong qua khu|tai thoi diem|quy dinh cu)\b", folded):
        return True
    years = [int(value) for value in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
    return any(year < date.today().year for year in years)


def _eligible(document: LegalDocument, historical: bool, requested_years: list[int]) -> tuple[bool, str | None]:
    if document.validity_status in {"expired", "repealed"} and not historical:
        return False, None
    if requested_years:
        if document.effective_from and all(year < document.effective_from.year for year in requested_years):
            return False, None
        if document.effective_to and all(year > document.effective_to.year for year in requested_years):
            return False, None
    if document.validity_status == "unverified":
        return True, f"Tình trạng hiệu lực của '{document.title}' chưa được xác minh trong metadata."
    if document.validity_status in {"expired", "repealed"}:
        return True, f"'{document.title}' được đánh dấu {document.validity_status}; chỉ đưa vào vì câu hỏi có phạm vi lịch sử."
    return True, None


def _bm25_scores(question: str, chunks: list[DocumentChunk]) -> dict[str, float]:
    query_terms = tokens(question)
    if not query_terms:
        return {}
    chunk_terms = {chunk.id: tokens(chunk.content) for chunk in chunks}
    avg_len = sum(map(len, chunk_terms.values())) / max(1, len(chunk_terms))
    document_frequency = Counter(term for terms in chunk_terms.values() for term in set(terms))
    total = len(chunks)
    scores: dict[str, float] = {}
    article_numbers = re.findall(r"(?i)\bĐiều\s+(\d+[a-zA-Z]?)", question)
    for chunk in chunks:
        terms = chunk_terms[chunk.id]
        frequencies = Counter(terms)
        score = 0.0
        for term in set(query_terms):
            df = document_frequency[term]
            inverse = max(0.0, math.log(1 + (total - df + 0.5) / (df + 0.5)))
            frequency = frequencies[term]
            denominator = frequency + 1.2 * (1 - 0.75 + 0.75 * len(terms) / max(1, avg_len))
            score += inverse * ((frequency * 2.2) / denominator) if frequency else 0.0
        if chunk.article_number and chunk.article_number in article_numbers:
            score += 4.0
        scores[chunk.id] = score
    maximum = max(scores.values(), default=0)
    return {chunk_id: score / maximum if maximum else 0.0 for chunk_id, score in scores.items()}


def retrieve(db: Session, question: str, top_k: int, mode: str = "graphrag") -> dict:
    started = time.perf_counter()
    documents = {doc.id: doc for doc in db.scalars(select(LegalDocument).where(LegalDocument.processing_status == "completed")).all()}
    historical = is_historical_query(question)
    years = [int(value) for value in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
    chunks = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id.in_(list(documents))).order_by(DocumentChunk.chunk_index)).all() if documents else []
    eligible_chunks: list[DocumentChunk] = []
    warnings: list[str] = []
    for chunk in chunks:
        doc = documents.get(chunk.document_id)
        if not doc:
            continue
        eligible, warning = _eligible(doc, historical, years)
        if eligible:
            eligible_chunks.append(chunk)
            if warning and warning not in warnings:
                warnings.append(warning)
    if not eligible_chunks:
        return {
            "mode": mode,
            "evidence": [],
            "warnings": warnings,
            "metrics": {"chunks_retrieved": 0, "graph_nodes_used": 0, "graph_relationships_used": 0, "retrieval_backend": "TF-IDF sparse vectors + BM25", "historical_scope": historical, "retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 2)},
        }

    lexical = _bm25_scores(question, eligible_chunks)
    vectors, vector_backend = vector_scores(question, eligible_chunks)
    vector_normalized: dict[str, float] = {}
    if vectors:
        low, high = min(vectors.values()), max(vectors.values())
        span = high - low or 1.0
        vector_normalized = {key: (value - low) / span for key, value in vectors.items()}
    retrieval_backend = f"{vector_backend} + BM25"
    base_scores = {
        chunk.id: 0.85 * vector_normalized.get(chunk.id, 0.0) + 0.15 * lexical.get(chunk.id, 0.0)
        for chunk in eligible_chunks
    }
    ranked = sorted(eligible_chunks, key=lambda item: base_scores[item.id], reverse=True)

    graph_nodes = graph_edges = 0
    graph_scores: dict[str, float] = {}
    if mode == "graphrag":
        seed_count = min(max(top_k, 3), len(ranked))
        graph_scores, graph_nodes, graph_edges = expand_chunk_ids(db, [chunk.id for chunk in ranked[:seed_count]], question)
        all_by_id = {chunk.id: chunk for chunk in eligible_chunks}
        for chunk_id in graph_scores:
            if chunk_id in all_by_id and chunk_id not in base_scores:
                ranked.append(all_by_id[chunk_id])
                base_scores[chunk_id] = 0.0
        ranked.sort(key=lambda item: base_scores.get(item.id, 0.0) + graph_scores.get(item.id, 0.0), reverse=True)
    selected = ranked[:top_k]
    evidence = []
    for index, chunk in enumerate(selected, start=1):
        doc = documents[chunk.document_id]
        evidence.append({
            "citation_id": f"C{index}",
            "chunk_id": chunk.id,
            "document_id": doc.id,
            "document_title": doc.title,
            "document_number": doc.document_number,
            "article_number": chunk.article_number,
            "location_label": chunk.location_label,
            "source_url": doc.source_url,
            "source_name": doc.source_name,
            "validity_status": doc.validity_status,
            "effective_from": doc.effective_from.isoformat() if doc.effective_from else None,
            "effective_to": doc.effective_to.isoformat() if doc.effective_to else None,
            "text": chunk.content,
            "score": round(base_scores.get(chunk.id, 0.0) + graph_scores.get(chunk.id, 0.0), 4),
            "graph_expanded": chunk.id in graph_scores,
        })
    metrics = {
        "chunks_retrieved": len(evidence),
        "graph_nodes_used": graph_nodes if mode == "graphrag" else 0,
        "graph_relationships_used": graph_edges if mode == "graphrag" else 0,
        "graph_expanded_chunks": sum(1 for item in evidence if item["graph_expanded"]),
        "retrieval_backend": retrieval_backend,
        "graph_backend": "SQLite-backed NetworkX" if mode == "graphrag" else None,
        "historical_scope": historical,
        "retrieval_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if evidence and all(item["score"] <= 0 for item in evidence):
        warnings.append("Không tìm thấy mức độ trùng khớp đáng kể; hãy kiểm tra nguồn hoặc bổ sung tài liệu phù hợp.")
    return {"mode": mode, "evidence": evidence, "warnings": warnings, "metrics": metrics}
