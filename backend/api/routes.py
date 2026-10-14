import logging
import platform
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, selectinload

from backend.core.config import settings
from backend.core.security import require_admin
from backend.database.db import get_db
from backend.models.tables import (
    ChatMessage,
    ChatSession,
    Citation,
    DocumentChunk,
    Entity,
    IndexJob,
    LegalDocument,
    Relation,
)
from backend.schemas import ChatAskRequest, EvaluationRequest, EvaluationReviewRequest, FeedbackRequest, RenameSessionRequest, RetrievalSettingsRequest, URLImportRequest
from backend.services.chat_service import ask, serialize_session
from backend.services.document_service import (
    create_upload_job,
    create_url_job,
    delete_document_storage,
    process_document_job,
    reindex_document,
)
from backend.services.embedding_service import embedding_status
from backend.services.evaluation_service import load_cases, run_evaluation
from backend.services.graph_service import graph_payload
from backend.services.llm_service import generate_answer, generate_baseline, llm_status
from backend.services.retrieval_service import retrieve

logger = logging.getLogger(__name__)
router = APIRouter()
Db = Annotated[Session, Depends(get_db)]
Admin = Annotated[None, Depends(require_admin)]


def _document_summary(document: LegalDocument) -> dict:
    return {
        "id": document.id,
        "title": document.title,
        "document_number": document.document_number,
        "issuing_body": document.issuing_body,
        "issued_at": document.issued_at,
        "effective_from": document.effective_from,
        "effective_to": document.effective_to,
        "validity_status": document.validity_status,
        "source_url": document.source_url,
        "source_name": document.source_name,
        "processing_status": document.processing_status,
        "processing_error": document.processing_error,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "chunk_count": len(document.chunks),
    }


@router.get("/health")
def health(db: Db) -> dict:
    try:
        db.execute(text("SELECT 1"))
        database_status = "connected"
    except Exception as exc:
        logger.exception("Database health check failed")
        database_status = f"error: {exc}"
    return {"status": "ok" if database_status == "connected" else "degraded", "database": database_status, "llm": llm_status(), "embeddings": embedding_status()}


@router.post("/chat/ask")
def chat_ask(payload: ChatAskRequest, db: Db) -> dict:
    return ask(db, payload.question, payload.session_id, payload.retrieval_mode, payload.top_k or settings.legal_top_k)


@router.get("/chat/sessions")
def list_sessions(db: Db, limit: int = Query(default=50, ge=1, le=200)) -> list[dict]:
    sessions = db.scalars(select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at, "updated_at": s.updated_at} for s in sessions]


@router.get("/chat/sessions/{session_id}")
def get_session(session_id: str, db: Db) -> dict:
    session = db.scalar(select(ChatSession).where(ChatSession.id == session_id).options(selectinload(ChatSession.messages).selectinload(ChatMessage.citations)))
    if not session:
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    return serialize_session(session, db)


@router.patch("/chat/sessions/{session_id}")
def rename_session(session_id: str, payload: RenameSessionRequest, db: Db) -> dict:
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    session.title = payload.title.strip()
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"id": session.id, "title": session.title}


@router.delete("/chat/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, db: Db) -> None:
    session = db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    db.delete(session)
    db.commit()


@router.patch("/chat/messages/{message_id}/feedback")
def message_feedback(message_id: str, payload: FeedbackRequest, db: Db) -> dict:
    message = db.get(ChatMessage, message_id)
    if not message or message.role != "assistant":
        raise HTTPException(status_code=404, detail="Assistant message was not found.")
    message.feedback = payload.value
    db.commit()
    return {"message_id": message.id, "feedback": message.feedback}


@router.get("/documents")
def list_documents(db: Db, q: str | None = None, processing_status: str | None = None, validity_status: str | None = None, limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0)) -> list[dict]:
    query = select(LegalDocument).options(selectinload(LegalDocument.chunks)).order_by(LegalDocument.created_at.desc()).limit(limit).offset(offset)
    if q:
        query = query.where(LegalDocument.title.ilike(f"%{q[:120]}%"))
    if processing_status:
        query = query.where(LegalDocument.processing_status == processing_status)
    if validity_status:
        query = query.where(LegalDocument.validity_status == validity_status)
    return [_document_summary(document) for document in db.scalars(query).all()]


@router.get("/documents/{document_id}")
def get_document(document_id: str, db: Db) -> dict:
    document = db.scalar(select(LegalDocument).where(LegalDocument.id == document_id).options(selectinload(LegalDocument.chunks)))
    if not document:
        raise HTTPException(status_code=404, detail="Document was not found.")
    data = _document_summary(document)
    data["content"] = document.raw_text
    data["chunks"] = [
        {"id": chunk.id, "chunk_index": chunk.chunk_index, "article_number": chunk.article_number, "location_label": chunk.location_label, "content": chunk.content}
        for chunk in sorted(document.chunks, key=lambda item: item.chunk_index)
    ]
    return data


@router.post("/admin/documents/upload", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_admin)])
async def upload_document(
    background_tasks: BackgroundTasks,
    db: Db,
    file: UploadFile = File(...),
    title: str = Form(..., min_length=2, max_length=500),
    document_number: str | None = Form(default=None),
    issuing_body: str | None = Form(default=None),
    issued_at: date | None = Form(default=None),
    effective_from: date | None = Form(default=None),
    effective_to: date | None = Form(default=None),
    validity_status: str = Form(default="unverified"),
    source_url: str | None = Form(default=None),
) -> dict:
    if validity_status not in {"unverified", "verified_active", "expired", "repealed"}:
        raise HTTPException(status_code=422, detail="Unsupported validity status.")
    content = await file.read(settings.legal_max_upload_mb * 1024 * 1024 + 1)
    metadata = {
        "title": title,
        "document_number": document_number,
        "issuing_body": issuing_body,
        "issued_at": issued_at,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "validity_status": validity_status,
        "source_url": source_url,
    }
    document, job = create_upload_job(db, file.filename or "upload", content, metadata)
    background_tasks.add_task(process_document_job, job.id)
    return {"document": _document_summary(document), "job_id": job.id, "job_status": job.status}


@router.post("/admin/documents/import-url", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_admin)])
def import_document_url(payload: URLImportRequest, background_tasks: BackgroundTasks, db: Db) -> dict:
    metadata = payload.model_dump(exclude={"url"})
    document, job = create_url_job(db, str(payload.url), metadata)
    background_tasks.add_task(process_document_job, job.id)
    return {"document": _document_summary(document), "job_id": job.id, "job_status": job.status}


@router.post("/admin/documents/{document_id}/reindex", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_admin)])
def reindex(document_id: str, background_tasks: BackgroundTasks, db: Db) -> dict:
    document = db.get(LegalDocument, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document was not found.")
    job = reindex_document(db, document)
    background_tasks.add_task(process_document_job, job.id)
    return {"document_id": document.id, "job_id": job.id, "job_status": job.status}


@router.delete("/admin/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_admin)])
def delete_document(document_id: str, db: Db) -> None:
    document = db.get(LegalDocument, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document was not found.")
    delete_document_storage(document)
    db.delete(document)
    db.commit()


@router.get("/admin/jobs/{job_id}", dependencies=[Depends(require_admin)])
def get_job(job_id: str, db: Db) -> dict:
    job = db.get(IndexJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Index job was not found.")
    return {"id": job.id, "document_id": job.document_id, "status": job.status, "error": job.error, "created_at": job.created_at, "started_at": job.started_at, "finished_at": job.finished_at}


@router.post("/admin/index/rebuild", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(require_admin)])
def rebuild_index(background_tasks: BackgroundTasks, db: Db) -> dict:
    documents = db.scalars(select(LegalDocument).where(LegalDocument.file_path.is_not(None))).all()
    jobs = []
    for document in documents:
        job = reindex_document(db, document)
        jobs.append(job.id)
        background_tasks.add_task(process_document_job, job.id)
    return {"queued_jobs": jobs, "count": len(jobs)}


@router.get("/admin/jobs/{job_id}/status", dependencies=[Depends(require_admin)])
def get_job_status_alias(job_id: str, db: Db) -> dict:
    return get_job(job_id, db)


@router.get("/graph/nodes")
def get_graph_nodes(db: Db, q: str | None = None, limit: int = Query(default=350, ge=1, le=2000)) -> dict:
    payload = graph_payload(db, q, limit)
    return {"nodes": payload["nodes"], "counts": payload["counts"]}


@router.get("/graph/relationships")
def get_graph_relationships(db: Db, q: str | None = None, limit: int = Query(default=350, ge=1, le=2000)) -> dict:
    payload = graph_payload(db, q, limit)
    return {"relationships": payload["relationships"], "counts": payload["counts"]}


@router.get("/graph/nodes/{entity_id}")
def get_graph_node(entity_id: str, db: Db) -> dict:
    entity = db.get(Entity, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Graph entity was not found.")
    relations = db.scalars(select(Relation).where((Relation.source_entity_id == entity_id) | (Relation.target_entity_id == entity_id))).all()
    chunk = db.get(DocumentChunk, entity.source_chunk_id) if entity.source_chunk_id else None
    document = db.get(LegalDocument, entity.document_id)
    return {
        "node": {"id": entity.id, "name": entity.name, "type": entity.entity_type, "metadata": entity.metadata_json, "document_id": entity.document_id, "chunk_id": entity.source_chunk_id},
        "document": {"id": document.id, "title": document.title, "document_number": document.document_number, "source_url": document.source_url, "validity_status": document.validity_status} if document else None,
        "source_chunk": {"id": chunk.id, "article_number": chunk.article_number, "location_label": chunk.location_label, "content": chunk.content} if chunk else None,
        "relationships": [{"type": rel.relation_type, "source_entity_id": rel.source_entity_id, "target_entity_id": rel.target_entity_id, "evidence_text": rel.evidence_text, "source_chunk_id": rel.source_chunk_id} for rel in relations],
    }


@router.get("/stats/overview")
def overview_stats(db: Db) -> dict:
    return {
        "documents": db.scalar(select(func.count()).select_from(LegalDocument)) or 0,
        "documents_processed": db.scalar(select(func.count()).select_from(LegalDocument).where(LegalDocument.processing_status == "completed")) or 0,
        "documents_failed": db.scalar(select(func.count()).select_from(LegalDocument).where(LegalDocument.processing_status == "failed")) or 0,
        "chunks": db.scalar(select(func.count()).select_from(DocumentChunk)) or 0,
        "entities": db.scalar(select(func.count()).select_from(Entity)) or 0,
        "relationships": db.scalar(select(func.count()).select_from(Relation)) or 0,
        "conversations": db.scalar(select(func.count()).select_from(ChatSession)) or 0,
        "messages": db.scalar(select(func.count()).select_from(ChatMessage)) or 0,
        "evaluation_cases": len(load_cases()),
        "llm": llm_status(),
        "embeddings": embedding_status(),
    }


@router.post("/retrieval/compare")
def compare_retrieval(payload: ChatAskRequest, db: Db) -> dict:
    top_k = payload.top_k or settings.legal_top_k
    vector_result = retrieve(db, payload.question, top_k, "vector")
    graph_result = retrieve(db, payload.question, top_k, "graphrag")
    return {
        "question": payload.question,
        "baseline": generate_baseline(payload.question),
        "vector_rag": {**vector_result, "answer": generate_answer(payload.question, vector_result["evidence"] )},
        "graph_rag": {**graph_result, "answer": generate_answer(payload.question, graph_result["evidence"] )},
    }


@router.post("/evaluation/run", dependencies=[Depends(require_admin)])
def evaluation_run(payload: EvaluationRequest, db: Db) -> dict:
    return run_evaluation(db, payload.retrieval_mode, payload.top_k)


@router.get("/evaluation/results", dependencies=[Depends(require_admin)])
def evaluation_results(db: Db, limit: int = Query(default=250, ge=1, le=1000)) -> dict:
    import json
    from backend.models.tables import EvaluationResult

    rows = db.scalars(select(EvaluationResult).order_by(EvaluationResult.created_at.desc()).limit(limit)).all()
    return {"results": [
        {"id": row.id, "run_id": row.run_id, "question": row.question, "retrieval_mode": row.retrieval_mode, "recall_at_k": row.recall_at_k, "reciprocal_rank": row.reciprocal_rank, "citation_accuracy": row.citation_accuracy, "answer_correctness_score": row.answer_correctness_score, "faithfulness_score": row.faithfulness_score, "review_notes": row.review_notes, "answer_text": row.answer_text, "details": json.loads(row.details_json or "{}"), "latency_ms": row.latency_ms, "created_at": row.created_at}
        for row in rows
    ]}


@router.patch("/evaluation/results/{result_id}/review", dependencies=[Depends(require_admin)])
def review_evaluation_result(result_id: str, payload: EvaluationReviewRequest, db: Db) -> dict:
    from backend.models.tables import EvaluationResult

    row = db.get(EvaluationResult, result_id)
    if not row:
        raise HTTPException(status_code=404, detail="Evaluation result was not found.")
    row.answer_correctness_score = payload.answer_correctness_score
    row.faithfulness_score = payload.faithfulness_score
    row.review_notes = payload.review_notes
    db.commit()
    return {"id": row.id, "answer_correctness_score": row.answer_correctness_score, "faithfulness_score": row.faithfulness_score, "review_notes": row.review_notes}


@router.get("/system/status")
def system_status(db: Db) -> dict:
    device = {"platform": platform.platform(), "cpu_count": __import__("os").cpu_count()}
    try:
        import psutil

        memory = psutil.virtual_memory()
        device["memory_total_gb"] = round(memory.total / (1024 ** 3), 2)
        device["memory_available_gb"] = round(memory.available / (1024 ** 3), 2)
    except ImportError:
        device["memory"] = "Install psutil to report RAM metrics."
    try:
        import torch

        device["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            device["gpu_name"] = torch.cuda.get_device_name(0)
            device["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    except ImportError:
        device["cuda_available"] = None
    return {
        "api": "available",
        "database_url_kind": "SQLite" if settings.database_url.startswith("sqlite") else "external database",
        "documents": db.scalar(select(func.count()).select_from(LegalDocument)) or 0,
        "chunks": db.scalar(select(func.count()).select_from(DocumentChunk)) or 0,
        "graph": {"backend": "NetworkX over SQLite entities and relationships", "nodes": db.scalar(select(func.count()).select_from(Entity)) or 0, "relationships": db.scalar(select(func.count()).select_from(Relation)) or 0},
        "llm": llm_status(),
        "embeddings": embedding_status(),
        "admin_enabled": bool(settings.admin_api_token),
        "training": {"status": "not run", "note": "QLoRA is an independent opt-in pipeline."},
        "device": device,
    }


@router.get("/admin/settings/retrieval", dependencies=[Depends(require_admin)])
def get_retrieval_settings() -> dict:
    return {"default_top_k": settings.legal_top_k, "chunk_size": settings.legal_chunk_size, "chunk_overlap": settings.legal_chunk_overlap, "persistence": "runtime only; set .env to keep across restart"}


@router.put("/admin/settings/retrieval", dependencies=[Depends(require_admin)])
def update_retrieval_settings(payload: RetrievalSettingsRequest) -> dict:
    if payload.chunk_overlap >= payload.chunk_size:
        raise HTTPException(status_code=422, detail="chunk_overlap must be smaller than chunk_size.")
    settings.legal_top_k = payload.default_top_k
    settings.legal_chunk_size = payload.chunk_size
    settings.legal_chunk_overlap = payload.chunk_overlap
    return get_retrieval_settings()


@router.get("/admin/logs", dependencies=[Depends(require_admin)])
def recent_logs(limit: int = Query(default=200, ge=1, le=1000)) -> dict:
    path = settings.data_path / "logs" / "app.log"
    if not path.exists():
        return {"lines": []}
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        return {"lines": list(deque(log_file, maxlen=limit))}
