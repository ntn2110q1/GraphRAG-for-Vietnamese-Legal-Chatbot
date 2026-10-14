import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.models.tables import ChatMessage, ChatSession, Citation, DocumentChunk, LegalDocument
from backend.services.llm_service import generate_answer
from backend.services.retrieval_service import retrieve

logger = logging.getLogger(__name__)


def ask(db: Session, question: str, session_id: str | None, retrieval_mode: str, top_k: int) -> dict:
    if session_id:
        session = db.scalar(select(ChatSession).where(ChatSession.id == session_id).options(selectinload(ChatSession.messages)))
        if not session:
            raise HTTPException(status_code=404, detail="Conversation was not found.")
    else:
        session = ChatSession(title=question[:80].strip())
        db.add(session)
        db.flush()

    result = retrieve(db, question, top_k, retrieval_mode)
    generated = generate_answer(question, result["evidence"])
    user_message = ChatMessage(session_id=session.id, role="user", content=question, retrieval_mode=retrieval_mode)
    assistant_message = ChatMessage(session_id=session.id, role="assistant", content=generated["answer"], retrieval_mode=retrieval_mode)
    db.add_all([user_message, assistant_message])
    db.flush()
    evidence_by_label = {item["citation_id"]: item for item in result["evidence"]}
    for label in generated["citation_ids"]:
        item = evidence_by_label.get(label)
        if item and db.get(DocumentChunk, item["chunk_id"]):
            db.add(Citation(message_id=assistant_message.id, chunk_id=item["chunk_id"], citation_label=label))
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    db.refresh(assistant_message)
    warnings = list(dict.fromkeys(result["warnings"] + generated["warnings"]))
    return {
        "session_id": session.id,
        "user_message": {"id": user_message.id, "role": "user", "content": question},
        "assistant_message": {"id": assistant_message.id, "role": "assistant", "content": generated["answer"]},
        "citations": [item for item in result["evidence"] if item["citation_id"] in generated["citation_ids"]],
        "retrieved_evidence": result["evidence"],
        "retrieval_mode": retrieval_mode,
        "generation_mode": generated["generation_mode"],
        "metrics": result["metrics"],
        "warnings": warnings,
    }


def serialize_session(session: ChatSession, db: Session | None = None) -> dict:
    messages = []
    for message in sorted(session.messages, key=lambda item: item.created_at):
        details = []
        if db:
            for citation in message.citations:
                chunk = db.get(DocumentChunk, citation.chunk_id)
                if not chunk:
                    continue
                document = db.get(LegalDocument, chunk.document_id)
                if document:
                    details.append({
                        "citation_id": citation.citation_label,
                        "chunk_id": chunk.id,
                        "document_id": document.id,
                        "document_title": document.title,
                        "document_number": document.document_number,
                        "article_number": chunk.article_number,
                        "location_label": chunk.location_label,
                        "source_url": document.source_url,
                        "validity_status": document.validity_status,
                        "text": chunk.content,
                    })
        messages.append({
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "retrieval_mode": message.retrieval_mode,
            "feedback": message.feedback,
            "created_at": message.created_at,
            "citations": [citation.citation_label for citation in message.citations],
            "citation_details": details,
        })
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "messages": messages,
    }
