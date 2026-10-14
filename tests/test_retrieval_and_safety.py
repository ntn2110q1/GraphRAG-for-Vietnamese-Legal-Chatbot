from fastapi import HTTPException

from backend.core.config import settings
from backend.models.tables import DocumentChunk, LegalDocument
from backend.services.llm_service import generate_answer, validate_citations
from backend.services.retrieval_service import retrieve
from backend.core.security import require_admin


def test_empty_evidence_abstains_instead_of_inventing_a_legal_answer():
    result = generate_answer("TEST QUESTION ONLY", [])
    assert result["generation_mode"] == "abstain_no_evidence"
    assert result["citation_ids"] == []
    assert "không có đoạn văn bản phù hợp" in result["answer"].lower()


def test_invalid_model_citation_is_removed_and_flagged():
    answer, accepted, warnings = validate_citations("Text [C1] and [C99].", ["C1", "C99"], {"C1"})
    assert accepted == ["C1"]
    assert "[C99]" not in answer
    assert warnings


def test_expired_documents_are_not_used_for_unscoped_current_questions(db_session):
    document = LegalDocument(title="TEST FIXTURE ONLY", processing_status="completed", validity_status="expired")
    db_session.add(document)
    db_session.flush()
    db_session.add(DocumentChunk(document_id=document.id, chunk_index=0, content="TEST FIXTURE ONLY token alpha", content_sha256="test-expired"))
    db_session.commit()
    result = retrieve(db_session, "token alpha", 3, "vector")
    assert result["evidence"] == []


def test_admin_token_is_required(monkeypatch):
    monkeypatch.setattr(settings, "admin_api_token", "test-secret")
    try:
        require_admin(x_admin_token="wrong")
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        raise AssertionError("Invalid token was accepted")
