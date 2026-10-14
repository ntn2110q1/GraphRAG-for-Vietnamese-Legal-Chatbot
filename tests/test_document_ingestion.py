import pytest
from fastapi import HTTPException

from backend.services.document_service import create_upload_job, create_url_job


def test_upload_job_rejects_unsupported_file_and_duplicates(monkeypatch, tmp_path, db_session):
    monkeypatch.setattr("backend.services.document_service._safe_upload_dir", lambda: tmp_path)
    metadata = {"title": "TEST FIXTURE ONLY", "validity_status": "unverified"}
    with pytest.raises(HTTPException) as unsupported:
        create_upload_job(db_session, "fixture.exe", b"test body", metadata)
    assert unsupported.value.status_code == 415

    document, job = create_upload_job(db_session, "fixture.txt", b"TEST FIXTURE ONLY source body", metadata)
    assert job.status == "queued"
    assert document.processing_status == "queued"
    with pytest.raises(HTTPException) as duplicate:
        create_upload_job(db_session, "copy.txt", b"TEST FIXTURE ONLY source body", metadata)
    assert duplicate.value.status_code == 409


def test_url_import_rejects_non_https_and_private_addresses(monkeypatch, db_session):
    with pytest.raises(HTTPException) as non_https:
        create_url_job(db_session, "http://example.com/law.pdf", {"title": "TEST FIXTURE ONLY"})
    assert non_https.value.status_code == 422

    monkeypatch.setattr("backend.services.document_service.socket.getaddrinfo", lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 443))])
    with pytest.raises(HTTPException) as private_host:
        create_url_job(db_session, "https://example.com/law.pdf", {"title": "TEST FIXTURE ONLY"})
    assert private_host.value.status_code == 422
