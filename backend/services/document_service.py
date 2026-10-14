import hashlib
import ipaddress
import logging
import mimetypes
import socket
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import requests
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.database.db import SessionLocal
from backend.models.tables import DocumentChunk, IndexJob, LegalDocument
from backend.services.embedding_service import encode, serialize_embedding
from backend.services.extraction_service import extract_file_text, make_chunks
from backend.services.graph_service import index_document

logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_URL_BYTES = settings.legal_max_upload_mb * 1024 * 1024


def _safe_source_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL import accepts public HTTPS addresses only.")
    host = parsed.hostname.rstrip(".").casefold()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("Private and local network URLs are not allowed.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    except OSError as exc:
        raise ValueError("The source host could not be resolved.") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("Private and local network URLs are not allowed.")
    return value


def _safe_upload_dir() -> Path:
    path = settings.data_path / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _validate_metadata(metadata: dict) -> None:
    start, end = metadata.get("effective_from"), metadata.get("effective_to")
    if start and end and end < start:
        raise HTTPException(status_code=422, detail="effective_to must be on or after effective_from.")
    source_url = metadata.get("source_url")
    if source_url:
        parsed = urlparse(str(source_url))
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise HTTPException(status_code=422, detail="source_url must be a public HTTPS URL without embedded credentials.")


def create_upload_job(db: Session, filename: str, content: bytes, metadata: dict) -> tuple[LegalDocument, IndexJob]:
    _validate_metadata(metadata)
    extension = Path(filename).suffix.casefold()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported file types are PDF, DOCX, and TXT.")
    max_bytes = settings.legal_max_upload_mb * 1024 * 1024
    if not content or len(content) > max_bytes:
        raise HTTPException(status_code=413 if content else 400, detail=f"File must be non-empty and no larger than {settings.legal_max_upload_mb} MB.")
    digest = hashlib.sha256(content).hexdigest()
    duplicate = db.scalar(select(LegalDocument).where(LegalDocument.file_sha256 == digest))
    if duplicate:
        raise HTTPException(status_code=409, detail=f"Duplicate document; it already exists as {duplicate.id}.")

    safe_name = Path(filename).name
    storage_path = (_safe_upload_dir() / f"{uuid4().hex}{extension}").resolve()
    storage_path.write_bytes(content)
    document = LegalDocument(
        title=metadata["title"],
        document_number=metadata.get("document_number"),
        issuing_body=metadata.get("issuing_body"),
        issued_at=metadata.get("issued_at"),
        effective_from=metadata.get("effective_from"),
        effective_to=metadata.get("effective_to"),
        validity_status=metadata.get("validity_status", "unverified"),
        source_url=metadata.get("source_url"),
        source_name=safe_name,
        file_path=str(storage_path),
        file_sha256=digest,
        processing_status="queued",
    )
    return _queue_document(db, document)


def create_url_job(db: Session, url: str, metadata: dict) -> tuple[LegalDocument, IndexJob]:
    _validate_metadata(metadata)
    try:
        safe_url = _safe_source_url(url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    document = LegalDocument(
        title=metadata["title"],
        document_number=metadata.get("document_number"),
        issuing_body=metadata.get("issuing_body"),
        issued_at=metadata.get("issued_at"),
        effective_from=metadata.get("effective_from"),
        effective_to=metadata.get("effective_to"),
        validity_status=metadata.get("validity_status", "unverified"),
        source_url=safe_url,
        source_name=urlparse(safe_url).path.rsplit("/", 1)[-1] or urlparse(safe_url).hostname,
        processing_status="queued",
    )
    return _queue_document(db, document)


def _queue_document(db: Session, document: LegalDocument) -> tuple[LegalDocument, IndexJob]:
    db.add(document)
    db.flush()
    job = IndexJob(document_id=document.id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(document)
    db.refresh(job)
    return document, job


def _download_source(url: str, destination: Path) -> Path:
    response = requests.get(url, stream=True, timeout=(8, 25), allow_redirects=False, headers={"User-Agent": "LegalGraphRAG-local/1.0"})
    if 300 <= response.status_code < 400:
        raise ValueError("The source redirected. Use the final HTTPS document URL directly.")
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";", 1)[0].casefold()
    suffix = Path(urlparse(url).path).suffix.casefold()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix_by_type = {"application/pdf": ".pdf", "text/plain": ".txt", "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"}
        suffix = suffix_by_type.get(content_type, "")
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Source did not return a supported PDF, DOCX, or TXT document.")
    if response.headers.get("content-length") and int(response.headers["content-length"]) > MAX_URL_BYTES:
        raise ValueError(f"Source exceeds the {settings.legal_max_upload_mb} MB limit.")
    try:
        with destination.open("wb") as output:
            size = 0
            for block in response.iter_content(64 * 1024):
                if not block:
                    continue
                size += len(block)
                if size > MAX_URL_BYTES:
                    raise ValueError(f"Source exceeds the {settings.legal_max_upload_mb} MB limit.")
                output.write(block)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    actual_path = destination if destination.suffix.casefold() == suffix else destination.with_suffix(suffix)
    if actual_path != destination:
        destination.replace(actual_path)
    return actual_path


def process_document_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(IndexJob, job_id)
        if not job:
            return
        document = db.get(LegalDocument, job.document_id)
        if not document:
            job.status = "failed"
            job.error = "Document no longer exists."
            db.commit()
            return
        job.status = document.processing_status = "processing"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        if document.source_url and not document.file_path:
            extension = Path(urlparse(document.source_url).path).suffix.casefold()
            if extension not in ALLOWED_EXTENSIONS:
                extension = ".bin"
            path = (_safe_upload_dir() / f"{uuid4().hex}{extension}").resolve()
            path = _download_source(document.source_url, path)
            document.file_path = str(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            duplicate = db.scalar(select(LegalDocument).where(LegalDocument.file_sha256 == digest, LegalDocument.id != document.id))
            if duplicate:
                path.unlink(missing_ok=True)
                raise ValueError(f"Duplicate document; it already exists as {duplicate.id}.")
            document.file_sha256 = digest
            document.source_name = path.name

        if not document.file_path or not Path(document.file_path).is_file():
            raise ValueError("Original file is missing from the local data directory.")
        text = extract_file_text(Path(document.file_path))
        specs = make_chunks(text)
        if not specs:
            raise ValueError("No indexable text chunks were extracted.")
        vectors = encode([spec.content for spec in specs])
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete(synchronize_session=False)
        db.flush()
        chunks: list[DocumentChunk] = []
        seen_chunk_hashes: set[str] = set()
        for spec in specs:
            content_hash = hashlib.sha256(spec.content.encode("utf-8")).hexdigest()
            if content_hash in seen_chunk_hashes:
                continue
            seen_chunk_hashes.add(content_hash)
            vector = vectors[spec.index] if vectors else None
            chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=spec.index,
                article_number=spec.article_number,
                location_label=spec.location_label,
                content=spec.content,
                content_sha256=content_hash,
                embedding_json=serialize_embedding(vector),
            )
            db.add(chunk)
            chunks.append(chunk)
        db.flush()
        index_document(db, document, chunks)
        document.raw_text = text
        document.processing_status = job.status = "completed"
        document.processing_error = job.error = None
        finished = datetime.now(timezone.utc)
        job.finished_at = finished
        db.commit()
        logger.info("Indexed document id=%s chunks=%s", document.id, len(chunks))
    except Exception as exc:
        db.rollback()
        logger.exception("Document indexing failed job_id=%s", job_id)
        job = db.get(IndexJob, job_id)
        if job:
            job.status = "failed"
            job.error = str(exc)[:2000]
            job.finished_at = datetime.now(timezone.utc)
            document = db.get(LegalDocument, job.document_id)
            if document:
                document.processing_status = "failed"
                document.processing_error = str(exc)[:2000]
            db.commit()
    finally:
        db.close()


def reindex_document(db: Session, document: LegalDocument) -> IndexJob:
    if not document.file_path or not Path(document.file_path).is_file():
        raise HTTPException(status_code=409, detail="Original file is unavailable; upload the source again before re-indexing.")
    document.processing_status = "queued"
    document.processing_error = None
    job = IndexJob(document_id=document.id, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def delete_document_storage(document: LegalDocument) -> None:
    if not document.file_path:
        return
    path = Path(document.file_path).resolve()
    raw_root = _safe_upload_dir()
    if path.parent == raw_root and path.is_file():
        path.unlink(missing_ok=True)
