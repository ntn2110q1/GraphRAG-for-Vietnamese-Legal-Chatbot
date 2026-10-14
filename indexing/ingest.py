"""Run indexing for a local file through the same extraction and graph pipeline as the API."""

import argparse
import hashlib
from datetime import date
from pathlib import Path

from backend.database.db import SessionLocal, init_db
from backend.models.tables import IndexJob, LegalDocument
from backend.services.document_service import process_document_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--number")
    parser.add_argument("--issuing-body")
    parser.add_argument("--issued-at", type=date.fromisoformat)
    parser.add_argument("--effective-from", type=date.fromisoformat)
    parser.add_argument("--effective-to", type=date.fromisoformat)
    parser.add_argument("--source-url")
    parser.add_argument("--validity", choices=["unverified", "verified_active", "expired", "repealed"], default="unverified")
    args = parser.parse_args()
    source = args.file.resolve()
    if not source.is_file() or source.suffix.casefold() not in {".pdf", ".docx", ".txt"}:
        raise SystemExit("Input must be an existing PDF, DOCX, or TXT file.")
    init_db()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with SessionLocal() as db:
        existing = db.query(LegalDocument).filter_by(file_sha256=digest).first()
        if existing and existing.processing_status == "completed":
            print(f"document={existing.id} status=completed (same source already indexed)")
            return
        if existing:
            doc = existing
            doc.title = args.title
            doc.document_number = args.number
            doc.issuing_body = args.issuing_body
            doc.issued_at = args.issued_at
            doc.effective_from = args.effective_from
            doc.effective_to = args.effective_to
            doc.source_url = args.source_url
            doc.source_name = source.name
            doc.file_path = str(source)
            doc.validity_status = args.validity
            doc.processing_status = "queued"
        else:
            doc = LegalDocument(
                title=args.title,
                document_number=args.number,
                issuing_body=args.issuing_body,
                issued_at=args.issued_at,
                effective_from=args.effective_from,
                effective_to=args.effective_to,
                source_url=args.source_url,
                source_name=source.name,
                file_path=str(source),
                file_sha256=digest,
                validity_status=args.validity,
                processing_status="queued",
            )
        db.add(doc)
        db.flush()
        job = IndexJob(document_id=doc.id, status="queued")
        db.add(job)
        db.commit()
        job_id = job.id
    process_document_job(job_id)
    with SessionLocal() as db:
        job = db.get(IndexJob, job_id)
        print(f"job={job.id} status={job.status} error={job.error or ''}")
        if job.status != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
