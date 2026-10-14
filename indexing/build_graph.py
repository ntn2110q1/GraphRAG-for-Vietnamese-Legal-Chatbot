"""Rebuild the SQLite-backed NetworkX graph from indexed chunk records."""

from sqlalchemy import select

from backend.database.db import SessionLocal, init_db
from backend.models.tables import DocumentChunk, LegalDocument
from backend.services.graph_service import index_document


def main() -> None:
    init_db()
    with SessionLocal() as db:
        documents = db.scalars(select(LegalDocument).where(LegalDocument.processing_status == "completed")).all()
        for document in documents:
            chunks = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.id).order_by(DocumentChunk.chunk_index)).all()
            index_document(db, document, chunks)
        db.commit()
        print(f"Graph rebuilt from {len(documents)} indexed documents.")


if __name__ == "__main__":
    main()
