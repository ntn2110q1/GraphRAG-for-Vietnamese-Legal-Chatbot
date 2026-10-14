"""Build and persist configured multilingual sentence embeddings for all chunks."""

from sqlalchemy import select

from backend.core.config import settings
from backend.database.db import SessionLocal, init_db
from backend.models.tables import DocumentChunk
from backend.services.embedding_service import encode, serialize_embedding


def main() -> None:
    if not settings.legal_embedding_model:
        raise SystemExit("Set LEGAL_EMBEDDING_MODEL in .env before building dense vectors.")
    init_db()
    with SessionLocal() as db:
        chunks = db.scalars(select(DocumentChunk).order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)).all()
        batch_size = 32
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            vectors = encode([chunk.content for chunk in batch])
            if vectors is None:
                raise RuntimeError("Embedding model could not be loaded.")
            for chunk, vector in zip(batch, vectors):
                chunk.embedding_json = serialize_embedding(vector)
            db.commit()
            print(f"Embedded {min(offset + len(batch), len(chunks))}/{len(chunks)} chunks")
    print("Vectors stored. Retrieval will use FAISS when faiss-cpu is installed; otherwise NumPy cosine similarity.")


if __name__ == "__main__":
    main()
