import json
import logging
from functools import lru_cache

import numpy as np

from backend.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    if not settings.legal_embedding_model:
        return None
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.legal_embedding_model, device="cpu")


def encode(texts: list[str]) -> list[list[float]] | None:
    model = _model()
    if model is None:
        return None
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32).tolist()


def serialize_embedding(vector: list[float] | None) -> str | None:
    return json.dumps(vector, separators=(",", ":")) if vector is not None else None


def dense_scores(query: str, chunks: list) -> dict[str, float] | None:
    stored = [(chunk.id, chunk.embedding_json) for chunk in chunks if chunk.embedding_json]
    if not stored or not settings.legal_embedding_model:
        return None


def vector_scores(query: str, chunks: list) -> tuple[dict[str, float], str]:
    dense = dense_scores(query, chunks)
    if dense is not None:
        return dense, "FAISS dense vectors" if _faiss_available() else "NumPy dense vectors"
    from sklearn.feature_extraction.text import TfidfVectorizer

    from backend.services.retrieval_service import tokens

    if not chunks:
        return {}, "TF-IDF sparse vectors"
    vectorizer = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False, sublinear_tf=True)
    matrix = vectorizer.fit_transform([chunk.content for chunk in chunks])
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    return {chunk.id: float(scores[index]) for index, chunk in enumerate(chunks)}, "TF-IDF sparse vectors"
    try:
        query_vector = encode([query])
        if not query_vector:
            return None
        ids = [item[0] for item in stored]
        matrix = np.asarray([json.loads(item[1]) for item in stored], dtype=np.float32)
        query_array = np.asarray(query_vector, dtype=np.float32)
        try:
            import faiss

            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            values, positions = index.search(query_array, len(ids))
            return {ids[int(pos)]: float(score) for score, pos in zip(values[0], positions[0]) if pos >= 0}
        except ImportError:
            scores = matrix @ query_array[0]
            return {ids[i]: float(scores[i]) for i in np.argsort(scores)[::-1]}
    except Exception:
        logger.exception("Dense retrieval unavailable; using lexical retrieval for this request")
        return None


def embedding_status() -> dict:
    if not settings.legal_embedding_model:
        return {"configured": False, "backend": "TF-IDF sparse vectors + BM25", "model": None}
    try:
        import sentence_transformers  # noqa: F401

        loaded = _model.cache_info().currsize > 0
        if not loaded:
            return {"configured": True, "available": False, "can_attempt_load": True, "loaded": False, "backend": "dense model configured; not loaded yet", "model": settings.legal_embedding_model}
        model = _model()
        return {
            "configured": True,
            "available": True,
            "loaded": True,
            "backend": "FAISS cosine similarity" if _faiss_available() else "NumPy cosine similarity",
            "model": settings.legal_embedding_model,
            "dimensions": int(model.get_sentence_embedding_dimension()),
        }
    except Exception as exc:
        return {"configured": True, "available": False, "can_attempt_load": False, "model": settings.legal_embedding_model, "error": str(exc)}


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False
