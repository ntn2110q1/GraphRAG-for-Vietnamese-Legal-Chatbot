"""Tests use explicitly synthetic TEST FIXTURE ONLY text, never legal source material."""

from backend.models.tables import DocumentChunk, LegalDocument
from backend.services.extraction_service import make_chunks
from backend.services.graph_service import expand_chunk_ids, graph_payload, index_document


def test_chunking_keeps_explicit_article_labels():
    text = "TEST FIXTURE ONLY intro paragraph long enough\n\nĐiều 1. Placeholder heading\nFixture body one.\n\nĐiều 2. Another placeholder\nFixture body two."
    chunks = make_chunks(text)
    assert [chunk.article_number for chunk in chunks] == [None, "1", "2"]
    assert chunks[1].content.startswith("Điều 1.")
    assert chunks[2].content.startswith("Điều 2.")


def test_graph_expands_only_an_explicit_reference(db_session):
    source = "TEST FIXTURE ONLY\n\nĐiều 1. Placeholder one\nFixture source text alpha.\n\nĐiều 2. Placeholder two\nFixture text explicitly refers to Điều 1 and mentions nghỉ hằng năm.\n\nĐiều 3. Unrelated placeholder\nFixture also mentions nghỉ hằng năm as an independent rule."
    document = LegalDocument(title="TEST FIXTURE ONLY", raw_text=source, processing_status="completed", validity_status="unverified")
    db_session.add(document)
    db_session.flush()
    chunks = []
    for spec in make_chunks(source):
        chunk = DocumentChunk(document_id=document.id, chunk_index=spec.index, article_number=spec.article_number, location_label=spec.location_label, content=spec.content, content_sha256=f"test-{spec.index}")
        db_session.add(chunk)
        chunks.append(chunk)
    db_session.flush()
    index_document(db_session, document, chunks)
    db_session.commit()

    by_article = {chunk.article_number: chunk for chunk in chunks if chunk.article_number}
    expanded, node_count, edge_count = expand_chunk_ids(db_session, [by_article["2"].id], "Điều 2")
    assert by_article["1"].id in expanded
    assert by_article["3"].id not in expanded
    topic_expanded, _, _ = expand_chunk_ids(db_session, [by_article["2"].id], "nghỉ hằng năm")
    assert by_article["3"].id not in topic_expanded
    assert node_count > 0
    assert edge_count > 0
    payload = graph_payload(db_session)
    assert payload["counts"]["nodes"] >= 5
    assert any(edge["type"] == "REFERENCES_ARTICLE" for edge in payload["relationships"])
