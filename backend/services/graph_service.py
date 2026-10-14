import json
import re
import unicodedata
from collections import defaultdict

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.tables import DocumentChunk, Entity, LegalDocument, Relation


CONCEPTS = {
    "employment_contract": ("Hợp đồng lao động", ("hợp đồng lao động", "thử việc", "giao kết hợp đồng")),
    "salary": ("Tiền lương", ("tiền lương", "trả lương", "mức lương", "tiền công")),
    "working_time": ("Thời giờ làm việc", ("thời giờ làm việc", "giờ làm việc", "làm thêm giờ", "tăng ca")),
    "rest_leave": ("Thời giờ nghỉ ngơi", ("nghỉ hằng năm", "nghỉ hàng năm", "nghỉ phép", "ngày nghỉ hằng tuần")),
    "holiday": ("Nghỉ lễ, tết", ("nghỉ lễ", "ngày lễ", "tết")),
    "termination": ("Chấm dứt hợp đồng", ("chấm dứt hợp đồng", "đơn phương chấm dứt", "sa thải")),
    "employee_rights": ("Quyền người lao động", ("người lao động có quyền", "quyền của người lao động")),
    "employer_duties": ("Nghĩa vụ người sử dụng lao động", ("người sử dụng lao động có nghĩa vụ", "trách nhiệm của người sử dụng lao động")),
}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _snippet(text: str, term: str, max_len: int = 500) -> str:
    folded, folded_term = _fold(text), _fold(term)
    index = folded.find(folded_term)
    if index < 0:
        return text[:max_len]
    start = max(0, index - 140)
    end = min(len(text), index + len(term) + 300)
    return text[start:end].strip()


def index_document(db: Session, document: LegalDocument, chunks: list[DocumentChunk]) -> None:
    db.query(Relation).filter(Relation.document_id == document.id).delete(synchronize_session=False)
    db.query(Entity).filter(Entity.document_id == document.id).delete(synchronize_session=False)
    entities_by_key: dict[str, Entity] = {}
    pending_edges: list[tuple[Entity, Entity, str, str | None, str, float]] = []

    def get_entity(key: str, name: str, kind: str, chunk_id: str | None = None, metadata: dict | None = None) -> Entity:
        if key in entities_by_key:
            return entities_by_key[key]
        entity = Entity(
            document_id=document.id,
            entity_key=key,
            name=name,
            entity_type=kind,
            source_chunk_id=chunk_id,
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
        )
        db.add(entity)
        db.flush()
        entities_by_key[key] = entity
        return entity

    doc_node = get_entity(f"doc:{document.id}", document.title, "LegalDocument", metadata={"document_number": document.document_number})
    articles: dict[str, Entity] = {}
    for chunk in chunks:
        chunk_node = get_entity(f"chunk:{chunk.id}", chunk.location_label or f"Đoạn {chunk.chunk_index + 1}", "Chunk", chunk.id)
        pending_edges.append((doc_node, chunk_node, "HAS_CHUNK", chunk.id, chunk.content[:500], 1.0))
        article_node = None
        if chunk.article_number:
            article_node = articles.get(chunk.article_number)
            if article_node is None:
                article_node = get_entity(
                    f"article:{document.id}:{chunk.article_number}",
                    f"Điều {chunk.article_number}",
                    "LegalArticle",
                    chunk.id,
                    {"article_number": chunk.article_number},
                )
                articles[chunk.article_number] = article_node
                pending_edges.append((doc_node, article_node, "CONTAINS_ARTICLE", chunk.id, chunk.content[:500], 1.0))
            pending_edges.append((article_node, chunk_node, "HAS_CHUNK", chunk.id, chunk.content[:500], 1.0))

        folded = _fold(chunk.content)
        for key, (name, aliases) in CONCEPTS.items():
            matched = next((alias for alias in aliases if _fold(alias) in folded), None)
            if matched:
                concept_node = get_entity(f"concept:{key}", name, "LegalConcept", metadata={"extraction": "deterministic phrase match"})
                pending_edges.append((article_node or chunk_node, concept_node, "ABOUT_TOPIC", chunk.id, _snippet(chunk.content, matched), 1.0))

    all_text = document.raw_text or ""
    referenced_numbers = set(re.findall(r"(?i)\bĐiều\s+(\d+[a-zA-Z]?)", all_text))
    for chunk in chunks:
        source_number = chunk.article_number
        if not source_number or source_number not in articles:
            continue
        source = articles[source_number]
        refs = set(re.findall(r"(?i)\bĐiều\s+(\d+[a-zA-Z]?)", chunk.content))
        for target_number in refs:
            if target_number == source_number or target_number not in referenced_numbers:
                continue
            target = articles.get(target_number)
            if target:
                pending_edges.append((source, target, "REFERENCES_ARTICLE", chunk.id, _snippet(chunk.content, f"Điều {target_number}"), 1.0))

    for source, target, relation_type, chunk_id, evidence, confidence in pending_edges:
        db.add(Relation(
            document_id=document.id,
            source_entity_id=source.id,
            target_entity_id=target.id,
            relation_type=relation_type,
            source_chunk_id=chunk_id,
            evidence_text=evidence,
            confidence=confidence,
        ))


def load_graph(db: Session, document_id: str | None = None) -> nx.MultiDiGraph:
    entity_query = select(Entity)
    relation_query = select(Relation)
    if document_id:
        entity_query = entity_query.where(Entity.document_id == document_id)
        relation_query = relation_query.where(Relation.document_id == document_id)
    entities = db.scalars(entity_query).all()
    relations = db.scalars(relation_query).all()
    graph = nx.MultiDiGraph()
    for entity in entities:
        graph.add_node(entity.id, key=entity.entity_key, name=entity.name, type=entity.entity_type, chunk_id=entity.source_chunk_id, document_id=entity.document_id)
    for relation in relations:
        graph.add_edge(
            relation.source_entity_id,
            relation.target_entity_id,
            key=relation.id,
            type=relation.relation_type,
            evidence=relation.evidence_text,
            chunk_id=relation.source_chunk_id,
        )
    return graph


def expand_chunk_ids(db: Session, seed_ids: list[str], question: str) -> tuple[dict[str, float], int, int]:
    graph = load_graph(db)
    if graph.number_of_nodes() == 0:
        return {}, 0, 0
    entity_by_chunk: dict[str, list[str]] = defaultdict(list)
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("chunk_id"):
            entity_by_chunk[attrs["chunk_id"]].append(node_id)
    seeds = {entity_id for chunk_id in seed_ids for entity_id in entity_by_chunk.get(chunk_id, [])}
    if not seeds:
        return {}, 0, 0

    question_folded = _fold(question)
    expanded: dict[str, float] = {}
    visited = set(seeds)
    frontier = set(seeds)
    used_edges = 0
    for depth in range(2):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in set(graph.successors(node)) | set(graph.predecessors(node)):
                connecting_edges = [
                    (source, target, attrs)
                    for source, target, _key, attrs in graph.out_edges(node, keys=True, data=True)
                    if target == neighbor
                ] + [
                    (source, target, attrs)
                    for source, target, _key, attrs in graph.in_edges(node, keys=True, data=True)
                    if source == neighbor
                ]
                for source, target, attrs in connecting_edges:
                    relation_type = attrs.get("type")
                    if relation_type == "REFERENCES_ARTICLE":
                        pass
                    elif relation_type == "ABOUT_TOPIC":
                        topic_node = neighbor if graph.nodes[neighbor].get("type") == "LegalConcept" else node
                        concept = _fold(graph.nodes[topic_node].get("name", ""))
                        # Concept nodes can connect many articles. Traverse only when
                        # the user names the full concept label, not on a generic word
                        # such as "nghỉ" that appears across unrelated provisions.
                        if not concept or concept not in question_folded:
                            continue
                    elif relation_type == "HAS_CHUNK" and graph.nodes[source].get("type") == "LegalArticle":
                        pass
                    else:
                        # Do not traverse document-to-every-chunk hub edges.
                        continue
                    next_frontier.add(neighbor)
                    used_edges += 1
                    chunk_id = graph.nodes[neighbor].get("chunk_id")
                    if chunk_id and chunk_id not in seed_ids:
                        expanded[chunk_id] = max(expanded.get(chunk_id, 0.0), 0.25 if depth == 0 else 0.15)
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return expanded, len(visited), used_edges


def graph_payload(db: Session, query: str | None = None, limit: int = 350) -> dict:
    graph = load_graph(db)
    nodes = list(graph.nodes(data=True))
    edges = list(graph.edges(keys=True, data=True))
    if query:
        folded = _fold(query)
        matched = {i for i, d in nodes if any(token in _fold(d.get("name", "")) for token in folded.split() if len(token) > 2)}
        expanded = set(matched)
        for node in matched:
            expanded.update(graph.successors(node))
            expanded.update(graph.predecessors(node))
        nodes = [(i, d) for i, d in nodes if i in expanded]
        kept = {i for i, _ in nodes}
        edges = [e for e in edges if e[0] in kept or e[1] in kept]
    nodes = nodes[:limit]
    kept = {i for i, _ in nodes}
    edges = [e for e in edges if e[0] in kept and e[1] in kept][: limit * 2]
    return {
        "nodes": [{"id": i, "name": d["name"], "type": d["type"], "chunk_id": d.get("chunk_id"), "document_id": d.get("document_id")} for i, d in nodes],
        "relationships": [{"source": s, "target": t, "type": d.get("type"), "evidence": d.get("evidence"), "chunk_id": d.get("chunk_id")} for s, t, _, d in edges],
        "counts": {"nodes": graph.number_of_nodes(), "relationships": graph.number_of_edges()},
    }
