"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query.strip() or top_k <= 0:
        return []

    collection = get_collection()
    embeddings = embed_texts([query])
    if not embeddings:
        return []

    query_vector = embeddings[0]
    try:
        response = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    if not response or not response.get("ids") or not response["ids"][0]:
        return []

    results = []
    seen_ids = set()

    ids = response["ids"][0]
    docs = response["documents"][0] if response.get("documents") else []
    metadatas = response["metadatas"][0] if response.get("metadatas") else []
    distances = response["distances"][0] if response.get("distances") else []

    for item_id, content, metadata, distance in zip(ids, docs, metadatas, distances):
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        meta = dict(metadata) if metadata else {}
        if "chunk_index" in meta and not isinstance(meta["chunk_index"], int):
            try:
                meta["chunk_index"] = int(meta["chunk_index"])
            except (ValueError, TypeError):
                meta["chunk_index"] = 0

        similarity = max(0.0, float(1.0 - distance))
        results.append({
            "id": str(item_id),
            "content": str(content),
            "score": similarity,
            "metadata": meta,
            "retrieval_method": "dense",
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("quy tắc ứng xử", top_k=3):
        print(result)
