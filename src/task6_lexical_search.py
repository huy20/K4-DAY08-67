"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import math
import numpy as np
from rank_bm25 import BM25Okapi


CORPUS: list[dict] = []
_CACHED_BM25 = None
_CACHED_CORPUS_LEN = 0


class RobustBM25Okapi(BM25Okapi):
    """BM25Okapi chuẩn với IDF không âm, đảm bảo corpus nhỏ vẫn tính điểm chính xác."""

    def _calc_idf(self, nd):
        super()._calc_idf(nd)
        for word, val in list(self.idf.items()):
            if val <= 0:
                freq = nd[word]
                self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)


def get_corpus() -> list[dict]:
    """Lấy corpus chunks từ Task 4 nếu CORPUS chưa có."""
    global CORPUS
    if not CORPUS:
        try:
            from .task4_chunking_indexing import chunk_documents, load_documents
            CORPUS = chunk_documents(load_documents())
        except Exception:
            CORPUS = []
    return CORPUS


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [item["content"].lower().split() for item in corpus]
    return RobustBM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if not query.strip() or top_k <= 0:
        return []

    corpus = CORPUS if CORPUS else get_corpus()
    if not corpus:
        return []

    global _CACHED_BM25, _CACHED_CORPUS_LEN
    if _CACHED_BM25 is None or _CACHED_CORPUS_LEN != len(corpus):
        _CACHED_BM25 = build_bm25_index(corpus)
        _CACHED_CORPUS_LEN = len(corpus)

    tokens = query.lower().split()
    if not tokens:
        return []

    scores = _CACHED_BM25.get_scores(tokens)
    indices = np.argsort(scores)[::-1]

    results = []
    seen_ids = set()
    for index in indices:
        if scores[index] <= 0:
            continue
        item = corpus[index]
        item_id = item["id"]
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        meta = dict(item.get("metadata", {}))
        if "chunk_index" in meta and not isinstance(meta["chunk_index"], int):
            try:
                meta["chunk_index"] = int(meta["chunk_index"])
            except (ValueError, TypeError):
                meta["chunk_index"] = 0

        results.append({
            "id": str(item_id),
            "content": str(item["content"]),
            "score": float(scores[index]),
            "metadata": meta,
            "retrieval_method": "bm25",
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("quy định", top_k=3):
        print(result)
