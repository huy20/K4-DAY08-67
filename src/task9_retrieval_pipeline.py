"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Lấy best cosine score gốc từ dense results.
    3. Nếu score dưới threshold, thử PageIndex fallback.
    4. Nếu fallback lỗi hoặc rỗng, fuse hai danh sách bằng RRF đúng một lần.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

import os
from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

_raw_threshold = os.getenv("SCORE_THRESHOLD", "").strip()
SCORE_THRESHOLD = float(_raw_threshold) if _raw_threshold else 0.3
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    if not query.strip() or top_k <= 0:
        return []

    candidate_k = max(top_k * 2, 10)
    dense = semantic_search(query, top_k=candidate_k)
    sparse = lexical_search(query, top_k=candidate_k)

    best_dense_score = dense[0]["score"] if dense else 0.0

    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            pass

    if use_reranking:
        hybrid = rerank_rrf([dense, sparse], top_k=top_k)
        return hybrid[:top_k]

    return dense[:top_k]


if __name__ == "__main__":
    for result in retrieve("quy tắc cấm gian lận", top_k=3):
        print(result)
