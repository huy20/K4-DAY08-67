"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CACHE_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"
TMP_PDF_DIR = Path(__file__).parent.parent / "pageindex_pdfs"


def upload_documents() -> dict[str, str]:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        print("PAGEINDEX_API_KEY not found in .env; skipping upload.")
        return {}

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids: dict[str, str] = {}
    if CACHE_FILE.exists():
        try:
            doc_ids = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            doc_ids = {}

    TMP_PDF_DIR.mkdir(parents=True, exist_ok=True)

    for md_path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        source_name = md_path.name
        if source_name in doc_ids:
            continue

        pdf_path = TMP_PDF_DIR / f"{md_path.stem}.pdf"
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=10)
            text = md_path.read_text(encoding="utf-8", errors="replace")
            safe_text = text.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 5, text=safe_text)
            pdf.output(str(pdf_path))

            res = client.submit_document(file_path=str(pdf_path))
            doc_id = res.get("doc_id")
            if doc_id:
                doc_ids[source_name] = doc_id
                print(f"Uploaded {source_name} -> doc_id: {doc_id}")
        except Exception as err:
            print(f"Failed to upload {source_name}: {err}")

    CACHE_FILE.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if not query.strip() or top_k <= 0 or not PAGEINDEX_API_KEY:
        return []

    if not CACHE_FILE.exists():
        return []

    try:
        doc_ids_map = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not doc_ids_map:
        return []

    try:
        from pageindex import PageIndexClient

        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        results: list[dict] = []
        rank = 0

        for source_name, doc_id in doc_ids_map.items():
            if len(results) >= top_k:
                break
            try:
                ocr_res = client.get_ocr(doc_id=doc_id, format="node")
                nodes = ocr_res.get("nodes", []) if isinstance(ocr_res, dict) else []
                for node in nodes:
                    content = node.get("text", "") or node.get("content", "")
                    if not content or query.lower() not in content.lower():
                        continue
                    rank += 1
                    score = max(0.1, 1.0 - (rank - 1) * 0.1)
                    results.append({
                        "id": f"pageindex::{doc_id}::{rank}",
                        "content": content[:1000],
                        "score": score,
                        "metadata": {
                            "source": source_name,
                            "title": source_name.replace(".md", "").replace("_", " ").title(),
                            "doc_type": "legal" if "legal" in source_name else "news",
                            "url": None,
                            "chunk_index": rank - 1,
                        },
                        "retrieval_method": "pageindex",
                    })
                    if len(results) >= top_k:
                        break
            except Exception:
                continue

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]
    except Exception:
        return []


if __name__ == "__main__":
    upload_documents()
