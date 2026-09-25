"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNKING_METHOD = os.getenv("CHUNKING_METHOD", "recursive")

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_ST_MODEL = None
_GENAI_CLIENT = None
_OPENAI_CLIENT = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Dispatch theo EMBEDDING_PROVIDER trong .env."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).strip().lower()

    if provider == "sentence_transformers":
        global _ST_MODEL
        if _ST_MODEL is None:
            from sentence_transformers import SentenceTransformer

            model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
            _ST_MODEL = SentenceTransformer(model_name)
        embeddings = _ST_MODEL.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    elif provider == "gemini":
        global _GENAI_CLIENT
        if _GENAI_CLIENT is None:
            from google import genai

            api_key = os.getenv("GEMINI_API_KEY")
            _GENAI_CLIENT = genai.Client(api_key=api_key)
        model_name = os.getenv("EMBEDDING_MODEL") or "text-embedding-004"
        response = _GENAI_CLIENT.models.embed_content(
            model=model_name,
            contents=texts,
        )
        return [item.values for item in response.embeddings]

    elif provider == "openai":
        global _OPENAI_CLIENT
        if _OPENAI_CLIENT is None:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            _OPENAI_CLIENT = OpenAI(api_key=api_key)
        model_name = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        response = _OPENAI_CLIENT.embeddings.create(
            model=model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document theo contract."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        doc_type = "legal" if "legal" in path.parts else "news"
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue

        title = path.stem.replace("_", " ").title()
        url = None
        for line in content.splitlines():
            line_str = line.strip()
            if line_str.startswith("# ") and title == path.stem.replace("_", " ").title():
                extracted = line_str[2:].strip()
                if extracted:
                    title = extracted
            elif line_str.startswith("**Source:**"):
                extracted_url = line_str.replace("**Source:**", "").strip()
                if extracted_url:
                    url = extracted_url

        doc_id = path.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append({
            "id": doc_id,
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index theo contract."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        raw_chunks = splitter.split_text(document["content"])
        chunk_idx = 0
        for text in raw_chunks:
            cleaned = text.strip()
            if not cleaned:
                continue
            chunks.append({
                "id": f"{document['id']}::chunk-{chunk_idx}",
                "content": cleaned,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": chunk_idx,
                },
            })
            chunk_idx += 1
    return chunks


def embed_chunks(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk["content"] for chunk in batch]
        embeddings = embed_texts(texts)
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding
    return chunks


def index_to_vectorstore(chunks: list[dict], batch_size: int = 100) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return

    collection = get_collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [chunk["id"] for chunk in batch]
        documents = [chunk["content"] for chunk in batch]
        embeddings = [chunk["embedding"] for chunk in batch]
        metadatas: list[dict[str, Any]] = []
        for chunk in batch:
            meta = dict(chunk["metadata"])
            if meta.get("url") is None:
                meta["url"] = ""
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    print("1. Loading documents from standardized directory...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print("2. Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("3. Generating embeddings...")
    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks.")

    print("4. Upserting to ChromaDB...")
    index_to_vectorstore(embedded_chunks)
    print(f"Successfully indexed {len(embedded_chunks)} chunks into ChromaDB at {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()
