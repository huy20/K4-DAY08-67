"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

SYSTEM_PROMPT = """Bạn là trợ lý giải đáp thắc mắc dựa trên tài liệu được cung cấp.
Quy tắc:
1. Trả lời CHÍNH XÁC và CHỈ DỰA VÀO context bên dưới.
2. Mỗi khẳng định trong câu trả lời phải kèm theo trích dẫn nguồn, ví dụ: [Document 1] hoặc [Title].
3. Nếu thông tin không đủ hoặc không có bằng chứng rõ ràng, hãy trả lời từ chối an toàn: "Tôi không thể xác minh thông tin này từ nguồn hiện có."
4. Tuyệt đối không bịa đặt hoặc suy đoán thông tin ngoài ngữ cảnh."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (giảm lost-in-the-middle)."""
    if len(chunks) <= 2:
        return [dict(chunk) for chunk in chunks]

    front = [dict(chunk) for chunk in chunks[::2]]
    back = [dict(chunk) for chunk in chunks[1::2]]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label rõ ràng cho LLM trích dẫn."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Document")
        source = metadata.get("source", "unknown")
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình trong .env."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    model = os.getenv("LLM_MODEL", LLM_MODEL).strip()

    if provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        model_name = model or "gemini-2.0-flash"
        response = client.models.generate_content(
            model=model_name,
            contents=user_message,
            config={
                "system_instruction": system_prompt,
                "temperature": TEMPERATURE,
            },
        )
        return response.text or ""

    elif provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        model_name = model or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    elif provider == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=api_key)
        model_name = model or "claude-3-5-sonnet-20241022"
        response = client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1024,
            temperature=TEMPERATURE,
        )
        return response.content[0].text or ""

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult theo hợp đồng."""
    if not query.strip() or top_k <= 0:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Dưới đây là các tài liệu tham khảo:\n\n{context}\n\nCâu hỏi: {query}"

    try:
        raw_answer = call_llm(SYSTEM_PROMPT, user_message)
        answer = raw_answer.strip() if raw_answer and raw_answer.strip() else "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    except Exception as err:
        print(f"LLM call failed: {err}")
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    primary_method = chunks[0].get("retrieval_method", "hybrid")
    retrieval_source = "pageindex" if primary_method == "pageindex" else "hybrid"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    print(generate_with_citation("PUBG có những quy định gì về gian lận?"))
