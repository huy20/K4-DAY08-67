import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import (
    SAFE_REFUSAL,
    SYSTEM_PROMPT,
    call_llm,
    format_context,
    generate_with_citation,
    reorder_for_llm,
)
from src.task9_retrieval_pipeline import retrieve


load_dotenv()

st.set_page_config(
    page_title="PUBG RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .rag-hero {
        background: linear-gradient(135deg, #f5a623 0%, #e8622a 100%);
        color: #ffffff; padding: 1.3rem 1.6rem; border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .rag-hero h1 { margin: 0; font-size: 1.7rem; }
    .rag-hero p { margin: .35rem 0 0; opacity: .92; }
    .src-card {
        border: 1px solid rgba(128,128,128,.28); border-radius: 12px;
        padding: .8rem 1rem; margin-bottom: .7rem;
    }
    .badge {
        display: inline-block; padding: .12rem .55rem; border-radius: 999px;
        font-size: .74rem; font-weight: 600; letter-spacing: .02em;
    }
    .badge-hybrid { background: #e8f0fe; color: #1a56c4; }
    .badge-dense { background: #e6f4ea; color: #137333; }
    .badge-bm25 { background: #fef7e0; color: #a05a00; }
    .badge-pageindex { background: #f3e8fd; color: #6b21a8; }
    .src-meta { color: #6b7280; font-size: .8rem; }
    .src-snippet { font-size: .88rem; color: #374151; }
    </style>
    """,
    unsafe_allow_html=True,
)


def answer_query(query: str, top_k: int, mode: str) -> dict:
    """Trả GenerationResult; mode 'hybrid' dùng pipeline công khai, 'dense' để A/B."""
    if mode == "Hybrid + RRF":
        return generate_with_citation(query, top_k=top_k)

    chunks = retrieve(query, top_k=top_k, use_reranking=False)
    if not chunks:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    context = format_context(reorder_for_llm(chunks))
    message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        answer = call_llm(SYSTEM_PROMPT, message).strip()
    except Exception:
        answer = ""
    method = chunks[0].get("retrieval_method", "dense")
    return {
        "answer": answer or SAFE_REFUSAL,
        "sources": chunks,
        "retrieval_source": "pageindex" if method == "pageindex" else "hybrid",
    }


def render_sources(sources: list[dict], retrieval_source: str, show_scores: bool) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn tham khảo · {retrieval_source} · {len(sources)} đoạn"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            title = metadata.get("title", "Tài liệu")
            source_file = metadata.get("source", "N/A")
            method = str(source.get("retrieval_method", "n/a"))
            score = float(source.get("score", 0.0))
            url = metadata.get("url")

            st.markdown(
                f"""<div class="src-card">
                <span class="badge badge-{method}">{method}</span>
                <b style="margin-left:.4rem">[{index}] {title}</b>
                <div class="src-meta">{source_file}{f' · <a href="{url}">link</a>' if url else ''}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if show_scores:
                st.progress(min(max(score, 0.0), 1.0), text=f"score: {score:.4f}")
            st.markdown(f"<div class='src-snippet'>{source.get('content', '')[:400]}</div>", unsafe_allow_html=True)
            st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Cấu hình")
    mode = st.selectbox("Chiến lược truy xuất", ["Hybrid + RRF", "Dense-only"])
    top_k = st.slider("Số chunks (top_k)", 3, 10, 5)
    show_scores = st.checkbox("Hiển thị điểm tương đồng", value=True)
    if st.button("Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Corpus: 4 tài liệu chính sách PUBG + 5 bài viết/bản tin.")
    st.caption(f"Chế độ: {mode} · top_k={top_k}")

st.markdown(
    """
    <div class="rag-hero">
        <h1>PUBG RAG Assistant</h1>
        <p>Hỏi đáp về quy tắc ứng xử, điều khoản dịch vụ, chính sách bảo mật và tin tức PUBG.
        Câu trả lời chỉ dựa trên tài liệu trích dẫn.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []),
                message.get("retrieval_source", "none"),
                show_scores,
            )

query = st.chat_input("Nhập câu hỏi về PUBG...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            try:
                result = answer_query(query, top_k, mode)
            except Exception as error:
                result = {
                    "answer": f"Lỗi khi xử lý câu hỏi: {error}",
                    "sources": [],
                    "retrieval_source": "none",
                }
        st.markdown(result["answer"])
        render_sources(result.get("sources", []), result.get("retrieval_source", "none"), show_scores)
        st.caption(f"retrieval_source: {result.get('retrieval_source')}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "retrieval_source": result.get("retrieval_source"),
        }
    )
