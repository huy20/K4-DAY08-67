import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation


load_dotenv()

st.set_page_config(
    page_title="PUBG Knowledge RAG Chatbot",
    page_icon="🎮",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("⚙️ Cấu hình RAG")
    st.caption("Chatbot tra cứu thông tin & quy định PUBG: BATTLEGROUNDS")
    top_k = st.slider("Số lượng Chunks truy xuất (top_k)", 1, 10, 5)
    
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🎮 PUBG Knowledge Assistant")
st.caption("Hỏi đáp về quy tắc ứng xử, điều khoản dịch vụ, chính sách bảo mật và tin tức sự kiện PUBG.")


def render_sources(sources: list[dict], retrieval_source: str) -> None:
    """Hiển thị thông tin nguồn trích dẫn và điểm số."""
    if not sources:
        return
    with st.expander(f"📚 Nguồn tham khảo ({retrieval_source.upper()} - {len(sources)} đoạn)", expanded=False):
        for idx, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            title = meta.get("title", "Tài liệu")
            source_file = meta.get("source", "N/A")
            score = src.get("score", 0.0)
            method = src.get("retrieval_method", "N/A")
            url = meta.get("url")

            header_text = f"**[{idx}] {title}** ({source_file})"
            if url:
                header_text += f" - [Link]({url})"
            st.markdown(header_text)
            st.caption(f"Phương thức: `{method}` | Điểm tương đồng: `{score:.4f}`")
            st.markdown(f"> {src.get('content', '')}")
            st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"], message.get("retrieval_source", "none"))

query = st.chat_input("Nhập câu hỏi về luật chơi, chính sách hoặc tin tức PUBG...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm dữ liệu và tạo câu trả lời..."):
            result = generate_with_citation(query, top_k=top_k)
            answer = result["answer"]
            sources = result["sources"]
            retrieval_source = result["retrieval_source"]

            st.markdown(answer)
            if sources:
                render_sources(sources, retrieval_source)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })
