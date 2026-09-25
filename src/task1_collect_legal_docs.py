"""Thu thập nguyên bản các chính sách PUBG từ nguồn công khai."""

from pathlib import Path

import requests

from src.task3_convert_markdown import HtmlToMarkdown


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

POLICY_URLS = {
    "privacy_policy.html": "https://pubg.com/vi/clause/privacy_policy/label_steam",
    "term_of_service.html": "https://pubg.com/vi/clause/term_of_service/label_steam",
    "rules_of_conduct.html": "https://pubg.com/vi/clause/rules_of_conduct/label_steam",
    "pubg_mods_policy.html": "https://pubg.com/vi/clause/pubg_mods_policy/label_steam",
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Lưu nguyên byte HTML, không tóm tắt hoặc chuyển đổi nội dung."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RAG-course-crawler/1.0)",
        "Accept-Language": "vi,en;q=0.8",
    }
    for filename, url in POLICY_URLS.items():
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        (DATA_DIR / filename).write_bytes(response.content)
        print(f"Saved: {DATA_DIR / filename} ({len(response.content)} bytes)")


def html_text(path: Path) -> str:
    """Trích nội dung HTML thành text sạch để dựng PDF."""
    parser = HtmlToMarkdown()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.markdown()


def render_pdf(html_path: Path) -> Path:
    """Dựng một PDF từ file HTML nguồn để đáp ứng yêu cầu tài liệu PDF/DOCX."""
    from fpdf import FPDF

    text = html_text(html_path)
    if len(text) < 200:
        raise ValueError(f"Rendered document is unexpectedly short: {html_path}")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = Path("C:/Windows/Fonts/arial.ttf")
    if font_path.exists():
        pdf.add_font("doc", "", str(font_path))
        pdf.set_font("doc", size=11)
    else:
        pdf.set_font("Helvetica", size=11)
        text = text.encode("latin-1", "replace").decode("latin-1")

    pdf.multi_cell(0, 6, text)
    output_path = html_path.with_suffix(".pdf")
    pdf.output(str(output_path))
    print(f"Saved: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def render_existing_html() -> None:
    """Chuyển toàn bộ HTML nguồn đã tải thành PDF tương ứng."""
    files = sorted(DATA_DIR.glob("*.html"))
    if not files:
        print(f"No HTML sources found in: {DATA_DIR}")
        return
    for path in files:
        render_pdf(path)


if __name__ == "__main__":
    setup_directory()
    download_documents()
    render_existing_html()
