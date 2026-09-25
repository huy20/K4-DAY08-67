"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

from pathlib import Path
from html import escape, unescape
from html.parser import HTMLParser
import json
import re


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


class HtmlToMarkdown(HTMLParser):
    """Chuyển phần nội dung HTML thành Markdown mà không gọi dịch vụ ngoài."""

    BLOCK_TAGS = {
        "article", "blockquote", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "p", "section", "tr",
    }
    SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0
        self.link_stack: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append(f"{'#' * int(tag[1])} ")
        elif tag == "li":
            self.parts.append("- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag == "a":
            href = dict(attrs).get("href")
            self.link_stack.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            return
        if tag == "a" and self.link_stack:
            href = self.link_stack.pop()
            if href:
                self.parts.append(f" ({escape(href)})")
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(unescape(data))

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ ]+", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_markdown(path: Path) -> str:
    parser = HtmlToMarkdown()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return parser.markdown()


def convert_legal_docs() -> None:
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(legal_dir.glob("*.html")):
        content = html_to_markdown(path)
        if len(content) < 200:
            raise ValueError(f"Converted document is unexpectedly short: {path}")
        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(f"# {path.stem.replace('_', ' ').title()}\n\n{content}\n", encoding="utf-8")
        print(f"Saved: {output_path}")


def convert_news_articles() -> None:
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"url", "title", "date_crawled", "content_markdown"}
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - data.keys()
        if missing:
            raise ValueError(f"{path} is missing metadata: {sorted(missing)}")
        content = str(data["content_markdown"]).strip()
        if len(content) < 200:
            raise ValueError(f"Converted article is unexpectedly short: {path}")
        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(header + content + "\n", encoding="utf-8")
        print(f"Saved: {output_path}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
