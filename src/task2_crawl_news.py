"""Crawl nguyên nội dung các bài viết PUBG và lưu theo schema của pipeline."""

import asyncio
from datetime import datetime, timezone
from html import unescape
import json
from pathlib import Path
import re

import requests

from src.task3_convert_markdown import HtmlToMarkdown


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://pubg.com/vi/news/11155",
    "https://pubg.com/vi/news/11057",
    "https://pubg.com/vi/news/11019",
    "https://pubg.com/vi/news/10994",
    "https://pubg.com/vi/news/10991",
]


def extract_title(html: str) -> str:
    """Lấy tiêu đề hiển thị từ Open Graph hoặc thẻ title."""
    match = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)',
        html,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    return unescape(re.sub(r"\s+", " ", match.group(1))).strip() if match else "PUBG News"


def html_to_markdown(html: str) -> str:
    parser = HtmlToMarkdown()
    parser.feed(html)
    return parser.markdown()


async def crawl_article(url: str) -> dict:
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; RAG-course-crawler/1.0)",
            "Accept-Language": "vi,en;q=0.8",
        },
        timeout=30,
    )
    response.raise_for_status()
    content_markdown = html_to_markdown(response.text)
    if len(content_markdown) < 200:
        raise ValueError(f"Crawled article is unexpectedly short: {url}")
    return {
        "url": url,
        "title": extract_title(response.text),
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
