"""Thu thập nguyên bản các chính sách PUBG từ nguồn công khai."""

from pathlib import Path

import requests


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


if __name__ == "__main__":
    setup_directory()
    download_documents()
