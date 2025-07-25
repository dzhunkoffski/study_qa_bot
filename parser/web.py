#!/usr/bin/env python3
"""
itmo_text_scraper.py
-----------------------------------
Download and extract all visible text from selected ITMO University
master's program pages.

Author : <Your Name>
Created: 2025-07-25
"""

from __future__ import annotations
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict
import os

import requests
from bs4 import BeautifulSoup, NavigableString, Comment
from slugify import slugify
from tqdm import tqdm

# ---------------------------- CONFIGURATION ---------------------------- #

# URLS: List[str] = [
#     "https://abit.itmo.ru/program/master/ai",
#     "https://abit.itmo.ru/program/master/ai_product",
# ]

HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "ru,en;q=0.9",
}

REQUEST_TIMEOUT: float = 15.0          # seconds
RETRY_PAUSE: float = 2.5               # seconds between retries
MAX_RETRIES: int = 3

# OUTPUT_DIR: Path = Path("scraped_output")
# OUTPUT_DIR.mkdir(exist_ok=True)

# Regular expression to collapse redundant whitespace
WHITESPACE_PATTERN = re.compile(r"\s+")

# ---------------------------- CORE LOGIC ---------------------------- #

def fetch_html(url: str) -> str:
    """Download raw HTML, retrying on common transient failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except (requests.Timeout, requests.ConnectionError) as e:
            logging.warning("Network error on %s (attempt %d/%d): %s",
                            url, attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_PAUSE)
            else:
                raise
        except requests.HTTPError as e:
            logging.error("HTTP %s on %s", e.response.status_code, url)
            raise

def is_visible(element) -> bool:
    """Filter out style/script/meta tags and hidden comments."""
    if element.parent.name in {"style", "script", "head", "title", "meta", "[document]"}:
        return False
    if isinstance(element, Comment):
        return False
    return True

def extract_visible_text(html: str) -> List[str]:
    """Parse HTML and return a list of cleaned visible text chunks."""
    soup = BeautifulSoup(html, "lxml")
    texts = soup.find_all(string=True)
    visible_texts = [t for t in texts if is_visible(t)]
    cleaned: List[str] = []
    for t in visible_texts:
        text = WHITESPACE_PATTERN.sub(" ", t.strip())
        if text:
            cleaned.append(text)
    return cleaned

def save_output(url: str, text_chunks: List[str], save_to) -> Path:
    """Save extracted text to a UTF-8 .txt and .json file."""
    slug = slugify(url, max_length=50)
    txt_path = Path(os.path.join(save_to, f"{slug}.txt"))
    json_path = Path(os.path.join(save_to, f"{slug}.json"))

    if not os.path.exists(txt_path):
        txt_path.write_text("\n".join(text_chunks), encoding="utf-8")
        json_path.write_text(json.dumps(text_chunks, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return txt_path

def parse_url(url, save_to):
    html = fetch_html(url)
    visible_text = extract_visible_text(html)
    file_path = save_output(url, visible_text, save_to)

# def main() -> None:
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()],
#     )

#     for url in tqdm(URLS, desc="Processing pages"):
#         html = fetch_html(url)
#         visible_text = extract_visible_text(html)
#         file_path = save_output(url, visible_text)
#         logging.info("Saved %d text fragments from %s to %s",
#                      len(visible_text), url, file_path)

# if __name__ == "__main__":
#     main()
