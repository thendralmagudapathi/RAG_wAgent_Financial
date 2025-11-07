#!/usr/bin/env python3
# HTML extraction and chunking (BeautifulSoup)

from bs4 import BeautifulSoup
import os
import re
from typing import List, Dict, Any


def extract_text_from_html(path: str) -> str:
    """Extract visible text from an HTML file."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    # remove scripts/styles
    for s in soup(['script', 'style', 'noscript']):
        s.decompose()
    text = soup.get_text(separator='\n')
    # normalize whitespace
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    return text


def guess_company_name(path: str, html_text: str) -> str:
    """Try to identify a company name: prefer title tag, otherwise filename prefix."""
    try:
        # try title from HTML
        soup = BeautifulSoup(html_text, 'html.parser')
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            # often title contains company name before a dash or newline
            return title.split(' - ')[0].split('\n')[0].strip()
    except Exception:
        pass
    # fallback to filename prefix
    base = os.path.basename(path)
    name = re.split(r'[_\-\.]', base)[0]
    return name


def chunk_text_with_lines(text: str, max_chars: int = 1000) -> List[Dict[str, Any]]:
    """Chunk text into passages, returning dicts with text and start_line (1-based).

    Returns: list of {'text': str, 'start_line': int}
    """
    lines = [ln for ln in text.split('\n')]
    paragraphs = []  # list of (paragraph_text, start_line)
    i = 0
    while i < len(lines):
        if lines[i].strip() == '':
            i += 1
            continue
        # start of paragraph
        start = i
        parts = []
        while i < len(lines) and lines[i].strip() != '':
            parts.append(lines[i].strip())
            i += 1
        paragraphs.append((" ".join(parts), start + 1))

    chunks = []
    cur_parts = []
    cur_len = 0
    cur_start = None
    for para, line_no in paragraphs:
        if cur_start is None:
            cur_start = line_no
        if cur_len + len(para) + 1 > max_chars and cur_parts:
            chunks.append({'text': '\n'.join(cur_parts), 'start_line': cur_start})
            cur_parts = [para]
            cur_len = len(para)
            cur_start = line_no
        else:
            cur_parts.append(para)
            cur_len += len(para) + 1
    if cur_parts:
        chunks.append({'text': '\n'.join(cur_parts), 'start_line': cur_start})
    return chunks


def chunk_text(text: str, max_chars: int = 1000):
    """Backward-compatible wrapper: returns list of texts only."""
    return [c['text'] for c in chunk_text_with_lines(text, max_chars=max_chars)]


if __name__ == '__main__':
    # quick demo when run directly
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if not p:
        print('Usage: python parser.py <html-file>')
    else:
        txt = extract_text_from_html(p)
        chunks = chunk_text_with_lines(txt)
        for c in chunks[:3]:
            print('start_line=', c['start_line'])
            print(c['text'][:500])