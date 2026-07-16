#!/usr/bin/env python3
"""
ocr_books.py - Turn folders of scanned book pages into per-book Markdown.

Each subfolder of the input directory is treated as one book. Every page image
is sent, one at a time and statelessly, to a local vision-language model served
by Ollama. The model transcribes the page into Markdown, preserving heading
hierarchy and tables, and describing figures/diagrams as `**Figure:** ...`.
Per-page output is cached so a long run is fully resumable; the cached pages are
then assembled (with cross-page tables stitched and running headers/footers
stripped) into one `<Book Name>.md` per folder.

Design notes
------------
* One page = one model call, no conversation carryover. Memory and context stay
  flat regardless of book length, which is what keeps a 200-page book from
  overrunning the context window.
* The Ollama *native* API (/api/chat) is used rather than the OpenAI-compat
  shim so num_ctx and other generation options can be set reliably per call.
* Resumability: each page is cached under <cache>/<book>/page-XXXX.md. Re-running
  skips pages already cached and books whose final .md already exists.

Author-selected defaults (tunable via flags):
    num_ctx      = 16384   # room for ~2.7k vision tokens + prompt + output
    num_predict  = 4096    # max Markdown tokens per page (dense page headroom)
    max_pixels   = 2_600_000  # guard; the 1436x1527 scans pass through untouched
    temperature  = 0       # deterministic transcription
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required. Install with: pip install pillow")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
BLANK_SENTINEL = "<<BLANK>>"
PAGE_BREAK = "\n<<<PAGE_BREAK>>>\n"

DEFAULTS = {
    "model": "qwen3-vl:30b",
    "host": os.environ.get("OLLAMA_HOST", "http://ollama.ash4d.com:11434"),
    "num_ctx": 16384,
    "num_predict": 4096,
    "max_pixels": 2_600_000,
    "temperature": 0.0,
    "timeout": 1800,      # seconds per page request (speed is not a concern)
    "retries": 3,
}

PROMPT = """You are transcribing a single scanned page from a printed book into clean Markdown.

Rules:
- Transcribe ALL body text exactly, preserving natural reading order.
- Represent the page's visual heading hierarchy with Markdown headings: use # for a chapter or major title, ## for a section heading, ### for a subsection. Judge levels from the relative size and prominence of the text.
- Render every table as a GitHub-Flavored Markdown table with a header row and separator. Keep columns aligned to the source. If a table clearly continues from a previous page (no repeated header), still output the rows as a GFM table.
- For any photograph, illustration, chart, or diagram, DO NOT embed the image. Instead insert a single line: **Figure:** followed by a concise, factual description of what the figure shows (include any visible caption text).
- Do NOT transcribe running headers, running footers, or standalone page numbers.
- Do NOT add commentary, notes, or explanations of your own. Output only the page's Markdown.
- Do NOT wrap your entire answer in a code fence.
- If the page has no readable content (blank or effectively empty), output exactly: {blank}

Transcribe the page now.""".format(blank=BLANK_SENTINEL)


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    cache_dir: Path
    model: str
    host: str
    num_ctx: int
    num_predict: int
    max_pixels: int
    temperature: float
    timeout: int
    retries: int
    force: bool
    only_book: Optional[str]
    merge_tables: bool
    strip_boilerplate: bool
    dry_run: bool


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_books(input_dir: Path) -> list[tuple[str, list[Path]]]:
    """Return [(book_name, [sorted page image paths]), ...] for each subfolder."""
    books: list[tuple[str, list[Path]]] = []
    for entry in sorted(input_dir.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue  # skip hidden and our own _output/_cache dirs
        pages = sorted(
            (p for p in entry.iterdir() if p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: p.name.lower(),
        )
        if pages:
            books.append((entry.name, pages))
    return books


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image(path: Path, max_pixels: int) -> str:
    """Load an image, downscale only if it exceeds max_pixels, return base64 JPEG."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def ocr_page(cfg: Config, image_b64: str) -> str:
    """Send one page to Ollama /api/chat and return the raw Markdown string."""
    payload = {
        "model": cfg.model,
        "stream": False,
        "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
        "options": {
            "num_ctx": cfg.num_ctx,
            "num_predict": cfg.num_predict,
            "temperature": cfg.temperature,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    url = cfg.host.rstrip("/") + "/api/chat"

    last_err: Optional[Exception] = None
    for attempt in range(1, cfg.retries + 1):
        try:
            req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urlrequest.urlopen(req, timeout=cfg.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["message"]["content"]
        except (URLError, HTTPError, KeyError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            wait = min(30, 2 ** attempt)
            logging.warning("  page call failed (attempt %d/%d): %s - retrying in %ds",
                            attempt, cfg.retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Ollama call failed after {cfg.retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Output cleanup
# ---------------------------------------------------------------------------

def unwrap_code_fence(text: str) -> str:
    """Strip a fence that wraps the ENTIRE answer (```markdown ... ```)."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return t


def is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def is_separator_line(line: str) -> bool:
    s = line.strip()
    return bool(re.match(r"^\|?[\s:|-]+\|?$", s)) and "-" in s


def strip_running_headers(pages: list[str], num_pages: int) -> tuple[list[str], list[str]]:
    """Remove short lines that repeat across many pages (running heads/feet).

    A line is treated as boilerplate if a normalized copy of it appears as the
    first or last non-empty line on at least threshold pages. Pure page-number
    lines are always removed.
    """
    threshold = max(10, int(0.4 * num_pages))
    edge_counts: dict[str, int] = {}

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip()).lower()

    for page in pages:
        lines = [l for l in page.split("\n") if l.strip()]
        if not lines:
            continue
        for edge in {lines[0], lines[-1]}:
            if len(edge) < 80:
                edge_counts[norm(edge)] = edge_counts.get(norm(edge), 0) + 1

    boilerplate = {k for k, v in edge_counts.items() if v >= threshold and k}
    removed_log = sorted(boilerplate)

    pagenum_re = re.compile(r"^\s*(?:page\s+)?[ivxlcdm\d]{1,6}\s*$", re.IGNORECASE)

    cleaned: list[str] = []
    for page in pages:
        out_lines = []
        for line in page.split("\n"):
            n = norm(line)
            if n in boilerplate:
                continue
            if line.strip() and pagenum_re.match(line.strip()) and not is_table_line(line):
                continue
            out_lines.append(line)
        cleaned.append("\n".join(out_lines).strip("\n"))
    return cleaned, removed_log


def merge_cross_page_tables(pages: list[str]) -> tuple[str, int]:
    """Assemble pages, stitching a table that continues onto the next page.

    Conservative: only merges when the previous page ends with a table row and
    the next page begins with a table row of the SAME column count. The
    continuation table's header+separator (if present) is dropped and its body
    rows are appended to the previous table.
    """
    merged: list[str] = []
    merge_count = 0

    for page in pages:
        page = page.strip("\n")
        if not page:
            continue
        if not merged:
            merged.append(page)
            continue

        prev_lines = merged[-1].split("\n")
        cur_lines = page.split("\n")

        # last non-empty line of previous page
        i = len(prev_lines) - 1
        while i >= 0 and not prev_lines[i].strip():
            i -= 1
        # first non-empty line of current page
        j = 0
        while j < len(cur_lines) and not cur_lines[j].strip():
            j += 1

        if i >= 0 and j < len(cur_lines) and is_table_line(prev_lines[i]) and is_table_line(cur_lines[j]):
            if prev_lines[i].count("|") == cur_lines[j].count("|"):
                # span of the leading table block on the current page
                m = j
                lead_idx = []
                while m < len(cur_lines) and is_table_line(cur_lines[m]):
                    lead_idx.append(m)
                    m += 1
                drop = 2 if len(lead_idx) >= 2 and is_separator_line(cur_lines[lead_idx[1]]) else 0
                body = [cur_lines[x] for x in lead_idx[drop:]]
                rest = "\n".join(cur_lines[m:]).strip("\n")

                merged[-1] = "\n".join(prev_lines[: i + 1] + body)
                if rest:
                    merged.append(rest)
                merge_count += 1
                continue

        merged.append(page)

    return "\n\n".join(merged), merge_count


# ---------------------------------------------------------------------------
# Per-book processing
# ---------------------------------------------------------------------------

@dataclass
class BookStats:
    name: str = ""
    total_pages: int = 0
    blank_pages: int = 0
    tables_merged: int = 0
    boilerplate_removed: list[str] = field(default_factory=list)
    seconds: float = 0.0
    output_path: str = ""


def process_book(cfg: Config, name: str, pages: list[Path]) -> BookStats:
    stats = BookStats(name=name, total_pages=len(pages))
    out_path = cfg.output_dir / f"{name}.md"

    if out_path.exists() and not cfg.force:
        logging.info("[%s] output exists - skipping (use --force to rebuild)", name)
        stats.output_path = str(out_path)
        stats.seconds = -1.0  # sentinel: skipped
        return stats

    book_cache = cfg.cache_dir / name
    book_cache.mkdir(parents=True, exist_ok=True)

    start = time.time()
    page_markdowns: list[str] = []

    for idx, page_path in enumerate(pages, start=1):
        cache_file = book_cache / f"page-{idx:04d}.md"
        if cache_file.exists() and not cfg.force:
            raw = cache_file.read_text(encoding="utf-8")
        else:
            logging.info("[%s] page %d/%d: %s", name, idx, len(pages), page_path.name)
            image_b64 = encode_image(page_path, cfg.max_pixels)
            raw = unwrap_code_fence(ocr_page(cfg, image_b64))
            cache_file.write_text(raw, encoding="utf-8")

        if raw.strip() == BLANK_SENTINEL or not raw.strip():
            stats.blank_pages += 1
            page_markdowns.append("")
        else:
            page_markdowns.append(raw)

    # ---- assemble ----
    non_blank = [p for p in page_markdowns if p.strip()]
    if cfg.strip_boilerplate:
        non_blank, removed = strip_running_headers(non_blank, len(pages))
        stats.boilerplate_removed = removed

    if cfg.merge_tables:
        document, merged = merge_cross_page_tables(non_blank)
        stats.tables_merged = merged
    else:
        document = "\n\n".join(p.strip("\n") for p in non_blank if p.strip())

    document = re.sub(r"\n{3,}", "\n\n", document).strip() + "\n"

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")
    stats.output_path = str(out_path)
    stats.seconds = time.time() - start
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> Config:
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve() if args.output else input_dir / "_output"
    cache_dir = Path(args.cache).expanduser().resolve() if args.cache else input_dir / "_cache"
    return Config(
        input_dir=input_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        model=args.model,
        host=args.host,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        max_pixels=args.max_pixels,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
        force=args.force,
        only_book=args.book,
        merge_tables=not args.no_merge_tables,
        strip_boilerplate=not args.keep_running_headers,
        dry_run=args.dry_run,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="OCR scanned book folders into per-book Markdown via a local Ollama VLM.")
    p.add_argument("--input", default=str(Path(__file__).resolve().parent),
                   help="Directory whose subfolders are books (default: this script's folder).")
    p.add_argument("--output", default=None, help="Where to write <Book>.md (default: <input>/_output).")
    p.add_argument("--cache", default=None, help="Per-page cache dir (default: <input>/_cache).")
    p.add_argument("--model", default=DEFAULTS["model"], help="Ollama model tag (e.g. qwen3-vl:30b or qwen2.5vl:7b).")
    p.add_argument("--host", default=DEFAULTS["host"], help="Ollama host URL.")
    p.add_argument("--num-ctx", type=int, default=DEFAULTS["num_ctx"], dest="num_ctx")
    p.add_argument("--num-predict", type=int, default=DEFAULTS["num_predict"], dest="num_predict")
    p.add_argument("--max-pixels", type=int, default=DEFAULTS["max_pixels"], dest="max_pixels")
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--timeout", type=int, default=DEFAULTS["timeout"], help="Per-page request timeout (s).")
    p.add_argument("--retries", type=int, default=DEFAULTS["retries"])
    p.add_argument("--book", default=None, help="Process only the book whose folder name matches this.")
    p.add_argument("--force", action="store_true", help="Reprocess even if output/cache already exist.")
    p.add_argument("--no-merge-tables", action="store_true", help="Disable cross-page table stitching.")
    p.add_argument("--keep-running-headers", action="store_true", help="Do not strip repeated headers/footers.")
    p.add_argument("--dry-run", action="store_true", help="List discovered books and page counts, then exit.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = build_config(args)

    if not cfg.input_dir.is_dir():
        logging.error("Input directory not found: %s", cfg.input_dir)
        return 2

    books = discover_books(cfg.input_dir)
    if cfg.only_book:
        books = [b for b in books if b[0] == cfg.only_book]
        if not books:
            logging.error("No book folder named %r under %s", cfg.only_book, cfg.input_dir)
            return 2

    if not books:
        logging.error("No book subfolders with images found under %s", cfg.input_dir)
        return 2

    logging.info("Discovered %d book(s) under %s:", len(books), cfg.input_dir)
    for name, pages in books:
        logging.info("  - %-45s %4d pages", name, len(pages))

    if cfg.dry_run:
        logging.info("Dry run - no model calls made.")
        return 0

    logging.info("Model: %s via %s  (num_ctx=%d, num_predict=%d, max_pixels=%d)",
                 cfg.model, cfg.host, cfg.num_ctx, cfg.num_predict, cfg.max_pixels)

    all_stats: list[BookStats] = []
    for name, pages in books:
        try:
            all_stats.append(process_book(cfg, name, pages))
        except Exception as e:  # keep going to the next book
            logging.error("[%s] FAILED: %s", name, e)

    # ---- verification summary ----
    logging.info("\n===== SUMMARY =====")
    for s in all_stats:
        if s.seconds == -1.0:
            logging.info("%-45s SKIPPED (output existed)", s.name)
            continue
        accounted = s.blank_pages  # blanks + non-blank should equal total
        logging.info("%-45s pages=%d blank=%d tables_merged=%d boilerplate_lines=%d  %.0fs",
                     s.name, s.total_pages, s.blank_pages, s.tables_merged,
                     len(s.boilerplate_removed), s.seconds)
        if s.boilerplate_removed:
            for line in s.boilerplate_removed:
                logging.info("      stripped repeated line: %r", line)
        logging.info("      -> %s", s.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
