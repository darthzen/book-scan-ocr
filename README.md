# book-scan → Markdown

Turns each subfolder of scanned book pages into a single Markdown file, using a
local vision-language model served by Ollama. Built for feeding Karakeep (and
whatever downstream pipeline ingests from there into Milvus).

## What it does

- Treats **each subfolder as one book**; pages are ordered by filename.
- Sends **one page at a time, statelessly** to the model — no growing context,
  so book length never overruns the context window.
- Preserves **heading hierarchy** (`#`/`##`/`###`) and renders **GFM tables**.
- Describes photos, charts, and diagrams as `**Figure:** …` instead of embedding
  images.
- **Stitches tables that span two pages** and strips repeated running
  headers/footers and bare page numbers.
- **Resumable**: every page is cached, so an interrupted run picks up where it
  left off. Books whose output already exists are **skipped** (rerun-friendly —
  drop in new folders and run again).

Output: `_output/<Book Name>.md` (one file per folder). Cache: `_cache/`.

## Setup

```bash
pip install -r requirements.txt          # just Pillow; HTTP is stdlib
ollama pull qwen3-vl:30b                  # primary model
ollama pull qwen2.5vl:7b                  # optional faster fallback
```

## Run

From inside the `book-scan` folder:

```bash
# See what it will process, no model calls:
python ocr_books.py --dry-run

# Process everything (default model qwen3-vl:30b, default host
# http://ollama.ash4d.com:11434 baked in - nothing to set):
python ocr_books.py

# Override the host only if needed (env OLLAMA_HOST or --host both work):
python ocr_books.py --host http://other-host:11434

# Switch to the fallback model:
python ocr_books.py --model qwen2.5vl:7b

# Rebuild a single book from scratch:
python ocr_books.py --book "A Whole New Mind" --force
```

## Key settings (chosen defaults)

| Flag | Default | Why |
|---|---|---|
| `--num-ctx` | 16384 | Fits ~2.7k vision tokens for a full-res page + prompt + output |
| `--num-predict` | 4096 | Headroom for a dense page of Markdown |
| `--max-pixels` | 2600000 | Guard for oversized scans; your 1436×1527 pages pass through untouched |
| `--temperature` | 0 | Deterministic transcription |
| `--timeout` | 1800 | Generous per-page wait; speed is not a concern |

Other flags: `--no-merge-tables`, `--keep-running-headers`, `--force`,
`--book NAME`, `--output DIR`, `--cache DIR`.

## Verification

After each run a summary prints per book: total pages, blank pages, tables
merged, and every repeated line it stripped (so you can confirm it only removed
boilerplate). Spot-check the table- and figure-heavy pages first — those are
where a VLM is most likely to slip.

## Container (optional) — nerdctl / containerd under Lima

```bash
# Build inside your Lima instance (containerd runtime):
nerdctl build -t book-ocr .

# Run, mounting this folder at /data:
nerdctl run --rm -v "$PWD:/data" \
    -e OLLAMA_HOST=http://<ollama-host>:11434 book-ocr --input /data
```

Base image is openSUSE BCI Python per standing policy; the container is only the
HTTP client that drives your Ollama server.

**Reaching Ollama from inside the Lima VM** — pick based on where Ollama runs:

- **Ollama on a lab server** (your setup): just use its hostname/IP, e.g.
  `-e OLLAMA_HOST=http://labserver:11434`. No gateway trickery needed.
- **Ollama on the Mac host**: the container can't use `localhost`. Either pass
  the Mac's LAN IP, or add a host-gateway mapping and use it:
  `nerdctl run --add-host host.docker.internal:host-gateway -e OLLAMA_HOST=http://host.docker.internal:11434 ...`
  (Lima's host is also reachable at `192.168.5.2` from the default VM network.)

Running the script directly with `python ocr_books.py` on the Mac (no container)
sidesteps all of this — the container is optional here.

## Notes / known limits

- **Heading levels** are inferred from visual styling and can occasionally drift
  (a section read as a chapter). This is the least reliable requirement — worth
  a spot-check.
- **Cross-page table merging** is conservative: it only joins when the last line
  of a page and the first line of the next are both table rows with matching
  column counts. Unusual splits may still need a manual touch-up.
- The model choice, prompt, and all thresholds live at the top of
  `ocr_books.py` if you want to tune them.
