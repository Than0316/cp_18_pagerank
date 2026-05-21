"""
Stream-parse the Simple English Wikipedia XML dump and save the link graph.

Output files (written next to this script):
  titles.txt       — one article title per line; line number = node index
  edges.npy        — int32 array of shape (E, 2), each row is (source, target)
  categories.json  — dict mapping node index (as string) to list of category names

Usage:
  python -m graph.parser
"""

import bz2
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import iterparse

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).parent.parent
DUMP   = HERE / "simplewiki-latest-pages-articles.xml.bz2"
DATA   = HERE / "data"
TITLES = DATA / "titles.txt"
EDGES  = DATA / "edges.npy"
CATS   = DATA / "categories.json"

# ── regex patterns ─────────────────────────────────────────────────────────────
LINK_RE     = re.compile(r'\[\[([^\]|#]+)')
CATEGORY_RE = re.compile(r'\[\[Category:([^\]|]+)', re.IGNORECASE)
SKIP_PREFIXES = ('File:', 'Image:', 'Category:', 'Special:', 'Template:',
                 'Wikipedia:', 'Help:', 'Portal:')

# ── helpers ────────────────────────────────────────────────────────────────────
def _iter_articles(dump_path: Path):
    """Yield (title, text) for every main-namespace, non-redirect article."""
    with bz2.open(dump_path, 'rb') as f:
        for _, elem in iterparse(f, events=('end',)):
            if elem.tag.split('}')[-1] != 'page':
                continue
            ns    = elem.findtext('.//{*}ns')
            title = (elem.findtext('.//{*}title') or '').strip()
            text  = elem.findtext('.//{*}text') or ''
            elem.clear()                        # release memory immediately
            if ns != '0' or not title:
                continue
            if text.lstrip().startswith('#REDIRECT'):
                continue
            yield title, text


# ── parse ──────────────────────────────────────────────────────────────────────
def parse(dump_path: Path):
    # ── pass 1: collect titles only (~30 MB) ──────────────────────────────────
    print("Pass 1: collecting titles ...")
    t0 = time.time()
    titles    = []
    title_idx = {}
    for title, _ in _iter_articles(dump_path):
        title_idx[title] = len(titles)
        titles.append(title)
    print(f"  {len(titles)} articles found  ({time.time()-t0:.0f}s)\n")

    # ── pass 2: extract links and categories, integers only (~80 MB) ──────────
    print("Pass 2: extracting links ...")
    t0 = time.time()
    flat_edges = []                # flat list [src, tgt, src, tgt, ...]
    categories = defaultdict(list) # index → [category names]
    n_raw = 0

    for n_pages, (title, text) in enumerate(_iter_articles(dump_path), 1):
        src = title_idx[title]

        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            # MediaWiki capitalises the first letter of link targets
            target = target[0].upper() + target[1:]
            n_raw += 1
            tgt = title_idx.get(target)
            if tgt is not None:             # drop links to unknown pages
                flat_edges.append(src)
                flat_edges.append(tgt)

        for m in CATEGORY_RE.finditer(text):
            categories[src].append(m.group(1).strip())

        if n_pages % 10_000 == 0:
            elapsed = time.time() - t0
            n_edges = len(flat_edges) // 2
            print(f"  {n_pages:>6} articles  |  {n_edges:>8} edges  |  {elapsed:.0f}s")

    n_edges = len(flat_edges) // 2
    print(f"\nPass 2 done: {n_edges} edges kept "
          f"({n_raw - n_edges} dropped — targets not in dump)")

    edges = np.array(flat_edges, dtype=np.int32).reshape(-1, 2)
    return titles, edges, dict(categories)


# ── save ───────────────────────────────────────────────────────────────────────
def save(titles, edges, categories):
    TITLES.write_text('\n'.join(titles), encoding='utf-8')
    print(f"Saved {len(titles)} titles  →  {TITLES.name}")

    np.save(EDGES, edges)
    print(f"Saved {len(edges)} edges   →  {EDGES.name}")

    with open(CATS, 'w', encoding='utf-8') as f:
        json.dump(categories, f)
    print(f"Saved categories          →  {CATS.name}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not DUMP.exists():
        raise FileNotFoundError(f"Dump not found: {DUMP}")
    DATA.mkdir(exist_ok=True)

    print(f"Parsing {DUMP.name}  ({DUMP.stat().st_size / 1e6:.0f} MB compressed)\n")
    t0 = time.time()

    titles, edges, categories = parse(DUMP)
    save(titles, edges, categories)

    print(f"\nDone in {time.time() - t0:.0f}s")
    print(f"  Pages    : {len(titles)}")
    print(f"  Edges    : {len(edges)}")
    print(f"  Avg out-degree : {len(edges)/len(titles):.1f} links per page")
