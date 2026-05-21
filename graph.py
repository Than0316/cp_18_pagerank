"""
Loads the Wikipedia link graph as compact numpy arrays.

    edges[k] = [src, tgt]    means page src links to page tgt
    out_degree[i]             number of outbound links of page i
    dangling[i]               True if page i has no outbound links

Use titles[i] to get the page name for index i.
"""

"""
Usage:
    from graph import load
    titles, edges, out_degree, dangling, categories = load()

──────────────────────────────────────────────────────────────
SYNTAX REFERENCE
──────────────────────────────────────────────────────────────
Get the index of a page by name:
    i = titles.index("April")

Get outbound links of page i  (arrows leaving i):
    edges[edges[:, 0] == i, 1]

Get inbound links of page i  (arrows arriving at i):
    edges[edges[:, 1] == i, 0]

Get out-degree of page i  (number of outbound links):
    out_degree[i]

Get in-degree of page i  (number of pages linking to i):
    np.sum(edges[:, 1] == i)

Check if page a links to page b:
    np.any((edges[:, 0] == a) & (edges[:, 1] == b))

Get all dangling nodes  (pages with no outbound links):
    np.where(dangling)[0]

Get all pages in a category:
    categories["Months"]          # list of indices
──────────────────────────────────────────────────────────────
"""

import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent / "data"


def load(data_dir: Path = DATA):
    titles   = (data_dir / "titles.txt").read_text(encoding="utf-8").splitlines()
    edges    = np.load(data_dir / "edges_dedup.npy")                  # (E, 2) int32
    raw_cats = json.loads((data_dir / "categories.json").read_text(encoding="utf-8"))

    n = len(titles)

    # out-degree: count how many times each source appears in edges
    out_degree = np.bincount(edges[:, 0], minlength=n).astype(np.int32)
    dangling   = out_degree == 0                                 # bool array

    # category name → list of page indices
    categories = {}
    for idx_str, cat_list in raw_cats.items():
        for cat in cat_list:
            categories.setdefault(cat, []).append(int(idx_str))

    return titles, edges, out_degree, dangling, categories


if __name__ == "__main__":
    import time, tracemalloc
    tracemalloc.start()
    t0 = time.time()

    titles, edges, out_degree, dangling, categories = load()

    current, peak = tracemalloc.get_traced_memory()
    print(f"Load time  : {time.time()-t0:.1f}s")
    print(f"RAM used   : {current/1e6:.0f} MB  (peak {peak/1e6:.0f} MB)")
    print(f"Pages      : {len(titles):,}")
    print(f"Edges      : {len(edges):,}")
    print(f"Dangling   : {dangling.sum():,}  ({100*dangling.mean():.1f}% of pages)")
    print()

    # example
    i = titles.index("April")
    targets = edges[edges[:, 0] == i, 1]
    print(f"Outlinks of 'April': {[titles[t] for t in targets]} ...")
