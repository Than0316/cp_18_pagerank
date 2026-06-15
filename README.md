# PageRank on Simple English Wikipedia

PageRank implemented via power iteration on the Simple English Wikipedia link graph (~294k pages, ~3.5M unique edges).

## Structure

```
graph/
  parser.py        — stream-parses the Wikipedia XML dump → titles.txt, edges.npy, categories.json
  filter.py        — deduplicates edges.npy → edges_dedup.npy
  loader.py        — loads the graph as numpy arrays (titles, edges, out_degree, dangling, categories)
data/
  titles.txt       — one article title per line; line index = node id
  edges.npy        — raw edges, (E, 2) int32 array
  edges_dedup.npy  — deduplicated edges
  categories.json  — per-page category memberships
pagerank.py               — baseline NumPy power iteration
pagerank_sparse.py        — sparse CSR version (SciPy + BLAS)
pagerank_personalized.py  — topic-personalized PageRank
pagerank_comparison.py    — benchmarks baseline vs sparse

pagerank_comparison.ipynb     — Extension A: baseline vs CSR benchmark + scaling experiments
pagerank_personalized.ipynb   — Extension B: topic-personalized PageRank (seed: Alan Turing)
dumping_number_analysis.ipynb — Extension C: convergence dynamics across damping factors
pagerank_checker.ipynb        — cross-validation against NetworkX
```

## Requirements

Python 3.8+ and:

```bash
pip install numpy scipy matplotlib networkx
```

All other dependencies (`bz2`, `xml`, `json`, etc.) are part of the Python standard library.

## How to run

The `data/` files are already included. Steps 1 and 2 are only needed to regenerate them from scratch. Commands use `python3`; replace with `python` if that is your system default.

**1. (Optional) Parse the dump:**

Download `simplewiki-latest-pages-articles.xml.bz2` from https://dumps.wikimedia.org/simplewiki/latest/, place it in the project root, then:

```bash
python3 -m graph.parser
```

**2. (Optional) Deduplicate edges:**

```bash
python3 -m graph.filter
# to also strip self-loops:
python3 -m graph.filter --drop-self-loops
```

**3. Run PageRank:**

```bash
python3 pagerank.py               # baseline (NumPy scatter-add)
python3 pagerank_sparse.py        # sparse CSR (SciPy + BLAS)
python3 pagerank_personalized.py  # topic-personalized PageRank (default seed: Alan Turing)
python3 pagerank_comparison.py    # benchmark baseline vs sparse and print speedup
```

**4. Run the notebooks:**

```bash
jupyter notebook
# or
jupyter lab
```

Open the `.ipynb` files and run all cells.
