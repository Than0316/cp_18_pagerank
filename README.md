# PageRank on Simple English Wikipedia

PageRank implemented via power iteration on the Simple English Wikipedia link graph (~294k pages, ~3.5M unique edges).

## Structure

```
graph/
  parser.py      — stream-parses the Wikipedia XML dump → titles.txt, edges.npy, categories.json
  filter.py      — deduplicates edges.npy → edges_dedup.npy
  loader.py      — loads the graph as numpy arrays (titles, edges, out_degree, dangling, categories)
data/
  titles.txt     — one article title per line; line index = node id
  edges.npy      — raw edges (E, 2) int32 array
  edges_dedup.npy — deduplicated edges
  categories.json
pagerank.py           — baseline NumPy power iteration
pagerank_sparse.py    — sparse CSR matrix version (SciPy)
pagerank_comparison.py — benchmarks both implementations against each other
```

## Usage

**1. Parse the dump** (only needed if regenerating the graph from scratch):

Download `simplewiki-latest-pages-articles.xml.bz2` from https://dumps.wikimedia.org/simplewiki/latest/ and place it in the project root, then:
```bash
python -m graph.parser
```

**2. Deduplicate edges** (~23% of raw edges are duplicates):
```bash
python -m graph.filter
# or to also strip self-loops:
python -m graph.filter --drop-self-loops
```

**3. Run PageRank:**
```bash
python pagerank.py           # baseline
python pagerank_sparse.py    # sparse
python pagerank_comparison.py  # benchmark both
```

## Algorithm

Standard PageRank with damping factor `d = 0.85`, convergence via L1 norm < 1e-6:

```
PR(i) = (1 - d) / N  +  d * Σ  PR(j) / out_degree(j)
                             j → i
```

Dangling nodes (no outbound links) redistribute their mass uniformly across all nodes each iteration.

## Results

| Implementation | Speed |
|---|---|
| NumPy baseline | `np.add.at` scatter over edge list |
| SciPy sparse   | BLAS-backed CSR matrix-vector multiply |
