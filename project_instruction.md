# 18. PageRank on Wikipedia – Graphs & Linear Algebra

PageRank ranks the nodes of a directed graph by the stationary distribution of a random walk on it — the algorithm that built Google. Apply it to Wikipedia.

## Goals

1. Use the Simple English Wikipedia dump (much smaller than `enwiki`). Stream-parse it to extract the `(page → page)` link graph.

2. Implement PageRank by power iteration with a damping factor; handle dangling nodes correctly. Discuss convergence (norm vs. iteration count).

3. Output the top-$k$ pages globally and the top-$k$ pages within a chosen category.

4. **Extensions**
   - *Personalised PageRank* from a seed page
   - Sparse-matrix linear-algebra version using `scipy.sparse`

## Relevant Courses

- **6**: Graphs — BFS, DFS, topological sorting, SCC
- **7**: MST & shortest paths
