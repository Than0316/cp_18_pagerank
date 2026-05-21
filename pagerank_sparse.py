import time
import numpy as np
import scipy.sparse as sp
from graph import load

def calculate_pagerank_sparse(titles, edges, out_degree, dangling, d=0.85, epsilon=1e-6, max_iter=100):
    """
    Computes PageRank using a SciPy CSR Sparse Matrix with decoupled structural metrics tracking.
    """
    N = len(titles)
    
    print("Constructing P^T transition matrix structure...")
    t_build_start = time.time()
    
    # Pack row/col pointer arrays for CSR matrix mapping
    sources = edges[:, 0]
    targets = edges[:, 1]
    data = 1.0 / out_degree[sources]
    
    # Build transposed transformation array: Shape (Targets, Sources)
    P_T = sp.csr_matrix((data, (targets, sources)), shape=(N, N))
    
    t_build = time.time() - t_build_start
    print(f"  --> Matrix Construction Time: {t_build:.4f} seconds")
    print(f"  --> Stored Non-zero Links: {P_T.nnz:,}")
    
    x = np.ones(N, dtype=np.float64) / N
    
    print("Starting BLAS-backed Sparse Power Iteration...")
    t_iter_start = time.time()
    
    for iteration in range(1, max_iter + 1):
        # High-performance sparse matrix-vector multiplication
        link_mass = P_T.dot(x)
        
        lost_mass = np.sum(x[dangling])
        teleport_mass = (d * lost_mass / N) + ((1.0 - d) / N)
        x_next = (d * link_mass) + teleport_mass
        
        error = np.sum(np.abs(x_next - x))
        x = x_next
        
        if error < epsilon:
            t_iter = time.time() - t_iter_start
            print(f"  --> Converged in {iteration} iterations.")
            print(f"  --> Pure Iteration Engine Time: {t_iter:.4f} seconds")
            return x, t_iter, t_build
            
    t_iter = time.time() - t_iter_start
    return x, t_iter, t_build

def print_top_k(x, titles, k=10, category_name=None, categories=None):
    if category_name:
        if category_name not in categories:
            print(f"\nCategory '{category_name}' not found!")
            return
        print(f"\n--- Top {k} Pages in Category: {category_name} ---")
        cat_indices = np.array(categories[category_name])
        cat_scores = x[cat_indices]
        local_top_indices = np.argsort(cat_scores)[-k:][::-1]
        top_indices = cat_indices[local_top_indices]
    else:
        print(f"\n--- Top {k} Pages Globally ---")
        top_indices = np.argsort(x)[-k:][::-1]
        
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank:>2}. {titles[idx]:<35} (Score: {x[idx]:.6f})")

if __name__ == "__main__":
    print("Loading graph arrays...")
    titles, edges, out_degree, dangling, categories = load()
    
    # Run sparse calculations
    scores_sparse, runtime_iter, runtime_build = calculate_pagerank_sparse(titles, edges, out_degree, dangling)
    
    # Output targeted deliverables (Identical to baseline script output)
    print_top_k(scores_sparse, titles, k=10)
    print_top_k(scores_sparse, titles, k=10, category_name="Computing", categories=categories)