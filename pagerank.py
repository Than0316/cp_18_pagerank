import time
import numpy as np
from graph import load

def calculate_pagerank_baseline(titles, edges, out_degree, dangling, d=0.85, epsilon=1e-6, max_iter=100):
    """
    Computes PageRank using native NumPy operations (baseline implementation)
    with precise timeline metrics tracking.
    """
    N = len(titles)
    x = np.ones(N, dtype=np.float64) / N
    src_out_degrees = out_degree[edges[:, 0]]
    
    print(f"Starting Power Iteration Baseline Engine (N = {N:,} nodes)...")
    t_start = time.time()
    
    for iteration in range(1, max_iter + 1):
        x_next = np.zeros(N, dtype=np.float64)
        
        # Distribute mass along edges
        link_mass = x[edges[:, 0]] / src_out_degrees
        np.add.at(x_next, edges[:, 1], link_mass)
        
        # Correct for lost mass at dangling nodes
        lost_mass = np.sum(x[dangling])
        teleport_mass = (d * lost_mass / N) + ((1.0 - d) / N)
        x_next = (d * x_next) + teleport_mass
        
        # Evaluate convergence via L1 norm
        error = np.sum(np.abs(x_next - x))
        x = x_next
        
        if error < epsilon:
            t_total = time.time() - t_start
            print(f"  --> Converged in {iteration} iterations.")
            print(f"  --> Core Execution Time: {t_total:.4f} seconds")
            print(f"  --> Average Iteration Speed: {t_total / iteration:.4f} seconds/iter")
            return x, t_total
            
    t_total = time.time() - t_start
    return x, t_total

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
    
    # Run loop execution and collect runtime metric data
    scores_base, runtime_base = calculate_pagerank_baseline(titles, edges, out_degree, dangling)
    
    # Output targeted deliverables
    print_top_k(scores_base, titles, k=10)
    print_top_k(scores_base, titles, k=10, category_name="Computing", categories=categories)