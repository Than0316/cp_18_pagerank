import time
import numpy as np
from graph import load

def calculate_pagerank(titles, edges, out_degree, dangling, d=0.85, epsilon=1e-6, max_iter=100):
    """
    Computes PageRank using native NumPy array operations (baseline implementation).
    """
    N = len(titles)
    
    # Initialize everyone with an equal share of the probability mass
    x = np.ones(N, dtype=np.float64) / N
    
    # Pre-fetch the out-degree for every source node in the edge list.
    # This prevents us from having to look it up repeatedly inside the loop.
    src_out_degrees = out_degree[edges[:, 0]]
    
    print(f"\nStarting Power Iteration Engine (N = {N:,} nodes)...")
    t_start = time.time()
    
    for iteration in range(1, max_iter + 1):
        # Create a fresh array of 0s for the next step's scores
        x_next = np.zeros(N, dtype=np.float64)
        
        # 1. DISTRIBUTE MASS ALONG EDGES
        # Calculate exactly how much rank each outgoing link carries
        # (Current rank of the source page / Total outbound links of the source page)
        link_mass = x[edges[:, 0]] / src_out_degrees
        
        # Add that mass to the target pages (edges[:, 1])
        # np.add.at is a highly efficient way to do this without a slow Python for-loop
        np.add.at(x_next, edges[:, 1], link_mass)
        
        # 2. CALCULATE LOST MASS (Sinks / Dangling Nodes)
        # Sum up the rank of all pages that have zero outgoing links
        lost_mass = np.sum(x[dangling])
        
        # 3. APPLY DAMPING AND TELEPORTATION
        # Add the teleportation chance + redistribute the lost mass uniformly
        teleport_mass = (d * lost_mass / N) + ((1.0 - d) / N)
        x_next = (d * x_next) + teleport_mass
        
        # 4. CHECK CONVERGENCE (L1 Norm)
        error = np.sum(np.abs(x_next - x))
        print(f"  Iteration {iteration:>2}: L1 Error = {error:.8f}")
        
        # Update the vector for the next loop
        x = x_next
        
        if error < epsilon:
            print(f"Converged successfully in {iteration} iterations! ({time.time() - t_start:.2f}s)")
            break
            
    return x

def print_top_k(x, titles, k=10, category_name=None, categories=None):
    """
    Retrieves and prints the top k pages, either globally or filtered by a category.
    """
    if category_name:
        if category_name not in categories:
            print(f"\nCategory '{category_name}' not found!")
            return
            
        print(f"\n--- Top {k} Pages in Category: {category_name} ---")
        # Isolate only the indices belonging to this category
        cat_indices = np.array(categories[category_name])
        cat_scores = x[cat_indices]
        
        # Sort locally within the category, get the top k, then map back to global indices
        local_top_indices = np.argsort(cat_scores)[-k:][::-1]
        top_indices = cat_indices[local_top_indices]
        
    else:
        print(f"\n--- Top {k} Pages Globally ---")
        # np.argsort sorts ascending, so we slice the last k and reverse it [::-1]
        top_indices = np.argsort(x)[-k:][::-1]
        
    # Print the results beautifully
    for rank, idx in enumerate(top_indices, 1):
        title = titles[idx]
        score = x[idx]
        print(f"{rank:>2}. {title:<35} (Score: {score:.6f})")

if __name__ == "__main__":
    # 1. Load the pre-processed data from graph.py
    print("Loading graph data...")
    titles, edges, out_degree, dangling, categories = load()
    
    # 2. Run the baseline engine
    pagerank_scores = calculate_pagerank(titles, edges, out_degree, dangling)
    
    # 3. Output Core Deliverables
    print_top_k(pagerank_scores, titles, k=15)
    
    # Try a few sample categories (make sure these categories actually exist in Simple English Wikipedia!)
    print_top_k(pagerank_scores, titles, k=10, category_name="Computing", categories=categories)
    print_top_k(pagerank_scores, titles, k=10, category_name="Mathematics", categories=categories)