import time
import numpy as np
import scipy.sparse as sp
from graph import load

def calculate_ppr(titles, edges, out_degree, dangling, seed_title, d=0.85, epsilon=1e-6, max_iter=100, personalize_dangling=True):
    """
    Computes Topic-Personalized PageRank (PPR) using a SciPy CSR Sparse Matrix.
    Includes a toggle to experiment with dangling node boundary conditions.
    """
    N = len(titles)
    
    # 1. Locate the Seed Page and build the Personalization Vector (v)
    try:
        seed_idx = titles.index(seed_title)
    except ValueError:
        print(f"Error: Seed page '{seed_title}' not found in the dataset.")
        return None, 0, 0
        
    v = np.zeros(N, dtype=np.float64)
    v[seed_idx] = 1.0
    
    print(f"\n--- [Extension B: Personalized PageRank] ---")
    print(f"Seed Page: '{seed_title}' (Index: {seed_idx})")
    print(f"Dangling Node Behavior: {'Snap to Seed' if personalize_dangling else 'Global Leak'}")
    
    # 2. Build the Sparse Matrix
    print("Constructing P^T transition matrix...")
    t_build_start = time.time()
    
    sources = edges[:, 0]
    targets = edges[:, 1]
    data = 1.0 / out_degree[sources]
    P_T = sp.csr_matrix((data, (targets, sources)), shape=(N, N))
    
    t_build = time.time() - t_build_start
    print(f"  --> Matrix Construction Time: {t_build:.4f} seconds")
    
    # 3. Initialization
    # We can start everyone equal, the engine will quickly pull the mass to the seed.
    x = np.ones(N, dtype=np.float64) / N
    
    print("Starting BLAS-backed Personalized Iteration...")
    t_iter_start = time.time()
    
    for iteration in range(1, max_iter + 1):
        # Matrix-vector multiplication (surfers clicking links)
        link_mass = P_T.dot(x)
        
        # Calculate trapped mass at dangling nodes
        lost_mass = np.sum(x[dangling])
        
        # --- THE BOUNDARY CONDITION TOGGLE ---
        if personalize_dangling:
            # Academic standard: Trapped surfers jump back to the seed page
            dangling_jump = (d * lost_mass) * v
        else:
            # Your experimental approach: Trapped surfers scatter globally
            dangling_jump = (d * lost_mass) / N
            
        # The 15% boredom teleportation ALWAYs goes to the seed page
        teleport_jump = (1.0 - d) * v
        
        # Combine all probability flows
        x_next = (d * link_mass) + dangling_jump + teleport_jump
        
        # Check Convergence
        error = np.sum(np.abs(x_next - x))
        x = x_next
        
        if error < epsilon:
            t_iter = time.time() - t_iter_start
            print(f"  --> Converged in {iteration} iterations.")
            print(f"  --> Iteration Engine Time: {t_iter:.4f} seconds")
            return x, t_iter, t_build
            
    t_iter = time.time() - t_iter_start
    return x, t_iter, t_build

def print_top_k(x, titles, k=10, title_prefix=""):
    print(f"\n--- Top {k} Pages {title_prefix} ---")
    top_indices = np.argsort(x)[-k:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"{rank:>2}. {titles[idx]:<35} (Score: {x[idx]:.6f})")

if __name__ == "__main__":
    print("Loading graph arrays...")
    titles, edges, out_degree, dangling, categories = load()
    
    seed = "Alan Turing"
    
    # Run 1: Strict Personalization (Dangling nodes snap to seed)
    scores_strict, t_iter_strict, t_build = calculate_ppr(
        titles, edges, out_degree, dangling, 
        seed_title=seed, personalize_dangling=True
    )
    if scores_strict is not None:
        print_top_k(scores_strict, titles, k=15, title_prefix="(Strict PPR)")
        
    # Run 2: Leaky Personalization (Dangling nodes jump globally)
    scores_leaky, t_iter_leaky, _ = calculate_ppr(
        titles, edges, out_degree, dangling, 
        seed_title=seed, personalize_dangling=False
    )
    if scores_leaky is not None:
        print_top_k(scores_leaky, titles, k=15, title_prefix="(Global Dangling Leak)")