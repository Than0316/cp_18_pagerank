import numpy as np
from graph import load

# Import execution modules from your adjacent files
from pagerank import calculate_pagerank_baseline
from pagerank_sparse import calculate_pagerank_sparse

if __name__ == "__main__":
    print("=========================================================")
    print("        PAGERANK ARCHITECTURE PERFORMANCE BENCHMARK       ")
    print("=========================================================\n")
    
    print("Loading data structures...")
    titles, edges, out_degree, dangling, _ = load()
    
    # 1. Run Baseline Test Configuration
    scores_base, time_base = calculate_pagerank_baseline(
        titles, edges, out_degree, dangling
    )
    
    # 2. Run Sparse Engine Configuration
    scores_sparse, time_sparse, time_build = calculate_pagerank_sparse(
        titles, edges, out_degree, dangling
    )
    
    # 3. Cross-Validation check to guarantee mathematical equality
    max_difference = np.max(np.abs(scores_base - scores_sparse))
    
    print("\n" + "="*57)
    print("                     BENCHMARK REPORT                    ")
    print("="*57)
    print(f"Baseline Iteration Core Runtime  : {time_base:.4f} seconds")
    print(f"Sparse Engine Iteration Runtime  : {time_sparse:.4f} seconds")
    print(f"Sparse CSR Matrix Assembly Overhead: {time_build:.4f} seconds")
    print("-" * 57)
    
    # Performance assessment
    speedup = time_base / time_sparse
    print(f"Pure Computation Acceleration Factor: {speedup:.2f}x Faster")
    print(f"Numerical Variance Tolerance Check : {max_difference:.2e}")
    
    if max_difference < 1e-10:
        print("Verification Status             : SUCCESS (Identical Arrays)")
    else:
        print("Verification Status             : WARNING (Precision Drift)")
    print("="*57)