"""
Removes duplicate edges from edges.npy and saves the result as
data/edges_dedup.npy.

A duplicate is any (src, tgt) pair that appears more than once.
Self-loops (src == tgt) are kept unless you pass --drop-self-loops.
"""

import argparse
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "data"


def filter_duplicates(data_dir: Path = DATA, drop_self_loops: bool = False) -> None:
    edges_path = data_dir / "edges.npy"
    out_path   = data_dir / "edges_dedup.npy"

    print(f"Loading {edges_path} ...")
    edges = np.load(edges_path)
    n_before = len(edges)
    print(f"  Edges before dedup: {n_before:,}")

    if drop_self_loops:
        self_loop_mask = edges[:, 0] == edges[:, 1]
        n_self = self_loop_mask.sum()
        edges = edges[~self_loop_mask]
        print(f"  Self-loops removed: {n_self:,}")

    # np.unique with axis=0 returns sorted unique rows
    edges_dedup = np.unique(edges, axis=0)
    n_after = len(edges_dedup)
    n_removed = n_before - n_after

    print(f"  Duplicates removed: {n_removed:,}  ({100 * n_removed / n_before:.1f}%)")
    print(f"  Edges after dedup:  {n_after:,}")

    np.save(out_path, edges_dedup)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate edges.npy")
    parser.add_argument(
        "--drop-self-loops",
        action="store_true",
        help="Also remove self-loops (edges where src == tgt)",
    )
    args = parser.parse_args()

    filter_duplicates(drop_self_loops=args.drop_self_loops)
