#!/usr/bin/env python3
"""
sample_benchmark.py

Benchmark and visualize the custom K-Means wrapper (CPU / GPU)
*and* the reference implementation from scikit-learn.

Artefacts expected in the working directory:

    GPU  →  kmeans_kernels.ptx   &   libkmeans.so (or kmeans.dll on Windows)
    CPU  →  libkmeans_cpu.so (or kmeans_cpu.dll on Windows)
"""

from pathlib import Path
import time
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as SKLearnKMeans

from k_means_loader import KMeans as CustomKMeans

# Define library and PTX file names here
if os.name == "nt":
    CPU_LIB = "kmeans_cpu.dll"
    GPU_LIB = "kmeans.dll"
else:
    CPU_LIB = "libkmeans_cpu.so"
    GPU_LIB = "libkmeans.so"

PTX_FILE = "kmeans_kernels.ptx"

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def run_custom_kmeans(backend: str, X: np.ndarray, *, k: int, d: int, lib_path: str, ptx_path: Optional[str] = None):
    """Fit the custom wrapper and return (model, elapsed_sec)."""
    km = CustomKMeans(
        n_clusters=k,
        n_features=d,
        backend=backend,            # "cpu" | "gpu"
        lib_path=lib_path,
        ptx_path=ptx_path,
        block_size=256,             # ignored on CPU
    )
    t0 = time.perf_counter()
    km.fit(X, n_iterations=5)
    return km, time.perf_counter() - t0


def run_sklearn_kmeans(X: np.ndarray, *, k: int):
    """Fit scikit-learn's KMeans and return (model, elapsed_sec)."""
    km = SKLearnKMeans(
        n_clusters=k,
        init="k-means++",
        n_init="auto",              # scikit-learn ≥1.4
        max_iter=10,
        algorithm="lloyd",
        random_state=42, tol=0.0
    )
    t0 = time.perf_counter()
    km.fit(X)
    return km, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
def main():
    # --------------------------- Synthetic Data -----------------------------
    rng_state = 42
    n_samples = 1000000
    n_features = 16
    n_clusters = 5

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=rng_state,
    )
    X = X.astype(np.float32)

    # --------------------------- Custom CPU ---------------------------------
    try:
        if not Path(CPU_LIB).exists():
            raise FileNotFoundError(f"Missing library: {CPU_LIB}")
        cpu_model, cpu_t = run_custom_kmeans("cpu", X, k=n_clusters, d=n_features, lib_path=CPU_LIB)
        print(f"Custom CPU : {cpu_t:.3f} s")
    except Exception as exc:
        print(f"[CPU] skipped → {exc}")
        cpu_model, cpu_t = None, float("inf")

    # --------------------------- Custom GPU ---------------------------------
    try:
        if not (Path(GPU_LIB).exists() and Path(PTX_FILE).exists()):
            raise FileNotFoundError(f"Missing artefacts: {GPU_LIB} and/or {PTX_FILE}")
        gpu_model, gpu_t = run_custom_kmeans("gpu", X, k=n_clusters, d=n_features, lib_path=GPU_LIB, ptx_path=PTX_FILE)
        if cpu_t < float("inf"):
            print(f"Custom GPU : {gpu_t:.3f} s  (×{cpu_t/gpu_t:.1f} faster than CPU)")
        else:
            print(f"Custom GPU : {gpu_t:.3f} s")
    except Exception as exc:
        print(f"[GPU] skipped → {exc}")
        gpu_model, gpu_t = None, float("inf")

    # --------------------------- Scikit-learn -------------------------------
    skl_model, skl_t = run_sklearn_kmeans(X, k=n_clusters)
    if cpu_t < float("inf"):
        faster_vs_cpu = cpu_t / skl_t
        print(f"scikit-learn : {skl_t:.3f} s  (×{faster_vs_cpu:.1f} faster than custom CPU)")
    else:
        print(f"scikit-learn : {skl_t:.3f} s")

    # --------------------------- Summary ------------------------------------
    times = {
        "Custom CPU": cpu_t,
        "Custom GPU": gpu_t,
        "scikit-learn": skl_t,
    }
    valid_times = {k: v for k, v in times.items() if v < float("inf")}
    if valid_times:
        best_time = min(valid_times.values())
        fastest = next(k for k, v in valid_times.items() if v == best_time)
        print(f"\n▶  Fastest overall: {fastest}")
    else:
        print("\n▶  No valid runs to compare.")

    # ---------------------- Quick Visual Sanity-Check -----------------------
    # (just plot the custom CPU clustering to keep the figure simple)
    if cpu_model is not None:
        pca = PCA(n_components=2, random_state=rng_state)
        X2d = pca.fit_transform(X)
        plt.figure(figsize=(5, 4))
        plt.scatter(X2d[:, 0], X2d[:, 1], c=cpu_model.labels_, s=2, cmap="tab10")
        plt.title("Custom CPU K-Means (PCA-2D)")
        plt.xlabel("PC-1")
        plt.ylabel("PC-2")
        plt.tight_layout()
        plt.savefig("kmeans_plot.png")      # saved for later inspection
    else:
        print("Skipping visualization as CPU model is not available")


if __name__ == "__main__":
    main()