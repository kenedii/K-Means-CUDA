#!/usr/bin/env python3
"""
sample_benchmark.py

Benchmark and visualize the custom K-Means wrapper in k_means_loader.py
against synthetic data generated with scikit-learn.

• Creates an (n_samples × n_features) dataset with make_blobs
• Runs the K-Means class on CPU and GPU
• Prints which back-end is faster
• Projects the data to 2-D with PCA and shows the clustered result(s)

Make sure the compiled artefacts are in the same directory:

    GPU  →  kmeans_kernels.ptx   &   libkmeans.so
    CPU  →  libkmeans_cpu.so
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

from k_means_loader import KMeans


def run_kmeans(backend: str, X: np.ndarray, n_clusters: int, n_features: int):
    """Fit and time the wrapper; return (model, elapsed_seconds)."""
    km = KMeans(
        n_clusters=n_clusters,
        n_features=n_features,
        backend=backend,
        backend_prefix="kmeans",       # -> libkmeans[ _cpu ].so + kmeans_kernels.ptx
        block_size=256,                # ignored on CPU
    )
    t0 = time.perf_counter()
    km.fit(X, n_iterations=100)
    elapsed = time.perf_counter() - t0
    return km, elapsed


def main():
    # --------------------------- data ---------------------------------
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
    X = X.astype(np.float32)      # <- cast only the data array

    # --------------------------- CPU ----------------------------------
    cpu_model, cpu_time = run_kmeans("cpu", X, n_clusters, n_features)
    print(f"CPU time : {cpu_time:.3f} s")

    # --------------------------- GPU ----------------------------------
    gpu_available = Path("kmeans_kernels.ptx").exists() and Path("libkmeans.so").exists()
    try:
        if gpu_available:
            gpu_model, gpu_time = run_kmeans("gpu", X, n_clusters, n_features)
            print(f"GPU time : {gpu_time:.3f} s ({cpu_time/gpu_time}x speedup)")
        else:
            raise FileNotFoundError("GPU artefacts not found")
    except Exception as exc:
        print(f"[GPU] skipped → {exc}")
        gpu_model, gpu_time = None, float("inf")

    faster = "GPU" if gpu_time < cpu_time else "CPU"
    print(f"\n▶  {faster} back-end was faster.")

    # ----------------------- PCA visualisation ------------------------
    pca = PCA(n_components=2, random_state=rng_state)
    X_2d = pca.fit_transform(X)

    if gpu_model:  # two sub-plots
        fig, (ax_cpu, ax_gpu) = plt.subplots(1, 2, figsize=(10, 4))
    else:          # only CPU
        fig, ax_cpu = plt.subplots(1, 1, figsize=(5, 4))

    s = 4  # marker size

    ax_cpu.scatter(X_2d[:, 0], X_2d[:, 1], c=cpu_model.labels_, s=s, cmap="tab10")
    ax_cpu.set_title("CPU K-Means (PCA-2D)")
    ax_cpu.set_xlabel("PC-1")
    ax_cpu.set_ylabel("PC-2")

    if gpu_model:
        ax_gpu.scatter(X_2d[:, 0], X_2d[:, 1], c=gpu_model.labels_, s=s, cmap="tab10")
        ax_gpu.set_title("GPU K-Means (PCA-2D)")
        ax_gpu.set_xlabel("PC-1")
        ax_gpu.set_ylabel("PC-2")

    fig.suptitle("Custom K-Means clustering")
    plt.tight_layout()
    plt.savefig("kmeans_plot.png")  # Save the plot to a file


if __name__ == "__main__":
    main()
