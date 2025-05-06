import ctypes
import os
from pathlib import Path
from typing import Literal, Optional
import numpy as np

class KMeans:
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        *,
        backend: Literal["gpu", "cpu"] = "gpu",
        lib_path: str,
        ptx_path: Optional[str] = None,
        block_size: int = 256,
        kernel_version: Literal["v1", "v2"] = "v1",
    ):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.block_size = block_size
        self.backend = backend.lower()
        self.kernel_version = kernel_version.lower()

        # Determine if shared memory is needed based on kernel_version
        if self.kernel_version not in ["v1", "v2"]:
            raise ValueError("kernel_version must be either 'v1' or 'v2'")
        self.requires_shared_memory = (self.kernel_version == "v2")  # True for v2, False for v1

        # Validate and load shared library
        lib_path = Path(lib_path).resolve()
        if not lib_path.is_file():
            raise FileNotFoundError(f"Shared library not found: {lib_path}")
        self._lib = ctypes.cdll.LoadLibrary(str(lib_path))

        # GPU backend setup
        if self.backend == "gpu":
            if ptx_path is None:
                raise ValueError("ptx_path must be provided for GPU backend")
            ptx_path = Path(ptx_path).resolve()
            if not ptx_path.is_file():
                raise FileNotFoundError(f"Missing PTX file: {ptx_path}")
            self._ptx_path = ptx_path

            import pycuda.driver as cuda
            import pycuda.autoinit
            import pycuda.gpuarray as gpuarray

            self.cuda = cuda
            self.gpuarray = gpuarray
            self._mod = None
            self._compute_distances_kernel = None

        elif self.backend != "cpu":
            raise ValueError('backend must be either "gpu" or "cpu"')

        # Common C entry-point
        self._kmeans_host = self._lib.kmeans
        self._kmeans_host.restype = ctypes.c_int
        self._kmeans_host.argtypes = [
            ctypes.c_void_p,  # data
            ctypes.c_void_p,  # centroids
            ctypes.c_void_p,  # labels
            ctypes.c_void_p,  # cluster_counts
            ctypes.c_int,     # n_samples
            ctypes.c_int,     # n_features
            ctypes.c_int,     # k_clusters
            ctypes.c_int,     # n_iterations
        ]

        # Backend-specific initialization
        if self.backend == "gpu":
            self._init_centroids_host = self._lib.initialize_centroids_host
            self._init_centroids_host.argtypes = [
                ctypes.c_void_p,  # data
                ctypes.c_void_p,  # centroids_out
                ctypes.c_void_p,  # random_indices
                ctypes.c_int,     # n_features
                ctypes.c_int,     # k_clusters
            ]
        elif self.backend == "cpu":
            self._init_centroids_host = self._lib.initialize_centroids
            self._init_centroids_host.argtypes = [
                ctypes.c_void_p,  # data
                ctypes.c_void_p,  # centroids_out
                ctypes.c_void_p,  # random_indices
                ctypes.c_int,     # n_features
                ctypes.c_int,     # k_clusters
            ]

        self.labels_ = None
        self.centroids_ = None

    def fit(self, X: np.ndarray, n_iterations: int = 20, *, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features}")
        n_samples = X.shape[0]

        if self.backend == "gpu":
            cuda = self.cuda
            gpuarray = self.gpuarray

            d_data = gpuarray.to_gpu(X)
            d_labels = gpuarray.zeros(n_samples, dtype=np.int32)
            d_centroids = gpuarray.zeros(
                (self.n_clusters, self.n_features), dtype=np.float32
            )
            d_cluster_counts = gpuarray.zeros(self.n_clusters, dtype=np.int32)

            # Step 1: Initialize centroids via host function
            rand_idx = rng.choice(n_samples, self.n_clusters, replace=False).astype(np.int32)
            d_rand_idx = gpuarray.to_gpu(rand_idx)
            self._init_centroids_host(
                int(d_data.gpudata),
                int(d_centroids.gpudata),
                int(d_rand_idx.gpudata),
                self.n_features,
                self.n_clusters,
            )

            # Main loop
            status = self._kmeans_host(
                int(d_data.gpudata),
                int(d_centroids.gpudata),
                int(d_labels.gpudata),
                int(d_cluster_counts.gpudata),
                n_samples,
                self.n_features,
                self.n_clusters,
                n_iterations,
            )
            if status:
                raise RuntimeError(f"kmeans() CUDA path returned {status}")

            self.labels_ = d_labels.get()
            self.centroids_ = d_centroids.get()

        else:  # CPU backend
            labels = np.empty(n_samples, dtype=np.int32)
            centroids = np.empty((self.n_clusters, self.n_features), dtype=np.float32)
            cluster_counts = np.empty(self.n_clusters, dtype=np.int32)

            rand_idx = rng.choice(n_samples, self.n_clusters, replace=False).astype(np.int32)
            self._init_centroids_host(
                X.ctypes.data,
                centroids.ctypes.data,
                rand_idx.ctypes.data,
                self.n_features,
                self.n_clusters,
            )

            status = self._kmeans_host(
                X.ctypes.data,
                centroids.ctypes.data,
                labels.ctypes.data,
                cluster_counts.ctypes.data,
                n_samples,
                self.n_features,
                self.n_clusters,
                n_iterations,
            )
            if status:
                raise RuntimeError(f"kmeans() CPU path returned {status}")

            self.labels_ = labels
            self.centroids_ = centroids

        return self

    def predict(self, X: np.ndarray):
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")

        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features}")
        n_samples = X.shape[0]

        if self.backend == "gpu":
            cuda = self.cuda
            gpuarray = self.gpuarray

            # Load PTX only when needed for predict
            if self._mod is None:
                with self._ptx_path.open("r") as f:
                    self._mod = cuda.module_from_buffer(f.read().encode())
                self._compute_distances_kernel = self._mod.get_function("compute_distances")

            d_data = gpuarray.to_gpu(X)
            d_labels = gpuarray.zeros(n_samples, dtype=np.int32)
            d_cluster_counts = gpuarray.zeros(self.n_clusters, dtype=np.int32)
            d_centroids = gpuarray.to_gpu(self.centroids_.astype(np.float32))

            grid = ((n_samples + self.block_size - 1) // self.block_size, 1, 1)
            # Set shared memory size based on kernel version
            shared_mem_size = (self.n_clusters * 4) if self.requires_shared_memory else 0  # 4 bytes per int
            self._compute_distances_kernel(
                d_data.gpudata,
                d_centroids.gpudata,
                np.int32(n_samples),
                np.int32(self.n_features),
                np.int32(self.n_clusters),
                d_labels.gpudata,
                d_cluster_counts.gpudata,
                block=(self.block_size, 1, 1),
                grid=grid,
                shared=shared_mem_size  # Dynamically set shared memory
            )
            cuda.Context.synchronize()
            return d_labels.get()

        # CPU: pure NumPy
        dists = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(dists, axis=1).astype(np.int32)

    @property
    def clusters(self):
        return self.n_clusters