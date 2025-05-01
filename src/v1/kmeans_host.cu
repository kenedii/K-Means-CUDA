// kmeans_host.cu  – host-side wrapper callable from Python via ctypes
//
// This file provides one exported function:
//
//     extern "C" int kmeans(...)
//
// It launches the four device kernels that implement the classic
// K-Means loop.  All device pointers are pre-allocated in Python with
// PyCUDA and passed in as raw void* pointers.

#include <cuda_runtime.h>
#include <float.h>

/* --------------------------------------------------------------------------
   Device-kernel prototypes  (must exactly match signatures in PTX)
   -------------------------------------------------------------------------- */

/*
 * Step-1  (initialised from Python): initialize_centroids
 * Each thread copies one randomly-chosen sample into the centroid array.
 */
__global__ void initialize_centroids(
    float *data,
    float *centroids_out,
    int *random_indices,
    int n_features,
    int k_clusters);

extern "C" void initialize_centroids_host(
    void *data,
    void *centroids_out,
    void *random_indices,
    int n_features,
    int k_clusters)
{
    // Cast void* to typed pointers
    float *d_data = reinterpret_cast<float *>(data);
    float *d_centroids_out = reinterpret_cast<float *>(centroids_out);
    int *d_random_indices = reinterpret_cast<int *>(random_indices);

    // Launch configuration
    const int blockSize = 256;
    const int numBlocks = (k_clusters + blockSize - 1) / blockSize;

    // Launch the kernel
    initialize_centroids<<<numBlocks, blockSize>>>(
        d_data,
        d_centroids_out,
        d_random_indices,
        n_features,
        k_clusters);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return; // Error in kernel launch
    }
}

/*
 * Step-2  – Assign points
 * For every sample i, compute its squared Euclidean distance to each
 * centroid j.  Store the index of the nearest centroid in labels_out[i]
 * and atomically increment cluster_counts[j].
 */
__global__ void compute_distances(
    float *data,
    float *centroids,
    int n_samples,
    int n_features,
    int k_clusters,
    int *labels_out,
    int *cluster_counts);

/*
 * Step-3  – Update centroids
 * Each thread adds its sample’s feature vector to the running sum for the
 * cluster it belongs to.  At the end of the kernel, centroids_out holds
 * per-cluster **sums** (not yet divided by cluster size).
 */
__global__ void update_centroids(
    float *data,
    int n_samples,
    int n_features,
    int k_clusters,
    int *labels,
    float *centroids_out);

/*
 * Step-4  – Normalize centroids
 * Divide each centroid sum by the corresponding cluster count so that
 * centroids_out holds the **mean** feature vector for every cluster.
 */
__global__ void normalize_centroids(
    int *cluster_counts,
    float *centroids_out,
    int k_clusters,
    int n_features);

/* ==========================================================================
   Host entry point – exposed to Python
   ========================================================================== */

extern "C" int kmeans(void *data,
                      void *centroids,
                      void *labels,
                      void *cluster_counts,
                      int n_samples,
                      int n_features,
                      int k_clusters,
                      int n_iterations)
{
    /* ------------------------------------------------------------------
       Cast raw void* addresses (originating in Python) to typed pointers
       ------------------------------------------------------------------ */
    float *d_data = reinterpret_cast<float *>(data);
    float *d_centroids = reinterpret_cast<float *>(centroids);
    int *d_labels = reinterpret_cast<int *>(labels);
    int *d_cluster_counts = reinterpret_cast<int *>(cluster_counts);

    /* Launch configuration */
    const int blockSize = 256;                                               // Threads per block
    const int numBlocks_samples = (n_samples + blockSize - 1) / blockSize;   // Blocks for samples
    const int numBlocks_clusters = (k_clusters + blockSize - 1) / blockSize; // Blocks for clusters

    /* -----------------------------  Main loop  ---------------------------- */
    for (int iter = 0; iter < n_iterations; ++iter)
    {
        /* Reset cluster-size counters */
        cudaMemset(d_cluster_counts, 0, k_clusters * sizeof(int));

        // ------------------------------------------------------------------
        // Step-2 : Assign each point to the closest centroid
        // ------------------------------------------------------------------
        compute_distances<<<numBlocks_samples, blockSize>>>(
            d_data,
            d_centroids,
            n_samples,
            n_features,
            k_clusters,
            d_labels,
            d_cluster_counts);
        cudaDeviceSynchronize();

        /* Clear centroid accumulators so we can sum fresh this iteration */
        cudaMemset(d_centroids, 0, k_clusters * n_features * sizeof(float));

        // ------------------------------------------------------------------
        // Step-3 : Sum the feature vectors per cluster
        // ------------------------------------------------------------------
        update_centroids<<<numBlocks_samples, blockSize>>>(
            d_data,
            n_samples,
            n_features,
            k_clusters,
            d_labels,
            d_centroids);
        cudaDeviceSynchronize();

        // ------------------------------------------------------------------
        // Step-4 : Average the sums to obtain new centroid positions
        // ------------------------------------------------------------------
        normalize_centroids<<<numBlocks_clusters, blockSize>>>(
            d_cluster_counts,
            d_centroids,
            k_clusters,
            n_features);
        cudaDeviceSynchronize();
    }

    return 0; // success
}