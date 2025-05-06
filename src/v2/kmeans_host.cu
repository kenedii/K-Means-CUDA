// kmeans_host.cu – host-side wrapper callable from Python via ctypes
//
// This file provides two exported functions:
//   - initialize_centroids_host: Launches the initialize_centroids kernel
//   - kmeans: Runs the main K-Means loop by launching compute_distances, update_centroids, and normalize_centroids kernels

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h> // For printf in error checking

/* --------------------------------------------------------------------------
   Device-kernel prototypes
   -------------------------------------------------------------------------- */

/*
 * Step-1: Initialize centroids
 * Each thread copies one randomly-chosen sample into the centroid array.
 */
__global__ void initialize_centroids(
    float *data,
    float *centroids_out,
    int *random_indices,
    int n_features,
    int k_clusters);

/*
 * Step-2: Assign points
 * For every sample i, compute its squared Euclidean distance to each centroid j.
 * Store the index of the nearest centroid in labels_out[i] and atomically increment cluster_counts[j].
 */
extern "C" __global__ void compute_distances(
    float *data,
    float *centroids,
    int n_samples,
    int n_features,
    int k_clusters,
    int *labels_out,
    int *cluster_counts);

/*
 * Step-3: Update centroids
 * Each thread adds its sample’s feature vector to the running sum for the cluster it belongs to.
 * At the end of the kernel, centroids_out holds per-cluster **sums** (not yet divided by cluster size).
 */
__global__ void update_centroids(
    float *data,
    int n_samples,
    int n_features,
    int k_clusters,
    int *labels,
    float *centroids_out);

/*
 * Step-4: Normalize centroids
 * Divide each centroid sum by the corresponding cluster count so that centroids_out holds the **mean** feature vector for every cluster.
 */
__global__ void normalize_centroids(
    int *cluster_counts,
    float *centroids_out,
    int k_clusters,
    int n_features);

/* ==========================================================================
   Host functions – exposed to Python
   ========================================================================== */

/*
 * Host function to initialize centroids
 * Launches the initialize_centroids kernel to copy randomly selected samples to centroids.
 * Parameters:
 *   - data: Device pointer to the input data (n_samples * n_features)
 *   - centroids_out: Device pointer to the output centroids (k_clusters * n_features)
 *   - random_indices: Device pointer to the randomly selected indices (k_clusters)
 *   - n_features: Number of features in the data
 *   - k_clusters: Number of clusters
 */
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
        printf("initialize_centroids kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

/*
 * Main K-Means host function
 * Runs the K-Means algorithm for a specified number of iterations.
 * Parameters:
 *   - data: Device pointer to the input data (n_samples * n_features)
 *   - centroids: Device pointer to the centroids (k_clusters * n_features)
 *   - labels: Device pointer to the output labels (n_samples)
 *   - cluster_counts: Device pointer to the cluster sizes (k_clusters)
 *   - n_samples: Number of samples in the data
 *   - n_features: Number of features in the data
 *   - k_clusters: Number of clusters
 *   - n_iterations: Number of iterations to run the algorithm
 * Returns:
 *   - 0 on success
 */
extern "C" int kmeans(
    void *data,
    void *centroids,
    void *labels,
    void *cluster_counts,
    int n_samples,
    int n_features,
    int k_clusters,
    int n_iterations)
{
    // Cast void* to typed pointers
    float *d_data = reinterpret_cast<float *>(data);
    float *d_centroids = reinterpret_cast<float *>(centroids);
    int *d_labels = reinterpret_cast<int *>(labels);
    int *d_cluster_counts = reinterpret_cast<int *>(cluster_counts);

    // Launch configuration
    const int blockSize = 256;
    const int numBlocks_samples = (n_samples + blockSize - 1) / blockSize;
    const int numBlocks_clusters = (k_clusters + blockSize - 1) / blockSize;

    // Shared memory size for compute_distances kernel
    const int shared_mem_size = k_clusters * sizeof(int);

    // Main K-Means loop
    for (int iter = 0; iter < n_iterations; ++iter)
    {
        // Reset cluster-size counters to zero
        cudaMemset(d_cluster_counts, 0, k_clusters * sizeof(int));

        // Step-2: Assign each point to the closest centroid
        compute_distances<<<numBlocks_samples, blockSize, shared_mem_size>>>(
            d_data,
            d_centroids,
            n_samples,
            n_features,
            k_clusters,
            d_labels,
            d_cluster_counts);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("compute_distances kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1; // Return error code
        }

        // Clear centroid accumulators for this iteration
        cudaMemset(d_centroids, 0, k_clusters * n_features * sizeof(float));

        // Step-3: Sum the feature vectors per cluster
        update_centroids<<<numBlocks_samples, blockSize>>>(
            d_data,
            n_samples,
            n_features,
            k_clusters,
            d_labels,
            d_centroids);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("update_centroids kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1; // Return error code
        }

        // Step-4: Average the sums to obtain new centroid positions
        normalize_centroids<<<numBlocks_clusters, blockSize>>>(
            d_cluster_counts,
            d_centroids,
            k_clusters,
            n_features);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("normalize_centroids kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1; // Return error code
        }
    }

    return 0; // Success
}