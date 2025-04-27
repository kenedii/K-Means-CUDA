#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <float.h>

void initialize_centroids(float *data, int n_samples, int n_features, int k_clusters, float *centroids_out)
{
    for (int i = 0; i < k_clusters; i++) // Get n_clusters clusters
    {
        int sample_idx = rand() % n_samples; // Randomly select a sample index
        for (int f = 0; f < n_features; f++)
        {
            centroids_out[i * n_features + f] = data[sample_idx * n_features + f]; // Copy the sample to centroids
        }
    }
}

__global__ void compute_distances(float *data, float *centroids, int n_samples, int n_features, int k_clusters, int *labels_out, int *cluster_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Get the sample index
    if (i < n_samples)                             // For each point
    {
        float min_dist = FLT_MAX;
        int cluster_idx = -1;
        for (int j = 0; j < k_clusters; j++) // For each cluster
        {
            // Computes the Euclidean distance between the sample and the centroid
            float dist = 0.0f;
            for (int f = 0; f < n_features; f++)
            {
                float diff = data[i * n_features + f] - centroids[j * n_features + f]; // Compute the distance
                dist += diff * diff;                                                   // Square the difference and accumulate
            }

            if (dist < min_dist) // Check if it's the minimum distance
            {
                min_dist = dist;
                cluster_idx = j; // Update the cluster index
            }
        }
        labels_out[i] = cluster_idx;                // Assign the closest cluster to the sample
        atomicAdd(&cluster_counts[cluster_idx], 1); // Increment the count for the cluster
    }
}

__global__ void update_centroids(float *data, int n_samples, int n_features, int k_clusters, int *labels, float *centroids_out, int *cluster_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_samples)
    {
        int cluster_idx = labels[i]; // Get the cluster index for the sample
        for (int f = 0; f < n_features; f++)
        {
            atomicAdd(&centroids_out[cluster_idx * n_features + f], data[i * n_features + f]); // Update the centroid for the cluster
        }
    }
}

__global__ void normalize_centroids(int *cluster_counts, float *centroids_out, int k_clusters, int n_features)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k_clusters)
    {
        if (cluster_counts[i] > 0) // Avoid division by zero
        {
            for (int f = 0; f < n_features; f++)
            {
                centroids_out[i * n_features + f] /= cluster_counts[i]; // Normalize the centroid
            }
        }
    }
}

int kmeans(float *data, int n_samples, int n_features, int k_clusters, int n_iterations, int *labels)
{
    // Step 1: Initialize: Randomly initialize K points as starting centroids
    float *centroids = (float *)malloc(k_clusters * n_features * sizeof(float));
    initialize_centroids(data, n_samples, n_features, k_clusters, centroids);

    // Step 2: Assign points: Compare the distance between each point and the centroids to assign it to a cluster

    // Allocate memory for distances and labels on the device
    float *d_data, *d_centroids;
    int *d_labels, *d_cluster_counts;

    cudaMalloc((void **)&d_data, n_samples * n_features * sizeof(float));
    cudaMalloc((void **)&d_centroids, k_clusters * n_features * sizeof(float));
    cudaMalloc((void **)&d_cluster_counts, k_clusters * sizeof(int));
    cudaMalloc((void **)&d_labels, n_samples * sizeof(int));

    cudaMemcpy(d_data, data, n_samples * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, k_clusters * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, n_samples * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_cluster_counts, 0, k_clusters * sizeof(int));

    int blockSize = 256;                                     // Number of threads per block
    int numBlocks = (n_samples + blockSize - 1) / blockSize; // Number of blocks needed (Total blocks in grid)

    for (int iter = 0; iter < n_iterations; iter++)
    {
        cudaMemset(d_cluster_counts, 0, k_clusters * sizeof(int)); // Reset cluster counts

        // Here we compute the distances between each point and the centroids. d_labels[i] has cluster index for point i. d_cluster_counts[j] has number of points in cluster j.
        compute_distances<<<numBlocks, blockSize>>>(d_data, d_centroids, n_samples, n_features, k_clusters, d_labels, d_cluster_counts); // Launch kernel to compute distances
        cudaDeviceSynchronize();                                                                                                         // Make sure all threads are done before moving on

        cudaMemset(d_centroids, 0, k_clusters * n_features * sizeof(float)); // clear centroids
        // Step 3: Update centroids: Compute the new centroids as the mean of the points assigned to each cluster
        update_centroids<<<numBlocks, blockSize>>>(d_data, n_samples, n_features, k_clusters, d_labels, d_centroids, d_cluster_counts); // Launch kernel to update centroids
        cudaDeviceSynchronize();                                                                                                        // Make sure all threads are done before moving on

        normalize_centroids<<<(k_clusters + blockSize - 1) / blockSize, blockSize>>>(d_cluster_counts, d_centroids, k_clusters, n_features); // Normalize the centroids
        cudaDeviceSynchronize();                                                                                                             // Make sure all threads are done before moving on
    }
}