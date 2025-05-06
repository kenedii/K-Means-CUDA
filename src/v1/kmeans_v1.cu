#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

// Step 1: Initialize - Randomly initialize K points as starting centroids
__global__ void initialize_centroids(float *data, float *centroids_out, int *random_indices, int n_features, int k_clusters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Each thread initializes one centroid
    if (i < k_clusters)
    {
        int sample_idx = random_indices[i]; // Randomly selected sample index passed from host
        for (int f = 0; f < n_features; f++)
        {
            centroids_out[i * n_features + f] = data[sample_idx * n_features + f]; // Copy sample to centroid
        }
    }
}

// Step 2: Assign points - Assign each point to the closest cluster
extern "C" __global__ void compute_distances(float *data, float *centroids, int n_samples, int n_features, int k_clusters, int *labels_out, int *cluster_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Get the sample index
    if (i < n_samples)
    {
        float min_dist = FLT_MAX;
        int cluster_idx = -1;
        for (int j = 0; j < k_clusters; j++) // For each cluster
        {
            float dist = 0.0f;
            for (int f = 0; f < n_features; f++)
            {
                float diff = data[i * n_features + f] - centroids[j * n_features + f]; // Compute distance
                dist += diff * diff;
            }

            if (dist < min_dist) // Check if it's the minimum distance
            {
                min_dist = dist;
                cluster_idx = j; // Update the closest cluster
            }
        }
        labels_out[i] = cluster_idx;                // Assign the point to a cluster
        atomicAdd(&cluster_counts[cluster_idx], 1); // Increment the cluster count
    }
}

// Step 3: Update centroids - Sum the points in each cluster
__global__ void update_centroids(float *data, int n_samples, int n_features, int k_clusters, int *labels, float *centroids_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Get the sample index
    if (i < n_samples)
    {
        int cluster_idx = labels[i]; // Get the assigned cluster
        for (int f = 0; f < n_features; f++)
        {
            atomicAdd(&centroids_out[cluster_idx * n_features + f], data[i * n_features + f]); // Sum the feature values
        }
    }
}

// Step 4: Normalize centroids - Average the sum to get the new centroid positions
__global__ void normalize_centroids(int *cluster_counts, float *centroids_out, int k_clusters, int n_features)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Get the cluster index
    if (i < k_clusters && cluster_counts[i] > 0)   // Avoid division by zero
    {
        for (int f = 0; f < n_features; f++)
        {
            centroids_out[i * n_features + f] /= cluster_counts[i]; // Normalize the centroid
        }
    }
}
