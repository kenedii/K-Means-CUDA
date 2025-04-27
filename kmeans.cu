#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
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

__global__ void compute_distances(float *data, float *centroids, int n_samples, int n_features, int k_clusters, int *labels_out)
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
                dist += pow(data[i * n_features + f] - centroids[j * n_features + f], 2); // Compute the distance
            }

            if (dist < min_dist) // Check if it's the minimum distance
            {
                min_dist = dist;
                cluster_idx = j; // Update the cluster index
            }
        }
        labels_out[i] = cluster_idx; // Assign the closest cluster to the sample
    }
}

int kmeans(float *data, int n_samples, int n_features, int k_clusters, int n_iterations, int *labels)
{
    // Step 1: Randomly initialize K points as starting centroids
    float *centroids = (float *)malloc(k_clusters * n_features * sizeof(float));
    initialize_centroids(data, n_samples, n_features, k_clusters, centroids);

    // Step 2: Allocate memory for distances and labels on the device
    float *d_data, *d_centroids, *d_distances;
    int *d_labels;
}