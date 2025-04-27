/* kmeans_cpu.c – naïve single-threaded K-Means ------------------ */
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ------------------------------------------------------------------------- */
/*  Random initialisation helper    */
void initialize_centroids(const float *data,
                          float *centroids_out,
                          const int *random_indices,
                          int n_features,
                          int k_clusters)
{
    for (int k = 0; k < k_clusters; ++k)
    {
        int idx = random_indices[k];
        memcpy(&centroids_out[k * n_features],
               &data[idx * n_features],
               n_features * sizeof(float));
    }
}

/* ------------------------------------------------------------------------- */
/*  Main K-Means loop – identical signature to the CUDA version.             */
int kmeans(float *data,
           float *centroids,
           int *labels,
           int *cluster_counts,
           int n_samples,
           int n_features,
           int k_clusters,
           int n_iterations)
{
    /* scratch space for accumulating new centroids each iteration */
    float *centroid_sums = (float *)calloc((size_t)k_clusters * n_features,
                                           sizeof(float));
    if (!centroid_sums)
        return -1; /* out-of-memory */

    for (int iter = 0; iter < n_iterations; ++iter)
    {
        /* zero counts & sums -------------------------------------------- */
        memset(cluster_counts, 0, k_clusters * sizeof(int));
        memset(centroid_sums, 0, (size_t)k_clusters * n_features * sizeof(float));

        /* step 1&2 – assign each sample to nearest centroid and sum ------ */
        for (int i = 0; i < n_samples; ++i)
        {
            float best_dist = FLT_MAX;
            int best_k = -1;

            for (int k = 0; k < k_clusters; ++k)
            {
                float dist = 0.0f;
                for (int f = 0; f < n_features; ++f)
                {
                    float diff = data[i * n_features + f] -
                                 centroids[k * n_features + f];
                    dist += diff * diff; /* squared Euclidean */
                }
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_k = k;
                }
            }

            labels[i] = best_k;
            ++cluster_counts[best_k];

            for (int f = 0; f < n_features; ++f)
                centroid_sums[best_k * n_features + f] +=
                    data[i * n_features + f];
        }

        /* step 3 – recompute centroids ---------------------------------- */
        for (int k = 0; k < k_clusters; ++k)
        {
            if (cluster_counts[k] == 0)
                continue; /* keep old position */
            for (int f = 0; f < n_features; ++f)
                centroids[k * n_features + f] =
                    centroid_sums[k * n_features + f] /
                    (float)cluster_counts[k];
        }
    }

    free(centroid_sums);
    return 0; /* success */
}
