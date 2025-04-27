# K-Means-CUDA
K Means Clustering Algorithm enhanced by CUDA C along with Python implementations using the C/Cuda Backend.

Implementations:
- CPU: Naive CPU implementation
- v1: Naive GPU implementation, 5x speedup over CPU on 1 million point dataset, 16 features, 5 clusters

How to run:
- pip install -r requirements.txt
- Ensure .so libraries and kernel PTX are in same directory as the k_means_loader.py
