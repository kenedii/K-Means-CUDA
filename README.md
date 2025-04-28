# K-Means-CUDA
K Means Clustering Algorithm enhanced by CUDA C along with Python implementations using the C/Cuda Backend.

Implementations:
- CPU: Naive CPU implementation
- v1: Naive GPU implementation

Stats - 1 million point dataset, 16 features, 5 clusters, 5 iter:
- Custom CPU : 1.534 s
- Custom GPU : 0.605 s  (×2.5 faster than CPU)
- scikit-learn : 1.031 s 

How to run:
- pip install -r requirements.txt
- Ensure .so libraries and kernel PTX are in same directory as the k_means_loader.py
