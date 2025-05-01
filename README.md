# K-Means-CUDA
K Means Clustering Algorithm enhanced by CUDA C along with Python implementations using the C/Cuda Backend.

Implementations:
- CPU: Naive CPU implementation in pure C
- v1: Naive GPU implementation in Cuda C
- v2: GPU Implementation in Cuda C - Uses shared memory over global in compute_distances kernel

Stats - 10 million point dataset, 25 features, 5 clusters, 5 iter:
- Custom CPU : 26.880 s
- Custom GPU v1 : 1.768 s  (×15.2 faster than CPU)
- Custom GPU v2 : 1.417 s  (×19.0 faster than CPU)
- scikit-learn : 4.883 s  (×5.5 faster than CPU)
- PyTorch : 3.978 s  (×6.8 faster than CPU)

How to run:
- pip install -r requirements.txt
- Ensure .so libraries and kernel PTX are in same directory as the k_means_loader.py
