# K-Means-CUDA
K Means Clustering Algorithm enhanced by CUDA C along with Python implementations using the C/Cuda Backend.


Implementations:
- CPU: Naive CPU implementation in pure C
- v1: Naive GPU implementation in Cuda C
- v2: GPU Implementation in Cuda C - Uses shared memory over global in compute_distances kernel

Stats - 10 million point dataset, 25 features, 5 clusters, 5 iter:
- Custom CPU : 35.220 s
- Custom GPU v1 : 2.459 s 
- Custom GPU v2 : 2.204 s 
- scikit-learn (Lloyd) : 5.629 s  
- scikit-learn (Elkan) : 6.504 s 
- PyTorch : 4.935 s  

How to run:
- pip install -r requirements.txt

To use the Python K Means class:
- Specify parameters n_clusters, n_features, backend (["gpu", "cpu"]) (mandatory)
- Parameter lib_path is the path to .so or .dll library (mandatory)
- Parameter ptx_path is the path to .ptx kernels for JIT compilation (needed if using predict function)
- Parameter block_size is the number of threads per block in the GPU CUDA grid (optional)
- Parameter kernel_version is either 'v1' or 'v2'. Set this if using the GPU implementations. v1 by default. Optional parameter if using CPU.

The K Means class functions are:
- .fit() - Params: X: np.ndarray, n_iterations: int
- .predict() - Params: X: np.ndarray
