# Compile shared library with shader for compute_61
nvcc -std=c++14 \
     -Xcompiler -fPIC \
     -shared \
     kmeans_host.cu kmeans_v2.cu \
     -o libkmeans.so \
     -lcudart \
     -gencode arch=compute_61,code=sm_61

# Generate PTX for compute_61
nvcc --ptx kmeans_v2.cu \
     -arch=compute_61 \
     -o kmeans_kernels.ptx