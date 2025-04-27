nvcc -std=c++14 \
     -Xcompiler -fPIC \
     -shared \
     kmeans_host.cu kmeans_v1.cu \
     -o libkmeans.so \
     -lcudart \
     && \
nvcc --ptx kmeans_v1.cu \
     -gencode arch=compute_61,code=sm_61 \
     -o kmeans_kernels.ptx