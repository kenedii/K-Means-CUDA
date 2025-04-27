#!/bin/bash

# Set output name
OUTPUT_LIB="libkmeans_cpu.so"

# Compile the C file into a shared object (.so)
gcc -fPIC -shared kmeans_cpu.c -o $OUTPUT_LIB

# Optional: print success message
echo "Built $OUTPUT_LIB successfully."
