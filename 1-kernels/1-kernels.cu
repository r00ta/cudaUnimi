/*
CUDA C extends C by allowing the programmer to define C functions, called kernels, that, when called, 
are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C functions.

A kernel is defined using the __global__ declaration specifier and the number of CUDA threads that execute that kernel 
for a given kernel call is specified using a new <<<...>>>execution configuration syntax (see C Language Extensions). 
Each thread that executes the kernel is given a unique thread ID that is accessible within the kernel through the 
built-in threadIdx variable.

As an illustration, the following sample code adds two vectors A and B of size N and stores the result into vector C:

Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz47mO3Jq1H

Here, each of the N threads that execute VecAdd() performs one pair-wise addition.

*/
#include <stdio.h>

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(){
	int N = 1024;
   	size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input vectors
    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(d_A, d_B, d_C);
}
