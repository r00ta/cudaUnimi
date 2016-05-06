
#include <stdio.h>
#define N 1024
// Kernel definition
__device__ int A[N][N];
__device__ int B[N][N];
__device__ int C[N][N];

__global__ void MatAdd()
{
	int n = 1933;
    int div = threadIdx.x + threadIdx.y + threadIdx.z;
    if()
}

int main()
{
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>();
}