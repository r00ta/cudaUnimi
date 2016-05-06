/*For convenience, threadIdx is a 3-component vector, so that threads can be identified using a one-dimensional, 
two-dimensional, or three-dimensional thread index, forming a one-dimensional, two-dimensional, or 
three-dimensional block of threads, called a thread block. This provides a natural way to invoke computation 
across the elements in a domain such as a vector, matrix, or volume.

The index of a thread and its thread ID relate to each other in a straightforward way: For a one-dimensional block, 
they are the same; for a two-dimensional block of size (Dx, Dy),the thread ID of a thread of index (x, y) is (x + y Dx); 
for a three-dimensional block of size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy).

As an example, the following code adds two matrices A and B of size NxN and stores the result into matrix C:

Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz47mgjEZkJ
*/
#include <stdio.h>
#define N 1024
// Kernel definition
__device__ int A[N][N];
__device__ int B[N][N];
__device__ int C[N][N];

__global__ void MatAdd()
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>();
}

/*There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same processor 
core and must share the limited memory resources of that core. On current GPUs, a thread block may contain up to 1024 threads.

However, a kernel can be executed by multiple equally-shaped thread blocks, so that the total number of threads is equal to the
 number of threads per block times the number of blocks.

Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional grid of thread blocks as illustrated by Figure 
6. The number of thread blocks in a grid is usually dictated by the size of the data being processed or the number of processors 
in the system, which it can greatly exceed.

Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz47rZpJHPt
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
*/