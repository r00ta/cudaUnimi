/*The number of threads per block and the number of blocks per grid specified in the <<<...>>> syntax can be of type int or dim3.
 Two-dimensional blocks or grids can be specified as in the example above.

Each block within the grid can be identified by a one-dimensional, two-dimensional, or three-dimensional index accessible 
within the kernel through the built-in blockIdx variable. The dimension of the thread block is accessible within the kernel
through the built-in blockDim variable.

Extending the previous MatAdd() example to handle multiple blocks, the code becomes as follows.

*/

#include <stdio.h>
#define N 1024
__device__ int A[N][N];
__device__ int B[N][N];
__device__ int C[N][N];

__global__ void MatAdd()
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>();
        cudaDeviceSynchronize();
}

/*A thread block size of 16x16 (256 threads), although arbitrary in this case, is a common choice. The grid is created with enough 
blocks to have one thread per matrix element as before. For simplicity, this example assumes that the number of threads per 
grid in each dimension is evenly divisible by the number of threads per block in that dimension, although that need not be 
the case.

Thread blocks are required to execute independently: It must be possible to execute them in any order, in parallel or in series. 
This independence requirement allows thread blocks to be scheduled in any order across any number of cores as illustrated by 
Figure 5, enabling programmers to write code that scales with the number of cores.

Threads within a block can cooperate by sharing data through some shared memory and by synchronizing their execution to
 coordinate memory accesses. More precisely, one can specify synchronization points in the kernel by calling the __syncthreads() 
 intrinsic function; __syncthreads() acts as a barrier at which all threads in the block must wait before any is allowed to
  proceed. Shared Memory gives an example of using shared memory.


For efficient cooperation, the shared memory is expected to be a low-latency memory near each processor core 
(much like an L1 cache) and __syncthreads() is expected to be lightweight.
*/