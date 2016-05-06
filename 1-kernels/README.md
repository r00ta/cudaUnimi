# CUDA C - Kernels

This chapter introduces the main concepts behind the CUDA programming model by outlining how they are exposed in C. An extensive description of CUDA C is given in Programming Interface.

Full code for the vector addition example used in this chapter and the next can be found in the vectorAdd CUDA sample.

Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz47mQwDf4I

My settings

```bash
$ nvidia-smi
Thu May  5 13:33:06 2016       
+------------------------------------------------------+                       
| NVIDIA-SMI 352.79     Driver Version: 352.79         |                       
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla M2050         Off  | 0000:06:00.0     Off |                    0 |
| N/A   N/A    P0    N/A /  N/A |      6MiB /  2687MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla M2090         Off  | 0000:11:00.0     Off |                    0 |
| N/A   N/A    P0    79W / 225W |     10MiB /  5375MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla M2090         Off  | 0000:14:00.0     Off |                    0 |
| N/A   N/A    P0    81W / 225W |     10MiB /  5375MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

For example compile `1-kernels.cu` with
```bash 
$ nvcc -arch sm_20 1-kernels.cu -o 1-kernels
```

CUDA C extends C by allowing the programmer to define C functions, called kernels, that, when called, 
are executed N times in parallel by N different CUDA threads, as opposed to only once like regular C functions.

A kernel is defined using the __global__ declaration specifier and the number of CUDA threads that execute that kernel 
for a given kernel call is specified using a new <<<...>>>execution configuration syntax (see C Language Extensions). 
Each thread that executes the kernel is given a unique thread ID that is accessible within the kernel through the 
built-in threadIdx variable.

As an illustration, the following sample code adds two vectors A and B of size N and stores the result into vector C:

Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz47mO3Jq1H

Here, each of the N threads that execute VecAdd() performs one pair-wise addition.

[VecAdd](1-kernels.cu)

For convenience, threadIdx is a 3-component vector, so that threads can be identified using a one-dimensional, 
two-dimensional, or three-dimensional thread index, forming a one-dimensional, two-dimensional, or 
three-dimensional block of threads, called a thread block. This provides a natural way to invoke computation 
across the elements in a domain such as a vector, matrix, or volume.

The index of a thread and its thread ID relate to each other in a straightforward way: For a one-dimensional block, 
they are the same; for a two-dimensional block of size (Dx, Dy),the thread ID of a thread of index (x, y) is (x + y Dx); 
for a three-dimensional block of size (Dx, Dy, Dz), the thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy).

As an example, the following code adds two matrices A and B of size NxN and stores the result into matrix C:

[threadHierarchy](2-threadHierarchy.cu)

There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same processor 
core and must share the limited memory resources of that core. On current GPUs, a thread block may contain up to 1024 threads.

However, a kernel can be executed by multiple equally-shaped thread blocks, so that the total number of threads is equal to the
 number of threads per block times the number of blocks.

Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional grid of thread blocks as illustrated by Figure 
6. The number of thread blocks in a grid is usually dictated by the size of the data being processed or the number of processors 
in the system, which it can greatly exceed.

The number of threads per block and the number of blocks per grid specified in the <<<...>>> syntax can be of type int or dim3.
 Two-dimensional blocks or grids can be specified as in the example above.

Each block within the grid can be identified by a one-dimensional, two-dimensional, or three-dimensional index accessible 
within the kernel through the built-in blockIdx variable. The dimension of the thread block is accessible within the kernel
through the built-in blockDim variable.

Extending the previous MatAdd() example to handle multiple blocks, the code becomes as follows.

[3-matAdd](3-matAdd.cu)

A thread block size of 16x16 (256 threads), although arbitrary in this case, is a common choice. The grid is created with enough 
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
