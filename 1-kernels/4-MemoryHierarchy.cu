/* CUDA threads may access data from multiple memory spaces during their execution as illustrated by Figure 7.
				thread
					|								 ___________
					\								|Per Thread |
					|<------------------------------>|     local |
					/								|__memory___|
					\ 
					
				thread block
				\\\\\\\\\\\								_________________
				///////////<-------------------------->|                 |
				\\\\\\\\\\\<-------------------------->|Per block shared |
				///////////<-------------------------->|    memory       |
				\\\\\\\\\\\							   |_________________|
													_____________________________
				Grid 0 <------------------------->  |							 |
				Grid 1<-------------------------->  |Global memory 				 |
				....<---------------------------->  |							 |
				Grid n<-------------------------->  |____________________________|


				Thread Block
Each thread has private local memory. Each thread block has shared memory visible to all threads of the block 
and with the same lifetime as the block. All threads have access to the same global memory.

There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces.
 The global, constant, and texture memory spaces are optimized for different memory usages (see Device Memory Accesses). 
 Texture memory also offers different addressing modes, as well as data filtering, for some specific data formats 
 (see Texture and Surface Memory).

The global, constant, and texture memory spaces are persistent across kernel launches by the same application.

 As illustrated by Figure 8, the CUDA programming model assumes that the CUDA threads execute on a physically separate device that operates 
 as a coprocessor to the host running the C program. This is the case, for example, when the kernels execute on a GPU and the rest of the C 
 program executes on a CPU.

The CUDA programming model also assumes that both the host and the device maintain their own separate memory spaces in DRAM, referred to as 
host memory and device memory, respectively. Therefore, a program manages the global, constant, and texture memory spaces visible to kernels 
through calls to the CUDA runtime (described in Programming Interface). This includes device memory allocation and deallocation as well as data 
transfer between host and device memory.

The compute capability of a device is represented by a version number, also sometimes called its "SM version". 
This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU.

The compute capability comprises a major revision number X and a minor revision number Y and is denoted by X.Y.

Devices with the same major revision number are of the same core architecture. The major revision number is 5 for devices based on the
 Maxwell architecture, 3 for devices based on the Kepler architecture, 2 for devices based on the Fermi architecture, and 1 for devices
  based on the Tesla architecture.

The minor revision number corresponds to an incremental improvement to the core architecture, possibly including new features.

As mentioned in Heterogeneous Programming, the CUDA programming model assumes a system composed of a host and a device, 
each with their own separate memory. Kernels operate out of device memory, so the runtime provides functions to allocate,
deallocate, and copy device memory, as well as transfer data between host memory and device memory.

Device memory can be allocated either as linear memory or as CUDA arrays.

CUDA arrays are opaque memory layouts optimized for texture fetching. They are described in Texture and Surface Memory.

Linear memory exists on the device in a 40-bit address space, so separately allocated entities can reference one another via pointers, 
for example, in a binary tree.

Linear memory is typically allocated using cudaMalloc() and freed using cudaFree() and data transfer between host memory and device memory 
are typically done using cudaMemcpy(). In the vector addition code sample of Kernels, the vectors need to be copied from host memory to
device memory:
*/

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}
            
// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input vectors
    ...

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

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
            
    // Free host memory
    ...
}

/*
Linear memory can also be allocated through cudaMallocPitch() and cudaMalloc3D(). These functions are recommended for allocations of 2D or 3D arrays as it makes sure that the allocation is appropriately padded to meet the alignment requirements described in Device Memory Accesses, therefore ensuring best performance when accessing the row addresses or performing copies between 2D arrays and other regions of device memory (using the cudaMemcpy2D() and cudaMemcpy3D() functions). The returned pitch (or stride) must be used to access array elements. The following code sample allocates a width x height 2D array of floating-point values and shows how to loop over the array elements in device code:

// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}

The following code sample allocates a width x height x depth 3D array of floating-point values and shows how to loop over the array elements in device code:

// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                    height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}

The reference manual lists all the various functions used to copy memory between linear memory allocated with cudaMalloc(), linear memory allocated with cudaMallocPitch() or cudaMalloc3D(), CUDA arrays, and memory allocated for variables declared in global or constant memory space.

The following code sample illustrates various ways of accessing global variables via the runtime API:

__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));

cudaGetSymbolAddress() is used to retrieve the address pointing to the memory allocated for a variable declared in global memory space. The size of the allocated memory is obtained through cudaGetSymbolSize()
*/