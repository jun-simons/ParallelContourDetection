This is an implementation of a *high-performance edge detection pipeline* designed for large scale batch image processing.

It uses a hybrid approach, with **MPI** used to run the algorithm across multi-GPU clusters, and **CUDA** within each GPU to splitl work across hundreds of thousands of threads.  It also implements **MPI I/O** for parallelized concurrent file writes to shared binary files.

The primary computer vision pipeline is implemented with CUDA kernels
- Greyscaling and gaussian blur
- Gradient Computation
- Non-maximum Surpression
- Thresholding
- Edge extraction

Since these operations are "embarassingly parallel" each CUDA kernel can process image pixels concurrently at each step of the process.

Strong and weak scaling tests were performed for the algorithm runtime and file write, and in general indicate strong parallel performance on large multi-GPU clusters running on NVIDIA hardware.


### Compiling and Running
For IBM Power9 hardware and NVIDIA Volta GPUs:

**Load Modules**:

module load xl_r spectrum-mpi cuda/11.2

**Compile**:

nvcc -arch=sm_70 -ccbin mpixlc  -I/usr/local/cuda-11.2/include  main_cuda.cu  -L/usr/local/cuda-11.2/lib64 -lcudadevrt -lcudart -lpng -o main_cuda

**Run directly**"

mpirun --bind-to core --map-by node -np [NUM_PROCESSES] ./main_cuda [IMAGE_FNAME] [NUM_FRAMES]

It is more sutable to run batches on slurm, or submit jobs as appropriate on the system.
