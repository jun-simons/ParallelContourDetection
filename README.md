This is an implementation of a high-performance edge detection pipeline designed for large scale batch image processing.

It uses a hybrid approach, with MPI used to run the algorithm across multi-GPU clusters, and CUDA within each GPU to splitl work across hundreds of thousands of threads.  It also implements MPI I/O for parallelized concurrent file writes to shared binary files.

The primary computer vision pipeline is implemented with CUDA kernels
- Greyscaling and gaussian blur
- Gradient Computation
- Non-maximum Surpression
- Thresholding
- Edge extraction

Since these operations are "embarassingly parallel" each CUDA kernel can process image pixels concurrently at each step of the process.

Strong and weak scaling tests were performed for the algorithm runtime and file write, and in general indicate strong parallel performance on large multi-GPU clusters running on NVIDIA hardware.
