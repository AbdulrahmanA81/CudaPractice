

#include "kernels.h"

/* This is your own kernel, so you should decide which parameters to 
   add here*/
TimingData run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height) {
  // Figure out how to split the work into threads and call the kernel below.
  // Allocate GPU memory
   int32_t *d_input, *d_output;
   int8_t *d_filter;
   TimingData timing_data;

   cudaMalloc((void **)&d_input, width * height * sizeof(int32_t));
   cudaMalloc((void **)&d_output, width * height * sizeof(int32_t));
   cudaMalloc((void **)&d_filter, dimension * dimension * sizeof(int8_t));


   Clock clock;
   // Copy data into GPU
   clock.start();
   cudaMemcpy(d_input, input, width * height * sizeof(int32_t), cudaMemcpyHostToDevice);
   cudaMemcpy(d_filter, filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);


   timing_data.time_gpu_transfer_in = clock.stop();

   // // Define grid and block sizes
   dim3 blockSize(512); 
   dim3 gridSize((width * height + blockSize.x * 8  - 1) / (blockSize.x * 8 ));



    if (dimension == 3) {
        clock.start();
        // Call the optimized kernel for 3x3 filter
        kernel5<<<gridSize, blockSize>>>(d_filter, d_input, d_output, width, height);
        cudaDeviceSynchronize(); // Ensure kernel execution completes
        timing_data.time_gpu_computation = clock.stop();
    } else {
        clock.start();
        // Call the optimized kernel for 3x3 filter
        kernel4<<<gridSize, blockSize>>>(d_filter, dimension, d_input, d_output, width, height);
        cudaDeviceSynchronize(); // Ensure kernel execution completes
        timing_data.time_gpu_computation = clock.stop();
    }

   



   // Now that we applied the kenel we need to normalize it

   // Reduction for min/max
   int numElements = width * height;
   int numBlocks = (numElements + blockSize.x - 1) / blockSize.x;
   int32_t *d_mins, *d_maxes;
   cudaMalloc((void **)&d_mins, numBlocks * sizeof(int32_t));
   cudaMalloc((void **)&d_maxes, numBlocks * sizeof(int32_t));

   // Launch findMinMax_row_major kernel
   clock.start();
   findMinMax_col_major<<<numBlocks, blockSize, 2 * blockSize.x * sizeof(int32_t)>>>(d_output, d_mins, d_maxes, width, height);
   cudaDeviceSynchronize();
   timing_data.time_gpu_computation = timing_data.time_gpu_computation + clock.stop();


   // Find the global min and max
   
   // Copy min/max arrays back to host
   int32_t *mins = new int32_t[numBlocks];
   int32_t *maxes = new int32_t[numBlocks];
   cudaMemcpy(mins, d_mins, numBlocks * sizeof(int32_t), cudaMemcpyDeviceToHost);
   cudaMemcpy(maxes, d_maxes, numBlocks * sizeof(int32_t), cudaMemcpyDeviceToHost);


   // CPU reduction to find global min/max
   int32_t globalMin = INT_MAX;
   int32_t globalMax = INT_MIN;
   for (int i = 0; i < numBlocks; i++) {
      globalMin = min(globalMin, mins[i]);
      globalMax = max(globalMax, maxes[i]);
   }


   clock.start();
   normalize5<<<gridSize, blockSize>>>(d_output, width, height, globalMin, globalMax);
   cudaDeviceSynchronize();
   timing_data.time_gpu_computation = timing_data.time_gpu_computation + clock.stop();

   // Copy the result back out
   clock.start();
   cudaMemcpy(output, d_output, width * height * sizeof(int32_t), cudaMemcpyDeviceToHost);
   timing_data.time_gpu_transfer_out = clock.stop();

   // free allocated memory
   cudaFree(d_input);
   cudaFree(d_output);
   cudaFree(d_filter);

   return timing_data;
}

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest) {
    if (biggest == smallest) {
        return;
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float range = static_cast<float>(biggest - smallest);
    for (int i = idx; i < width * height; i += stride) {
        float pixel = static_cast<float>(image[i]);
        image[i] = static_cast<int32_t>(((pixel - smallest) * 255.0f) / range);
    }
}

__global__ void kernel5(const int8_t * __restrict__ filter, 
                                   const int32_t * __restrict__ input, int32_t * __restrict__ output, 
                                   int32_t width, int32_t height) {
    // Load the 3x3 filter into shared memory
    __shared__ int8_t sharedFilter[9];
    int threadId = threadIdx.x;

    if (threadId < 9) {
        sharedFilter[threadId] = filter[threadId];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < width * height; i += stride) {
        int row = i / width;
        int col = i % width;

       // Check if the thread is within the image boundaries
        if (col < width && row < height) {
            int result = 0;
            int filterIdx, imgIdx;
            // Apply the filter centered around the current pixel
            // Top-left corner
            filterIdx = 0; // Corresponds to (i = -1, j = -1)
            imgIdx = (row - 1) * width + (col - 1);
            if (row > 0 && col > 0) result += input[imgIdx] * sharedFilter[filterIdx];

            // Top-middle
            filterIdx = 1; // Corresponds to (i = -1, j = 0)
            imgIdx = (row - 1) * width + col;
            if (row > 0) result += input[imgIdx] * sharedFilter[filterIdx];

            // Top-right corner
            filterIdx = 2; // Corresponds to (i = -1, j = 1)
            imgIdx = (row - 1) * width + (col + 1);
            if (row > 0 && col < width - 1) result += input[imgIdx] * sharedFilter[filterIdx];

            // Middle-left
            filterIdx = 3; // Corresponds to (i = 0, j = -1)
            imgIdx = row * width + (col - 1);
            if (col > 0) result += input[imgIdx] * sharedFilter[filterIdx];

            // Center
            filterIdx = 4; // Corresponds to (i = 0, j = 0)
            imgIdx = row * width + col;
            result += input[imgIdx] * sharedFilter[filterIdx];

            // Middle-right
            filterIdx = 5; // Corresponds to (i = 0, j = 1)
            imgIdx = row * width + (col + 1);
            if (col < width - 1) result += input[imgIdx] * sharedFilter[filterIdx];

            // Bottom-left corner
            filterIdx = 6; // Corresponds to (i = 1, j = -1)
            imgIdx = (row + 1) * width + (col - 1);
            if (row < height - 1 && col > 0) result += input[imgIdx] * sharedFilter[filterIdx];

            // Bottom-middle
            filterIdx = 7; // Corresponds to (i = 1, j = 0)
            imgIdx = (row + 1) * width + col;
            if (row < height - 1) result += input[imgIdx] * sharedFilter[filterIdx];

            // Bottom-right corner
            filterIdx = 8; // Corresponds to (i = 1, j = 1)
            imgIdx = (row + 1) * width + (col + 1);
            if (row < height - 1 && col < width - 1) result += input[imgIdx] * sharedFilter[filterIdx];
            // Store the result in the output image
            output[row * width + col] = result;
        } 
    }
}
