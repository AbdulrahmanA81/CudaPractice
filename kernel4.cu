
#include "kernels.h"

TimingData run_kernel4(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height) {
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

    // Define grid and block sizes
    dim3 blockSize(512); 
    dim3 gridSize((width * height + blockSize.x * 4 - 1) / (blockSize.x * 4));

    clock.start();
    kernel4<<<gridSize, blockSize>>>(d_filter, dimension, d_input, d_output, width, height);
    cudaDeviceSynchronize(); // Ensure kernel execution completes
    timing_data.time_gpu_computation = clock.stop();



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
    normalize4<<<gridSize, blockSize>>>(d_output, width, height, globalMin, globalMax);
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


__global__ void kernel4(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, 
                        int32_t width, int32_t height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    
    for (int i = idx; i < width * height; i += stride) {
        int row = i / width;
        int col = i % width;

       // Check if the thread is within the image boundaries
        if (col < width && row < height) {
            int result = 0;
            int center = dimension / 2;

            // Apply the filter centered around the current pixel
            for (int i = -center; i <= center; i++) {
                for (int j = -center; j <= center; j++) {
                    int curRow = row + i;
                    int curCol = col + j;

                    // Check if the neighboring pixel is within the image boundaries
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        int imgIdx = curRow * width + curCol;
                        int filterIdx = (i + center) * dimension + (j + center);
                        
                        result += input[imgIdx] * filter[filterIdx];
                    }
                }
            }
            // Store the result in the output image
            output[row * width + col] = result;
        } 
    }
}




__global__ void normalize4(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest) {
    if (biggest == smallest) {
        return;

    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < width * height; i += stride) {
        int32_t pixel = image[i];

        // Normalize the pixel value
        
        image[i] = ((pixel - smallest) * 255) / (biggest - smallest);
    }
}


__global__ void findMinMax_col_major(const int32_t *input, int32_t *blockMin, int32_t *blockMax, 
                           int32_t width, int32_t height) {
    extern __shared__ int32_t sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int threadId = threadIdx.x;

    // Initialize shared memory for min and max
    int32_t minVal = INT_MAX;
    int32_t maxVal = INT_MIN;

    // Strided loop similar to Kernel 4
    for (int i = idx; i < width * height; i += stride) {
        minVal = min(minVal, input[i]);
        maxVal = max(maxVal, input[i]);
    }

    // Each thread puts its result in shared memory
    sdata[threadId] = minVal;
    sdata[blockDim.x + threadId] = maxVal;
    __syncthreads();

    // Reduction within a block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadId < s) {
            sdata[threadId] = min(sdata[threadId], sdata[threadId + s]);
            sdata[blockDim.x + threadId] = max(sdata[blockDim.x + threadId], sdata[blockDim.x + threadId + s]);
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (threadId == 0) {
        blockMin[blockIdx.x] = sdata[0];
        blockMax[blockIdx.x] = sdata[blockDim.x];
    }
}
