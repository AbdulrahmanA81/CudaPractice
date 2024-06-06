/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2022 Bogdan Simion
 * -------------
 */

#include "kernels.h"


TimingData run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height) {
    // Allocate device memory
    int8_t *d_filter;
    int32_t *d_input, *d_output;
    TimingData timing_data;
    size_t imageSize = width * height * sizeof(int32_t);
    size_t filterSize = dimension * dimension * sizeof(int8_t);

    cudaMalloc((void **)&d_filter, filterSize);
    cudaMalloc((void **)&d_input, imageSize);
    cudaMalloc((void **)&d_output, imageSize);


    Clock clock;


    // Copy data into GPU
    clock.start();
    cudaMemcpy(d_input, input, width * height * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);

    timing_data.time_gpu_transfer_in = clock.stop();



    // Determine grid and block sizes
    int threadsPerBlock = 1024; 
    int blocksPerGrid = (imageSize + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blockSize(threadsPerBlock, 1, 1);
    dim3 gridSize(blocksPerGrid, 1, 1);
    // Launch kernel
    clock.start();
    kernel3<<<gridSize, blockSize>>>(d_filter, dimension, d_input, d_output, width, height);
    cudaDeviceSynchronize();  // Ensure kernel execution completes
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
    findMinMax_row_major<<<numBlocks, blockSize, 2 * blockSize.x * sizeof(int32_t)>>>(d_output, d_mins, d_maxes, width, height);
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
    normalize3<<<gridSize, blockSize>>>(d_output, width, height, globalMin, globalMax);
    cudaDeviceSynchronize();
    timing_data.time_gpu_computation = timing_data.time_gpu_computation + clock.stop();


    // Copy the result back out
    clock.start();
    cudaMemcpy(output, d_output, width * height * sizeof(int32_t), cudaMemcpyDeviceToHost);
    timing_data.time_gpu_transfer_out = clock.stop();
    
    // Clean up
    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);

    return timing_data;
}

__global__ void kernel3(const int8_t *filter, int32_t dimension, const int32_t *input, int32_t *output, int32_t width, int32_t height) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x; // Unique ID of the thread
    int totalThreads = gridDim.x * blockDim.x; // Total number of threads

    int pixelsPerThread = (width * height + totalThreads - 1) / totalThreads; // Handle remainder pixels
    int startPixel = threadId * pixelsPerThread;
    int endPixel = min(startPixel + pixelsPerThread, width * height); // Ensure not to exceed image size

    for (int idx = startPixel; idx < endPixel; ++idx) {
        int row = idx / width; // Row index of the pixel
        int col = idx % width; // Column index of the pixel

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



__global__ void normalize3(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest) {
    if (smallest == biggest) {
        return;
    }
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    int pixelsPerThread = (width * height + totalThreads - 1) / totalThreads;
    int startPixel = threadId * pixelsPerThread;
    int endPixel = min(startPixel + pixelsPerThread, width * height);

    for (int idx = startPixel; idx < endPixel; ++idx) {
        int pixelValue = image[idx];

        // Normalize the pixel value
        float normalizedValue = (float)(pixelValue - smallest) / (biggest - smallest);

        image[idx] = static_cast<int32_t>(normalizedValue * 255);
    }
 }


__global__ void findMinMax_row_major(const int32_t *input, int32_t *minPerBlock, int32_t *maxPerBlock, int32_t width, int32_t height) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = 64 * gridDim.x;

    int localMin = INT_MAX;
    int localMax = INT_MIN;

    // Loop over the image data assigned to this block
    while (i < width * height) {
        localMin = min(localMin, input[i]);
        localMax = max(localMax, input[i]);

        // Ensure we're not reading out of bounds
        if (i + blockDim.x < width * height) {
            localMin = min(localMin, input[i + blockDim.x]);
            localMax = max(localMax, input[i + blockDim.x]);
        }

        i += gridSize;
    }

    // Each thread puts its local min and max in shared memory
    sdata[tid] = localMin;
    sdata[tid + blockDim.x] = localMax;
    __syncthreads();

    // Do reduction in shared memory
    unsigned int maxIndex = blockDim.x; // Maximum index for shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < maxIndex) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
            sdata[tid + blockDim.x] = max(sdata[tid + blockDim.x], sdata[tid + s + blockDim.x]);
        }
        __syncthreads();
    }


    // Write the result for this block to global memory
    if (tid == 0) {
        minPerBlock[blockIdx.x] = sdata[0];
        maxPerBlock[blockIdx.x] = sdata[blockDim.x];
    }
}


