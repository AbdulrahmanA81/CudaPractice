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
#include <iostream>

#define MAX_FILTER_DIM 9 

TimingData run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
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

    // Initializing block size
    dim3 blockSize(16, 16); 
    dim3 gridSize((height + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);


    clock.start();
    kernel1<<<gridSize, blockSize>>>(d_filter, dimension, d_input, d_output, width, height);
    cudaDeviceSynchronize();  // Ensure kernel execution completes
    timing_data.time_gpu_computation = clock.stop();

    int32_t *d_mins, *d_maxes;

    // Reduction for min/max
    int blocksPerColumn = (width + blockSize.x - 1) / blockSize.x;
    int blocksPerRow = height; // One block for each row
    dim3 gridDimReduction(blocksPerColumn, blocksPerRow);

    int totalBlocks = blocksPerColumn * blocksPerRow;
    cudaMalloc((void **)&d_mins, totalBlocks * sizeof(int32_t));
    cudaMalloc((void **)&d_maxes, totalBlocks * sizeof(int32_t));

    // Launch findMinMax kernel with 2D grid
    clock.start();
    findMinMax<<<gridDimReduction, blockSize, 2 * blockSize.x * sizeof(int32_t)>>>(d_output, d_mins, d_maxes, width, height);
    cudaDeviceSynchronize();
    timing_data.time_gpu_computation = timing_data.time_gpu_computation + clock.stop();

    // Copy min/max arrays back to host
    int32_t *mins = new int32_t[totalBlocks];
    int32_t *maxes = new int32_t[totalBlocks];
    cudaMemcpy(mins, d_mins, totalBlocks * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(maxes, d_maxes, totalBlocks * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // CPU reduction to find global min/max
    int32_t globalMin = INT_MAX;
    int32_t globalMax = INT_MIN;
    for (int i = 0; i < totalBlocks; i++) {
        globalMin = min(globalMin, mins[i]);
        globalMax = max(globalMax, maxes[i]);
    }

    clock.start();
    normalize1<<<gridSize, blockSize>>>(d_output, width, height, globalMin, globalMax);
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

__global__ void kernel1(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height) {


    __shared__ int8_t sharedFilter[MAX_FILTER_DIM][MAX_FILTER_DIM];


    // Load the filter into shared memory
    // Each thread loads one element of the filter
    int filterX = threadIdx.x;
    int filterY = threadIdx.y;

    if (filterX < dimension && filterY < dimension) {
      sharedFilter[filterY][filterX] = filter[filterY * dimension + filterX];
    }

    // Synchronize to make sure the filter is fully loaded
    __syncthreads();


    // In a column-major approach, the 'col' index is determined by the block index and thread index in Y direction.
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    // The 'row' index is determined by the block index and thread index in X direction.
    int row = blockIdx.x * blockDim.x + threadIdx.x;

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
                    
                    result += input[imgIdx] *  sharedFilter[i + center][j + center];
                }
            }
        }
        // Store the result in the output image
        output[row * width + col] = result;
    }    
}

__global__ void findMinMax(const int32_t *input, int32_t *mins, int32_t *maxes, int32_t width, int32_t height) {
    extern __shared__ int32_t sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y;

    unsigned int index = y + x * height; // Convert to column-major index

    // Load input into shared memory.
    sdata[tid] = (index < width * height) ? input[index] : INT_MAX;
    sdata[blockDim.x + tid] = (index < width * height) ? input[index] : INT_MIN;
    __syncthreads();

    // Perform reduction in shared memory.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
            sdata[blockDim.x + tid] = max(sdata[blockDim.x + tid], sdata[blockDim.x + tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global memory.
    if (tid == 0) {
        mins[blockIdx.x + blockIdx.y * gridDim.x] = sdata[0]; // Index adjusted for column-major
        maxes[blockIdx.x + blockIdx.y * gridDim.x] = sdata[blockDim.x]; // Index adjusted for column-major
    }
}

__global__ void normalize1(int32_t *image, int32_t width, int32_t height, int32_t smallest, int32_t biggest) {
    // Return if there is no range to normalize.
    if (smallest == biggest) {
        return;
    }

    // In a column-major approach, the 'col' index is determined by the block index and thread index in Y direction.
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    // The 'row' index is determined by the block index and thread index in X direction.
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int index = row * width + col;

    
    if (col < width && row < height) {
        // Perform normalization.
        image[index] = ((image[index] - smallest) * 255) / (biggest - smallest);
    }
   
}
