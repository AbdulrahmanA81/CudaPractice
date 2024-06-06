

#include "filters.h"
#include "clock.h"
#ifndef __KERNELS__H
#define __KERNELS__H

/* TODO: you may want to change the signature of some or all of those functions,
 * depending on your strategy to compute min/max elements.
 * Be careful: "min" and "max" are names of CUDA library functions
 * unfortunately, so don't use those for variable names.*/
struct TimingData {
   float time_gpu_computation;
   float time_gpu_transfer_in;
   float time_gpu_transfer_out;
};


void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height);

TimingData run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel1(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest);
__global__ void findMinMax(const int32_t *input, int32_t *mins, int32_t *maxes, int32_t width, int32_t height);

TimingData run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel2(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize2(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest);
__global__ void findMinMax2(const int32_t *input, int32_t *mins, int32_t *maxes, int32_t n);

TimingData run_kernel3(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel3(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize3(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest);
__global__ void findMinMax_row_major(const int32_t *input, int32_t *minPerBlock, int32_t *maxPerBlock, int32_t width, int32_t height);

TimingData run_kernel4(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
__global__ void kernel4(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);
__global__ void normalize4(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest);
__global__ void findMinMax_col_major(const int32_t *input, int32_t *blockMin, int32_t *blockMax, 
                           int32_t width, int32_t height);

TimingData run_kernel5(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height);
/* This is your own kernel, you should decide which parameters to add
   here*/
__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                           int32_t smallest, int32_t biggest) ;
__global__ void kernel5(const int8_t * __restrict__ filter,  const int32_t * __restrict__ input, int32_t * __restrict__ output,  int32_t width, int32_t height);


#endif
