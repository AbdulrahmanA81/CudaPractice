
#include "kernels.h"

#include <pthread.h>
#include <iostream>
#include <queue>
#include <vector>
#include "filters.h"

/************** WORK STRUCTS *****************/
typedef struct common_work_t
{
  filter *f;
  const int8_t *filter;
  int32_t dimension;
  const int32_t *original_image;
  int32_t *output_image;
  int32_t width;
  int32_t height;
  int32_t max_threads;
  int32_t globle_min;
  int32_t globle_max;
  pthread_barrier_t barrier;
} common_work;


typedef struct work_t
{
  common_work *common;
  int32_t id;
} work;

// Used for row and col sharding 
pthread_mutex_t mutex; 
filter laplacian_filter;

/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) {
  if (smallest == largest) {
    return;
  }

  target[pixel_idx] =
      ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}
/*************** COMMON WORK ***********************/
/* Processes a single pixel and returns the value of processed pixel
 * */
int32_t apply_transformation(const filter *f, const int32_t *original, int32_t *target,
                int32_t width, int32_t height, int row, int column) {
  int32_t result = 0;
  int center = f->dimension / 2;

  for (int i = -center; i <= center; i++) {
      for (int j = -center; j <= center; j++) {
          int curRow = row + i;
          int curCol = column + j;

          if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
              int imgIdx = curRow * width + curCol;
              int filterIdx = (i + center) * f->dimension + (j + center);
              
              result += original[imgIdx] * f->matrix[filterIdx];
          }
      }
  }

  return result;
}

/****************** ROW SHARDING ************/
// ROW SHARDING IMPLEMENTATION
/* Recall that, once the filter is applied, all threads need to wait for
 * each other to finish before computing the smallest/largets elements
 * in the resulting matrix. To accomplish that, we declare a barrier variable:
 *      pthread_barrier_t barrier;
 * And then initialize it specifying the number of threads that need to call
 * wait() on it:
 *      pthread_barrier_init(&barrier, NULL, num_threads);
 * Once a thread has finished applying the filter, it waits for the other
 * threads by calling:
 *      pthread_barrier_wait(&barrier);
 * This function only returns after *num_threads* threads have called it.
 */
void *sharding_work(void *arg) {
    /* Your algorithm is essentially:
    *  1- Apply the filter on the image
    *  2- Wait for all threads to do the same
    *  3- Calculate global smallest/largest elements on the resulting image
    *  4- Scale back the pixels of the image. For the non work queue
    *      implementations, each thread should scale the same pixels
    *      that it worked on step 1.
    */
    work *w = (work *) arg;
    common_work * cw = w->common;
    int32_t local_min = INT32_MAX;
    int32_t local_max = INT32_MIN;

    // Calculate row  and col range for this thread
    int32_t rows_per_thread;
    int32_t start_row;
    int32_t end_row;

    // Formula from the assignment sheet
    rows_per_thread = (cw->height + cw->max_threads - 1) / cw->max_threads;
    start_row = w->id * rows_per_thread;
    end_row = start_row + rows_per_thread;

    // Make sure the last thread doesn't overshoot the total number of rows
    if (end_row > cw->height) {
        end_row = cw->height;
    }


    // Apply filter on designated rows
    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < cw->width; x++) {
            int idx = y * cw->width + x;
            int32_t new_val = apply_transformation(cw->f, cw->original_image, cw->output_image, cw->width, cw->height, y, x);
            cw->output_image[idx] = new_val;
            if (new_val < local_min) local_min = new_val;
            if (new_val > local_max) local_max = new_val;
        }
    }
  

    pthread_mutex_lock(&mutex);
    if(local_min < cw->globle_min) cw->globle_min = local_min;
    if(local_max > cw->globle_max) cw->globle_max= local_max;
    pthread_mutex_unlock(&mutex);


    pthread_barrier_wait(&cw->barrier);


    // Normalize designated rows
    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < cw->width; x++) {
            int idx = y * cw->width + x;
            normalize_pixel(cw->output_image, idx,  cw->globle_min, cw->globle_max);
        }
    }

  return NULL;
}


// The best CPU implementation according to my a2 analysis was the row sharding format.
void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {

    int num_threads = 16;
    pthread_t workers[num_threads];
    work work_thread[num_threads];
    common_work w;

    // In your run_best_cpu function:
    
    laplacian_filter.dimension = dimension;
    laplacian_filter.matrix = filter;

    w.f = &laplacian_filter;
    w.original_image = input;
    w.output_image = output;
    w.max_threads = num_threads;
    w.height = height;
    w.width = width;
    w.globle_max = INT32_MIN;
    w.globle_min = INT32_MAX;
    pthread_barrier_init(&w.barrier, NULL, num_threads);
    pthread_mutex_init(&mutex, NULL);


    for (int i =0; i < num_threads; i ++){
        work_thread[i].common  = &w;
        work_thread[i].id = i;
        pthread_create(&workers[i], NULL, sharding_work, &work_thread[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(workers[i], NULL);
    }

    pthread_mutex_destroy(&mutex);
    pthread_barrier_destroy(&w.barrier);
}
