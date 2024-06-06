
#ifndef __FILTERS__H
#define __FILTERS__H

#include <stdint.h>

/**************FILTER STRUCT DEFINITIONS*****************/
/* Filters are square matrices with odd dimension.
 */
typedef struct filter_t {
  int32_t dimension;
  const int8_t *matrix;
} filter;

/* Filter constants */
#define NUM_FILTERS 4
#define LAPLACIAN_FILTER_3 0 /* 3x3 filter */
#define LAPLACIAN_FILTER_5 1 /* 5x5 filter */
#define LAP_OF_GAUS_FILTER 2 /* 9x9 filter */
#define IDENTITY_FILTER 3    /* 1x1 filter */

extern filter *builtin_filters[NUM_FILTERS];

/* parallel methods*/
typedef enum {
  SHARDED_ROWS,
  SHARDED_COLUMNS_COLUMN_MAJOR,
  SHARDED_COLUMNS_ROW_MAJOR,
  WORK_QUEUE
} parallel_method;

#endif
