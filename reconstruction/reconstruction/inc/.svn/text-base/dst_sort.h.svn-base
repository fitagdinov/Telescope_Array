#ifndef _dst_sort_h_
#define _dst_sort_h_
#include "dst_std_types.h"

#ifdef __cplusplus
extern "C" {
#endif
  /* Sort the array of real8 (double precision floats) in an increasing order
     and store the indices that put the array 'data_array' into an increasing order into the
     integer array called 'index'.  Both data_array and index_array must be of dimension
     nelements. */
  void dst_sort_real8(integer4 nelements, real8* data_array, integer4* index_array);
  /* sorting all other sortable types in a similar way */
  void dst_sort_real4(integer4 nelements, real4* data_array, integer4* index_array);
  void dst_sort_integer4(integer4 nelements, integer4* data_array, integer4* index_array);
  void dst_sort_integer2(integer4 nelements, integer2* data_array, integer4* index_array);
  void dst_sort_integer1(integer4 nelements, integer1* data_array, integer4* index_array);
#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif
