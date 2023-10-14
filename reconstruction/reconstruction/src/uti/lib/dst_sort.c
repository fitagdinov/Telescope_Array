#include <stdlib.h>
#include <math.h>
#include "dst_sort.h"

#define DST_SORT_TYPE(DST_TYPE)						\
  static DST_TYPE* dst_sort_##DST_TYPE##_array = 0;			\
  static int dst_sort_##DST_TYPE##_cmp_fun(const void * index1, const void * index2) \
  {									\
   int i_f = *((int*)index1);						\
   int i_s = *((int*)index2);						\
   real8 f =  (real8)dst_sort_##DST_TYPE##_array[i_f];			\
   real8 s =  (real8)dst_sort_##DST_TYPE##_array[i_s];		\
   if(isnan(f))								\
     f = -1;								\
   if(isnan(s))								\
     s = -1;								\
   if (f > s) return  1;						\
   if (f < s) return -1;						\
   return 0;								\
  }									\
  void dst_sort_##DST_TYPE(integer4 nelements, DST_TYPE* data_array, integer4* index_array) \
  {									\
    int i;								\
    for (i=0; i<nelements; i++)						\
      index_array[i] = i;						\
    dst_sort_##DST_TYPE##_array = data_array;				\
      qsort (index_array,nelements,sizeof(int),dst_sort_##DST_TYPE##_cmp_fun); \
  }									\
  

DST_SORT_TYPE(real8);
DST_SORT_TYPE(real4);
DST_SORT_TYPE(integer4);
DST_SORT_TYPE(integer2);
DST_SORT_TYPE(integer1);

