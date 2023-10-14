/**
 * DST bank for GDAS data
 *
 * GDAS: Global Data Assimilation System
 * http://www.ready.noaa.gov/gdas1.php
 *
 * 2014/03/07 Y.Tsunesada
 * 
 * 2015/Mar/12 Y. Tsunesada
 *   Now this bank contains only profiles of the TA-nearest grid N113W39.
 *   The common struct of "fdatmos_param_t" is used.
 */
#ifndef _GDAS_H_
#define _GDAS_H_

#include "fdatmos_param_dst.h"

#define GDAS_BANKID 12460
#define GDAS_BANKVERSION 001

#define GDAS_MAXITEM 23
#define GDAS_NGRID 35 
// so that gdas DST bank follows standard convention of dst2k-ta as well:
// other applications, such as TDSTio, rely on this fact.
#define gdas_ gdasbank_

#ifdef __cplusplus
extern "C" {
#endif
integer4 gdas_bank_to_common_(integer1 *bank);
integer4 gdas_common_to_dst_(integer4 *unit);
integer4 gdas_common_to_bank_();
integer4 gdas_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* gdas_bank_buffer_ (integer4* gdas_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


/**
 * The atmospheric data of GDAS were extracted and stored in the bank
 * for 35 grids of latitudes 37~41 degrees and longitudes 110~116 degrees (35 grids).
 * The grid number is given by
 *   n = (lat - 37)*7 + (lon - 110)
 * For example n is 0 for the grid (37, 110), and 34 for (41, 116). 
 * The nearest grid to the TA site is (39, 113), which corresponds to n = 17.
 * 
 */
/* Commented out 2015/Mar/12
typedef struct _gdas_t {

  // uniq ID 
   integer4 uniqID;

  // available date from 
   integer4 dateFrom; //sec from 1970/1/1
  // available date to 
   integer4 dateTo; //sec from 1970/1/1

   // number of data lines 
   integer4 nItem;

   // height [km] 
   real4 height[GDAS_NGRID][GDAS_MAXITEM];
   // pressure [hPa] 
   real4 pressure[GDAS_NGRID][GDAS_MAXITEM];
   // pressure error [hPa] 
   real4 pressureError[GDAS_NGRID][GDAS_MAXITEM];
   // temperature [degree] 
   real4 temperature[GDAS_NGRID][GDAS_MAXITEM];
   // temperature error [degree] 
   real4 temperatureError[GDAS_NGRID][GDAS_MAXITEM];
   // dew point [degree] 
   real4 dewPoint[GDAS_NGRID][GDAS_MAXITEM];
   // dew point error [degree] 
   real4 dewPointError[GDAS_NGRID][GDAS_MAXITEM];


} gdas_dst_common;
*/

typedef struct _fdatmos_param_t gdas_dst_common;
extern gdas_dst_common gdasbank_;

#ifdef __cplusplus
# define __BEGIN_DECLS extern "C" {
# define __END_DECLS }
#else
# define __BEGIN_DECLS 
# define __END_DECLS 
#endif

__BEGIN_DECLS

/** Returns the grid index of a grid (lat, lon) */
int latlon2index(int lat, int lon);

__END_DECLS

#endif
