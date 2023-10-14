/*
 * $Source:$
 * $Log:$
*/

#ifndef _MCRAW_
#define _MCRAW_

#include "univ_dst.h"
#include "mc04_detector.h"

#define MCRAW_BANKID 15041 
#define MCRAW_BANKVERSION 0 
#define MCRAW_BAD_THCAL1 -1000.0
#define MCRAW_BAD_PRXF -100000.0

#ifdef __cplusplus
extern "C" {
#endif
integer4 mcraw_common_to_bank_();
integer4 mcraw_bank_to_dst_(integer4 *NumUnit);
integer4 mcraw_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 mcraw_bank_to_common_(integer1 *bank);
integer4 mcraw_common_to_dump_(integer4 *long_output);
integer4 mcraw_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* mcraw_bank_buffer_ (integer4* mcraw_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct
{
  /* 2440000 subtracted from the julian day to give room for millisecond 
     precision.
     checks: "1/1/1985,0.0hr UT" gives 6066 + 0.5;
     jday=0 gives 1968 May 23.5 UT
  */
  integer4 jday;        /* mean julian day - 2.44e6 */
  integer4 jsec;        /* second into Julian day */
  integer4 msec;        /* milli sec of julian day (NOT since UT0:00) */

  integer4 neye;        /* number of sites triggered */
  integer4 nmir;        /* number of mirrors for this event */
  integer4 ntube;       /* total number of tubes for this event */
	
  /* -------------- eye info  ---------------*/

  integer4 eyeid     [MC04_MAXEYE];

  /* -------------- mir info, one for each of nmir mirrors  ---------------*/

  integer4 mirid     [MC04_MAXMIR];  /* mirror # (id), saved as short */
  integer4 mir_eye   [MC04_MAXMIR];  /* eye # (id), saved as short */
  integer4 mir_rev   [MC04_MAXMIR];  /* mirror version (rev3 or rev4) */
  integer4 mirevtno  [MC04_MAXMIR];  /* event # from mirror packet */
  integer4 mir_ntube [MC04_MAXMIR];  /* # of tubes for that mir */
  integer4 mirtime_ns[MC04_MAXMIR];  /* time of mir. holdoff in ns from sec. */

  /* -------------- tube info, one for each of ntube tubes  ---------------*/

  integer4 tube_eye[MC04_MAXTUBE]; /* eye #, saved with tube as short */
  integer4 tube_mir[MC04_MAXTUBE]; /* mirror #, saved with tube as short */
  integer4 tubeid  [MC04_MAXTUBE]; /* tube # */
  integer4 qdca    [MC04_MAXTUBE]; /* digitized channel A charge integral */
  integer4 qdcb    [MC04_MAXTUBE]; /* digitized channel B charge integral */
  integer4 tdc     [MC04_MAXTUBE]; /* digitized tube trigger to holdoff time */
  integer4 tha     [MC04_MAXTUBE]; /* trigger threshold (mV) on minute ch A */
  integer4 thb     [MC04_MAXTUBE]; /* trigger threshold (mV) on minute ch B */
  real4    prxf    [MC04_MAXTUBE]; /* # of photons according to RXF calib. */
  real4    thcal1  [MC04_MAXTUBE]; /* time according to HCAL1 calib. */

} mcraw_dst_common ;

extern mcraw_dst_common mcraw_ ; 

#endif
