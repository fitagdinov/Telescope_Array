/*
 * $Source:$
 * $Log:$
*/

#ifndef _STNPE_
#define _STNPE_

#include "univ_dst.h"
#include "mc04_detector.h"
/* #include "hires_const.h" */

#define STNPE_BANKID 15045 
#define STNPE_BANKVERSION 0 
#define STNPE_BAD_THCAL1 -1000.0
#define STNPE_BAD_PRXF -100000.0

#define STNPE_MAXEYE  2
#define STNPE_PASS1_CALIB  1
#define STNPE_DBASE_CALIB  2

#ifdef __cplusplus
extern "C" {
#endif
integer4 stnpe_common_to_bank_();
integer4 stnpe_bank_to_dst_(integer4 *NumUnit);
integer4 stnpe_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 stnpe_bank_to_common_(integer1 *bank);
integer4 stnpe_common_to_dump_(integer4 *long_output);
integer4 stnpe_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* stnpe_bank_buffer_ (integer4* stnpe_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct
{
  integer4 jday;    /* ( from hraw1_ ) mean julian day - 2.44e6 */
  integer4 jsec;    /* ( from hraw1_ ) second into Julian day */
  integer4 msec;    /* ( from hraw1_ ) milli sec of julian day  */

  integer4 julian;  /* ( from fraw1_ ) */
  integer4 jsecond; /* ( from fraw1_ ) */
  integer4 jclkcnt; /* ( from fraw1_ ) this number is nanoseconds */

  integer4 calib_source; /* pass1 or data base */

  integer4 neye;    /* number of sites triggered */
  integer4 nmir;    /* number of mirrors for this event */
  integer4 ntube;   /* total number of tubes for this event */
	
  /* -------------- eye info  ---------------*/

  integer4 eyeid     [MC04_MAXEYE];
  integer4 eye_nmir  [MC04_MAXEYE];  /* number of triggered mirrors in eye */
  integer4 eye_ntube [MC04_MAXEYE];  /* number of triggered mirrors in eye */

  /* -------------- mir info, one for each of nmir mirrors  ---------------*/

  integer4 mirid     [MC04_MAXMIR]; /* mirror # (id), saved as short */
  integer4 mir_eye   [MC04_MAXMIR]; /* eye # (id), saved as short */
  integer4 mir_rev   [MC04_MAXMIR]; /* mirror version (rev3 or rev4 or fadc) */

  integer4 mirevtno  [MC04_MAXMIR]; /* event # from mirror packet */
  integer4 mir_ntube [MC04_MAXMIR]; /* # of tubes for that mir */

  integer4 mirtime_ns[MC04_MAXMIR]; /* ( from hraw1_ ) */
  integer4 second    [MC04_MAXMIR]; /* ( from fraw1_ ) */
  integer4 clkcnt    [MC04_MAXMIR]; /* ( from fraw1_ ) */

  /* -------------- tube info, one for each of ntube tubes  ---------------*/

  integer4 tube_eye  [MC04_MAXTUBE]; /* eye #, saved with tube as short */
  integer4 tube_mir  [MC04_MAXTUBE]; /* mirror #, saved with tube as short */
  integer4 tubeid    [MC04_MAXTUBE]; /* tube # */
  integer4 thresh    [MC04_MAXTUBE]; /* ( from hraw1_ ( hraw1_.thb ) ) */

  real4    npe       [MC04_MAXTUBE]; /* # photo-electrons */
  real4    time      [MC04_MAXTUBE]; /* (ns), with respect to event start  */
  integer4 dtime     [MC04_MAXTUBE]; /* ( from fpho1_ ) width of pulse (ns) */

} stnpe_dst_common ;

extern stnpe_dst_common stnpe_ ; 

#endif
