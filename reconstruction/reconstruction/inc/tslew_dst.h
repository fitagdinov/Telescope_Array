/*
 * $Source:$
 * $Log:$
*/

#ifndef _TSLEW_
#define _TSLEW_

#include "univ_dst.h"
#include "mc04_detector.h"

#define TSLEW_BANKID 15046 
#define TSLEW_BANKVERSION 0 
#define TSLEW_BAD_THCAL1 -1000.0
#define TSLEW_BAD_PRXF -100000.0

#ifdef __cplusplus
extern "C" {
#endif
integer4 tslew_common_to_bank_();
integer4 tslew_bank_to_dst_(integer4 *NumUnit);
integer4 tslew_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 tslew_bank_to_common_(integer1 *bank);
integer4 tslew_common_to_dump_(integer4 *long_output);
integer4 tslew_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* tslew_bank_buffer_ (integer4* tslew_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct
{
  integer4 neye;        /* number of sites triggered */
  integer4 nmir;        /* number of mirrors for this event */
  integer4 ntube;       /* total number of tubes for this event */
	
  /* -------------- eye info  ---------------*/

  integer4 eyeid     [MC04_MAXEYE];

  /* -------------- mir info, one for each of nmir mirrors  ---------------*/

  integer4 mirid     [MC04_MAXMIR];  /* mirror # (id), saved as short */
  integer4 mir_eye   [MC04_MAXMIR];  /* eye # (id), saved as short */
  integer4 mir_rev   [MC04_MAXMIR];  /* mirror version (rev3 or rev4) */
  integer4 mir_ntube [MC04_MAXMIR];  /* # of tubes for that mir */

  /* -------------- tube info, one for each of ntube tubes  ---------------*/

  integer4 tube_eye[MC04_MAXTUBE]; /* eye #, saved with tube as short */
  integer4 tube_mir[MC04_MAXTUBE]; /* mirror #, saved with tube as short */
  integer4 tubeid  [MC04_MAXTUBE]; /* tube # */
  integer4 thb     [MC04_MAXTUBE]; /* trigger threshold (mV) on minute ch B */
  real4    thcal1  [MC04_MAXTUBE]; /* time according to HCAL1 calib. */
  real4    tcorr   [MC04_MAXTUBE]; /* time slewing correction */
} tslew_dst_common ;

extern tslew_dst_common tslew_ ; 

#endif
