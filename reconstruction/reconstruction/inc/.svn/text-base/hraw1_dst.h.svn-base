/*
 * $Source: /hires_soft/uvm2k/bank/hraw1_dst.h,v $
 * $Log: hraw1_dst.h,v $
 * Revision 1.8  1997/08/28 16:34:40  jui
 * changed HRAW1_BAD_THCAL1 to -1000.0
 *
 * Revision 1.7  1997/08/28  01:07:59  jui
 * added macro defs HRAW1_BAD_THCAL1 -100000.0 HRAW1_BAD_PRXF -100000.0
 *
 * Revision 1.6  1997/08/17  22:49:09  jui
 * added pxref (roving xenon flasher calibration calculated photon count)
 * and thcal1 (time from hcal1 calibration)
 *
 * Revision 1.5  1997/06/28  00:01:56  jui
 * changed jday from r*8 to jday(i*4) and jsec(i*4)
 *
 * Revision 1.4  1997/05/19  16:38:16  jui
 * removed mirscaler...no longer supported under Big-H
 * added mir_rev. Big-H is a mixed detector.
 *
 * Revision 1.3  1997/04/29  23:31:11  tareq
 * removed idth field from tube info
 *
 * Revision 1.2  1997/04/28  21:44:37  tareq
 * *** empty log message ***
 *
*/

#ifndef _HRAW1_
#define _HRAW1_

#include "univ_dst.h"

#define HRAW1_BANKID 15001 
#define HRAW1_BANKVERSION 0 
#define HRAW1_BAD_THCAL1 -1000.0
#define HRAW1_BAD_PRXF -100000.0

#ifdef __cplusplus
extern "C" {
#endif
integer4 hraw1_common_to_bank_();
integer4 hraw1_bank_to_dst_(integer4 *NumUnit);
integer4 hraw1_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 hraw1_bank_to_common_(integer1 *bank);
integer4 hraw1_common_to_dump_(integer4 *long_output);
integer4 hraw1_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hraw1_bank_buffer_ (integer4* hraw1_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct
{
  /* 2440000 subtracted from the julian day to give room for millisecond precision */
  /* checks: "1/1/1985,0.0hr UT" gives 6066 + 0.5; jday=0 gives 1968 May 23.5 UT */
  integer4 jday;        /* mean julian day - 2.44e6 */
  integer4 jsec;        /* second into Julian day */
  integer4 msec;        /* milli sec of julian day (NOT since UT0:00) */
  integer4 status;      /* set bit 0 to 1 when converts from .pln file */
                                /* set bit 1 to 1 for Monte Carlo events */
  integer4 nmir;        /* number of mirrors for this event */
  integer4 ntube;			/* total number of tubes for this event */
	
  /* -------------- mir info, one for each of nmir mirrors  ---------------*/
  integer4 mir [HR_UNIV_MAXMIR];	    /* mirror # (id), saved as short */
  integer4 mir_rev[HR_UNIV_MAXMIR];	    /* mirror version (rev3 or rev4) */
  integer4 mirevtno [HR_UNIV_MAXMIR];	    /* event # from mirror packet */
  integer4 mirntube [HR_UNIV_MAXMIR];	    /* # of tubes for that mir, saved as short */
  integer4 miraccuracy_ns[HR_UNIV_MAXMIR];  /* clock accuracy (gps or wwvb) in nsec */
  integer4 mirtime_ns[HR_UNIV_MAXMIR];	    /* time of mirror holdoff in nsec from second */

  /* -------------- tube info, one for each of ntube tubes  ---------------*/
  integer4 tubemir [HR_UNIV_MAXTUBE];	    /* mirror #, saved with tube as short */
  integer4 tube [HR_UNIV_MAXTUBE];    /* tube # */
  integer4 qdca [HR_UNIV_MAXTUBE];    /* digitized channel A charge integral */
  integer4 qdcb [HR_UNIV_MAXTUBE];    /* digitized channel B charge integral */
  integer4 tdc [HR_UNIV_MAXTUBE];    /* digitized tube trigger to holdoff time */
  integer4 tha [HR_UNIV_MAXTUBE];    /* trigger threshold in millivolts on minute ch A */
  integer4 thb [HR_UNIV_MAXTUBE];    /* trigger threshold in millivolts on minute ch B */
					    /* thb[] = 0 for hr1 mirrors */
  real4 prxf [HR_UNIV_MAXTUBE];      /* # of photons according to RXF calib. */
  real4 thcal1 [HR_UNIV_MAXTUBE];    /* time according to HCAL1 calib. */
} hraw1_dst_common ;

extern hraw1_dst_common hraw1_ ; 

#endif
