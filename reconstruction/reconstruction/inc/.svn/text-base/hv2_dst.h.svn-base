/*
 * $Source: /hires_soft/cvsroot/bank/hv2_dst.h,v $
 * $Log: hv2_dst.h,v $
 * Revision 1.1  1995/10/27 18:25:22  wiencke
 * Initial revision
 *
*/

#ifndef _HV2_
#define _HV2_

#define HV2_BANKID 21006 
#define HV2_BANKVERSION 0 

#define HV2_MAXMEAS 500

#ifdef __cplusplus
extern "C" {
#endif
integer4 hv2_common_to_bank_();
integer4 hv2_bank_to_dst_(integer4 *NumUnit);
integer4 hv2_common_to_dst_(integer4 *NumUnit);
integer4 hv2_bank_to_common_(integer1 *bank);
integer4 hv2_common_to_dump_(integer4 *long_form);
integer4 hv2_common_to_dumpf_(FILE* fp, integer4 *long_form);
/* get (packed) buffer pointer and size */
integer1* hv2_bank_buffer_ (integer4* hv2_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct
{
  integer4 nmeas;			/* # of HV measurments */
  integer4 mir[HV2_MAXMEAS];		/* mirror number */
  integer4 stat[HV2_MAXMEAS];		/* status word (not used) */
  integer4 sclu[HV2_MAXMEAS];		/* subcluster number (1 to 16) */
  real4 volts[HV2_MAXMEAS];		/* voltage parameter */
  integer4 fdate[HV2_MAXMEAS];		/* begin date for valid meas. YYMMDD */
  integer4 ldate[HV2_MAXMEAS];		/* end date for valid measure YYMMDD */
} hv2_dst_common ;

extern hv2_dst_common hv2_ ; 

#endif
