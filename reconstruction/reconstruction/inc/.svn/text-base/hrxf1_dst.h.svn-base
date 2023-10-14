/*
 * hrxf1_dst.h
 *
 * $Source: /hires_soft/uvm2k/bank/hrxf1_dst.h,v $
 * $Log: hrxf1_dst.h,v $
 * Revision 1.3  2001/03/15 18:05:59  reil
 * Added conditional inclusion.
 *
 * Revision 1.2  1997/08/28 01:03:16  jui
 * added macro defs HRXF1_BAD_PED 100000.0 and HRXF1_BAD_GAIN 0.0
 *
 * Revision 1.1  1997/08/24  22:06:15  jui
 * Initial revision
 *
 *
 * Big-H "xenon roving flasher" calibration.
 * QDC vs. photon count responses are assumed perfectly linear
 * no error estimates are made for the conversion factors
 * the gains are using filter factor (0-1) as the INDEPENDENT
 * variable, and the QDC result as the DEPENDENT variable
 *
 */

#ifndef __HRXF1_BANK__
#define __HRXF1_BANK__

#define HRXF1_BANKID 15011
#define HRXF1_BANKVERSION 0 
#define HRXF1_BAD_PED 100000.0
#define HRXF1_BAD_GAIN 0.0

#ifdef __cplusplus
extern "C" {
#endif
integer4 hrxf1_common_to_bank_(void);
integer4 hrxf1_bank_to_dst_(integer4 *NumUnit);
integer4 hrxf1_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 hrxf1_bank_to_common_(integer1 *bank);
integer4 hrxf1_common_to_dump_(integer4 *long_output);
integer4 hrxf1_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hrxf1_bank_buffer_ (integer4* hrxf1_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct 
{
   integer4 jday1;      /* valid time range starting Julian day */
   integer4 jsec1;      /* valid time range starting second into day */
   integer4 jday2;      /* valid time range ending Julian day */
   integer4 jsec2;      /* valid time range ending second into day */
   
   /* jday and jsec values are set to -1 if no valid limit */

   real4    prxf;  /* # of photons for 100% (no filter) flasher strength */
   real4    xg [HR_UNIV_MAXMIR][HR_UNIV_MIRTUBE];        /* rxf gain */
   real4    xp [HR_UNIV_MAXMIR][HR_UNIV_MIRTUBE];        /* rxf pedestal */
   
   /*
    * NOTE: here the tube numbers are according to row/column, and in C
    * we need to subtract 1 from mirror # and tube # to get the storage
    * index. This subtraction is not necessary in FORTRAN.
    *
    * ALSO: if xp=HRXF1_BAD_PED, then the rxf calib. is not valid for this channel
    * under these circumstances the gain is set to HRXF1_BAD_GAIN
    */

} hrxf1_dst_common ;

extern hrxf1_dst_common hrxf1_ ; 

#endif
