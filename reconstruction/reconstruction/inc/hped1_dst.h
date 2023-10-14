/*
 * hped1_dst.h
 *
 * $Source: /hires_soft/uvm2k/bank/hped1_dst.h,v $
 * $Log: hped1_dst.h,v $
 * Revision 1.3  2001/03/15 18:05:57  reil
 * Added conditional inclusion.
 *
 * Revision 1.2  1997/08/28 00:58:07  jui
 * added macro def HPED1_BAD_PED (=100000.0)
 * for bad/default pedestals
 *
 * Revision 1.1  1997/08/25  16:46:48  jui
 * Initial revision
 *
 * Revision 1.1  1997/08/17  18:24:21  jui
 * Initial revision
 *
 *
 * Big-H "simple" (i.e. quick and crude) electronic calibration bank
 * both QDC (channel B only)  and TDC are assumed perfectly linear
 * no error estimates are made for the conversion factors,
 * and phototube gains are considered elsewhere
 * 
 */

#ifndef __HPED1__BANK__
#define __HPED1__BANK__

#define HPED1_BANKID 15012
#define HPED1_BANKVERSION 0 
#define HPED1_BAD_PED 100000.0

#ifdef __cplusplus
extern "C" {
#endif
integer4 hped1_common_to_bank_(void);
integer4 hped1_bank_to_dst_(integer4 *NumUnit);
integer4 hped1_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 hped1_bank_to_common_(integer1 *bank);
integer4 hped1_common_to_dump_(integer4 *long_output);
integer4 hped1_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hped1_bank_buffer_ (integer4* hped1_bank_buffer_size);
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

   real4    pA [HR_UNIV_MAXMIR][HR_UNIV_MIRTUBE];        /* qdcA pedestal */
   real4    pB [HR_UNIV_MAXMIR][HR_UNIV_MIRTUBE];        /* qdcB pedestal */
   
   /*
    * NOTE: here the tube numbers are according to row/column, and in C
    * we need to subtract 1 from mirror # and tube # to get the storage
    * index. This subtraction is not necessary in FORTRAN.
    *
    * ALSO: if pA or pB=HPED1_BAD_PED, then the pedestal is not valid
    */

} hped1_dst_common ;

extern hped1_dst_common hped1_ ; 

#endif
