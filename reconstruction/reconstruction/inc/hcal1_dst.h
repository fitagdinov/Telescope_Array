/*
 * hcal1_dst.h
 *
 * $Source: /hires_soft/uvm2k/bank/hcal1_dst.h,v $
 * $Log: hcal1_dst.h,v $
 * Revision 1.3  2001/03/15 18:05:54  reil
 * Added conditional inclusion.
 *
 * Revision 1.2  1997/08/28 00:50:30  jui
 * added macro defs HCAL1_BAD_PED and HCAL1_BAD_GAIN for default/bad
 * pedestals and gains
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

#ifndef __HCAL_BANK__
#define __HCAL_BANK__

#define HCAL1_BANKID 15010
#define HCAL1_BANKVERSION 0 
#define HCAL1_BAD_PED 100000.0
#define HCAL1_BAD_GAIN 0.0

#ifdef __cplusplus
extern "C" {
#endif
integer4 hcal1_common_to_bank_(void);
integer4 hcal1_bank_to_dst_(integer4 *NumUnit);
integer4 hcal1_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 hcal1_bank_to_common_(integer1 *bank);
integer4 hcal1_common_to_dump_(integer4 *long_output);
integer4 hcal1_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hcal1_bank_buffer_ (integer4* hcal1_bank_buffer_size);
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

   real4    tg [HR_UNIV_MAXMIR][HR_UNIV_MIRTUBE];        /* tdc gain */
   real4    tp [HR_UNIV_MAXMIR][HR_UNIV_MIRTUBE];        /* tdc pedestal */
   real4    qg [HR_UNIV_MAXMIR][HR_UNIV_MIRTUBE];        /* qdc gain */
   real4    qp [HR_UNIV_MAXMIR][HR_UNIV_MIRTUBE];        /* qdc pedestal */
   
   /*
    * NOTE: here the tube numbers are according to row/column, and in C
    * we need to subtract 1 from mirror # and tube # to get the storage
    * index. This subtraction is not necessary in FORTRAN.
    *
    * ALSO: if tp=HCAL1_BAD_PED, then the tdc calibration is not valid for this channel
    * AND:  if qp=HCAL1_BAD_PED, then the qdc calibration is not valid for this channel
    * under these circumstances the gains are set to HCAL1_BAD_GAIN
    */

} hcal1_dst_common ;

extern hcal1_dst_common hcal1_ ; 
#endif
