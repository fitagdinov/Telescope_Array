
/*
 * hbar_dst.h
 *
 * $Source: /hires_soft/uvm2k/bank/hbar_dst.h,v $
 * $Log: hbar_dst.h,v $
 * Revision 1.2  2000/05/31 20:33:09  ben
 * Added QE sigma variables.
 *
 * Revision 1.1  2000/02/04 22:44:16  ben
 * Initial revision
 *
 *
 * This is a calibration bank for BigH.  It reads in the Roving Xenon Flasher 
 * and/or YAG calibration coefficients from the hnpe bank and adds them to the 
 * data stream in hpass1.
 *
 */

#ifndef HBAR_BANKID

#define HBAR_BANKID 15023
#define HBAR_BANKVERSION 0 

#define HBAR_FNC	256   /*  Maximum number of filename characters  */
#define HBAR_MAXSRC     20

#ifndef __HBARBIT
#define __HBARBIT
#define HBARBIT(x) (1 << (x))
#endif

  /* An enumerated constant to describe the source that was used to create 
     this bank  */ 

enum source_desc {YAG, RXF}; 


#define SOURCE_NUM      10    /* number of sources in source_desc */

#ifdef __cplusplus
extern "C" {
#endif
integer4 hbar_common_to_bank_(void);
integer4 hbar_bank_to_dst_(integer4 *NumUnit);
integer4 hbar_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 hbar_bank_to_common_(integer1 *bank);
integer4 hbar_common_to_dump_(integer4 *long_output);
integer4 hbar_common_to_dumpf_(FILE *fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hbar_bank_buffer_ (integer4* hbar_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



typedef struct 
{
  /* 2440000 subtracted from the julian day to give room for millisecond 
     precison */
  /* checks: "1/1/1985,0.0hr UT" gives 6066 + 0.5; jday=0 gives 1968 May 23.5 
     UT*/
  
  integer4 jday;        /* mean julian day - 2.44e6 */
  integer4 jsec;        /* second into Julian day */
  integer4 msec;        /* milli sec of julian day (NOT since UT0:00) */

  /* 
     Jday of hnpe dst bank from which data for this bank is extracted. 
     One jday for each mirror.
  */
  real8 hnpe_jday[HR_UNIV_MAXMIR];  

  /* Filled with an enum source type */  

  integer4 source;
 
  integer4 nmir;        /* number of mirrors for this event */
  integer4 ntube;       /* total number of tubes for this event */

 
  /*---------------mirror info, one for each of nmir mirrors-------------*/

  integer4 mir[HR_UNIV_MAXMIR];	  /* mirror # (id), one for each of nmir
				   mirrors */
  
  /* Mirror reflectivity coefficient, one for each mirror */
  
  real8 mir_reflect[HR_UNIV_MAXMIR];

  /*---------------tube info, one for each of ntube tubes----------------*/

  integer4 tubemir[HR_UNIV_MAXTUBE];  /* mirror #, saved with tube as short */
  integer4 tube[HR_UNIV_MAXTUBE];    /* tube # */
  integer4 qdcb[HR_UNIV_MAXTUBE];    /* digitized channel B charge integral */

   /*
     Number of photo-electrons that is calculated by taking the qdcb from 
     above and multipling it by the first order gain from the hnpe dst bank
  */
  
  real8 npe[HR_UNIV_MAXTUBE];

  /*
     Sigma of photo-electrons from above calculated by taking the sigma of the 
     first order gain and multiplying it by the qdcb from above
  */

  real8 sigma_npe[HR_UNIV_MAXTUBE];  
 
 /*
     flag used to determine if a given tube has had problems in the fitting 
     process.
  */

  unsigned char first_order_gain_flag[HR_UNIV_MAXTUBE];

 
  /*
     Second order gain obtained from the electronic calibration and relates a
     nanovolt-second pulse width to photo-electrons.

     nVs(QDCB[i][t], width) * second_order_gain[i][t] = NPE[i][t] 
  */
     
  real8 second_order_gain[HR_UNIV_MAXTUBE]; 
						
  /*
     The second order fit goodness is a "goodness of fit" estimate for the 
     NVs(QDCB, w) vs NPE fit.
  */

  real8 second_order_gain_sigma[HR_UNIV_MAXTUBE];

 /*
     flag used to determine if a given tube has had problems in the second 
     order gain fitting process.
  */

  unsigned char second_order_gain_flag[HR_UNIV_MAXTUBE];

  /*
     These values are the quantum efficiency factor for each tube at 337 nm.
     The values are to be used in scaling a standard quantum curve. 
  */

  real8 qe_337[HR_UNIV_MAXTUBE];
  real8 sigma_qe_337[HR_UNIV_MAXTUBE];
  
  /*
     These values modify the given HiRes UV filter curve for every tube.
     They are applied as an exponent to the curve being used.
  */

  real8 uv_exp[HR_UNIV_MAXTUBE];

} 
hbar_dst_common;

extern hbar_dst_common hbar_; 

#endif







