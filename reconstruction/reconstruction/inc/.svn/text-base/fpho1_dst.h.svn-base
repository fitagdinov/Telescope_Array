/*
 * fpho1_dst.h
 *
 * $Source: /hires_soft/uvm2k/bank/fpho1_dst.h,v $
 * $Log: fpho1_dst.h,v $
 * Revision 1.4  2003/05/28 15:27:02  hires
 * update to nevis version to eliminate tar file dependency (boyer)
 *
 * Revision 1.3  1997/02/21  00:33:05  mjk
 * Updated prototypes to ANSI standard. Added Source and Log directives.
 *
 *
 * FADC equivalent version of pho1_dst.h
 * Created from pho1_dst by replacing all occurences of "pho1" with "fpho1".
 *
 * Author:  J. Boyer 11 July 1995
*/

#ifndef _FPHO1_
#define _FPHO1_

#define FPHO1_BANKID 12004 
#define FPHO1_BANKVERSION 1 

#ifdef __cplusplus
extern "C" {
#endif
integer4 fpho1_common_to_bank_(void);
integer4 fpho1_bank_to_dst_(integer4 *NumUnit);
integer4 fpho1_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 fpho1_bank_to_common_(integer1 *bank);
integer4 fpho1_common_to_dump_(integer4 *long_output) ;
integer4 fpho1_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* fpho1_bank_buffer_ (integer4* fpho1_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct  {
  real8    jday;                      /* mean julian day - 2.44e6 */
  integer4 ntube;                     /* number of tubes */
  integer1 calfile [32];              /* calibration file name */

  /* tube info  */

  integer4 mirtube[HR_UNIV_MAXTUBE];  /* mir * 1000 + tubeid */
  integer4 pha    [HR_UNIV_MAXTUBE];  /* channel A photons */
  integer4 phb    [HR_UNIV_MAXTUBE];  /* channel B photons */

  /* tab is a 3-digit number for tdc,qdca,qdcb calib status */

  integer4 tab    [HR_UNIV_MAXTUBE];
  integer4 time   [HR_UNIV_MAXTUBE];  /* time of pulse mean, nsec since event store start time */ 
  integer4 dtime  [HR_UNIV_MAXTUBE];  /* width of pulse (nsec) */

  /* more accurate calibrated values, added by DI on  2016/11/17 */
  real4    adc    [HR_UNIV_MAXTUBE];  /* tube signal in ADC counts, pedestal subtracted */
  real4    npe    [HR_UNIV_MAXTUBE];  /* tube signal in NPE units, pedestal subtracted */
  real4    ped    [HR_UNIV_MAXTUBE];  /* tube threshold in ADC counts */

} fpho1_dst_common ;

extern fpho1_dst_common fpho1_ ; 

#endif
