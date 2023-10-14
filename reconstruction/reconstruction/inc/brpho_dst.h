/*
 * brpho_dst.h
 *
 * $Source: /hires_soft/uvm2k/bank/brpho_dst.h,v $
 * $Log: brpho_dst.h,v $
 * Revision 1.4  2003/05/28 15:27:02  hires
 * update to nevis version to eliminate tar file dependency (boyer)
 *
 * Revision 1.3  1997/02/21  00:33:05  mjk
 * Updated prototypes to ANSI standard. Added Source and Log directives.
 *
 *
 * FADC equivalent version of pho1_dst.h
 * Created from pho1_dst by replacing all occurences of "pho1" with "brpho".
 *
 * Author:  J. Boyer 11 July 1995
*/

#ifndef _BRPHO_
#define _BRPHO_

#define BRPHO_BANKID 12105
#define BRPHO_BANKVERSION 0 

#ifdef __cplusplus
extern "C" {
#endif
integer4 brpho_common_to_bank_(void);
integer4 brpho_bank_to_dst_(integer4 *NumUnit);
integer4 brpho_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 brpho_bank_to_common_(integer1 *bank);
integer4 brpho_common_to_dump_(integer4 *long_output) ;
integer4 brpho_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* brpho_bank_buffer_ (integer4* brpho_bank_buffer_size);
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

} brpho_dst_common ;

extern brpho_dst_common brpho_ ; 

#endif
