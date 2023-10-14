/*
 * Created 2000/04/22 DRB
*/

#ifndef _FSCN1_
#define _FSCN1_

#define FSCN1_BANKID 12013
#define FSCN1_BANKVERSION 0 

#ifdef __cplusplus
extern "C" {
#endif
integer4 fscn1_common_to_bank_();
integer4 fscn1_bank_to_dst_(integer4 *NumUnit);
integer4 fscn1_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 fscn1_bank_to_common_(integer1 *bank);
integer4 fscn1_common_to_dump_(integer4 *long_output);
integer4 fscn1_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* fscn1_bank_buffer_ (integer4* fscn1_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct  {
	integer4 ntube;
	integer4 mir[HR_UNIV_MAXTUBE];    /* mir number */ 
	integer4 tube[HR_UNIV_MAXTUBE];	  /* tube number */
	integer4 ig[HR_UNIV_MAXTUBE];     /* tube flag
                                    ig=1: good tube  
                                    ig=0: rejected by scan or has no signal */
	real4    ped[HR_UNIV_MAXTUBE];    /* pedestal */
	real4    pedrms[HR_UNIV_MAXTUBE]; /* RMS of pedestal */
	integer4 pamp[HR_UNIV_MAXTUBE];   /* Max amplitude of filtered pulse */
	integer4 pmaxt[HR_UNIV_MAXTUBE];  /* Time index of max */
	real4    pnpe[HR_UNIV_MAXTUBE];   /* Integrated pulse above ped */
        integer4 pt0[HR_UNIV_MAXTUBE];    /* Index of first slice in pulse */
	integer4 pnt[HR_UNIV_MAXTUBE];    /* Number of slices in pulse */
	real4    ptav[HR_UNIV_MAXTUBE];   /* Weighted average time */
	real4    pfilt[HR_UNIV_MAXTUBE];  /* Filter time scale in slices */
} fscn1_dst_common ;

extern fscn1_dst_common fscn1_ ; 

#endif


