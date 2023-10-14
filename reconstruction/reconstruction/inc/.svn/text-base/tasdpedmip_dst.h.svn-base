/*
 *     Bank for SD MIP analysis
 *     written by a student
 *     Time-stamp: Thu Apr 23 19:21:35 2009 JST
*/

#ifndef _TASDPEDMIP_
#define _TASDPEDMIP_

#define TASDPEDMIP_BANKID  13012
#define TASDPEDMIP_BANKVERSION   003

#ifdef __cplusplus
extern "C" {
#endif
int tasdpedmip_common_to_bank_();
int tasdpedmip_bank_to_dst_(int *NumUnit);
int tasdpedmip_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdpedmip_bank_to_common_(char *bank);
int tasdpedmip_common_to_dump_(int *opt1) ;
int tasdpedmip_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdpedmip_bank_buffer_ (integer4* tasdpedmip_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdpedmip_ndmax 512    /* maximum number of detector per event      */


typedef struct {
  int lid;
  int livetime;		/* livetime in 10 min */

  float upedAmp;	/* amplitude of pedestal (upper layer) */
  float upedAvr;	/* average of pedestal (upper layer) */
  float upedStdev;      /* standard deviation of pedestal
			  (upper layer) */
  float upedAmpError;   /* error of amplitude of pedestal
			  (upper layer) */
  float upedAvrError;   /* error of average of pedestal
			  (upper layer) */
  float upedStdevError; /* error of standard deviation of pedestal
			   (upper layer) */
  float upedChisq;      /* Chi square value (upper layer) */
  float upedDof;	/* degree of freedom (upper layer) */
  int ubroken;		/* FADC broken flag */

  float lpedAmp;	/* amplitude of pedestal (lower layer) */
  float lpedAvr;	/* average of pedestal (lower layer) */
  float lpedStdev;	/* standard deviation of pedestal
			  (lower layer) */
  float lpedAmpError;   /* error of amplitude of pedestal
			  (lower layer) */
  float lpedAvrError;   /* error of average of pedestal
			  (lowerr layer) */
  float lpedStdevError; /* error of standard deviation of pedestal
			  (lower layer) */
  float lpedChisq;      /* Chi square value (lower layer) */
  float lpedDof;	/* degree of freedom (lower layer) */
  int lbroken;		/* FADC broken flag */


  float umipAmp;	  /* amplitude of MIP histogram
			     (upper layer) */
  float umipNonuni;       /* Non-uniformity (upper layer) */
  float umipMev2cnt;      /* Mev to count conversion factor (upper
			     layer) */
  float umipMev2pe;       /* Mev to photo-electron conversion
			     factor (upper layer) */
  float umipAmpError;     /* error of amplitude of MIP histogram
			     (upper layer)  */
  float umipNonuniError;  /* error of non-uniformity
			     (upper layer) */
  float umipMev2cntError; /* error of Mev to count conversion
			     factor (upper layer) */
  float umipMev2peError;  /* error of Mev to photo-electron
			     conversion factor (upper layer) */
  float umipChisq;	  /* Chi square value (upper layer) */
  float umipDof;	  /* degree of freedom (upper layer) */

  float lmipAmp;	  /* amplitude of MIP histogram
			     (lower layer) */
  float lmipNonuni;       /* Non-uniformity (lower layer) */
  float lmipMev2cnt;      /* Mev to count conversion factor (lower
			     layer) */
  float lmipMev2pe;       /* Mev to photo-electron conversion
			     factor (lower layer) */
  float lmipAmpError;     /* error of amplitude of MIP histogram
			     (lower layer)  */
  float lmipNonuniError;  /* error of non-uniformity
			     (lower layer) */
  float lmipMev2cntError; /* error of Mev to count conversion
			     factor (lower layer) */
  float lmipMev2peError;  /* error of Mev to photo-electron
			     conversion factor (lower layer) */
  float lmipChisq;	  /* Chi square value (lower layer) */
  float lmipDof;	  /* degree of freedom (lower layer) */

  float lvl0Rate[10];	  /* level-0 trigger rate */
  float lvl1Rate[10];	  /* level-1 trigger rate */


} SDPedMipData;


typedef struct {
  int num_det;   /* the number of detectors       */
  int date;      /* year month day */
  int time;      /* hour minute second */

  SDPedMipData sub[tasdpedmip_ndmax];

  int footer;

} tasdpedmip_dst_common;

extern tasdpedmip_dst_common tasdpedmip_;

#endif


