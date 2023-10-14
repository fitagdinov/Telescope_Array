/*
 *     Bank for SD calibration information needed for Rutgers SD reconstruction
 *     Dmitri Ivanov (ivanov@physics.rutgers.edu)
 *     Created: Aug 17, 2009
 *     Last modified: Aug 17, 2009
*/

#ifndef _RUSDCAL_
#define _RUSDCAL_
#define RUSDCAL_BANKID  13110
#define RUSDCAL_BANKVERSION   000

#ifdef __cplusplus
extern "C" {
#endif
integer4 rusdcal_common_to_bank_ ();
integer4 rusdcal_bank_to_dst_ (integer4 * NumUnit);
integer4 rusdcal_common_to_dst_ (integer4 * NumUnit);	/* combines above 2 */
integer4 rusdcal_bank_to_common_ (integer1 * bank);
integer4 rusdcal_common_to_dump_ (integer4 * opt1);
integer4 rusdcal_common_to_dumpf_ (FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* rusdcal_bank_buffer_ (integer4* rusdcal_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



#define RUSDCAL_NSDS 512 // Max. number of SDs
#define RUSDCAL_NMONCHAN 512 // Max. number of channels in a monitoring histogram


typedef struct
{

  integer4 nsds;                        /* total number of SDs present */
  integer4 xxyy[RUSDCAL_NSDS];          /* counter position ID (LID) */
  integer4 date;
  integer4 time;
  
  // [*][0] - lower layer, [*][1] - upper layer
  integer4 pchmip[RUSDCAL_NSDS][2];     /* peak channel of 1MIP histograms */
  integer4 pchped[RUSDCAL_NSDS][2];     /* peak channel of pedestal histograms */
  integer4 lhpchmip[RUSDCAL_NSDS][2];   /* left half-peak channel for 1mip histogram */
  integer4 lhpchped[RUSDCAL_NSDS][2];   /* left half-peak channel of pedestal histogram */
  integer4 rhpchmip[RUSDCAL_NSDS][2];   /* right half-peak channel for 1mip histogram */
  integer4 rhpchped[RUSDCAL_NSDS][2];   /* right half-peak channel of pedestal histograms */
  
  /* Results from fitting 1MIP histograms */
  integer4 mftndof[RUSDCAL_NSDS][2]; /* number of degrees of freedom in 1MIP fit */
  real8 mip[RUSDCAL_NSDS][2];        /* 1MIP value (ped. subtracted) */
  real8 mftchi2[RUSDCAL_NSDS][2];    /* chi2 of the 1MIP fit */
  
  /* 
     1MIP Fit function: 
     [3]*(1+[2]*(x-[0]))*exp(-(x-[0])*(x-[0])/2/[1]/[1])/sqrt(2*PI)/[1]
     4 fit parameters:
     [0]=Gauss Mean
     [1]=Gauss Sigma
     [2]=Linear Coefficient
     [3]=Overall Scalling Factor
  */
  real8 mftp[RUSDCAL_NSDS][2][4];    /* 1MIP fit parameters */
  real8 mftpe[RUSDCAL_NSDS][2][4];   /* Errors on 1MIP fit parameters */
  

} rusdcal_dst_common;

extern rusdcal_dst_common rusdcal_;

#endif
