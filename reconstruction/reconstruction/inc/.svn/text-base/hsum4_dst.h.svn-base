/*
 * hsum4_dst.h
 *
 * $Source: $
 * $Log: $
 *
 * summary of hctim_dst, prfc_dst banks
 * used for analysis of BigH data
 *
 */

#ifndef _HSUM4_
#define _HSUM4_

#define HSUM4_BANKID 15035
#define HSUM4_BANKVERSION 0 

/* plog and invv cuts should be implemnted earlier in analysis */

#define  HSUM4_CUTS_MIN_PLOG   (real4) 2.0
#define  HSUM4_CUTS_MIN_INVV   (real4) 0.3

#define  HSUM4_CUTS_MIN_TRK    (real4) 7.9
#define  HSUM4_CUTS_MAX_XF     (real4) 1000.0
#define  HSUM4_CUTS_MIN_AVGCFC (real4) 0.9
#define  HSUM4_CUTS_MAX_PSI    (real4) 120.0

#define  HSUM4_FAIL_PLOG   17
#define  HSUM4_FAIL_INVV   19

#define  HSUM4_FAIL_IG     2
#define  HSUM4_FAIL_TRK    3
#define  HSUM4_FAIL_XF     5
#define  HSUM4_FAIL_AVGCFC 7
#define  HSUM4_FAIL_PSI    11
#define  HSUM4_FAIL_LASER  13


#ifdef __cplusplus
extern "C" {
#endif
integer4 hsum4_common_to_bank_(void);
integer4 hsum4_bank_to_dst_(integer4 *NumUnit);
integer4 hsum4_common_to_dst_(integer4 *NumUnit);
integer4 hsum4_bank_to_common_(integer1 *bank);
integer4 hsum4_common_to_dump_(integer4 *long_output);
integer4 hsum4_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hsum4_bank_buffer_ (integer4* hsum4_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



typedef struct  {

  real8    chi2_com;  /* combined profile / time fit chi2 */
  real8    chi2_pfl;  /* profile fit chi2 */
  real8    chi2_tim;  /* time fit chi2 */

  real8    jd;        /* unmodified julian day */
  real8    energy;    /* total shower energy (EeV) */
  real8    Nmax;      /* shower Nmax */
  real8    x0;        /* shower x0 (gm/cm^2) */
  real8    xmax;      /* shower xmax (gm/cm^2) */

  real8    xfirst;    /* atmos. depth (gm/cm^2) at first observed bin */
  real8    xlast;     /* atmos. depth (gm/cm^2) at last  observed bin */

  real8    rp;        /* impact parameter Rp (m) */
  real8    psi;       /* psi angle from time fit (deg) */
  real8    theta;     /* shower theta angle (deg) */

  real8    shower[3]; /* origin direction vector */
  real8    plnnorm[3]; /* reconstruction plane normal vector */

  real8    dec;       /* origin declination (deg) */
  real8    ra;        /* origin right ascension (deg) */

  integer4 imin_com;  /* prfc fit index (ifit) with min. combined chi2 */
  integer4 imin_tim;  /* prfc fit index (ifit) with min. time fit chi2 */
  integer4 imin_pfl;  /* prfc fit index (ifit) with min. profile chi2 */

  integer4 failmode;  /* failmode */

} hsum4_dst_common ;

extern hsum4_dst_common hsum4_ ; 

#endif









