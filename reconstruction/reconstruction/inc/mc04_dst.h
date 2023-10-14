/*  mc04_dst.h
 *
 * $Source: /hires_soft/cvsroot/bank/mc04_dst.h,v $
 * $Log: mc04_dst.h,v $
 * Revision 1.2  2008/11/13 22:41:21  doug
 * Updated to include current parameters of GH fit.
 *
 *
 * output of mc04 event simulation
 *
 */

#ifndef _MC04_
#define _MC04_

#define  MC04_BANKID 15040
#define  MC04_BANKVERSION 2

#include "univ_dst.h"
#include "mc04_detector.h"

/* define event types */

#ifndef MC04_PROTON
#  define MC04_PROTON   1
#  define MC04_IRON     2
#  define MC04_GAMMA    3
#  define MC04_He       4
#  define MC04_CNO      5
#  define MC04_MgSi     6
#  define MC04_LASER   10
#  define MC04_FLASHER 11
#endif

/***********************************************/

#ifdef __cplusplus
extern "C" {
#endif
integer4  mc04_common_to_bank_(void);
integer4  mc04_bank_to_dst_(integer4 *NumUnit);
integer4  mc04_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4  mc04_bank_to_common_(integer1 *bank);
integer4  mc04_common_to_dump_(integer4 *long_output);
integer4  mc04_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* mc04_bank_buffer_ (integer4* mc04_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


/***********************************************/
/* Define common block structures    */

typedef struct {

  /* define geometry parameters */

  real8    energy;                  /* shower energy (EeV), (mJ) if laser */
  real8    csmax;                   /* shower size at shower max. / 1e9 */
  real8    x0;                      /* G-H fit paramater (g/cm^2) */
  real8    x1;                      /* depth of first interaction (g/cm^2) */
  real8    xmax;                    /* depth of shower max        (g/cm^2) */
  real8    lambda;                  /* G-H fit paramater          (g/cm^2) */
  real8    xfin;                    /* depth at rfin (from rini)  (g/cm^2) */
  real8    rini [3];                /* vector position of x0 point */
  real8    rfin [3];                /* vector position of final point */

  real8    uthat[3];                /* track direction unit vector */
  real8    theta;                   /* shower track zenith angle */
  real8    Rpvec[3];                /* Rp vector to track (m) from origin */
  real8    Rcore[3];                /* shower core (from origin),z=0 (m) */
  real8    Rp;                      /* magnitude of Rpvec */

  real8    rsite[MC04_MAXEYE][3];   /* site location with respect to origin */

  real8    rpvec[MC04_MAXEYE][3];   /* Rp vector to track (meters) */
  real8    rcore[MC04_MAXEYE][3];   /* shower core vector,z=0 (meters) */
  real8    shwn [MC04_MAXEYE][3];   /* shower-detector plane */
  real8    rp   [MC04_MAXEYE];      /* magnitude of rpvec */
  real8    psi  [MC04_MAXEYE];      /* psi angle in SD plane */

  real8    aero_vod;                /* aerosols vertical optical depth */
  real8    aero_hal;                /* aerosols horiz. attenuation length (m)*/
  real8    aero_vsh;                /* aerosols vertical scale height (m) */
  real8    aero_mlh;                /* aerosols mixing layer height */

  real8    la_site[3];              /* laser or flasher site (meters) */
  real8    la_wavlen;               /* laser wave length (nm) */
  real8    fl_totpho;               /* total number of photons */
  real8    fl_twidth;               /* flasher pulse width (ns) */

  integer4 iprim;                   /* primary particle: 1=proton, 2=iron */

  integer4 eventNr;                 /* event number in mc file (set) */
  integer4 setNr;                   /* set identifier YYMMDDPP */

  integer4 iseed1;                  /* iseed before event */
  integer4 iseed2;                  /* iseed after event */

  integer4 detid;                   /* detector id ( Hires, TA, TALE, ... ) */
  integer4 maxeye;                  /* number of sites in detector */
  integer4 if_eye [MC04_MAXEYE];    /* if (site[ieye] != 1) ignore site */

  integer4 neye;                    /* number of sites triggered */
  integer4 nmir;                    /* number of mirrors in event */
  integer4 ntube;                   /* total number of tubes in event */

  integer4 eyeid    [MC04_MAXEYE];  /* triggered site id */
  integer4 eye_nmir [MC04_MAXEYE];  /* number of triggered mirrors in eye */
  integer4 eye_ntube[MC04_MAXEYE];  /* number of triggered tube in eye */

  integer4 mirid    [MC04_MAXMIR];  /* triggered mirrors id */
  integer4 mir_eye  [MC04_MAXMIR];  /* triggered mirrors id */
  integer4 thresh   [MC04_MAXMIR];  /* mir. average tube threshold in mV */

  integer4 tubeid   [MC04_MAXTUBE]; /* tube id */
  integer4 tube_mir [MC04_MAXTUBE]; /* mirror id for each tube */
  integer4 tube_eye [MC04_MAXTUBE]; /* eye id for each tube */
  integer4 pe       [MC04_MAXTUBE]; /* pe's received by tube from shower */
  integer4 triggered[MC04_MAXTUBE]; /* 1 if tube is part of triggered event, 0 otherwise*/

  real4    t_tmean  [MC04_MAXTUBE]; /* pe's mean arrival time */
  real4    t_trms   [MC04_MAXTUBE]; /* pe's RMS of arrival times */
  real4    t_tmin   [MC04_MAXTUBE]; /* pe's min. arrival time */
  real4    t_tmax   [MC04_MAXTUBE]; /* pe's max. arrival time */

}  mc04_dst_common ;

extern  mc04_dst_common  mc04_ ; 

#endif

