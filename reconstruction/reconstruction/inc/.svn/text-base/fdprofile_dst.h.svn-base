/* Created 2008/11 LMS */

#ifndef _FDPROFILE_DST_
#define _FDPROFILE_DST_

#include "fdraw_dst.h"

#define FDPROFILE_BANKID        12094
#define FDPROFILE_BANKVERSION   000

#define FDPROF_MAXTSLICE        (fdraw_nt_chan_max)

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdprofile_common_to_bank_();
integer4 fdprofile_bank_to_dst_(integer4 *unit);
integer4 fdprofile_common_to_dst_(integer4 *unit); // combines above 2
integer4 fdprofile_bank_to_common_(integer1 *bank);
integer4 fdprofile_common_to_dump_(integer4 *opt);
integer4 fdprofile_common_to_dumpf_(FILE *fp, integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* fdprofile_bank_buffer_ (integer4* fdprofile_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct {
  integer4 siteid;			// site ID (BR = 0, LR = 1)
  integer4 ntslice;			// number of time slices (FADC bins)
  integer4 ngtslice[3];			// number of good time slices (acceptance)
  integer4 status[3];                   // status[0] is for fdplane_.psi
                                        // status[1] is for fdplane_.psi - fdplane_.epsi
                                        // status[2] is for fdplane_.psi + fdplane_.epsi
                                        // (-2 if bank is not filled
					//  -1 if bad geometry fit
                                        //   0 if bad profile fit
                                        //   1 if good profile fit)


  integer4 timebin[FDPROF_MAXTSLICE];	// FADC bin time slice

  real8 rp[3];                          // Impact parameter (meters)
  real8 psi[3];                         // Shower-detector plane angle (radians)
  real8 t0[3];                          // Detection time at Rp, less Rp travel time (ns)

  real8 Xmax[3];			// Shower maximum (g/cm2)
  real8 eXmax[3];			// uncertainty on xmax
  real8 Nmax[3];			// Number of charged particles at shower maximum
  real8 eNmax[3];			// uncertainty on nmax
  real8 Energy[3];			// Initial cosmic-ray energy
  real8 eEnergy[3];			// uncertainty on energy
  real8 chi2[3];			// Total chi2 of fit

  real8 npe[FDPROF_MAXTSLICE];		// number of photoelectrons by time slice
  real8 enpe[FDPROF_MAXTSLICE];		// uncertainty on npe

  real8 x[3][FDPROF_MAXTSLICE];		// slant depth at middle of time slice (g/cm2)

  real8 dtheta[3][FDPROF_MAXTSLICE];	// angular size of bin (radians)
  real8 darea[3][FDPROF_MAXTSLICE];	// cosine-corrected active area of mirror (sq. meter)

  real8 acpt[3][FDPROF_MAXTSLICE];	// PMT acceptance by time slice
  real8 eacpt[3][FDPROF_MAXTSLICE];	// binomial uncertainty on acceptance

  real8 flux[3][FDPROF_MAXTSLICE];	// flux at the mirror [photons / (m2 * radian)]
  real8 eflux[3][FDPROF_MAXTSLICE];	// uncertainty on flux

  real8 nfl[3][FDPROF_MAXTSLICE];	// Flux of simulated fluorescence photons
  real8 ncvdir[3][FDPROF_MAXTSLICE];	// Flux of simulated direct cerenkov photons
  real8 ncvmie[3][FDPROF_MAXTSLICE];	// Flux of simulated Mie scattered cerenkov photons
  real8 ncvray[3][FDPROF_MAXTSLICE];	// Flux of simulated Rayleigh scattered cerenkov photons
  real8 simflux[3][FDPROF_MAXTSLICE];	// Total flux of simluated photons

  real8 tres[3][FDPROF_MAXTSLICE];	// Time-slice fit residual
  real8 tchi2[3][FDPROF_MAXTSLICE];	// Time-slice fit chi2 contribution

  real8 ne[3][FDPROF_MAXTSLICE];	// Number of charged particles
  real8 ene[3][FDPROF_MAXTSLICE];	// uncertainty on ne

  integer4 mc;				// [0 = don't use trumpmc bank info, 1 = use trumpmc bank]
} fdprofile_dst_common;

extern fdprofile_dst_common fdprofile_;
extern integer4 fdprofile_blen; /* needs to be accessed by the c files of the derived banks */ 

integer4 fdprofile_struct_to_abank_(fdprofile_dst_common *fdprofile, integer1 *(*pbank), integer4 id, integer4 ver);
integer4 fdprofile_abank_to_dst_(integer1 *bank, integer4 *unit);
integer4 fdprofile_struct_to_dst_(fdprofile_dst_common *fdprofile, integer1 *bank, integer4 *unit, integer4 id, integer4 ver);
integer4 fdprofile_abank_to_struct_(integer1 *bank, fdprofile_dst_common *fdprofile);
integer4 fdprofile_struct_to_dump_(fdprofile_dst_common *fdprofile, integer4 *opt);
integer4 fdprofile_struct_to_dumpf_(fdprofile_dst_common *fdprofile, FILE *fp, integer4 *opt);

#endif
