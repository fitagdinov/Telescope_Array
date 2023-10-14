// DSTbank definition for scaling between different energies at different
// angles in a shower library.
// It is envisioned that one azgh_showscale DST bank will be at the beginning
// of a DST file, followed by ALL the individual shower entries in ALL drawers
//
// azgh_showscale_dst.h: DRB - 2008/09/30

#ifndef _AZGH_SHOWSCALE_
#define _AZGH_SHOWSCALE_

#define AZGH_SHOWSCALE_BANKID      12802
#define AZGH_SHOWSCALE_BANKVERSION   000

#include "azgh_showlib_dst.h"

#define AZGH_SHOWLIB_MAX_LEBIN  10
#define AZGH_SHOWLIB_MAX_LEEDGE 11
#define AZGH_SHOWLIB_MAX_THBIN   5
#define AZGH_SHOWLIB_MAX_THEDGE  6

#ifdef __cplusplus
extern "C" {
#endif
integer4 azgh_showscale_common_to_bank_();
integer4 azgh_showscale_bank_to_dst_(integer4 *unit);
integer4 azgh_showscale_common_to_dst_(integer4 *units); // combines above 2
integer4 azgh_showscale_bank_to_common_(integer1 *bank);
integer4 azgh_showscale_common_to_dump_(integer4 *opt) ;
integer4 azgh_showscale_common_to_dumpf_(FILE* fp,integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* azgh_showscale_bank_buffer_ (integer4* azgh_showscale_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct {
  integer4 nlE;                        // Number of log(energy) bins in this library
  integer4 nth;                        // Number of angle bins in this library
  real8    lEEdge[AZGH_SHOWLIB_MAX_LEEDGE]; // Edges of log(energy) bins
  real8    secTh[AZGH_SHOWLIB_MAX_THBIN];   // Values of sec(th) where shower was generated
  real8    thEdge[AZGH_SHOWLIB_MAX_THEDGE]; // Edges sec(th) of angle bins in this library 
  real8    nMaxScaleFactor;   
  /* Correction factor to account for below threshold particles in CORSIKA; 
   * This factor multiplies the stored nMax to get a better nMax for this shower,
   * unlike the definition in Andreas Zech thesis */
  real8    nMaxLESlope[AZGH_SHOWLIB_MAX_THBIN];
  /* nMaxLESLope allows us to scale the nMax for the current energy, given 
   * a particular angular bin */
  real8    xMaxLESlope[AZGH_SHOWLIB_MAX_THBIN]; // to scale xMax (like for nMax)
  real8    x0LESlope[AZGH_SHOWLIB_MAX_THBIN];   // to scale x0
} azgh_showscale_dst_common;

extern azgh_showscale_dst_common azgh_showscale_;

// Function prototypes for various shower library utility functions

/* These routines return the scaled GH parameter, given a particular
 * library/scaling and a particular entry.  This is meant to include
 * any overall scaling due to CORSIKA thresholds */
real8 azgh_showscale_getScaledNMax(azgh_showscale_dst_common* library, 
				   azgh_showlib_dst_common* entry,
				   real8 lE); 
real8 azgh_showscale_getScaledXMax(azgh_showscale_dst_common* library, 
				   azgh_showlib_dst_common* entry,
				   real8 lE);
real8 azgh_showscale_getScaledX0(azgh_showscale_dst_common* library, 
				 azgh_showlib_dst_common* entry,
				 real8 lE);

/* These routines are identical to the above ones, but use the
 * global scped DST structures, and are thus FORTRAN callable */
real8 azgh_showscale_getScaledNMax_(real8 lE); 
real8 azgh_showscale_getScaledXMax_(real8 lE);
real8 azgh_showscale_getScaledX0_(real8 lE);

/* This routine returns a new azgh_showlib_dst_common structure
 * cloned from the original and scaled */
azgh_showlib_dst_common* azgh_showscale_scaledClone(azgh_showscale_dst_common* library, 
						     azgh_showlib_dst_common* entry,
						     real8 lE);
/* This routine scales a azgh_showlib_dst_common structure in place */
void azgh_showscale_scale(azgh_showscale_dst_common* library, 
			  azgh_showlib_dst_common* entry,
			  real8 lE);
/* This routine scales the azgh_showlib_dst_common structure in place 
 * and is Fortran callable (it uses only the common's)*/
void azgh_showscale_scale_(real8 lE);

/* This routines returns the number of shower particles using GH at a given depth
 * the entry is assumed to be unscaled */
real8 azgh_showscale_getScaledGHNe(azgh_showscale_dst_common* library, 
				   azgh_showlib_dst_common* entry,
				   real8 lE, real8 X);

/* This routines returns the number of shower particles using GH at a given depth
 * The entry is assumed scaled */
real8 azgh_showscale_getGHNe(azgh_showlib_dst_common* entry, real8 X);

/* This routines returns the number of shower particles using GH at a given depth 
 * using the (previously scaled) common */
real8 azgh_showscale_getghne_(real8 x);

real8 azgh_showscale_gh_(real8 nmax, real8 xmax, real8 x0, real8 lambda, real8 x);

#endif
