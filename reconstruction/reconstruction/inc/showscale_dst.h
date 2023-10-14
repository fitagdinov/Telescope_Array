// DSTbank definition for scaling between different energies at different
// angles in a shower library.
// It is envisioned that one azgh_showscale DST bank will be at the beginning
// of a DST file, followed by ALL the individual shower entries in ALL drawers
//
// showscale_dst.h: SS - 2009/01/07

#ifndef _SHOWSCALE_
#define _SHOWSCALE_

#define SHOWSCALE_BANKID      12812
#define SHOWSCALE_BANKVERSION   001

#include "showlib_dst.h"

#define SHOWLIB_MAX_LEBIN  10
#define SHOWLIB_MAX_LEEDGE 11
#define SHOWLIB_MAX_THBIN   6
#define SHOWLIB_MAX_THEDGE  7

#ifdef __cplusplus
extern "C" {
#endif
integer4 showscale_common_to_bank_();
integer4 showscale_bank_to_dst_(integer4 *unit);
integer4 showscale_common_to_dst_(integer4 *units); // combines above 2
integer4 showscale_bank_to_common_(integer1 *bank);
integer4 showscale_common_to_dump_(integer4 *opt) ;
integer4 showscale_common_to_dumpf_(FILE* fp,integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* showscale_bank_buffer_ (integer4* showscale_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct {
  integer4 nlE;                        // Number of log(energy) bins in this library
  integer4 nth;                        // Number of angle bins in this library
  real8    lEEdge[SHOWLIB_MAX_LEEDGE]; // Edges of log(energy) bins
  real8    secTh[SHOWLIB_MAX_THBIN];   // Values of sec(th) where shower was generated
  real8    thEdge[SHOWLIB_MAX_THEDGE]; // Edges sec(th) of angle bins in this library 
  real8    nMaxScaleFactor;   
  /* Correction factor to account for below threshold particles in CORSIKA; 
   * This factor multiplies the stored nMax to get a better nMax for this shower,
   * unlike the definition in Andreas Zech thesis */
  real8    nMaxLESlope[SHOWLIB_MAX_THBIN];
  /* nMaxLESLope allows us to scale the nMax for the current energy, given 
   * a particular angular bin */
  real8    xMaxLESlope[SHOWLIB_MAX_THBIN]; // to scale xMax (like for nMax)
  real8    x0LESlope[SHOWLIB_MAX_THBIN];   // to scale x0, OR (TAS 20150124),
                                           // if nMaxScaleFactor < 0, to scale WIDTH!
  integer4  numEntries[SHOWLIB_MAX_LEBIN][SHOWLIB_MAX_THBIN];
} showscale_dst_common;

extern showscale_dst_common showscale_;

// Function prototypes for various shower library utility functions

/* These routines return the scaled GH parameter, given a particular
 * library/scaling and a particular entry.  This is meant to include
 * any overall scaling due to CORSIKA thresholds */

#ifdef __cplusplus
extern "C" real8 showscale_getScaledNMax(showscale_dst_common* library, 
					 showlib_dst_common* entry,
					 real8 lE); 
extern "C" real8 showscale_getScaledXMax(showscale_dst_common* library, 
					 showlib_dst_common* entry,
					 real8 lE);
extern "C" real8 showscale_getScaledX0(showscale_dst_common* library, 
				       showlib_dst_common* entry,
				       real8 lE);

/* These routines are identical to the above ones, but use the
 * global scped DST structures, and are thus FORTRAN callable */
extern "C" real8 showscale_getScaledNMax_(real8 lE); 
extern "C" real8 showscale_getScaledXMax_(real8 lE);
extern "C" real8 showscale_getScaledX0_(real8 lE);

/* This routine returns a new showlib_dst_common structure
 * cloned from the original and scaled */
extern "C" showlib_dst_common* showscale_scaledClone(showscale_dst_common* library, 
						     showlib_dst_common* entry,
						     real8 lE);
/* This routine scales a showlib_dst_common structure in place */
extern "C" void showscale_scale(showscale_dst_common* library, 
				showlib_dst_common* entry,
				real8 lE);
/* This routine scales the showlib_dst_common structure in place 
 * and is Fortran callable (it uses only the common's)*/
extern "C" void showscale_scale_(real8 lE);

/* This routines returns the number of shower particles using GH at a given depth
 * the entry is assumed to be unscaled */
extern "C" real8 showscale_getScaledGHNe(showscale_dst_common* library, 
					 showlib_dst_common* entry,
					 real8 lE, real8 X);

/* This routines returns the number of shower particles using GH at a given depth
 * The entry is assumed scaled */
extern "C" real8 showscale_getGHNe(showlib_dst_common* entry, real8 X);

/* This routines returns the number of shower particles using GH at a given depth 
 * using the (previously scaled) common */
extern "C" real8 showscale_getghne_(real8 x);

extern "C" real8 showscale_gh_(real8 nmax, real8 xmax, real8 x0, real8 lambda, real8 x);

// new 20150124, to scale shower width instead of X0 (T. A. Stroman)
extern "C" real8 getWidthFromLambdaX0(showlib_dst_common *entry);
extern "C" real8 getRatioFromLambdaX0(showlib_dst_common *entry);
extern "C" real8 getLambdaFromScaled(real8 width, real8 ratio);
extern "C" real8 getX0FromScaled(real8 width, real8 ratio, real8 xmax);
extern "C" real8 scaleWidth(real8 width, real8 lE, real8 lE0, real8 slope);
extern "C" real8 showscale_getScaledWidth(showscale_dst_common *library,
					  showlib_dst_common *entry,
					  real8 lE);
#else
real8 showscale_getScaledNMax(showscale_dst_common* library, 
				   showlib_dst_common* entry,
				   real8 lE); 
real8 showscale_getScaledXMax(showscale_dst_common* library, 
				   showlib_dst_common* entry,
				   real8 lE);
real8 showscale_getScaledX0(showscale_dst_common* library, 
				 showlib_dst_common* entry,
				 real8 lE);

/* These routines are identical to the above ones, but use the
 * global scped DST structures, and are thus FORTRAN callable */
real8 showscale_getScaledNMax_(real8 lE); 
real8 showscale_getScaledXMax_(real8 lE);
real8 showscale_getScaledX0_(real8 lE);

/* This routine returns a new showlib_dst_common structure
 * cloned from the original and scaled */
showlib_dst_common* showscale_scaledClone(showscale_dst_common* library, 
						     showlib_dst_common* entry,
						     real8 lE);
/* This routine scales a showlib_dst_common structure in place */
void showscale_scale(showscale_dst_common* library, 
			  showlib_dst_common* entry,
			  real8 lE);
/* This routine scales the showlib_dst_common structure in place 
 * and is Fortran callable (it uses only the common's)*/
void showscale_scale_(real8 lE);

/* This routines returns the number of shower particles using GH at a given depth
 * the entry is assumed to be unscaled */
real8 showscale_getScaledGHNe(showscale_dst_common* library, 
				   showlib_dst_common* entry,
				   real8 lE, real8 X);

/* This routines returns the number of shower particles using GH at a given depth
 * The entry is assumed scaled */
real8 showscale_getGHNe(showlib_dst_common* entry, real8 X);

/* This routines returns the number of shower particles using GH at a given depth 
 * using the (previously scaled) common */
real8 showscale_getghne_(real8 x);

real8 showscale_gh_(real8 nmax, real8 xmax, real8 x0, real8 lambda, real8 x);

// new 20150124, to scale shower width instead of X0 (T. A. Stroman)
real8 getWidthFromLambdaX0(showlib_dst_common *entry);
real8 getRatioFromLambdaX0(showlib_dst_common *entry);
real8 getLambdaFromScaled(real8 width, real8 ratio);
real8 getX0FromScaled(real8 width, real8 ratio, real8 xmax);
real8 scaleWidth(real8 width, real8 lE, real8 lE0, real8 slope);
real8 showscale_getScaledWidth(showscale_dst_common *library,
                               showlib_dst_common *entry,
                               real8 lE);
#endif


#endif
