// DSTbank definition for scaling between different energies at different
// angles in a shower library.
// It is envisioned that one azgh_showscale DST bank will be at the beginning
// of a DST file, followed by ALL the individual shower entries in ALL drawers
//
// azgh_showscale_dst.c: DRB - 2008/09/30

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "azgh_showscale_dst.h"
#include "azgh_showlib_dst.h"

azgh_showscale_dst_common azgh_showscale_;

static integer4 azgh_showscale_blen = 0;
static integer4 azgh_showscale_maxlen = sizeof(integer4) * 2 + sizeof(azgh_showscale_dst_common);
static integer1 *azgh_showscale_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* azgh_showscale_bank_buffer_ (integer4* azgh_showscale_bank_buffer_size)
{
  (*azgh_showscale_bank_buffer_size) = azgh_showscale_blen;
  return azgh_showscale_bank;
}



static void azgh_showscale_bank_init() {
  azgh_showscale_bank = (integer1 *)calloc(azgh_showscale_maxlen, sizeof(integer1));
  if (azgh_showscale_bank==NULL) {
      fprintf (stderr,"azgh_showscale_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 azgh_showscale_common_to_bank_() {
  static integer4 id = AZGH_SHOWSCALE_BANKID, ver = AZGH_SHOWSCALE_BANKVERSION;
  integer4 rcode, nobj;

  if (azgh_showscale_bank == NULL) azgh_showscale_bank_init();

  rcode = dst_initbank_(&id, &ver, &azgh_showscale_blen, &azgh_showscale_maxlen, azgh_showscale_bank);

// Initialize azgh_showscale_blen and pack the id and version to bank

  nobj = 1;
  rcode += dst_packi4_(&azgh_showscale_.nlE,             &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  rcode += dst_packi4_(&azgh_showscale_.nth,             &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = azgh_showscale_.nlE+1;
  rcode += dst_packr8_(&azgh_showscale_.lEEdge[0],       &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = azgh_showscale_.nth;
  rcode += dst_packr8_(&azgh_showscale_.secTh[0],        &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = azgh_showscale_.nth+1;
  rcode += dst_packr8_(&azgh_showscale_.thEdge[0],       &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = 1;
  rcode += dst_packr8_(&azgh_showscale_.nMaxScaleFactor, &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = azgh_showscale_.nth;
  rcode += dst_packr8_(&azgh_showscale_.nMaxLESlope[0],  &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  rcode += dst_packr8_(&azgh_showscale_.xMaxLESlope[0],  &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  rcode += dst_packr8_(&azgh_showscale_.x0LESlope[0],    &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);

  return rcode;
}

integer4 azgh_showscale_bank_to_dst_ (integer4 *unit) {
  return dst_write_bank_(unit, &azgh_showscale_blen, azgh_showscale_bank);
}

integer4 azgh_showscale_common_to_dst_(integer4 *unit) {
  integer4 rcode;
    if ( (rcode = azgh_showscale_common_to_bank_()) ) {
      fprintf(stderr, "azgh_showscale_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
    if ( (rcode = azgh_showscale_bank_to_dst_(unit)) ) {
      fprintf(stderr, "azgh_showscale_bank_to_dst_ ERROR : %ld\n", (long)rcode);           
      exit(0);
  }
  return 0;
}

integer4 azgh_showscale_bank_to_common_(integer1 *azgh_showscale_bank) {
  integer4 rcode = 0 ;
  integer4 nobj;
  azgh_showscale_blen = 2 * sizeof(integer4);   /* skip id and version  */

  nobj = 1;
  rcode += dst_unpacki4_(&azgh_showscale_.nlE,             &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  rcode += dst_unpacki4_(&azgh_showscale_.nth,             &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = azgh_showscale_.nlE+1;
  rcode += dst_unpackr8_(&azgh_showscale_.lEEdge[0],       &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = azgh_showscale_.nth;
  rcode += dst_unpackr8_(&azgh_showscale_.secTh[0],        &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = azgh_showscale_.nth+1;
  rcode += dst_unpackr8_(&azgh_showscale_.thEdge[0],       &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = 1;
  rcode += dst_unpackr8_(&azgh_showscale_.nMaxScaleFactor, &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  nobj = azgh_showscale_.nth;
  rcode += dst_unpackr8_(&azgh_showscale_.nMaxLESlope[0],  &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  rcode += dst_unpackr8_(&azgh_showscale_.xMaxLESlope[0],  &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);
  rcode += dst_unpackr8_(&azgh_showscale_.x0LESlope[0],    &nobj, azgh_showscale_bank, &azgh_showscale_blen, &azgh_showscale_maxlen);

  return rcode;
}

integer4 azgh_showscale_common_to_dump_(integer4 *long_output) {
  return azgh_showscale_common_to_dumpf_(stdout, long_output);       
}

integer4 azgh_showscale_common_to_dumpf_(FILE* fp, integer4 *long_output) {
  int i,nle,nth;
  (void)(long_output);
  fprintf (fp, "AZ Shower Library:\n");

  nle = azgh_showscale_.nlE;
  fprintf (fp, "  %2d bins in lE: ",nle);
  for (i=0;i<nle+1;i++)
    fprintf(fp, "%4.1f ",azgh_showscale_.lEEdge[i]);
  fprintf (fp, "\n");

  nth = azgh_showscale_.nth;
  fprintf (fp, "  %2d bins in th: ",nth);
  for (i=0;i<nth;i++)
    fprintf(fp, "%5.3f ",azgh_showscale_.thEdge[i]);
  fprintf(fp, " %7.1e",azgh_showscale_.thEdge[nth]);
  fprintf (fp, "\n");

  fprintf (fp, "  nMax scale (correction) factor: %5.3f\n",azgh_showscale_.nMaxScaleFactor);

  fprintf (fp, "  lE slopes for scaling to given energies:\n");

  fprintf (fp, "      sec(th) ");
  for (i=0;i<nth;i++)
    fprintf(fp, "    %4.2f ",azgh_showscale_.secTh[i]);
  fprintf (fp, "\n");

  fprintf (fp, "      nMax    ");
  for (i=0;i<nth;i++)
    fprintf(fp, "  %7.3f",azgh_showscale_.nMaxLESlope[i]);
  fprintf (fp, "\n");

  fprintf (fp, "      xMax    ");
  for (i=0;i<nth;i++)
    fprintf(fp, "  %7.3f",azgh_showscale_.xMaxLESlope[i]);
  fprintf (fp, "\n");

  fprintf (fp, "      x0      ");
  for (i=0;i<nth;i++)
    fprintf(fp, "  %7.3f",azgh_showscale_.x0LESlope[i]);
  fprintf (fp, "\n");

  return SUCCESS;
}

// Various shower library utility functions

static real8 correctNMax(real8 nMax, real8 scale) {return scale*nMax;}
static real8 scaleNMax(real8 nMax, real8 lE0, real8 lE, real8 slope) {return nMax * exp(slope*(lE-lE0));}
static real8 scaleXMax(real8 xMax, real8 lE0, real8 lE, real8 slope) {return xMax +     slope*(lE-lE0); }
static real8 scaleX0  (real8 x0,   real8 lE0, real8 lE, real8 slope) {return x0   +     slope*(lE-lE0); }

static integer4 thBin(azgh_showscale_dst_common* library, real8 theta) {
  int i;
  double secTheta = 1./cos(theta);
  for (i=0;i<library->nth;i++)
    if ( (secTheta>=library->thEdge[i]) && (secTheta<library->thEdge[i+1]) )
      return i;
  fprintf(stderr,"azgh_showscale_dst::thBin: No theta bin found, secTheta %5.3f, bin edges",secTheta);
  for (i=0;i<library->nth+1;i++)
    fprintf(stderr," %5.3f",library->thEdge[i]);
  fprintf(stderr,"\n");
  return -1;
}
  
real8 azgh_showscale_getScaledNMax(azgh_showscale_dst_common* library, 
				   azgh_showlib_dst_common* entry,
				   real8 lE) {
  real8 nmax = correctNMax(entry->nmax,library->nMaxScaleFactor);
  real8 lE0 = log10(entry->energy)+9.;
  integer4 ith = thBin(library,entry->angle);
  real8 slope = library->nMaxLESlope[ith];
  return scaleNMax(nmax,lE0,lE,slope);
}

real8 azgh_showscale_getScaledXMax(azgh_showscale_dst_common* library, 
				   azgh_showlib_dst_common* entry,
				   real8 lE) {
  real8 lE0 = log10(entry->energy)+9.;
  integer4 ith = thBin(library,entry->angle);
  real8 slope = library->xMaxLESlope[ith];
  return scaleXMax(entry->xmax,lE0,lE,slope);
}

real8 azgh_showscale_getScaledX0(azgh_showscale_dst_common* library, 
				 azgh_showlib_dst_common* entry,
				 real8 lE) {
  real8 lE0 = log10(entry->energy)+9.;
  integer4 ith = thBin(library,entry->angle);
  real8 slope = library->x0LESlope[ith];
  return scaleX0(entry->x0,lE0,lE,slope);
}

real8 azgh_showscale_getScaledNMax_(real8 lE) {
  return azgh_showscale_getScaledNMax(&azgh_showscale_, &azgh_showlib_, lE); 
}
real8 azgh_showscale_getScaledXMax_(real8 lE) {
  return azgh_showscale_getScaledXMax(&azgh_showscale_, &azgh_showlib_, lE); 
}
real8 azgh_showscale_getScaledX0_(real8 lE) {
  return azgh_showscale_getScaledX0(&azgh_showscale_, &azgh_showlib_, lE); 
}

azgh_showlib_dst_common* azgh_showscale_scaledClone(azgh_showscale_dst_common* library, 
						    azgh_showlib_dst_common* entry,
						    real8 lE) {
  azgh_showlib_dst_common* newEntry;
  newEntry = (azgh_showlib_dst_common*) calloc(1, sizeof(azgh_showlib_dst_common));
  newEntry->code     = entry->code;
  newEntry->number   = entry->number;
  newEntry->angle    = entry->angle;
  newEntry->particle = entry->particle;
  newEntry->energy   = entry->energy;
  newEntry->first    = entry->first;
  newEntry->nmax     = entry->nmax;
  newEntry->x0       = entry->x0;
  newEntry->xmax     = entry->xmax;
  newEntry->lambda   = entry->lambda;
  newEntry->chi2     = entry->chi2;
  azgh_showscale_scale(library, newEntry, lE);
  return newEntry;
}

void azgh_showscale_scale(azgh_showscale_dst_common* library, 
			  azgh_showlib_dst_common* entry,
			  real8 lE) {
  real8 nmax = azgh_showscale_getScaledNMax(library,entry,lE);
  real8 x0   = azgh_showscale_getScaledX0(library,entry,lE);
  real8 xmax = azgh_showscale_getScaledXMax(library,entry,lE);
  entry->energy = pow(10.,lE-9.);
  entry->nmax = nmax;
  entry->x0   = x0;
  entry->xmax = xmax;
}

void azgh_showscale_scale_(real8 lE) {azgh_showscale_scale(&azgh_showscale_,&azgh_showlib_,lE);}

real8 azgh_showscale_gh_(real8 nmax, real8 xmax, real8 x0, real8 lambda, real8 x) {
  real8 lf1 = (xmax-x0)/lambda * log((x-x0)/(xmax-x0));
  real8 lf2 = (xmax-x)/lambda;
  return nmax * exp( lf1 + lf2 );
}

real8 azgh_showscale_getGHNe(azgh_showlib_dst_common* entry, real8 x) {
  real8 nmax   = entry->nmax;
  real8 x0     = entry->x0;
  real8 xmax   = entry->xmax;
  real8 lambda = entry->lambda;
  return azgh_showscale_gh_(nmax,xmax,x0,lambda,x);
}

real8 azgh_showscale_getghne_(real8 x) {return azgh_showscale_getGHNe(&azgh_showlib_, x);}

real8 azgh_showscale_getScaledGHNe(azgh_showscale_dst_common* library,
				   azgh_showlib_dst_common* entry, real8 lE, real8 x) {
  real8 nmax   = azgh_showscale_getScaledNMax(library,entry,lE);
  real8 x0     = azgh_showscale_getScaledX0(library,entry,lE);
  real8 xmax   = azgh_showscale_getScaledXMax(library,entry,lE);
  real8 lambda = entry->lambda;
  return azgh_showscale_gh_(nmax,xmax,x0,lambda,x);
}
