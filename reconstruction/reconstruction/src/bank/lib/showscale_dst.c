// DSTbank definition for scaling between different energies at different
// angles in a shower library.
// It is envisioned that one showscale DST bank will be at the beginning
// of a DST file, followed by ALL the individual shower entries in ALL drawers
//
// showscale_dst.c: SS - 2009/01/07

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "showscale_dst.h"
#include "showlib_dst.h"

showscale_dst_common showscale_;

static integer4 showscale_blen = 0;
static integer4 showscale_maxlen = sizeof(integer4) * 2 + sizeof(showscale_dst_common);
static integer1 *showscale_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* showscale_bank_buffer_ (integer4* showscale_bank_buffer_size)
{
  (*showscale_bank_buffer_size) = showscale_blen;
  return showscale_bank;
}



static void showscale_bank_init() {
  showscale_bank = (integer1 *)calloc(showscale_maxlen, sizeof(integer1));
  if (showscale_bank==NULL) {
      fprintf (stderr,"showscale_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 showscale_common_to_bank_() {
  static integer4 id = SHOWSCALE_BANKID, ver = SHOWSCALE_BANKVERSION;
  integer4 i, rcode, nobj;

  if (showscale_bank == NULL) showscale_bank_init();

  rcode = dst_initbank_(&id, &ver, &showscale_blen, &showscale_maxlen, showscale_bank);

// Initialize showscale_blen and pack the id and version to bank

  nobj = 1;
  rcode += dst_packi4_(&showscale_.nlE,             &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  rcode += dst_packi4_(&showscale_.nth,             &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = showscale_.nlE+1;
  rcode += dst_packr8_(&showscale_.lEEdge[0],       &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = showscale_.nth;
  rcode += dst_packr8_(&showscale_.secTh[0],        &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = showscale_.nth+1;
  rcode += dst_packr8_(&showscale_.thEdge[0],       &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = 1;
  rcode += dst_packr8_(&showscale_.nMaxScaleFactor, &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = showscale_.nth;
  rcode += dst_packr8_(&showscale_.nMaxLESlope[0],  &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  rcode += dst_packr8_(&showscale_.xMaxLESlope[0],  &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  rcode += dst_packr8_(&showscale_.x0LESlope[0],    &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);

  nobj = showscale_.nth;
  for ( i=0; i<showscale_.nlE; i++ )
    rcode += dst_packi4_(&showscale_.numEntries[i][0], &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);

  return rcode;
}

integer4 showscale_bank_to_dst_ (integer4 *unit) {
  return dst_write_bank_(unit, &showscale_blen, showscale_bank);
}

integer4 showscale_common_to_dst_(integer4 *unit) {
  integer4 rcode;
    if ( (rcode = showscale_common_to_bank_()) ) {
      fprintf(stderr, "showscale_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
    if ( (rcode = showscale_bank_to_dst_(unit)) ) {
      fprintf(stderr, "showscale_bank_to_dst_ ERROR : %ld\n", (long)rcode);           
      exit(0);
  }
  return 0;
}

integer4 showscale_bank_to_common_(integer1 *showscale_bank) {
  integer4 rcode = 0 ;
  integer4 i, j, nobj, id, ver;
  // showscale_blen = 2 * sizeof(integer4);   /* skip id and version  */

  /* check ID and version */
  nobj = 1;
  showscale_blen = 0;
  rcode += dst_unpacki4_(&id, &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  rcode += dst_unpacki4_(&ver, &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);

  nobj = 1;
  rcode += dst_unpacki4_(&showscale_.nlE,             &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  rcode += dst_unpacki4_(&showscale_.nth,             &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = showscale_.nlE+1;
  rcode += dst_unpackr8_(&showscale_.lEEdge[0],       &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = showscale_.nth;
  rcode += dst_unpackr8_(&showscale_.secTh[0],        &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = showscale_.nth+1;
  rcode += dst_unpackr8_(&showscale_.thEdge[0],       &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = 1;
  rcode += dst_unpackr8_(&showscale_.nMaxScaleFactor, &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  nobj = showscale_.nth;
  rcode += dst_unpackr8_(&showscale_.nMaxLESlope[0],  &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  rcode += dst_unpackr8_(&showscale_.xMaxLESlope[0],  &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  rcode += dst_unpackr8_(&showscale_.x0LESlope[0],    &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);

  if ( ver >= 1 ) {
    nobj = showscale_.nth;
    for ( i=0; i<showscale_.nlE; i++ )
      rcode += dst_unpacki4_(&showscale_.numEntries[i][0], &nobj, showscale_bank, &showscale_blen, &showscale_maxlen);
  }
  else { // this is the original TRUMP shower library
    for ( i=0; i<showscale_.nlE; i++ )
      for ( j=0; j<showscale_.nth; j++ )
        showscale_.numEntries[i][j] = 500;
  }
  return rcode;
}

integer4 showscale_common_to_dump_(integer4 *long_output) {
  return showscale_common_to_dumpf_(stdout, long_output);       
}

integer4 showscale_common_to_dumpf_(FILE* fp, integer4 *long_output) {
  int i,j, nle,nth;
  (void)(long_output);
  fprintf (fp, "AZ Shower Library:\n");

  nle = showscale_.nlE;
  fprintf (fp, "  %2d bins in lE: ",nle);
  for (i=0;i<nle+1;i++)
    fprintf(fp, "%4.1f ",showscale_.lEEdge[i]);
  fprintf (fp, "\n");

  nth = showscale_.nth;
  fprintf (fp, "  %2d bins in th: ",nth);
  for (i=0;i<nth;i++)
    fprintf(fp, "%5.3f ",showscale_.thEdge[i]);
  fprintf(fp, " %7.1e",showscale_.thEdge[nth]);
  fprintf (fp, "\n");

  fprintf (fp, "  nMax scale (correction) factor: %5.3f\n",showscale_.nMaxScaleFactor);

  fprintf (fp, "  lE slopes for scaling to given energies:\n");

  fprintf (fp, "      sec(th) ");
  for (i=0;i<nth;i++)
    fprintf(fp, "    %4.2f ",showscale_.secTh[i]);
  fprintf (fp, "\n");

  fprintf (fp, "      nMax    ");
  for (i=0;i<nth;i++)
    fprintf(fp, "  %7.3f",showscale_.nMaxLESlope[i]);
  fprintf (fp, "\n");

  fprintf (fp, "      xMax    ");
  for (i=0;i<nth;i++)
    fprintf(fp, "  %7.3f",showscale_.xMaxLESlope[i]);
  fprintf (fp, "\n");

  fprintf (fp, "      x0      ");
  for (i=0;i<nth;i++)
    fprintf(fp, "  %7.3f",showscale_.x0LESlope[i]);
  fprintf (fp, "\n");

  fprintf(fp, "Number of drawer entries:\n");
  for ( i=0; i<showscale_.nlE; i++ ) {
    for ( j=0; j<showscale_.nth; j++ ) 
      fprintf(fp, "  %4d", showscale_.numEntries[i][j]);
    fprintf(fp, "\n");
  }

  return SUCCESS;
}

// Various shower library utility functions

// static real8 correctNMax(real8 nMax, real8 scale) {return scale*nMax;}
static real8 scaleNMax(real8 nMax, real8 lE0, real8 lE, real8 slope) {return nMax * exp(slope*(lE-lE0));}
static real8 scaleXMax(real8 xMax, real8 lE0, real8 lE, real8 slope) {return xMax +     slope*(lE-lE0); }
static real8 scaleX0  (real8 x0,   real8 lE0, real8 lE, real8 slope) {return x0   +     slope*(lE-lE0); }

static integer4 thBin(showscale_dst_common* library, real8 theta) {
  int i;
  double secq = 1.0 / cos(theta);
  for (i=0;i<library->nth;i++)
    if ( (secq>=library->thEdge[i]) && (secq<library->thEdge[i+1]) )
      return i;
  return -1;
}
  
real8 showscale_getScaledNMax(showscale_dst_common* library, 
				   showlib_dst_common* entry,
				   real8 lE) {
  // real8 nmax = correctNMax(entry->nmax,library->nMaxScaleFactor);
  real8 nmax = entry->nmax;  // do not automatically scale Nmax (SS - 2 Mar 2010)
  real8 lE0 = log10(entry->energy)+9.;
  integer4 ith = thBin(library,entry->angle);
  real8 slope = library->nMaxLESlope[ith];
  return scaleNMax(nmax,lE0,lE,slope);
}

real8 showscale_getScaledXMax(showscale_dst_common* library, 
				   showlib_dst_common* entry,
				   real8 lE) {
  real8 lE0 = log10(entry->energy)+9.;
  integer4 ith = thBin(library,entry->angle);
  real8 slope = library->xMaxLESlope[ith];
  return scaleXMax(entry->xmax,lE0,lE,slope);
}

real8 showscale_getScaledX0(showscale_dst_common* library, 
				 showlib_dst_common* entry,
				 real8 lE) {
  real8 lE0 = log10(entry->energy)+9.;
  integer4 ith = thBin(library,entry->angle);
  real8 slope = library->x0LESlope[ith];
  return scaleX0(entry->x0,lE0,lE,slope);
}

real8 showscale_getScaledNMax_(real8 lE) {
  return showscale_getScaledNMax(&showscale_, &showlib_, lE); 
}
real8 showscale_getScaledXMax_(real8 lE) {
  return showscale_getScaledXMax(&showscale_, &showlib_, lE); 
}
real8 showscale_getScaledX0_(real8 lE) {
  return showscale_getScaledX0(&showscale_, &showlib_, lE); 
}

showlib_dst_common* showscale_scaledClone(showscale_dst_common* library, 
						    showlib_dst_common* entry,
						    real8 lE) {
  showlib_dst_common* newEntry;
  newEntry = (showlib_dst_common*) calloc(1, sizeof(showlib_dst_common));
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
  showscale_scale(library, newEntry, lE);
  return newEntry;
}

void showscale_scale(showscale_dst_common* library, 
			  showlib_dst_common* entry,
			  real8 lE) {
  real8 nmax = showscale_getScaledNMax(library,entry,lE);
  real8 x0   = showscale_getScaledX0(library,entry,lE);
  real8 xmax = showscale_getScaledXMax(library,entry,lE);
  entry->energy = pow(10.,lE-9.);
  entry->nmax = nmax;
  entry->x0   = x0;
  entry->xmax = xmax;
}

void showscale_scale_(real8 lE) {showscale_scale(&showscale_,&showlib_,lE);}

real8 showscale_gh_(real8 nmax, real8 xmax, real8 x0, real8 lambda, real8 x) {
  real8 lf1 = (xmax-x0)/lambda * log((x-x0)/(xmax-x0));
  real8 lf2 = (xmax-x)/lambda;
  return nmax * exp( lf1 + lf2 );
}

real8 showscale_getGHNe(showlib_dst_common* entry, real8 x) {
  real8 nmax   = entry->nmax;
  real8 x0     = entry->x0;
  real8 xmax   = entry->xmax;
  real8 lambda = entry->lambda;
  return showscale_gh_(nmax,xmax,x0,lambda,x);
}

real8 showscale_getghne_(real8 x) {return showscale_getGHNe(&showlib_, x);}

real8 showscale_getScaledGHNe(showscale_dst_common* library,
				   showlib_dst_common* entry, real8 lE, real8 x) {
  real8 nmax   = showscale_getScaledNMax(library,entry,lE);
  real8 x0     = showscale_getScaledX0(library,entry,lE);
  real8 xmax   = showscale_getScaledXMax(library,entry,lE);
  real8 lambda = entry->lambda;
  return showscale_gh_(nmax,xmax,x0,lambda,x);
}

// new 20150124, to scale shower width instead of X0 (T. A. Stroman)
#define LN256 5.54517744447956229
#define SQRTLN256 2.35482004503094933
real8 getWidthFromLambdaX0(showlib_dst_common *entry) {
  return sqrt(entry->lambda*(entry->xmax-entry->x0)*LN256); // use original Xmax, not scaled
}
real8 getRatioFromLambdaX0(showlib_dst_common *entry) {
  return sqrt(entry->lambda/(entry->xmax-entry->x0));
}



real8 scaleWidth(real8 width, real8 lE, real8 lE0, real8 slope) {
  return width + (lE-lE0)*slope;
}

real8 showscale_getScaledWidth(showscale_dst_common *library,
                               showlib_dst_common *entry,
                               real8 lE) {
  real8 width = getWidthFromLambdaX0(entry);
  
  real8 lE0 = log10(entry->energy)+9.;
  integer4 ith = thBin(library,entry->angle);
  real8 slope = library->x0LESlope[ith]; // if nMaxScaleFactor < -1, this value mean width.
  
  return scaleWidth(width,lE,lE0,slope);
}

real8 getLambdaFromScaled(real8 width, real8 ratio) {
  return width*ratio/SQRTLN256;
}
real8 getX0FromScaled(real8 width, real8 ratio, real8 xmax) {
  return xmax - width/ratio/SQRTLN256;
}
