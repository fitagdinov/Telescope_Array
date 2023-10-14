/*
 * hsum4_dst.c 
 *
 * $Source: $
 * $Log: $
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "hsum4_dst.h"  

hsum4_dst_common hsum4_;  /* allocate memory to hsum4_common */

static integer4 hsum4_blen = 0; 
static integer4 hsum4_maxlen = sizeof(integer4) * 2 + sizeof(hsum4_dst_common);
static integer1 *hsum4_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hsum4_bank_buffer_ (integer4* hsum4_bank_buffer_size)
{
  (*hsum4_bank_buffer_size) = hsum4_blen;
  return hsum4_bank;
}



static void hsum4_bank_init(void) {

  hsum4_bank = (integer1 *)calloc(hsum4_maxlen, sizeof(integer1));

  if(hsum4_bank==NULL) {
    fprintf(stderr,"hsum4_bank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

integer4 hsum4_common_to_bank_(void) {
	
  static integer4 id  = HSUM4_BANKID;
  static integer4 ver = HSUM4_BANKVERSION;

  integer4 rcode;
  integer4 nobj;

  if (hsum4_bank == NULL) hsum4_bank_init();

  /* Initialize hsum4_blen, and pack the id and version to bank */
  
  if((rcode = dst_initbank_(&id, &ver, &hsum4_blen, &hsum4_maxlen, hsum4_bank)))
    return rcode;

  // reals
  if ((rcode = dst_packr8_( &hsum4_.chi2_com, (nobj=21, &nobj),
			    hsum4_bank, &hsum4_blen, &hsum4_maxlen)))
    return rcode;
  
  // integers
  if ((rcode = dst_packi4_( &hsum4_.imin_com, (nobj=4, &nobj),
			    hsum4_bank, &hsum4_blen, &hsum4_maxlen)))
    return rcode;
  
  return SUCCESS;
}


integer4 hsum4_bank_to_dst_(integer4 *NumUnit) {
	
  return dst_write_bank_(NumUnit, &hsum4_blen, hsum4_bank );
}

integer4 hsum4_common_to_dst_(integer4 *NumUnit) {

  integer4 rcode;

  if((rcode = hsum4_common_to_bank_())) {
    fprintf (stderr,"hsum4_common_to_bank_ ERROR : %ld\n", (long) rcode);
    exit(0);			 	
  }             
  
  if((rcode = hsum4_bank_to_dst_(NumUnit))) {
    fprintf (stderr,"hsum4_bank_to_dst_ ERROR : %ld\n", (long) rcode);
    exit(0);
  }
  
  return SUCCESS;
}

integer4 hsum4_bank_to_common_(integer1 *bank) {

  integer4 rcode = 0;
  integer4 nobj;

  hsum4_blen = 2 * sizeof(integer4); /* skip id and version  */

  // reals
  if ((rcode = dst_unpackr8_( &hsum4_.chi2_com, (nobj=21, &nobj),
			      bank, &hsum4_blen, &hsum4_maxlen)))
    return rcode;
  
  // integers
  if ((rcode = dst_unpacki4_( &hsum4_.imin_com, (nobj=4, &nobj),
			      bank, &hsum4_blen, &hsum4_maxlen )))
    return rcode;
  
  return SUCCESS;
}

integer4 hsum4_common_to_dump_(integer4 *long_output) {

  return hsum4_common_to_dumpf_(stdout, long_output);
}

integer4 hsum4_common_to_dumpf_(FILE* fp, integer4 *long_output) {
  (void)(long_output);
  fprintf( fp, "\nHSUM4 bank ________________________\n");
  fprintf( fp, "failmode = %d\n", hsum4_.failmode );

  if ( hsum4_.failmode != SUCCESS ) return SUCCESS;

  fprintf( fp, "Chi2 (com)       : %10.2f\n", hsum4_.chi2_com );
  fprintf( fp, "Chi2 (pfl)       : %10.2f\n", hsum4_.chi2_pfl );
  fprintf( fp, "Chi2 (tim)       : %10.2f\n", hsum4_.chi2_tim );

  fprintf( fp, "Julian Day       : %13.6f\n", hsum4_.jd );
  fprintf( fp, "energy           : %10.2f EeV\n", hsum4_.energy );
  fprintf( fp, "Nmax             : %20.2g\n", hsum4_.Nmax );
  fprintf( fp, "x0               : %10.2f g/cm^2\n", hsum4_.x0 );
  fprintf( fp, "xmax             : %10.2f g/cm^2\n", hsum4_.xmax );

  fprintf( fp, "xfirst           : %10.2f g/cm^2\n", hsum4_.xfirst );
  fprintf( fp, "xlast            : %10.2f g/cm^2\n", hsum4_.xlast );

  fprintf( fp, "rp               : %10.2f km\n", 1e-3*hsum4_.rp );
  fprintf( fp, "psi              : %10.2f deg\n", hsum4_.psi );
  fprintf( fp, "theta            : %10.2f deg\n", hsum4_.theta );
  fprintf( fp, "Declination      : %10.2f deg\n", hsum4_.dec );
  fprintf( fp, "Right Ascension  : %10.2f deg\n", hsum4_.ra );
  fprintf( fp, "Origin Vector:   : (%6.5f, %6.5f, %6.5f)\n", hsum4_.shower[0],
	   hsum4_.shower[1], hsum4_.shower[2]);
  fprintf( fp, "Plane Norm:      : (%6.5f, %6.5f, %6.5f)\n", hsum4_.plnnorm[0],
	   hsum4_.plnnorm[1], hsum4_.plnnorm[2]);


  fprintf( fp, "imin (com)       : %2d\n", hsum4_.imin_com );
  fprintf( fp, "imin (pfl)       : %2d\n", hsum4_.imin_pfl );
  fprintf( fp, "imin (tim)       : %2d\n", hsum4_.imin_tim );

  return SUCCESS;
}





