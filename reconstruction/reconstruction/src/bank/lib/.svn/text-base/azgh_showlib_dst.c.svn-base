// DST bank implementation for shower library entries.
// Based on shower library entries originally created by Andreas Zech for HiRes.
// The Gaisser-Hillas parameterization of a hadronic shower is
//    n(x) = nmax * ((x-x0)/(xmax-x0))^((xmax-x0)/lambda) * exp((xmax-x)/lambda)
//
// asgh_showlib_dst.c DRB - 2008/09/18

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "azgh_showlib_dst.h"

azgh_showlib_dst_common azgh_showlib_;

static integer4 azgh_showlib_blen = 0;
static integer4 azgh_showlib_maxlen = sizeof(integer4) * 2 + sizeof(azgh_showlib_dst_common);
static integer1 *azgh_showlib_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* azgh_showlib_bank_buffer_ (integer4* azgh_showlib_bank_buffer_size)
{
  (*azgh_showlib_bank_buffer_size) = azgh_showlib_blen;
  return azgh_showlib_bank;
}



static void azgh_showlib_bank_init() {
  azgh_showlib_bank = (integer1 *)calloc(azgh_showlib_maxlen, sizeof(integer1));
  if (azgh_showlib_bank==NULL) {
      fprintf (stderr,"azgh_showlib_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 azgh_showlib_common_to_bank_() {
  static integer4 id = AZGH_SHOWLIB_BANKID, ver = AZGH_SHOWLIB_BANKVERSION;
  integer4 rcode, nobj;

  if (azgh_showlib_bank == NULL) azgh_showlib_bank_init();

  rcode = dst_initbank_(&id, &ver, &azgh_showlib_blen, &azgh_showlib_maxlen, azgh_showlib_bank);

// Initialize azgh_showlib_blen and pack the id and version to bank

  nobj = 1;
  rcode += dst_packi4_(&azgh_showlib_.code,     &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packi4_(&azgh_showlib_.number,   &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packr4_(&azgh_showlib_.angle,    &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packi4_(&azgh_showlib_.particle, &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packr4_(&azgh_showlib_.energy,   &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packr4_(&azgh_showlib_.first,    &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packr4_(&azgh_showlib_.nmax,     &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packr4_(&azgh_showlib_.x0,       &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packr4_(&azgh_showlib_.xmax,     &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packr4_(&azgh_showlib_.lambda,   &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_packr4_(&azgh_showlib_.chi2,     &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);

  return rcode;
}

integer4 azgh_showlib_bank_to_dst_ (integer4 *unit) {
  return dst_write_bank_(unit, &azgh_showlib_blen, azgh_showlib_bank);
}

integer4 azgh_showlib_common_to_dst_(integer4 *unit) {
  integer4 rcode;
    if ( (rcode = azgh_showlib_common_to_bank_()) ) {
      fprintf(stderr, "azgh_showlib_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
    if ( (rcode = azgh_showlib_bank_to_dst_(unit)) ) {
      fprintf(stderr, "azgh_showlib_bank_to_dst_ ERROR : %ld\n", (long)rcode);           
      exit(0);
  }
  return 0;
}

integer4 azgh_showlib_bank_to_common_(integer1 *azgh_showlib_bank) {
  integer4 rcode = 0 ;
  integer4 nobj;
  azgh_showlib_blen = 2 * sizeof(integer4);   /* skip id and version  */

  nobj = 1;
  rcode += dst_unpacki4_(&azgh_showlib_.code,     &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpacki4_(&azgh_showlib_.number,   &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpackr4_(&azgh_showlib_.angle,    &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpacki4_(&azgh_showlib_.particle, &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpackr4_(&azgh_showlib_.energy,   &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpackr4_(&azgh_showlib_.first,    &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpackr4_(&azgh_showlib_.nmax,     &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpackr4_(&azgh_showlib_.x0,       &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpackr4_(&azgh_showlib_.xmax,     &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpackr4_(&azgh_showlib_.lambda,   &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);
  rcode += dst_unpackr4_(&azgh_showlib_.chi2,     &nobj, azgh_showlib_bank, &azgh_showlib_blen, &azgh_showlib_maxlen);

  return rcode;
}

integer4 azgh_showlib_common_to_dump_(integer4 *long_output) {
  return azgh_showlib_common_to_dumpf_(stdout, long_output);       
}

integer4 azgh_showlib_common_to_dumpf_(FILE* fp, integer4 *long_output) {
  char particle[0x10];
  char model[0x10];
  (void)(long_output);
  switch (azgh_showlib_.particle) {
  case 14:
    sprintf(particle,"prot");
    break;
  case 5626:
    sprintf(particle,"iron");
    break;
  default:
    sprintf(particle,"%4d",azgh_showlib_.particle);
    break;
  }

  switch (azgh_showlib_.code%10) {
  case 0:
    sprintf(model," QGSJet 01");
    break;
  case 1:
    sprintf(model,"SIBYLL 1.6");
    break;
  case 2:
    sprintf(model,"SIBYLL 2.1");
    break;
  default:
    sprintf(model,"   MODEL %d",azgh_showlib_.code%10);
    break;
  }
    
  fprintf (fp, "AZShower %6d %3d: %4s %9s %2.0f\n", 
	   azgh_showlib_.code,azgh_showlib_.number,particle,model,57.296*azgh_showlib_.angle);
  fprintf (fp, "         energy: %6.2f EeV, first int: %6.1f g/cm2\n",
	   azgh_showlib_.energy/1e9,azgh_showlib_.first);
  fprintf (fp, "         nMx: %10.3e x0: %6.1f g/cm2 xMx: %6.1f g/cm2 lam: %4.1f g/cm2\n",
	   azgh_showlib_.nmax,azgh_showlib_.x0,azgh_showlib_.xmax,azgh_showlib_.lambda);
  
  return SUCCESS;
}
