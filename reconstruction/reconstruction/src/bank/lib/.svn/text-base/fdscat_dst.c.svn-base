/*
 * C functions for fdscat
 * MRM July 27
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fdscat_dst.h"  

fdscat_dst_common fdscat_;  /* allocate memory to fdscat_common */

static integer4 fdscat_blen = 0; 
static integer4 fdscat_maxlen = sizeof(integer4) * 2 + sizeof(fdscat_dst_common);
static integer1 *fdscat_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdscat_bank_buffer_ (integer4* fdscat_bank_buffer_size)
{
  (*fdscat_bank_buffer_size) = fdscat_blen;
  return fdscat_bank;
}



static integer4 fdscat_ver = FDSCAT_BANKVERSION;

static void fdscat_bank_init() {
  fdscat_bank = (integer1 *)calloc(fdscat_maxlen, sizeof(integer1));
  if (fdscat_bank==NULL) {
    fprintf (stderr,"fdscat_bank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}    

integer4 fdscat_common_to_bank_() {
  static integer4 id = FDSCAT_BANKID;
  integer4 rcode, nobj;

  fdscat_ver = FDSCAT_BANKVERSION;

  if (fdscat_bank == NULL) fdscat_bank_init();
    
  rcode = dst_initbank_(&id, &fdscat_ver, &fdscat_blen, &fdscat_maxlen, fdscat_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj=1;
  rcode += dst_packr8_(&fdscat_.startTime, &nobj, fdscat_bank, &fdscat_blen, &fdscat_maxlen);
  rcode += dst_packr8_(&fdscat_.endTime, &nobj, fdscat_bank, &fdscat_blen, &fdscat_maxlen);
  rcode += dst_packr8_(&fdscat_.hzalen, &nobj, fdscat_bank, &fdscat_blen, &fdscat_maxlen);
  rcode += dst_packr8_(&fdscat_.vaodep, &nobj, fdscat_bank, &fdscat_blen, &fdscat_maxlen);
  rcode += dst_packr8_(&fdscat_.schght, &nobj, fdscat_bank, &fdscat_blen, &fdscat_maxlen);

  if (fdscat_ver >= 1)
  {
    rcode += dst_packi4_(&fdscat_.uniqID,     &nobj, fdscat_bank, &fdscat_blen,
			 &fdscat_maxlen);
    rcode += dst_packi4_(&fdscat_.vaodID,     &nobj, fdscat_bank, &fdscat_blen,
			 &fdscat_maxlen);
    rcode += dst_packi4_(&fdscat_.vaodDate,   &nobj, fdscat_bank, &fdscat_blen,
			 &fdscat_maxlen);
    rcode += dst_packr8_(&fdscat_.vaodep_rms, &nobj, fdscat_bank, &fdscat_blen,
			 &fdscat_maxlen);
    rcode += dst_packi4_(&fdscat_.qual,       &nobj, fdscat_bank, &fdscat_blen,
			 &fdscat_maxlen);
    rcode += dst_packi4_(&fdscat_.method,     &nobj, fdscat_bank, &fdscat_blen,
			 &fdscat_maxlen);
    rcode += dst_packi4_(&fdscat_.nMeas,      &nobj, fdscat_bank, &fdscat_blen,
			 &fdscat_maxlen);
    nobj = fdscat_.nMeas;
    if (nobj < 0 || nobj >= FDSCAT_MAX_MEAS)
    {
      fprintf(stderr, "%s: %s: fdscat_.nMeas: illegal value (%d)\n",
	      __FILE__, __func__, nobj);
      exit(EXIT_FAILURE);
    }
    rcode += dst_packr8_(&fdscat_.hzalen_meas[0], &nobj, fdscat_bank,
			 &fdscat_blen, &fdscat_maxlen);
    rcode += dst_packr8_(&fdscat_.vaodep_meas[0], &nobj, fdscat_bank,
			 &fdscat_blen, &fdscat_maxlen);
    rcode += dst_packr8_(&fdscat_.vaodep_rms_meas[0], &nobj, fdscat_bank,
			 &fdscat_blen, &fdscat_maxlen);
    rcode += dst_packr8_(&fdscat_.schght_meas[0], &nobj, fdscat_bank,
			 &fdscat_blen, &fdscat_maxlen);
    rcode += dst_packi4_(&fdscat_.site_meas[0], &nobj, fdscat_bank,
			 &fdscat_blen, &fdscat_maxlen);
    rcode += dst_packi4_(&fdscat_.qual_meas[0], &nobj, fdscat_bank,
			 &fdscat_blen, &fdscat_maxlen);
  }
  
  return rcode ;
}

integer4 fdscat_bank_to_dst_ (integer4 *unit) {
  return dst_write_bank_(unit, &fdscat_blen, fdscat_bank);
}

integer4 fdscat_common_to_dst_(integer4 *unit) {
  integer4 rcode;
  if ( (rcode = fdscat_common_to_bank_()) ) {
    fprintf(stderr, "fdscat_common_to_bank_ ERROR : %ld\n", (long)rcode);
    exit(0);
  }
  if ( (rcode = fdscat_bank_to_dst_(unit) )) {
    fprintf(stderr, "fdscat_bank_to_dst_ ERROR : %ld\n", (long)rcode);
    exit(0);
  }

  return 0;
}

integer4 fdscat_bank_to_common_(integer1 *bank) {
  integer4 rcode = 0 ;
  integer4 nobj;
  fdscat_blen = 1 * sizeof(integer4);	/* skip id  */

  nobj=1;
  rcode += dst_unpacki4_(&fdscat_ver, &nobj, bank, &fdscat_blen, &fdscat_maxlen);
  rcode += dst_unpackr8_(&fdscat_.startTime, &nobj, bank, &fdscat_blen, &fdscat_maxlen);
  rcode += dst_unpackr8_(&fdscat_.endTime, &nobj, bank, &fdscat_blen, &fdscat_maxlen);
  rcode += dst_unpackr8_(&fdscat_.hzalen, &nobj, bank, &fdscat_blen, &fdscat_maxlen);
  rcode += dst_unpackr8_(&fdscat_.vaodep, &nobj, bank, &fdscat_blen, &fdscat_maxlen);
  rcode += dst_unpackr8_(&fdscat_.schght, &nobj, bank, &fdscat_blen, &fdscat_maxlen);

  if (fdscat_ver >= 1)
  {
    rcode += dst_unpacki4_(&fdscat_.uniqID,     &nobj, bank, &fdscat_blen,
			   &fdscat_maxlen);
    rcode += dst_unpacki4_(&fdscat_.vaodID,     &nobj, bank, &fdscat_blen,
			   &fdscat_maxlen);
    rcode += dst_unpacki4_(&fdscat_.vaodDate,   &nobj, bank, &fdscat_blen,
			   &fdscat_maxlen);
    rcode += dst_unpackr8_(&fdscat_.vaodep_rms, &nobj, bank, &fdscat_blen,
			   &fdscat_maxlen);
    rcode += dst_unpacki4_(&fdscat_.qual,       &nobj, bank, &fdscat_blen,
			   &fdscat_maxlen);
    rcode += dst_unpacki4_(&fdscat_.method,     &nobj, bank, &fdscat_blen,
			   &fdscat_maxlen);
    rcode += dst_unpacki4_(&fdscat_.nMeas,      &nobj, bank, &fdscat_blen,
			   &fdscat_maxlen);
    nobj = fdscat_.nMeas;
    if (nobj < 0 || nobj >= FDSCAT_MAX_MEAS)
    {
      fprintf(stderr, "%s: %s: fdscat_.nMeas: illegal value (%d)\n",
	      __FILE__, __func__, nobj);
      exit(EXIT_FAILURE);
    }
    rcode += dst_unpackr8_(&fdscat_.hzalen_meas[0], &nobj, bank,
			   &fdscat_blen, &fdscat_maxlen);
    rcode += dst_unpackr8_(&fdscat_.vaodep_meas[0], &nobj, bank,
			   &fdscat_blen, &fdscat_maxlen);
    rcode += dst_unpackr8_(&fdscat_.vaodep_rms_meas[0], &nobj, bank,
			   &fdscat_blen, &fdscat_maxlen);
    rcode += dst_unpackr8_(&fdscat_.schght_meas[0], &nobj, bank,
			   &fdscat_blen, &fdscat_maxlen);
    rcode += dst_unpacki4_(&fdscat_.site_meas[0], &nobj, bank,
			   &fdscat_blen, &fdscat_maxlen);
    rcode += dst_unpacki4_(&fdscat_.qual_meas[0], &nobj, bank,
			   &fdscat_blen, &fdscat_maxlen);
  }

  return rcode ;
}

integer4 fdscat_common_to_dump_(integer4 *long_output) {
  return fdscat_common_to_dumpf_(stdout,long_output);
}

void dateToString(double jsec, char *str) {
  const int dayinmo[2][12] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}, 
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
  };
  const int dayinyr[2] = { 365, 366 };

  int i, j, yr, mo, da, hr, mn, sc;

  da = (int)( jsec / 86400.0 );
  sc = (int)jsec % 86400;

  yr = 1970;
  i = 0;
  while ( da >= dayinyr[i] ) {
    da -= dayinyr[i];
    i = ( (++yr)%4 == 0 ) ? 1 : 0;
  }

  j = 0;
  while ( da >= dayinmo[i][j] )
    da -= dayinmo[i][j++];

  mo = j + 1;
  da = da + 1;

  hr = sc / 3600;
  mn = ( sc / 60 ) % 60;
  sc = sc % 60;

  sprintf(str, "%2d/%02d/%4d -- %2d:%02d:%02d", mo, da, yr, hr, mn, sc);
}

integer4 fdscat_common_to_dumpf_(FILE* fp,integer4 *long_output) {
  char sstr[24], estr[24];

  fprintf(fp, "FDSCAT :");

  fprintf(fp, "\n");

  dateToString(fdscat_.startTime, sstr);
  dateToString(fdscat_.endTime, estr);

  if (fdscat_ver >= 1)
  {
    fprintf(fp, "  uniqID: %d, vaodID: %d, vaodDate: %d\n",
	    fdscat_.uniqID, fdscat_.vaodID, fdscat_.vaodDate);
  }

  fprintf(fp, "  Valid from:  %s  to  %s\n", sstr, estr);

  fprintf(fp, "  Horizontal Attenuation Length: %lf (m)\n", fdscat_.hzalen);
  fprintf(fp, "  Vertical Aerosol Optical Depth: %lf ", fdscat_.vaodep);
  if (fdscat_ver >= 1)
    fprintf(fp, "+/- %lf", fdscat_.vaodep_rms);
  fprintf(fp, "\n");
  fprintf(fp, "  Scale Height: %lf (m)\n", fdscat_.schght);
  if (fdscat_ver >= 1)
  {
    fprintf(fp, "  Combined measurement method: %d, quality: %d\n",
	    fdscat_.method, fdscat_.qual);
    fprintf(fp, "  Number of sites measured: %d\n", fdscat_.nMeas);

    if (*long_output > 0)
    {
      fprintf(fp, "  %4s %9s %9s %9s %10s %4s\n", "site", "schght",
	      "hzalen", "vaodep", "vaodep_rms", "qual");
      fprintf(fp, "  %4s %9s %9s %9s %10s %4s\n", "----", "---------",
	      "---------", "---------", "----------", "----");
      int i;
      for (i = 0; i < fdscat_.nMeas; i++)
      {
	fprintf(fp, "  %4d %9.2lf %9.2lf %9.5lf %10.5lf %4d\n",
		fdscat_.site_meas[i], fdscat_.schght_meas[i],
		fdscat_.hzalen_meas[i], fdscat_.vaodep_meas[i],
		fdscat_.vaodep_rms_meas[i], fdscat_.qual_meas[i]);
      }
    }
  }





  fprintf(fp, "\n");

  return 0;
}










