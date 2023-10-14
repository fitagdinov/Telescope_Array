/*
 * C functions for fdraw
 * DRB 2008/09/23
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fdfft_dst.h"  
#include "convtime.h"  

fdfft_dst_common fdfft_;  /* allocate memory to fdfft_common */

integer4 fdfft_blen = 0; /* not static because it needs to be accessed by the c files of the derived banks */ 
static integer4 fdfft_maxlen = sizeof(integer4) * 2 + sizeof(fdfft_dst_common);
static integer1 *fdfft_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdfft_bank_buffer_ (integer4* fdfft_bank_buffer_size)
{
  (*fdfft_bank_buffer_size) = fdfft_blen;
  return fdfft_bank;
}



static void fdfft_abank_init(integer1* (*pbank) ) {
  *pbank = (integer1 *)calloc(fdfft_maxlen, sizeof(integer1));
  if (*pbank==NULL) {
    fprintf (stderr,"fdfft_abank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  } 
}    
static void fdfft_bank_init() {fdfft_abank_init(&fdfft_bank);}

integer4 fdfft_common_to_bank_() {
  if (fdfft_bank == NULL) fdfft_bank_init();
  return fdfft_struct_to_abank_(&fdfft_, &fdfft_bank, FDFFT_BANKID, FDFFT_BANKVERSION);
}
integer4 fdfft_bank_to_dst_ (integer4 *unit) {return fdfft_abank_to_dst_(fdfft_bank, unit);}
integer4 fdfft_common_to_dst_(integer4 *unit) {
  if (fdfft_bank == NULL) fdfft_bank_init();
  return fdfft_struct_to_dst_(&fdfft_, &fdfft_bank, unit, FDFFT_BANKID, FDFFT_BANKVERSION);
}
integer4 fdfft_bank_to_common_(integer1 *bank) {return fdfft_abank_to_struct_(bank, &fdfft_);}
integer4 fdfft_common_to_dump_(integer4 *opt) {return fdfft_struct_to_dumpf_(-1, &fdfft_, stdout, opt);}
integer4 fdfft_common_to_dumpf_(FILE* fp, integer4 *opt) {return fdfft_struct_to_dumpf_(-1, &fdfft_, fp, opt);}

integer4 fdfft_struct_to_abank_(fdfft_dst_common *fdfft, integer1* (*pbank), integer4 id, integer4 ver)
{
  integer4 rcode, nobj, i, j;
  integer1 *bank;

  if ( *pbank == NULL ) fdfft_abank_init(pbank);
    
  bank = *pbank;
  rcode = dst_initbank_(&id, &ver, &fdfft_blen, &fdfft_maxlen, bank);

  /* Initialize test_blen, and pack the id and version to bank */

  nobj=1;
  rcode += dst_packi4_(&fdfft->startsec,              &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_packi4_(&fdfft->stopsec,               &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_packi4_(&fdfft->startnsec,             &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_packi4_(&fdfft->stopnsec,              &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_packi4_(&fdfft->ncamera,               &nobj, bank, &fdfft_blen, &fdfft_maxlen);

  nobj = fdfft->ncamera;
  rcode += dst_packi4_(&fdfft->camera[0],             &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_packi4_(&fdfft->nchan[0],              &nobj, bank, &fdfft_blen, &fdfft_maxlen);

  for ( i=0; i<fdfft->ncamera; i++ ) {
    nobj = fdfft->nchan[i];
    rcode += dst_packi4_(&fdfft->chan[i][0],          &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  }

  nobj = fdfft_nt_chan_max;
  for ( i=0; i<fdfft->ncamera; i++ ) {
    for ( j=0; j<fdfft->nchan[i]; j++ ) {
      rcode += dst_packr8_(&fdfft->powerspec[i][j][0],&nobj, bank, &fdfft_blen, &fdfft_maxlen);
    }
  }

  for ( i=0; i<fdfft->ncamera; i++ ) {
    for ( j=0; j<fdfft->nchan[i]; j++ ) {
      rcode += dst_packr8_(&fdfft->powerspecerr[i][j][0],&nobj, bank, &fdfft_blen, &fdfft_maxlen);
    }
  }

  return rcode ;
}

integer4 fdfft_abank_to_dst_(integer1 *bank, integer4 *unit) {return dst_write_bank_(unit, &fdfft_blen, bank);}

integer4 fdfft_struct_to_dst_(fdfft_dst_common *fdfft, integer1* (*pbank), integer4 *unit, integer4 id, integer4 ver) {
  integer4 rcode;
  if ( (rcode = fdfft_struct_to_abank_(fdfft, pbank, id, ver)) ) {
      fprintf(stderr, "fdfft_struct_to_abank_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  if ( (rcode = fdfft_abank_to_dst_(*pbank, unit)) ) {
      fprintf(stderr, "fdfft_abank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  return 0;
}

integer4 fdfft_abank_to_struct_(integer1 *bank, fdfft_dst_common *fdfft) {
  integer4 rcode = 0 ;
  integer4 nobj ,i ,j;

  fdfft_blen = 2 * sizeof(integer4);      /*  skip id and version  */

  nobj=1;
  rcode += dst_unpacki4_(&fdfft->startsec,              &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_unpacki4_(&fdfft->stopsec,               &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_unpacki4_(&fdfft->startnsec,             &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_unpacki4_(&fdfft->stopnsec,              &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_unpacki4_(&fdfft->ncamera,               &nobj, bank, &fdfft_blen, &fdfft_maxlen);

  nobj = fdfft->ncamera;
  rcode += dst_unpacki4_(&fdfft->camera[0],             &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  rcode += dst_unpacki4_(&fdfft->nchan[0],              &nobj, bank, &fdfft_blen, &fdfft_maxlen);

  for ( i=0; i<fdfft->ncamera; i++ ) {
    nobj = fdfft->nchan[i];
    rcode += dst_unpacki4_(&fdfft->chan[i][0],          &nobj, bank, &fdfft_blen, &fdfft_maxlen);
  }

  nobj = fdfft_nt_chan_max;
  for ( i=0; i<fdfft->ncamera; i++ ) {
    for ( j=0; j<fdfft->nchan[i]; j++ ) {
      rcode += dst_unpackr8_(&fdfft->powerspec[i][j][0],&nobj, bank, &fdfft_blen, &fdfft_maxlen);
    }
  }

  for ( i=0; i<fdfft->ncamera; i++ ) {
    for ( j=0; j<fdfft->nchan[i]; j++ ) {
      rcode += dst_unpackr8_(&fdfft->powerspecerr[i][j][0],&nobj, bank, &fdfft_blen, &fdfft_maxlen);
    }
  }

  return rcode ;
}

integer4 fdfft_struct_to_dump_(integer4 siteid, fdfft_dst_common *fdfft, integer4 *long_output) {
  return fdfft_struct_to_dumpf_(siteid, fdfft, stdout, long_output);
}

integer4 fdfft_struct_to_dumpf_(integer4 siteid, fdfft_dst_common *fdfft, FILE* fp,integer4 *long_output)
{
  integer4 i,j,k;
  integer4 ymd, yr, mo, da, hr, mn, ss;
  real8 f, sec;

  if ( siteid == 0 )
    fprintf(fp, "BRFFT :\n");
  else if ( siteid == 1 )
    fprintf(fp, "LRFFT :\n");
  else
    fprintf (fp, "FDFFT :\n");

  eposec2ymdsec((double)fdfft->startsec, &ymd, &sec);
  yr = ymd / 10000;
  mo = ( ymd / 100 ) % 100;
  da = ymd % 100;
  hr = ( (int)sec ) / 3600;
  mn = ( (int)sec/60 ) % 60;
  ss = (int)sec % 60;
  fprintf(fp, "  time:  %2d/%02d/%4d %2d:%02d:%02d.%09d ", mo, da, yr, hr, mn, 
	  ss, fdfft->startnsec);

  eposec2ymdsec((double)fdfft->stopsec, &ymd, &sec);
  yr = ymd / 10000;
  mo = ( ymd / 100 ) % 100;
  da = ymd % 100;
  hr = ( (int)sec ) / 3600;
  mn = ( (int)sec/60 ) % 60;
  ss = (int)sec % 60;
  fprintf(fp, "to %2d/%02d/%4d %2d:%02d:%02d.%09d\n", mo, da, yr, hr, mn, ss, 
	  fdfft->stopnsec);


  fprintf(fp, "  stored data for %d cameras\n", fdfft->ncamera);
  if ( *long_output == TRUE ) {
    fprintf(fp, "    cam  tube  idx    freq      amplitude  "
            "ampl.RMS\n");
    for ( i=0; i<fdfft->ncamera; i++ ) {
      for ( j=0; j<fdfft->nchan[i]; j++ ) {
	for ( k=0; k<fdfft_nt_chan_max; k++ ) {
	  if ( k == 0 )
	    f = 0.00;
	  else if ( k <= fdfft_nt_chan_max/2 )
	    f = (float)k / 51.2;
	  else
	    f = (float)(k-fdfft_nt_chan_max) / 51.2;

	  fprintf(fp, "      %2d   %3d  %3d  %lf  %lf  %lf\n", 
		  fdfft->camera[i], fdfft->chan[i][j], k, f, 
		  fdfft->powerspec[i][j][k], fdfft->powerspecerr[i][j][k]);
	}
      }
    }
  }
  else {
    fprintf(fp, "    data not written in short output.\n");
  }


  return 0;
} 

