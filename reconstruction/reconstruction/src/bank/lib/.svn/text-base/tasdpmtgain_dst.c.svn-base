/*
  A test and an example program to learn the DST system
  Y. Tsunesada 2008/Jul/18
  Modified by N.Sakruai for tasdpmtgain_dst
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
#include "tasdpmtgain_dst.h"

tasdpmtgain_dst_common tasdpmtgain_;

static integer4 tasdpmtgain_blen = 0;
static int tasdpmtgain_maxlen=sizeof(int)*2+sizeof(tasdpmtgain_dst_common);

// Data block for read/write 
static char *tasdpmtgain_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdpmtgain_bank_buffer_ (integer4* tasdpmtgain_bank_buffer_size)
{
  (*tasdpmtgain_bank_buffer_size) = tasdpmtgain_blen;
  return tasdpmtgain_bank;
}



static void tasdpmtgain_init()
{
  tasdpmtgain_bank = (char *) calloc(tasdpmtgain_maxlen, sizeof(char));
  if (tasdpmtgain_bank == NULL) {
    fprintf(stderr,
	    "tasdpmtgain_init: fail to assign memory to bank. Abort.\n");
    exit(1);
  }
  //  fprintf(stderr, "ALLOCATED the memory\n");
}

// Pack the data from struct to the data block
int tasdpmtgain_common_to_bank_()
{
  static int id=TASDPMTGAIN_BANKID, ver=TASDPMTGAIN_BANKVERSION;
  int rcode, nobj, ii;

  if (tasdpmtgain_bank == NULL) tasdpmtgain_init();
  /* Initialize test_blen, and pack the id and version to bank */
  rcode = dst_initbank_(&id, &ver, &tasdpmtgain_blen,
			&tasdpmtgain_maxlen,
			tasdpmtgain_bank);


  /* Change here for each bank structure */
  nobj = 1;
  rcode += dst_packi4_(&tasdpmtgain_.npmt, &nobj,
		       tasdpmtgain_bank, &tasdpmtgain_blen,
		       &tasdpmtgain_maxlen);

  nobj = sizeof(SDPmtData)/sizeof(int);
  for(ii=0;ii<tasdpmtgain_.npmt;ii++){
    rcode += dst_packi4_((int*)&tasdpmtgain_.pmt[ii], &nobj,
			 tasdpmtgain_bank, &tasdpmtgain_blen,
			 &tasdpmtgain_maxlen);
  }

  return rcode;
}

// Unpack the data from the data block to struct
int tasdpmtgain_bank_to_common_(char *block)
{
  int rcode = 0;
  int nobj,ii;

  tasdpmtgain_blen = 2*sizeof(int); /* skip id and version */

  nobj = 1;
  rcode += dst_unpacki4_( &tasdpmtgain_.npmt, &nobj, block,
			  &tasdpmtgain_blen, &tasdpmtgain_maxlen);

  nobj = sizeof(SDPmtData)/sizeof(int);
  for(ii=0;ii<tasdpmtgain_.npmt;ii++){
    rcode+=dst_unpacki4_((int*)&tasdpmtgain_.pmt[ii],&nobj,block,
			 &tasdpmtgain_blen, &tasdpmtgain_maxlen);
  }

  return rcode;
}

int tasdpmtgain_common_to_dumpf_(FILE *fp, int *long_output)
{
  /* Change here for each bank */
  /*
  fprintf(fp, "Pmt serial#: %d\n", tasdpmtgain_.serial);
  fprintf(fp, "Minumum voltage: %f\n", tasdpmtgain_.minv);
  fprintf(fp, "Maximum voltage: %f\n", tasdpmtgain_.maxv);
  */
  (void)(fp);
  (void)(long_output);
  return 0;
}


int tasdpmtgain_bank_to_dst_(int *NumUnit)
{
  int rcode;
  rcode = dst_write_bank_(NumUnit, &tasdpmtgain_blen, tasdpmtgain_bank);
  free(tasdpmtgain_bank);
  tasdpmtgain_bank = NULL;
  return rcode;
}

int tasdpmtgain_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ((rcode = tasdpmtgain_common_to_bank_())) {
    fprintf(stderr, "tasdpmtgain_common_to_bank_ ERROR: %ld\n", (long) rcode);
    exit(1);
  } 
  if ((rcode = tasdpmtgain_bank_to_dst_(NumUnit))) {
    fprintf(stderr, "tasdpmtgain_bank_to_dst_ ERROR: %ld\n", (long) rcode);
    exit(2);
  }
  return 0;
}
