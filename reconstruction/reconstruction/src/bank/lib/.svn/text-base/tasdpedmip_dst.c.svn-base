/*
 * tasdevent_dst.c
 *
 * C functions for TASDPEDMIP bank
 * a student - 20080619
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
#include "tasdpedmip_dst.h"

tasdpedmip_dst_common tasdpedmip_;  /* allocate memory to tasdpedmip_common */

static integer4 tasdpedmip_blen = 0;
static int tasdpedmip_maxlen = sizeof(int)*2+sizeof(tasdpedmip_dst_common);
static char *tasdpedmip_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdpedmip_bank_buffer_ (integer4* tasdpedmip_bank_buffer_size)
{
  (*tasdpedmip_bank_buffer_size) = tasdpedmip_blen;
  return tasdpedmip_bank;
}



static void tasdpedmip_bank_init(void)
{
  tasdpedmip_bank = (char *)calloc(tasdpedmip_maxlen, sizeof(char));
  if (tasdpedmip_bank==NULL){
    fprintf(stderr,"tasdpedmip_bank_init: "
	    "fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

int tasdpedmip_common_to_bank_(void)
{
  static int id = TASDPEDMIP_BANKID, ver = TASDPEDMIP_BANKVERSION;
  int rc, nobj, ii;

  if (tasdpedmip_bank == NULL) tasdpedmip_bank_init();

  /* Initialize tasdpedmip_blen, and pack the id and version to bank */
  if ( (rc=dst_initbank_(&id,&ver,&tasdpedmip_blen,&tasdpedmip_maxlen,tasdpedmip_bank)) )
    return rc;

  nobj = 1;
  rc+=dst_packi4_(&tasdpedmip_.num_det,&nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
  rc+=dst_packi4_(&tasdpedmip_.date,   &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
  rc+=dst_packi4_(&tasdpedmip_.time,   &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

  for(ii=0;ii<tasdpedmip_.num_det;ii++){
    nobj = 1;
    rc+=dst_packi4_(&tasdpedmip_.sub[ii].lid,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packi4_(&tasdpedmip_.sub[ii].livetime,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    rc+=dst_packr4_(&tasdpedmip_.sub[ii].upedAmp,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].upedAvr,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].upedStdev,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].upedAmpError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].upedAvrError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].upedStdevError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].upedChisq,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].upedDof,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packi4_(&tasdpedmip_.sub[ii].ubroken,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lpedAmp,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lpedAvr,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lpedStdev,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lpedAmpError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lpedAvrError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lpedStdevError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lpedChisq,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lpedDof,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packi4_(&tasdpedmip_.sub[ii].lbroken,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipAmp,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipNonuni,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipMev2cnt,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipMev2pe,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipAmpError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipNonuniError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipMev2cntError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipMev2peError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipChisq,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].umipDof,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipAmp,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipNonuni,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipMev2cnt,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipMev2pe,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipAmpError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipNonuniError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipMev2cntError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipMev2peError,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipChisq,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lmipDof,
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    nobj = 10;
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lvl0Rate[0],
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_packr4_(&tasdpedmip_.sub[ii].lvl1Rate[0],
		    &nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
  }

  nobj = 1;
  rc+=dst_packi4_(&tasdpedmip_.footer,&nobj,tasdpedmip_bank,&tasdpedmip_blen,&tasdpedmip_maxlen);


  return rc;
}


int tasdpedmip_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdpedmip_blen, tasdpedmip_bank );
}

int tasdpedmip_common_to_dst_(int *NumUnit)
{
  int rc;
  if ( (rc = tasdpedmip_common_to_bank_()) ){
    fprintf(stderr,"tasdpedmip_common_to_bank_ ERROR : %ld\n",
	    (long) rc);
    exit(0);
  }
  if ( (rc = tasdpedmip_bank_to_dst_(NumUnit)) ){
    fprintf (stderr,"tasdpedmip_bank_to_dst_ ERROR : %ld\n",
	     (long) rc);
    exit(0);
  }
  return SUCCESS;
}

int tasdpedmip_bank_to_common_(char *bank)
{
  int rc = 0;
  int nobj, ii;

  tasdpedmip_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rc+=dst_unpacki4_(&tasdpedmip_.num_det,&nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
  rc+=dst_unpacki4_(&tasdpedmip_.date,   &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
  rc+=dst_unpacki4_(&tasdpedmip_.time,   &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

  nobj = sizeof(SDPedMipData)/sizeof(int);
  for(ii=0;ii<tasdpedmip_.num_det;ii++){
    nobj = 1;
    rc+=dst_unpacki4_(&tasdpedmip_.sub[ii].lid,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpacki4_(&tasdpedmip_.sub[ii].livetime,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].upedAmp,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].upedAvr,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].upedStdev,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].upedAmpError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].upedAvrError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].upedStdevError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].upedChisq,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].upedDof,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpacki4_(&tasdpedmip_.sub[ii].ubroken,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lpedAmp,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lpedAvr,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lpedStdev,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lpedAmpError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lpedAvrError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lpedStdevError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lpedChisq,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lpedDof,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpacki4_(&tasdpedmip_.sub[ii].lbroken,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipAmp,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipNonuni,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipMev2cnt,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipMev2pe,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipAmpError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipNonuniError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipMev2cntError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipMev2peError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipChisq,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].umipDof,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipAmp,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipNonuni,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipMev2cnt,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipMev2pe,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipAmpError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipNonuniError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipMev2cntError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipMev2peError,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipChisq,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lmipDof,
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

    nobj = 10;
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lvl0Rate[0],
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
    rc+=dst_unpackr4_(&tasdpedmip_.sub[ii].lvl1Rate[0],
		      &nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);
  }

  nobj = 1;
  rc+=dst_unpacki4_(&tasdpedmip_.footer,&nobj,bank,&tasdpedmip_blen,&tasdpedmip_maxlen);

  return rc;

}

int tasdpedmip_common_to_dump_(int *long_output)
{
  return tasdpedmip_common_to_dumpf_(stdout, long_output);
}

int tasdpedmip_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
