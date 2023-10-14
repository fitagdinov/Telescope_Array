/*
 * tasdtrgmode_dst.c
 *
 * C functions for TASDTRGMODE bank
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
#include "tasdtrgmode_dst.h"

tasdtrgmode_dst_common tasdtrgmode_;/* allocate memory to
				 tasdtrgmode_common */

static integer4 tasdtrgmode_blen = 0;
static int tasdtrgmode_maxlen = sizeof(int)*2+sizeof(tasdtrgmode_dst_common);
static char *tasdtrgmode_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdtrgmode_bank_buffer_ (integer4* tasdtrgmode_bank_buffer_size)
{
  (*tasdtrgmode_bank_buffer_size) = tasdtrgmode_blen;
  return tasdtrgmode_bank;
}



static void tasdtrgmode_bank_init(void)
{
  tasdtrgmode_bank = (char *)calloc(tasdtrgmode_maxlen, sizeof(char));
  if (tasdtrgmode_bank==NULL){
    fprintf(stderr,
	    "tasdtrgmode_bank_init: "
	    "fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

int tasdtrgmode_common_to_bank_(void)
{
  static int id = TASDTRGMODE_BANKID, ver = TASDTRGMODE_BANKVERSION;
  int rc, nobj, ii;

  if (tasdtrgmode_bank == NULL) tasdtrgmode_bank_init();

  /*Initialize tasdtrgmode_blen, and pack the id and version to bank*/
  if ( (rc=dst_initbank_(&id,&ver,&tasdtrgmode_blen,&tasdtrgmode_maxlen,tasdtrgmode_bank)) )
    return rc;

  nobj = 1;
  rc+=dst_packi4_(&tasdtrgmode_.year,  &nobj,tasdtrgmode_bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
  rc+=dst_packi4_(&tasdtrgmode_.run_id,&nobj,tasdtrgmode_bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
  rc+=dst_packi4_(&tasdtrgmode_.npoint,&nobj,tasdtrgmode_bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);

  for(ii=0;ii<tasdtrgmode_.npoint;ii++){
    nobj = 1;
    rc+=dst_packi2_(&tasdtrgmode_.data[ii].sec,
		    &nobj,tasdtrgmode_bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
    rc+=dst_packi2_(&tasdtrgmode_.data[ii].trgmode,
		    &nobj,tasdtrgmode_bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
    nobj = tasdtrgmode_nhmax;
    rc+=dst_packi4_(&tasdtrgmode_.data[ii].strial[0],
		    &nobj,tasdtrgmode_bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
    rc+=dst_packi4_(&tasdtrgmode_.data[ii].etrial[0],
		    &nobj,tasdtrgmode_bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
  }

  nobj = 1;
  rc+=dst_packi4_(&tasdtrgmode_.footer,&nobj,tasdtrgmode_bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);

  return rc;
}


int tasdtrgmode_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdtrgmode_blen, tasdtrgmode_bank );
}

int tasdtrgmode_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdtrgmode_common_to_bank_()) ){
    fprintf (stderr,"tasdtrgmode_common_to_bank_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  if ( (rcode = tasdtrgmode_bank_to_dst_(NumUnit)) ) {
    fprintf (stderr,"tasdtrgmode_bank_to_dst_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  return SUCCESS;
}

int tasdtrgmode_bank_to_common_(char *bank)
{
  int rc = 0;
  int nobj, ii;


  tasdtrgmode_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rc+=dst_unpacki4_(&tasdtrgmode_.year,  &nobj,bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
  rc+=dst_unpacki4_(&tasdtrgmode_.run_id,&nobj,bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
  rc+=dst_unpacki4_(&tasdtrgmode_.npoint,&nobj,bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);

  for(ii=0;ii<tasdtrgmode_.npoint;ii++){
    nobj = 1;
    rc+=dst_unpacki2_(&tasdtrgmode_.data[ii].sec,
		    &nobj,bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
    rc+=dst_unpacki2_(&tasdtrgmode_.data[ii].trgmode,
		    &nobj,bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
    nobj = tasdtrgmode_nhmax;
    rc+=dst_unpacki4_(&tasdtrgmode_.data[ii].strial[0],
		    &nobj,bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
    rc+=dst_unpacki4_(&tasdtrgmode_.data[ii].etrial[0],
		    &nobj,bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);
  }

  nobj = 1;
  rc+=dst_unpacki4_(&tasdtrgmode_.footer,&nobj,bank,&tasdtrgmode_blen,&tasdtrgmode_maxlen);


  return rc;

}

int tasdtrgmode_common_to_dump_(int *long_output)
{
  return tasdtrgmode_common_to_dumpf_(stdout, long_output);
}

int tasdtrgmode_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
