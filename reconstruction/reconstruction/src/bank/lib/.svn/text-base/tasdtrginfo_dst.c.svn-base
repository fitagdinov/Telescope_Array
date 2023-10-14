/*
 * tasdtrginfo_dst.c
 *
 * C functions for TASDTRGINFO bank
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
#include "tasdtrginfo_dst.h"

tasdtrginfo_dst_common tasdtrginfo_;/* allocate memory to
				 tasdtrginfo_common */

static integer4 tasdtrginfo_blen = 0;
static int tasdtrginfo_maxlen = sizeof(int)*2+sizeof(tasdtrginfo_dst_common);
static char *tasdtrginfo_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdtrginfo_bank_buffer_ (integer4* tasdtrginfo_bank_buffer_size)
{
  (*tasdtrginfo_bank_buffer_size) = tasdtrginfo_blen;
  return tasdtrginfo_bank;
}



static void tasdtrginfo_bank_init(void)
{
  tasdtrginfo_bank = (char *)calloc(tasdtrginfo_maxlen, sizeof(char));
  if (tasdtrginfo_bank==NULL){
    fprintf(stderr,
	    "tasdtrginfo_bank_init: "
	    "fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

int tasdtrginfo_common_to_bank_(void)
{
  static int id = TASDTRGINFO_BANKID, ver = TASDTRGINFO_BANKVERSION;
  int rc, nobj, ii;

  if (tasdtrginfo_bank == NULL) tasdtrginfo_bank_init();

  /*Initialize tasdtrginfo_blen, and pack the id and version to bank*/
  if ( (rc=dst_initbank_(&id,&ver,&tasdtrginfo_blen,&tasdtrginfo_maxlen,tasdtrginfo_bank)) )
    return rc;

  nobj = 1;
  rc+=dst_packi4_(&tasdtrginfo_.site,  &nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  rc+=dst_packi4_(&tasdtrginfo_.run_id,&nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  rc+=dst_packi4_(&tasdtrginfo_.year,  &nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  rc+=dst_packi4_(&tasdtrginfo_.npoint,&nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  for(ii=0;ii<tasdtrginfo_.npoint;ii++){
    rc+=dst_packi4_(&tasdtrginfo_.data[ii].bank,
		    &nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
    rc+=dst_packi2_(&tasdtrginfo_.data[ii].pos,
		    &nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
    rc+=dst_packi2_(&tasdtrginfo_.data[ii].command,
		    &nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
    rc+=dst_packi1_(&tasdtrginfo_.data[ii].trgcode,
		    &nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
    rc+=dst_packi1_(&tasdtrginfo_.data[ii].daqcode,
		    &nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  }
  rc+=dst_packi4_(&tasdtrginfo_.footer,&nobj,tasdtrginfo_bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);

  return rc;
}


int tasdtrginfo_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdtrginfo_blen, tasdtrginfo_bank );
}

int tasdtrginfo_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdtrginfo_common_to_bank_()) ){
    fprintf (stderr,"tasdtrginfo_common_to_bank_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  if ( (rcode = tasdtrginfo_bank_to_dst_(NumUnit)) ) {
    fprintf (stderr,"tasdtrginfo_bank_to_dst_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  return SUCCESS;
}

int tasdtrginfo_bank_to_common_(char *bank)
{
  int rc = 0;
  int nobj, ii;


  tasdtrginfo_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rc+=dst_unpacki4_(&tasdtrginfo_.site,  &nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  rc+=dst_unpacki4_(&tasdtrginfo_.run_id,&nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  rc+=dst_unpacki4_(&tasdtrginfo_.year,  &nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  rc+=dst_unpacki4_(&tasdtrginfo_.npoint,&nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  for(ii=0;ii<tasdtrginfo_.npoint;ii++){
    rc+=dst_unpacki4_(&tasdtrginfo_.data[ii].bank,
		    &nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
    rc+=dst_unpacki2_(&tasdtrginfo_.data[ii].pos,
		    &nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
    rc+=dst_unpacki2_(&tasdtrginfo_.data[ii].command,
		    &nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
    rc+=dst_unpacki1_(&tasdtrginfo_.data[ii].trgcode,
		    &nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
    rc+=dst_unpacki1_(&tasdtrginfo_.data[ii].daqcode,
		    &nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);
  }
  rc+=dst_unpacki4_(&tasdtrginfo_.footer,&nobj,bank,&tasdtrginfo_blen,&tasdtrginfo_maxlen);

  return rc;

}

int tasdtrginfo_common_to_dump_(int *long_output)
{
  return tasdtrginfo_common_to_dumpf_(stdout, long_output);
}

int tasdtrginfo_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
