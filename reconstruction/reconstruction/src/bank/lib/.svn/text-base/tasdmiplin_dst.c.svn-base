/*
 * tasdevent_dst.c
 *
 * C functions for TASDMIPLIN bank
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
#include "tasdmiplin_dst.h"

tasdmiplin_dst_common tasdmiplin_;  /* allocate memory to tasdmiplin_common */

static integer4 tasdmiplin_blen = 0;
static int tasdmiplin_maxlen =
  sizeof(int)*2+sizeof(tasdmiplin_dst_common);
static char *tasdmiplin_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdmiplin_bank_buffer_ (integer4* tasdmiplin_bank_buffer_size)
{
  (*tasdmiplin_bank_buffer_size) = tasdmiplin_blen;
  return tasdmiplin_bank;
}



static void tasdmiplin_bank_init(void)
{
  tasdmiplin_bank = (char *)calloc(tasdmiplin_maxlen,sizeof(char));
  if (tasdmiplin_bank==NULL){
    fprintf(stderr,
	    "tasdmiplin_bank_init: "
	    "fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

int tasdmiplin_common_to_bank_(void)
{
  static int id = TASDMIPLIN_BANKID, ver = TASDMIPLIN_BANKVERSION;
  int rcode, nobj, ii;

  if (tasdmiplin_bank == NULL) tasdmiplin_bank_init();

  /* Initialize tasdmiplin_blen, and pack the id and version to bank */
  if ( (rcode=dst_initbank_(&id,&ver,&tasdmiplin_blen,
			    &tasdmiplin_maxlen,tasdmiplin_bank)) ){
    return rcode;
  }

  nobj = 1;
  rcode += dst_packi4_(&tasdmiplin_.num_det,&nobj,tasdmiplin_bank,
		       &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_packi4_(&tasdmiplin_.dateFrom,&nobj,tasdmiplin_bank,
		       &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_packi4_(&tasdmiplin_.dateTo,  &nobj,tasdmiplin_bank,
		       &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_packi4_(&tasdmiplin_.timeFrom,&nobj,tasdmiplin_bank,
		       &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_packi4_(&tasdmiplin_.timeTo,  &nobj,tasdmiplin_bank,
		       &tasdmiplin_blen, &tasdmiplin_maxlen);

  nobj = tasdmiplin_nhmax;
  rcode += dst_packi4_(tasdmiplin_.first_run_id,
		       &nobj,tasdmiplin_bank,
		       &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_packi4_(tasdmiplin_.last_run_id,
		       &nobj,tasdmiplin_bank,
		       &tasdmiplin_blen, &tasdmiplin_maxlen);

  nobj = sizeof(SDMiplinData);
  for(ii=0;ii<tasdmiplin_.num_det;ii++){
    rcode += dst_packi1_((char*)(&tasdmiplin_.sub[ii]), &nobj,
			 tasdmiplin_bank,
			 &tasdmiplin_blen, &tasdmiplin_maxlen);
  }

  nobj = 1;
  rcode += dst_packi4_(&tasdmiplin_.footer,&nobj,tasdmiplin_bank,
		       &tasdmiplin_blen, &tasdmiplin_maxlen);


  return rcode;
}


int tasdmiplin_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit,&tasdmiplin_blen,tasdmiplin_bank);
}

int tasdmiplin_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdmiplin_common_to_bank_()) )
    {
      fprintf (stderr,"tasdmiplin_common_to_bank_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  if ( (rcode = tasdmiplin_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdmiplin_bank_to_dst_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  return SUCCESS;
}

int tasdmiplin_bank_to_common_(char *bank)
{
  int rcode = 0;
  int nobj, ii;

  tasdmiplin_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rcode += dst_unpacki4_( &tasdmiplin_.num_det, &nobj, bank,
			  &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_unpacki4_( &tasdmiplin_.dateFrom, &nobj, bank,
			  &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_unpacki4_( &tasdmiplin_.dateTo,   &nobj, bank,
			  &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_unpacki4_( &tasdmiplin_.timeFrom, &nobj, bank,
			  &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_unpacki4_( &tasdmiplin_.timeTo,   &nobj, bank,
			  &tasdmiplin_blen, &tasdmiplin_maxlen);

  nobj = tasdmiplin_nhmax;
  rcode += dst_unpacki4_( tasdmiplin_.first_run_id,  &nobj, bank,
			  &tasdmiplin_blen, &tasdmiplin_maxlen);
  rcode += dst_unpacki4_( tasdmiplin_.last_run_id,   &nobj, bank,
			  &tasdmiplin_blen, &tasdmiplin_maxlen);

  nobj = sizeof(SDMiplinData);
  for(ii=0;ii<tasdmiplin_.num_det;ii++){
    rcode+=dst_unpacki1_((char*)(&tasdmiplin_.sub[ii]),&nobj,bank,
			 &tasdmiplin_blen, &tasdmiplin_maxlen);
  }

  nobj = 1;
  rcode += dst_unpacki4_( &tasdmiplin_.footer, &nobj, bank,
			  &tasdmiplin_blen, &tasdmiplin_maxlen);

  return rcode;

}

int tasdmiplin_common_to_dump_(int *long_output)
{
  return tasdmiplin_common_to_dumpf_(stdout, long_output);
}

int tasdmiplin_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
