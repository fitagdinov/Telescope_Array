/*
 * tasdevent_dst.c 
 *
 * C functions for TASDLEDLIN bank
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
#include "tasdledlin_dst.h"

tasdledlin_dst_common tasdledlin_;  /* allocate memory to tasdledlin_common */

static integer4 tasdledlin_blen = 0;
static int tasdledlin_maxlen = sizeof(int)*2+sizeof(tasdledlin_dst_common);
static char *tasdledlin_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdledlin_bank_buffer_ (integer4* tasdledlin_bank_buffer_size)
{
  (*tasdledlin_bank_buffer_size) = tasdledlin_blen;
  return tasdledlin_bank;
}



static void tasdledlin_bank_init(void)
{
  tasdledlin_bank = (char *)calloc(tasdledlin_maxlen,sizeof(char));
  if (tasdledlin_bank==NULL)
    {
      fprintf(stderr,
	      "tasdledlin_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

int tasdledlin_common_to_bank_(void)
{
  static int id = TASDLEDLIN_BANKID, ver = TASDLEDLIN_BANKVERSION;
  int rcode, nobj, ii;

  if (tasdledlin_bank == NULL) tasdledlin_bank_init();

  /* Initialize tasdledlin_blen, and pack the id and version to bank */
  if ( (rcode=dst_initbank_(&id,&ver,&tasdledlin_blen,&tasdledlin_maxlen,tasdledlin_bank)) )
    return rcode;

  nobj = 1;
  rcode+=dst_packi4_(&tasdledlin_.ndet, &nobj, tasdledlin_bank,
		     &tasdledlin_blen, &tasdledlin_maxlen);

  nobj = 1;
  rcode+=dst_packi4_(&tasdledlin_.first_date,&nobj,
		     tasdledlin_bank,
		     &tasdledlin_blen, &tasdledlin_maxlen);
  nobj = tasdledlin_nhmax;
  rcode+=dst_packi4_(tasdledlin_.first_run_id,&nobj,
		     tasdledlin_bank,
		     &tasdledlin_blen, &tasdledlin_maxlen);
  nobj = 1;
  rcode+=dst_packi4_(&tasdledlin_.last_date,&nobj,
		     tasdledlin_bank,
		     &tasdledlin_blen, &tasdledlin_maxlen);
  nobj = tasdledlin_nhmax;
  rcode+=dst_packi4_(tasdledlin_.last_run_id,&nobj,
		     tasdledlin_bank,
		     &tasdledlin_blen, &tasdledlin_maxlen);

  nobj = sizeof(SDLEDSubData)/sizeof(int);
  for(ii=0;ii<tasdledlin_.ndet;ii++){
    rcode+=dst_packi4_((int*)&tasdledlin_.sub[ii], &nobj,
		       tasdledlin_bank, &tasdledlin_blen,
		       &tasdledlin_maxlen);
  }

  nobj = 1;
  rcode+=dst_packi4_(&tasdledlin_.footer, &nobj, tasdledlin_bank,
		     &tasdledlin_blen, &tasdledlin_maxlen);
  return rcode;
}


int tasdledlin_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdledlin_blen, tasdledlin_bank );
}

int tasdledlin_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdledlin_common_to_bank_()) )
    {
      fprintf(stderr,"tasdledlin_common_to_bank_ ERROR : %ld\n",
	      (long) rcode);
      exit(0);
    }
  if ( (rcode = tasdledlin_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdledlin_bank_to_dst_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  return SUCCESS;
}

int tasdledlin_bank_to_common_(char *bank)
{
  int rcode = 0;
  int nobj, ii;


  tasdledlin_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rcode+=dst_unpacki4_(&tasdledlin_.ndet,&nobj,bank,
		       &tasdledlin_blen, &tasdledlin_maxlen);

  nobj = 1;
  rcode+=dst_unpacki4_(&tasdledlin_.first_date,&nobj,bank,
		       &tasdledlin_blen, &tasdledlin_maxlen);
  nobj = tasdledlin_nhmax;
  rcode+=dst_unpacki4_(tasdledlin_.first_run_id,&nobj,bank,
		       &tasdledlin_blen, &tasdledlin_maxlen);
  nobj = 1;
  rcode+=dst_unpacki4_(&tasdledlin_.last_date,&nobj,bank,
		       &tasdledlin_blen, &tasdledlin_maxlen);
  nobj = tasdledlin_nhmax;
  rcode+=dst_unpacki4_(tasdledlin_.last_run_id,&nobj,bank,
		       &tasdledlin_blen, &tasdledlin_maxlen);

  nobj = sizeof(SDLEDSubData)/sizeof(int);
  for(ii=0;ii<tasdledlin_.ndet;ii++){
    rcode+=dst_unpacki4_((int*)&tasdledlin_.sub[ii],&nobj,bank,
			 &tasdledlin_blen, &tasdledlin_maxlen);
  }

  nobj = 1;
  rcode+=dst_unpacki4_(&tasdledlin_.footer,&nobj,bank,
		       &tasdledlin_blen, &tasdledlin_maxlen);


  return rcode;

}

int tasdledlin_common_to_dump_(int *long_output)
{
  return tasdledlin_common_to_dumpf_(stdout, long_output);
}

int tasdledlin_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
