/*
 * tasdgps_dst.c
 *
 * C functions for TASDGPS bank
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
#include "tasdgps_dst.h"

tasdgps_dst_common tasdgps_; /* allocate memory to
					  tasdgps_common */

static integer4 tasdgps_blen = 0;
static int tasdgps_maxlen = sizeof(int)*2+sizeof(tasdgps_dst_common);
static char *tasdgps_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdgps_bank_buffer_ (integer4* tasdgps_bank_buffer_size)
{
  (*tasdgps_bank_buffer_size) = tasdgps_blen;
  return tasdgps_bank;
}



static void tasdgps_bank_init(void)
{
  tasdgps_bank = (char *)calloc(tasdgps_maxlen, sizeof(char));
  if (tasdgps_bank==NULL){
    fprintf(stderr,
	    "tasdgps_bank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

int tasdgps_common_to_bank_(void)
{
  static int id = TASDGPS_BANKID, ver = TASDGPS_BANKVERSION;
  int rcode, nobj, ii;

  if (tasdgps_bank == NULL) tasdgps_bank_init();

  /* Initialize tasdgps_blen, and pack the id and version to bank */
  if ( (rcode=dst_initbank_(&id,&ver,&tasdgps_blen,&tasdgps_maxlen,tasdgps_bank)) )
    return rcode;

  nobj = 1;
  rcode += dst_packi4_(&tasdgps_.ndet,&nobj,tasdgps_bank,
		       &tasdgps_blen, &tasdgps_maxlen);
  rcode += dst_packi4_(&tasdgps_.first_date,&nobj,tasdgps_bank,
		       &tasdgps_blen, &tasdgps_maxlen);
  rcode += dst_packi4_(&tasdgps_.last_date,&nobj,tasdgps_bank,
		       &tasdgps_blen, &tasdgps_maxlen);

  nobj = tasdgps_nhmax;
  rcode+= dst_packi4_(&tasdgps_.first_run_id[0],&nobj,tasdgps_bank,
		      &tasdgps_blen, &tasdgps_maxlen);
  rcode+= dst_packi4_(&tasdgps_.last_run_id[0],&nobj,tasdgps_bank,
		      &tasdgps_blen, &tasdgps_maxlen);

  nobj = sizeof(SDGpsSubData)/sizeof(int);
  for(ii=0;ii<tasdgps_.ndet;ii++){
    rcode += dst_packi4_((int*)(&tasdgps_.sub[ii]),&nobj,
			 tasdgps_bank, &tasdgps_blen,
			 &tasdgps_maxlen);
  }

  nobj = 1;
  rcode += dst_packi4_(&tasdgps_.footer,&nobj,tasdgps_bank,
		       &tasdgps_blen, &tasdgps_maxlen);
  return rcode;
}


int tasdgps_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdgps_blen, tasdgps_bank );
}

int tasdgps_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdgps_common_to_bank_()) )
    {
      fprintf (stderr,"tasdgps_common_to_bank_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  if ( (rcode = tasdgps_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdgps_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);
    }
  return SUCCESS;
}

int tasdgps_bank_to_common_(char *bank)
{
  int rcode = 0;
  int nobj, ii;

  tasdgps_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rcode+=dst_unpacki4_(&tasdgps_.ndet, &nobj, bank,
		       &tasdgps_blen, &tasdgps_maxlen);
  rcode+=dst_unpacki4_(&tasdgps_.first_date, &nobj, bank,
		       &tasdgps_blen, &tasdgps_maxlen);
  rcode+=dst_unpacki4_(&tasdgps_.last_date, &nobj, bank,
		       &tasdgps_blen, &tasdgps_maxlen);

  nobj = tasdgps_nhmax;
  rcode+=dst_unpacki4_(&tasdgps_.first_run_id[0], &nobj, bank,
		       &tasdgps_blen, &tasdgps_maxlen);
  rcode+=dst_unpacki4_(&tasdgps_.last_run_id[0], &nobj, bank,
		       &tasdgps_blen, &tasdgps_maxlen);

  nobj = sizeof(SDGpsSubData)/sizeof(int);
  for(ii=0;ii<tasdgps_.ndet;ii++){
    rcode+=dst_unpacki4_((int*)(&tasdgps_.sub[ii]),&nobj,
			 bank, &tasdgps_blen,
			 &tasdgps_maxlen);
  }

  nobj = 1;
  rcode+=dst_unpacki4_(&tasdgps_.footer, &nobj, bank,
		       &tasdgps_blen, &tasdgps_maxlen);

  return rcode;

}

int tasdgps_common_to_dump_(int *long_output)
{
  return tasdgps_common_to_dumpf_(stdout, long_output);
}

int tasdgps_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
