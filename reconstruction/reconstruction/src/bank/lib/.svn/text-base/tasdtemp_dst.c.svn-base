/*
 * tasdevent_dst.c
 *
 * C functions for TASDTEMP bank
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
#include "tasdtemp_dst.h"

tasdtemp_dst_common tasdtemp_;  /* allocate memory to tasdtemp_common */

static integer4 tasdtemp_blen = 0;
static int tasdtemp_maxlen = sizeof(int)*2+sizeof(tasdtemp_dst_common);
static char *tasdtemp_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdtemp_bank_buffer_ (integer4* tasdtemp_bank_buffer_size)
{
  (*tasdtemp_bank_buffer_size) = tasdtemp_blen;
  return tasdtemp_bank;
}



static void tasdtemp_bank_init(void)
{
  tasdtemp_bank = (char *)calloc(tasdtemp_maxlen, sizeof(char));
  if (tasdtemp_bank==NULL)
    {
      fprintf(stderr,
	      "tasdtemp_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

int tasdtemp_common_to_bank_(void)
{
  static int id = TASDTEMP_BANKID, ver = TASDTEMP_BANKVERSION;
  int rcode, nobj, ii;

  if (tasdtemp_bank == NULL) tasdtemp_bank_init();

  /* Initialize tasdtemp_blen, and pack the id and version to bank */
  if ( (rcode=dst_initbank_(&id,&ver,&tasdtemp_blen,&tasdtemp_maxlen,tasdtemp_bank)) )
    return rcode;

  nobj = 1;
  rcode += dst_packi4_(&tasdtemp_.num_det,&nobj,tasdtemp_bank,
		       &tasdtemp_blen, &tasdtemp_maxlen);
  rcode += dst_packi4_(&tasdtemp_.date,   &nobj,tasdtemp_bank,
		       &tasdtemp_blen, &tasdtemp_maxlen);
  rcode += dst_packi4_(&tasdtemp_.time,   &nobj,tasdtemp_bank,
		       &tasdtemp_blen, &tasdtemp_maxlen);

  nobj = sizeof(SDTempData)/sizeof(int);
  for(ii=0;ii<tasdtemp_.num_det;ii++){
    rcode += dst_packi4_((int*)(&tasdtemp_.sub[ii]), &nobj,
			 tasdtemp_bank,
			 &tasdtemp_blen, &tasdtemp_maxlen);
  }

  nobj = 1;
  rcode += dst_packi4_(&tasdtemp_.footer,&nobj,tasdtemp_bank,
		       &tasdtemp_blen, &tasdtemp_maxlen);


  return rcode;
}


int tasdtemp_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdtemp_blen, tasdtemp_bank );
}

int tasdtemp_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdtemp_common_to_bank_()) )
    {
      fprintf (stderr,"tasdtemp_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);
    }
  if ( (rcode = tasdtemp_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdtemp_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);
    }
  return SUCCESS;
}

int tasdtemp_bank_to_common_(char *bank)
{
  int rcode = 0;
  int nobj, ii;

  tasdtemp_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rcode += dst_unpacki4_( &tasdtemp_.num_det, &nobj, bank,
			  &tasdtemp_blen, &tasdtemp_maxlen);
  rcode += dst_unpacki4_( &tasdtemp_.date, &nobj, bank,
			  &tasdtemp_blen, &tasdtemp_maxlen);
  rcode += dst_unpacki4_( &tasdtemp_.time, &nobj, bank,
			  &tasdtemp_blen, &tasdtemp_maxlen);

  nobj = sizeof(SDTempData)/sizeof(int);
  for(ii=0;ii<tasdtemp_.num_det;ii++){
    rcode += dst_unpacki4_((int*)(&tasdtemp_.sub[ii]), &nobj, bank,
			   &tasdtemp_blen, &tasdtemp_maxlen);
  }

  nobj = 1;
  rcode += dst_unpacki4_( &tasdtemp_.footer, &nobj, bank,
			  &tasdtemp_blen, &tasdtemp_maxlen);

  return rcode;

}

int tasdtemp_common_to_dump_(int *long_output)
{
  return tasdtemp_common_to_dumpf_(stdout, long_output);
}

int tasdtemp_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
