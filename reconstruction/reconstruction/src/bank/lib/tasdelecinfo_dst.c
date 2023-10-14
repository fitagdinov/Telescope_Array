/*
 * tasdelecinfo_dst.c
 *
 * C functions for TASDELECINFO bank
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
#include "tasdelecinfo_dst.h"

tasdelecinfo_dst_common tasdelecinfo_; /* allocate memory to
					  tasdelecinfo_common */

static integer4 tasdelecinfo_blen = 0;
static int tasdelecinfo_maxlen =
  sizeof(int)*2+sizeof(tasdelecinfo_dst_common);
static char *tasdelecinfo_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdelecinfo_bank_buffer_ (integer4* tasdelecinfo_bank_buffer_size)
{
  (*tasdelecinfo_bank_buffer_size) = tasdelecinfo_blen;
  return tasdelecinfo_bank;
}



static void tasdelecinfo_bank_init(void)
{
  tasdelecinfo_bank =
    (char*)calloc(tasdelecinfo_maxlen,sizeof(char));
  if (tasdelecinfo_bank==NULL){
    fprintf(stderr,
	    "tasdelecinfo_bank_init: "
	    "fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

int tasdelecinfo_common_to_bank_(void)
{
  static int id = TASDELECINFO_BANKID, ver = TASDELECINFO_BANKVERSION;
  int rcode, nobj, ii;

  if (tasdelecinfo_bank == NULL) tasdelecinfo_bank_init();

  /* Initialize tasdelecinfo_blen, and pack the id and version to bank */
  if((rcode=dst_initbank_(&id,&ver,&tasdelecinfo_blen,
			  &tasdelecinfo_maxlen,tasdelecinfo_bank)))
    return rcode;

  nobj = 1;
  rcode += dst_packi4_(&tasdelecinfo_.ndet,&nobj,tasdelecinfo_bank,
		       &tasdelecinfo_blen, &tasdelecinfo_maxlen);

  for(ii=0;ii<tasdelecinfo_.ndet;ii++){
    nobj = 1;
    rcode += dst_packi4_((int*)&tasdelecinfo_.sub[ii].wlanidmsb,
			 &nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode += dst_packi4_((int*)&tasdelecinfo_.sub[ii].wlanidlsb,
			 &nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode += dst_packi4_(&tasdelecinfo_.sub[ii].ccid,     &nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode += dst_packi4_(&tasdelecinfo_.sub[ii].error_flag,&nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);

    rcode += dst_packr4_(&tasdelecinfo_.sub[ii].uoffset,  &nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode += dst_packr4_(&tasdelecinfo_.sub[ii].loffset,  &nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode += dst_packr4_(&tasdelecinfo_.sub[ii].uslope,   &nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode += dst_packr4_(&tasdelecinfo_.sub[ii].lslope,   &nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);

    nobj = 8;
    rcode += dst_packi1_(&tasdelecinfo_.sub[ii].elecid[0],&nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode += dst_packi1_(&tasdelecinfo_.sub[ii].gpsid[0], &nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode += dst_packi1_(&tasdelecinfo_.sub[ii].cpldver[0],&nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode += dst_packi1_(&tasdelecinfo_.sub[ii].ccver[0], &nobj,
			 tasdelecinfo_bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);

  }

  return rcode;
}


int tasdelecinfo_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdelecinfo_blen, tasdelecinfo_bank );
}

int tasdelecinfo_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdelecinfo_common_to_bank_()) )
    {
      fprintf (stderr,"tasdelecinfo_common_to_bank_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  if ( (rcode = tasdelecinfo_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdelecinfo_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);
    }
  return SUCCESS;
}

int tasdelecinfo_bank_to_common_(char *bank)
{
  int rcode = 0;
  int nobj, ii;

  tasdelecinfo_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rcode+=dst_unpacki4_(&tasdelecinfo_.ndet, &nobj, bank,
			 &tasdelecinfo_blen, &tasdelecinfo_maxlen);

  for(ii=0;ii<tasdelecinfo_.ndet;ii++){
    nobj = 1;
    rcode+=dst_unpacki4_((int*)&tasdelecinfo_.sub[ii].wlanidmsb,
			 &nobj,bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode+=dst_unpacki4_((int*)&tasdelecinfo_.sub[ii].wlanidlsb,
			 &nobj,bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode+=dst_unpacki4_(&tasdelecinfo_.sub[ii].ccid, &nobj,bank,
			 &tasdelecinfo_blen, &tasdelecinfo_maxlen);
    rcode+=dst_unpacki4_(&tasdelecinfo_.sub[ii].error_flag,&nobj,
			 bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);

    rcode+=dst_unpackr4_(&tasdelecinfo_.sub[ii].uoffset,&nobj,bank,
			 &tasdelecinfo_blen, &tasdelecinfo_maxlen);
    rcode+=dst_unpackr4_(&tasdelecinfo_.sub[ii].loffset,&nobj,bank,
			 &tasdelecinfo_blen, &tasdelecinfo_maxlen);
    rcode+=dst_unpackr4_(&tasdelecinfo_.sub[ii].uslope, &nobj,bank,
			 &tasdelecinfo_blen, &tasdelecinfo_maxlen);
    rcode+=dst_unpackr4_(&tasdelecinfo_.sub[ii].lslope, &nobj,bank,
			 &tasdelecinfo_blen, &tasdelecinfo_maxlen);

    nobj = 8;
    rcode+=dst_unpacki1_(&tasdelecinfo_.sub[ii].elecid[0],&nobj,
			 bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode+=dst_unpacki1_(&tasdelecinfo_.sub[ii].gpsid[0], &nobj,
			 bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode+=dst_unpacki1_(&tasdelecinfo_.sub[ii].cpldver[0],&nobj,
			 bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);
    rcode+=dst_unpacki1_(&tasdelecinfo_.sub[ii].ccver[0],  &nobj,
			 bank, &tasdelecinfo_blen,
			 &tasdelecinfo_maxlen);

  }

  return rcode;

}

int tasdelecinfo_common_to_dump_(int *long_output)
{
  return tasdelecinfo_common_to_dumpf_(stdout, long_output);
}

int tasdelecinfo_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
