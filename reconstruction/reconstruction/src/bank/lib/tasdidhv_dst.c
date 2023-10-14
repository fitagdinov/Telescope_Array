/*
 * tasdevent_dst.c
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
#include "tasdidhv_dst.h"

tasdidhv_dst_common tasdidhv_;/* allocate memory to
				 tasdidhv_common */

static integer4 tasdidhv_blen = 0;
static int tasdidhv_maxlen = sizeof(int)*2+sizeof(tasdidhv_dst_common);
static char *tasdidhv_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdidhv_bank_buffer_ (integer4* tasdidhv_bank_buffer_size)
{
  (*tasdidhv_bank_buffer_size) = tasdidhv_blen;
  return tasdidhv_bank;
}



static void tasdidhv_bank_init(void)
{
  tasdidhv_bank = (char *)calloc(tasdidhv_maxlen, sizeof(char));
  if (tasdidhv_bank==NULL)
    {
      fprintf(stderr,
	      "tasdidhv_bank_init: "
	      "fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

int tasdidhv_common_to_bank_(void)
{
  static int id = TASDIDHV_BANKID, ver = TASDIDHV_BANKVERSION;
  int rcode, nobj, ii;

  if (tasdidhv_bank == NULL) tasdidhv_bank_init();

  /* Initialize tasdidhv_blen, and pack the id and version to bank */
  if ( (rcode=dst_initbank_(&id,&ver,&tasdidhv_blen,
			    &tasdidhv_maxlen,tasdidhv_bank)) )
    return rcode;

  nobj = 1;
  rcode += dst_packi4_( &tasdidhv_.ndet,   &nobj, tasdidhv_bank,
			&tasdidhv_blen, &tasdidhv_maxlen);
  rcode += dst_packi4_( &tasdidhv_.site,   &nobj, tasdidhv_bank,
			&tasdidhv_blen, &tasdidhv_maxlen);
  rcode += dst_packi4_( &tasdidhv_.run_id, &nobj, tasdidhv_bank,
			&tasdidhv_blen, &tasdidhv_maxlen);
  rcode += dst_packi4_( &tasdidhv_.year, &nobj, tasdidhv_bank,
			&tasdidhv_blen, &tasdidhv_maxlen);

  for(ii=0;ii<tasdidhv_.ndet;ii++){
    nobj = 1;
    rcode+=dst_packi4_( &tasdidhv_.sub[ii].lid,&nobj,tasdidhv_bank,
			&tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_packi4_( (int*)&tasdidhv_.sub[ii].wlanidmsb,&nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);
    rcode+=dst_packi4_( (int*)&tasdidhv_.sub[ii].wlanidlsb,&nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);
    rcode+=dst_packi4_( &tasdidhv_.sub[ii].error_flag, &nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);

    rcode+=dst_packi2_( &tasdidhv_.sub[ii].trig_mode0, &nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);
    rcode+=dst_packi2_( &tasdidhv_.sub[ii].trig_mode1, &nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);
    rcode+=dst_packi2_( &tasdidhv_.sub[ii].uthre_lvl0, &nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);
    rcode+=dst_packi2_( &tasdidhv_.sub[ii].lthre_lvl0, &nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);
    rcode+=dst_packi2_( &tasdidhv_.sub[ii].uthre_lvl1, &nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);
    rcode+=dst_packi2_( &tasdidhv_.sub[ii].lthre_lvl1, &nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);
    rcode+=dst_packi2_( &tasdidhv_.sub[ii].uhv,	       &nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);
    rcode+=dst_packi2_( &tasdidhv_.sub[ii].lhv,	       &nobj,
			tasdidhv_bank, &tasdidhv_blen,
			&tasdidhv_maxlen);

    nobj = 32;
    rcode += dst_packi1_( &tasdidhv_.sub[ii].firm_version[0],&nobj,
			  tasdidhv_bank, &tasdidhv_blen,
			  &tasdidhv_maxlen);

  }

  return rcode;
}


int tasdidhv_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdidhv_blen, tasdidhv_bank );
}

int tasdidhv_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdidhv_common_to_bank_()) )
    {
      fprintf (stderr,"tasdidhv_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);
    }
  if ( (rcode = tasdidhv_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdidhv_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);
    }
  return SUCCESS;
}

int tasdidhv_bank_to_common_(char *bank)
{
  int rcode = 0;
  int nobj, ii;


  tasdidhv_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rcode += dst_unpacki4_( &tasdidhv_.ndet,   &nobj, bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
  rcode += dst_unpacki4_( &tasdidhv_.site,   &nobj, bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
  rcode += dst_unpacki4_( &tasdidhv_.run_id, &nobj, bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
  rcode += dst_unpacki4_( &tasdidhv_.year, &nobj, bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);

  for(ii=0;ii<tasdidhv_.ndet;ii++){
    nobj = 1;
    rcode+=dst_unpacki4_(&tasdidhv_.sub[ii].lid,       &nobj, bank,
			 &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki4_((int*)&tasdidhv_.sub[ii].wlanidmsb,
			 &nobj, bank,
			 &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki4_((int*)&tasdidhv_.sub[ii].wlanidlsb,
			 &nobj, bank,
			 &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki4_( &tasdidhv_.sub[ii].error_flag,&nobj,bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);

    rcode+=dst_unpacki2_( &tasdidhv_.sub[ii].trig_mode0,&nobj,bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki2_( &tasdidhv_.sub[ii].trig_mode1,&nobj,bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki2_( &tasdidhv_.sub[ii].uthre_lvl0,&nobj,bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki2_( &tasdidhv_.sub[ii].lthre_lvl0,&nobj,bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki2_( &tasdidhv_.sub[ii].uthre_lvl1,&nobj,bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki2_( &tasdidhv_.sub[ii].lthre_lvl1,&nobj,bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki2_( &tasdidhv_.sub[ii].uhv,       &nobj,bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);
    rcode+=dst_unpacki2_( &tasdidhv_.sub[ii].lhv,       &nobj,bank,
			  &tasdidhv_blen, &tasdidhv_maxlen);

    nobj = 32;
    rcode+=dst_unpacki1_( &tasdidhv_.sub[ii].firm_version[0],&nobj,
			  bank, &tasdidhv_blen, &tasdidhv_maxlen);

  }

  return rcode;

}

int tasdidhv_common_to_dump_(int *long_output)
{
  return tasdidhv_common_to_dumpf_(stdout, long_output);
}

int tasdidhv_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
