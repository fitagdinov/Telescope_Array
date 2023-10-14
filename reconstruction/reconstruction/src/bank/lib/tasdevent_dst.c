/*
 * tasdevent_dst.c
 *
 * C functions for TASDEVENT bank
 * J. Belz - 20080516
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
#include "tasdevent_dst.h"

tasdevent_dst_common tasdevent_;  /* allocate memory to
				     tasdevent_common */

static integer4 tasdevent_blen = 0;
static int tasdevent_maxlen = sizeof(int)*2+sizeof(tasdevent_dst_common);
static char *tasdevent_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdevent_bank_buffer_ (integer4* tasdevent_bank_buffer_size)
{
  (*tasdevent_bank_buffer_size) = tasdevent_blen;
  return tasdevent_bank;
}



static void tasdevent_bank_init(void)
{
  tasdevent_bank = (char *)calloc(tasdevent_maxlen, sizeof(char));
  if (tasdevent_bank==NULL)
    {
      fprintf(stderr,
	      "tasdevent_bank_init: fail to assign memory to bank."
	      " Abort.\n");
      exit(0);
    }
}

int tasdevent_common_to_bank_(void)
{
  static int id = TASDEVENT_BANKID, ver = TASDEVENT_BANKVERSION;
  int rcode, nobj, ii;

  if (tasdevent_bank == NULL) tasdevent_bank_init();

  /* Initialize tasdevent_blen, and pack the id and version
     to bank */
  if ( (rcode=dst_initbank_(&id,&ver,&tasdevent_blen,
			    &tasdevent_maxlen,tasdevent_bank)) )
    return rcode;

  /* integers */
  nobj = 1;
  rcode+=dst_packi4_(&tasdevent_.event_code, &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.run_id,     &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.site,       &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.trig_id,    &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);

  rcode+=dst_packi4_(&tasdevent_.trig_code,  &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.code,  &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.num_trgwf,  &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.num_wf,     &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);

  rcode+=dst_packi4_(&tasdevent_.bank,       &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.date,       &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.time,       &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.date_org,   &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.time_org,   &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.usec,       &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.gps_error,  &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_packi4_(&tasdevent_.pos,	     &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);

  /* 1D integer arrays */
  nobj = 16 ;
  rcode+=dst_packi4_(&tasdevent_.pattern[0], &nobj, tasdevent_bank,
		     &tasdevent_blen, &tasdevent_maxlen);

  /* 2D integer arrays */
  if(tasdevent_.num_trgwf>tasdevent_ndmax){
    tasdevent_.num_trgwf=tasdevent_ndmax;
  }
  for(ii=0;ii<tasdevent_.num_trgwf;ii++){
    nobj = 1;
    rcode+=dst_packi4_( &tasdevent_.sub[ii].clock, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi4_( &tasdevent_.sub[ii].max_clock, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].lid, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].usum, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].lsum, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].uavr, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].lavr, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].wf_id, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].num_trgwf, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].bank, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].num_retry, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].trig_code, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].wf_error, &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    nobj = tasdevent_nfadc;
    rcode+=dst_packi2_( &tasdevent_.sub[ii].uwf[0], &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
    rcode+=dst_packi2_( &tasdevent_.sub[ii].lwf[0], &nobj,
			tasdevent_bank, &tasdevent_blen,
			&tasdevent_maxlen);
  }

  return rcode;
}


int tasdevent_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdevent_blen, tasdevent_bank);
}

int tasdevent_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdevent_common_to_bank_()) )
    {
      fprintf (stderr,"tasdevent_common_to_bank_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  if ( (rcode = tasdevent_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdevent_bank_to_dst_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  return SUCCESS;
}

int tasdevent_bank_to_common_(char *bank)
{
  int rcode = 0;
  int nobj, ii;

  tasdevent_blen = 2 * sizeof(int); /* skip id and version  */

  /* integers */
  nobj = 1;
  rcode+=dst_unpacki4_(&tasdevent_.event_code, &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.run_id,     &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.site,       &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.trig_id,    &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);

  rcode+=dst_unpacki4_(&tasdevent_.trig_code,  &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.code,  &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.num_trgwf,  &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.num_wf,     &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);

  rcode+=dst_unpacki4_(&tasdevent_.bank,       &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.date,       &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.time,       &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.date_org,   &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.time_org,   &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.usec,       &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.gps_error,  &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);
  rcode+=dst_unpacki4_(&tasdevent_.pos,	     &nobj, bank,
		       &tasdevent_blen, &tasdevent_maxlen);


  /* 1D integer arrays */
  nobj = 16 ;
  rcode+=dst_unpacki4_( &tasdevent_.pattern[0], &nobj, bank,
			&tasdevent_blen, &tasdevent_maxlen);

  if(tasdevent_.num_trgwf>tasdevent_ndmax){
    tasdevent_.num_trgwf=tasdevent_ndmax;
  }

  /* 2D integer arrays */
  for(ii=0;ii<tasdevent_.num_trgwf;ii++){
    nobj = 1;
    rcode+=dst_unpacki4_( &tasdevent_.sub[ii].clock, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki4_( &tasdevent_.sub[ii].max_clock, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].lid, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].usum, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].lsum, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].uavr, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].lavr, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].wf_id, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].num_trgwf, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].bank, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].num_retry, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].trig_code, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].wf_error, &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    nobj = tasdevent_nfadc;
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].uwf[0], &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
    rcode+=dst_unpacki2_( &tasdevent_.sub[ii].lwf[0], &nobj,
			  bank, &tasdevent_blen,
			  &tasdevent_maxlen);
  }

  return rcode;

}

int tasdevent_common_to_dump_(int *long_output)
{
  return tasdevent_common_to_dumpf_(stdout, long_output);
}

int tasdevent_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
