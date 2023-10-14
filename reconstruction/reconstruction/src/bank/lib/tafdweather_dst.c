/*
 * tafdweather_dst.c
 *
 * C functions for TAFDWEATHER bank
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
#include "tafdweather_dst.h"

tafdweather_dst_common tafdweather_; /* allocate memory to
					  tafdweather_common */

static integer4 tafdweather_blen = 0;
static int tafdweather_maxlen = sizeof(int)*2+sizeof(tafdweather_dst_common);
static char *tafdweather_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tafdweather_bank_buffer_ (integer4* tafdweather_bank_buffer_size)
{
  (*tafdweather_bank_buffer_size) = tafdweather_blen;
  return tafdweather_bank;
}



static void tafdweather_bank_init(void)
{
  tafdweather_bank=(char *)calloc(tafdweather_maxlen,sizeof(char));
  if (tafdweather_bank==NULL){
    fprintf(stderr,
	    "tafdweather_bank_init: fail to assign memory to bank."
	    " Abort.\n");
    exit(0);
  }
}

int tafdweather_common_to_bank_(void)
{
  static int id = TAFDWEATHER_BANKID, ver=TAFDWEATHER_BANKVERSION;
  int rc, nobj, ii, jj;

  if (tafdweather_bank == NULL) tafdweather_bank_init();

  /* Initialize tafdweather_blen, and pack the id and version
     to bank */
  if( (rc=dst_initbank_(&id,&ver,&tafdweather_blen,
			   &tafdweather_maxlen,tafdweather_bank)) )
    return rc;

  nobj = 1;
  rc+=dst_packi4_(&tafdweather_.nsite,&nobj,tafdweather_bank,
		  &tafdweather_blen, &tafdweather_maxlen);
  rc+=dst_packi4_(&tafdweather_.date,&nobj,tafdweather_bank,
		  &tafdweather_blen, &tafdweather_maxlen);

  for(ii=0;ii<tafdweather_.nsite;ii++){
    rc += dst_packi4_(&tafdweather_.st[ii].site, &nobj,
		      tafdweather_bank, &tafdweather_blen,
		      &tafdweather_maxlen);
    rc += dst_packi4_(&tafdweather_.st[ii].num_data, &nobj,
		      tafdweather_bank, &tafdweather_blen,
		      &tafdweather_maxlen);
    if(tafdweather_.st[ii].num_data>0){
      for(jj=0;jj<tafdweather_npmax;jj++){
	FDWeather10mData *ptr = &tafdweather_.st[ii].data[jj];
	rc+=dst_packi4_(&ptr->timeFrom,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packi4_(&ptr->timeTo,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packr4_(&ptr->averageWindSpeed,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packr4_(&ptr->maximumWindSpeed,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packr4_(&ptr->windDirection,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packr4_(&ptr->atmosphericPressure,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packr4_(&ptr->temperature,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packr4_(&ptr->humidity,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packr4_(&ptr->totalRainfall,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packr4_(&ptr->rainfall,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
	rc+=dst_packr4_(&ptr->numberOfHails,
			&nobj,tafdweather_bank,&tafdweather_blen,
			&tafdweather_maxlen);
      }
    }
  }

  nobj = 1;
  rc += dst_packi4_(&tafdweather_.footer,&nobj,tafdweather_bank,
		       &tafdweather_blen, &tafdweather_maxlen);
  return rc;
}


int tafdweather_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tafdweather_blen,
			 tafdweather_bank );
}

int tafdweather_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tafdweather_common_to_bank_()) )
    {
      fprintf (stderr,"tafdweather_common_to_bank_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  if ( (rcode = tafdweather_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tafdweather_bank_to_dst_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  return SUCCESS;
}

int tafdweather_bank_to_common_(char *bank)
{
  int rc = 0;
  int nobj, ii, jj;

  tafdweather_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rc+=dst_unpacki4_(&tafdweather_.nsite,&nobj,bank,
		    &tafdweather_blen, &tafdweather_maxlen);
  rc+=dst_unpacki4_(&tafdweather_.date,&nobj,bank,
		    &tafdweather_blen, &tafdweather_maxlen);

  for(ii=0;ii<tafdweather_.nsite;ii++){
    rc += dst_unpacki4_(&tafdweather_.st[ii].site, &nobj,
			bank, &tafdweather_blen,
			&tafdweather_maxlen);
    rc += dst_unpacki4_(&tafdweather_.st[ii].num_data, &nobj,
			bank, &tafdweather_blen,
			&tafdweather_maxlen);
    if(tafdweather_.st[ii].num_data>0){
      for(jj=0;jj<tafdweather_npmax;jj++){
	FDWeather10mData *ptr = &tafdweather_.st[ii].data[jj];
	rc+=dst_unpacki4_(&ptr->timeFrom,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpacki4_(&ptr->timeTo,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpackr4_(&ptr->averageWindSpeed,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpackr4_(&ptr->maximumWindSpeed,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpackr4_(&ptr->windDirection,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpackr4_(&ptr->atmosphericPressure,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpackr4_(&ptr->temperature,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpackr4_(&ptr->humidity,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpackr4_(&ptr->totalRainfall,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpackr4_(&ptr->rainfall,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
	rc+=dst_unpackr4_(&ptr->numberOfHails,
			  &nobj,bank,&tafdweather_blen,
			  &tafdweather_maxlen);
      }
    }
  }


  nobj = 1;
  rc+=dst_unpacki4_(&tafdweather_.footer, &nobj, bank,
		    &tafdweather_blen, &tafdweather_maxlen);

  return rc;

}

int tafdweather_common_to_dump_(int *long_output)
{
  return tafdweather_common_to_dumpf_(stdout, long_output);
}

int tafdweather_common_to_dumpf_(FILE* fp, int *long_output)
{
  /*
  int ii;
  fprintf(fp,"#time windSpeed(avr) windSpeed(max)  "
  for(ii=0;ii<144;ii++){
  }
  */
  (void)(fp);
  (void)(long_output);
  return 0;
}
