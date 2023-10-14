/*
 * tasdevent_dst.c
 *
 * C functions for TASDCOND bank
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
#include "tasdcond_dst.h"

tasdcond_dst_common tasdcond_;  /* allocate memory to
				   tasdcond_common */

static integer4 tasdcond_blen = 0;
static int tasdcond_maxlen = sizeof(int)*2+sizeof(tasdcond_dst_common);
static char *tasdcond_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdcond_bank_buffer_ (integer4* tasdcond_bank_buffer_size)
{
  (*tasdcond_bank_buffer_size) = tasdcond_blen;
  return tasdcond_bank;
}



static void tasdcond_bank_init(void)
{
  tasdcond_bank = (char *)calloc(tasdcond_maxlen, sizeof(char));
  if (tasdcond_bank==NULL){
    fprintf(stderr,
	    "tasdcond_bank_init: fail to assign memory to bank."
	    " Abort.\n");
    exit(0);
  }
}

int tasdcond_common_to_bank_(void)
{
  static int id = TASDCOND_BANKID, ver = TASDCOND_BANKVERSION;
  int rcode, nobj, ii;

  if (tasdcond_bank == NULL) tasdcond_bank_init();

  /* Initialize tasdcond_blen, and pack the id and version to
     bank */
  if ( (rcode=dst_initbank_(&id,&ver,&tasdcond_blen,
			    &tasdcond_maxlen,tasdcond_bank)) )
    return rcode;

  nobj = 1;
  rcode += dst_packi4_( &tasdcond_.num_det, &nobj, tasdcond_bank,
			&tasdcond_blen, &tasdcond_maxlen);
  rcode += dst_packi4_( &tasdcond_.date, &nobj, tasdcond_bank,
			&tasdcond_blen, &tasdcond_maxlen);
  rcode += dst_packi4_( &tasdcond_.time, &nobj, tasdcond_bank,
			&tasdcond_blen, &tasdcond_maxlen);
  nobj = 600;
  rcode += dst_packi1_( &tasdcond_.trgMode[0],&nobj, tasdcond_bank,
			&tasdcond_blen, &tasdcond_maxlen);
  for(ii=0;ii<tasdcond_nhmax;ii++){
    nobj = 1;
    rcode += dst_packi4_(&tasdcond_.host[ii].numTrg,
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    nobj = tasdcond_.host[ii].numTrg;
    rcode += dst_packi4_(&tasdcond_.host[ii].trgBank[0],
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packi4_(&tasdcond_.host[ii].trgSec[0],
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packi2_(&tasdcond_.host[ii].trgPos[0],
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packi4_(&tasdcond_.host[ii].daqMode[0],
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    nobj = 600;
    rcode += dst_packi1_(&tasdcond_.host[ii].miss[0],
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packi1_(&tasdcond_.host[ii].gpsError[0],
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packi2_(&tasdcond_.host[ii].run_id[0],
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
  }
  for(ii=0;ii<tasdcond_.num_det;ii++){
    nobj = 10;
    rcode += dst_packi4_(&tasdcond_.sub[ii].slowCond[0],
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    nobj = 1;
    rcode += dst_packr4_(&tasdcond_.sub[ii].clockFreq,
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packr4_(&tasdcond_.sub[ii].clockChirp,
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packr4_(&tasdcond_.sub[ii].clockError,
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    nobj = 600;
    rcode += dst_packi1_(&tasdcond_.sub[ii].miss[0],
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    nobj = 1;
    rcode += dst_packi2_(&tasdcond_.sub[ii].site,
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packi2_(&tasdcond_.sub[ii].lid,
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packi1_(&tasdcond_.sub[ii].gpsHealth,
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_packi1_(&tasdcond_.sub[ii].gpsRunMode,
			 &nobj,tasdcond_bank,
			 &tasdcond_blen, &tasdcond_maxlen);
  }
  nobj = 1;
  rcode += dst_packi4_( &tasdcond_.footer, &nobj, tasdcond_bank,
			&tasdcond_blen, &tasdcond_maxlen);


  return rcode;
}


int tasdcond_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdcond_blen, tasdcond_bank );
}


int tasdcond_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdcond_common_to_bank_()) )
    {
      fprintf (stderr,"tasdcond_common_to_bank_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  if ( (rcode = tasdcond_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdcond_bank_to_dst_ ERROR : %ld\n",
	       (long) rcode);
      exit(0);
    }
  return SUCCESS;
}


int tasdcond_bank_to_common_(char *bank)
{
  int rcode = 0;
  int nobj, ii;
  tasdcond_blen = 2 * sizeof(int); /* skip id and version  */
  nobj = 1;
  rcode += dst_unpacki4_( &tasdcond_.num_det, &nobj, bank,
			  &tasdcond_blen, &tasdcond_maxlen);
  rcode += dst_unpacki4_( &tasdcond_.date, &nobj, bank,
			  &tasdcond_blen, &tasdcond_maxlen);
  rcode += dst_unpacki4_( &tasdcond_.time, &nobj, bank,
			  &tasdcond_blen, &tasdcond_maxlen);
  nobj = 600;
  rcode += dst_unpacki1_( &tasdcond_.trgMode[0],&nobj, bank,
			  &tasdcond_blen, &tasdcond_maxlen);
  for(ii=0;ii<tasdcond_nhmax;ii++){
    nobj = 1;
    rcode += dst_unpacki4_(&tasdcond_.host[ii].numTrg,&nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    nobj = tasdcond_.host[ii].numTrg;
    rcode += dst_unpacki4_(&tasdcond_.host[ii].trgBank[0],
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpacki4_(&tasdcond_.host[ii].trgSec[0],
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpacki2_(&tasdcond_.host[ii].trgPos[0],
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpacki4_(&tasdcond_.host[ii].daqMode[0],
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    nobj = 600;
    rcode += dst_unpacki1_(&tasdcond_.host[ii].miss[0],
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpacki1_(&tasdcond_.host[ii].gpsError[0],
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpacki2_(&tasdcond_.host[ii].run_id[0],
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
  }
  for(ii=0;ii<tasdcond_.num_det;ii++){
    nobj = 10;
    rcode += dst_unpacki4_(&tasdcond_.sub[ii].slowCond[0],
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    nobj = 1;
    rcode += dst_unpackr4_(&tasdcond_.sub[ii].clockFreq,
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpackr4_(&tasdcond_.sub[ii].clockChirp,
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpackr4_(&tasdcond_.sub[ii].clockError,
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    nobj = 600;
    rcode += dst_unpacki1_(&tasdcond_.sub[ii].miss[0],
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    nobj = 1;
    rcode += dst_unpacki2_(&tasdcond_.sub[ii].site,
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpacki2_(&tasdcond_.sub[ii].lid,
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpacki1_(&tasdcond_.sub[ii].gpsHealth,
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
    rcode += dst_unpacki1_(&tasdcond_.sub[ii].gpsRunMode,
			   &nobj,bank,
			   &tasdcond_blen, &tasdcond_maxlen);
  }
  nobj = 1;
  rcode += dst_unpacki4_( &tasdcond_.footer, &nobj, bank,
			  &tasdcond_blen, &tasdcond_maxlen);
  return rcode;
}


int tasdcond_common_to_dump_(int *long_output)
{
  return tasdcond_common_to_dumpf_(stdout, long_output);
}


int tasdcond_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
