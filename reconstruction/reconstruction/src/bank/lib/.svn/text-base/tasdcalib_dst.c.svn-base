/*
 * tasdcalib_dst.c
 *
 * C functions for TASDCALIB bank
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
#include "tasdcalib_dst.h"

tasdcalib_dst_common tasdcalib_;  /* allocate memory to
				   tasdcalib_common */

static integer4 tasdcalib_blen = 0;
static int tasdcalib_maxlen = sizeof(int)*2+sizeof(tasdcalib_dst_common);
static char *tasdcalib_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdcalib_bank_buffer_ (integer4* tasdcalib_bank_buffer_size)
{
  (*tasdcalib_bank_buffer_size) = tasdcalib_blen;
  return tasdcalib_bank;
}



static void tasdcalib_bank_init(void)
{
  tasdcalib_bank = (char *)calloc(tasdcalib_maxlen, sizeof(char));
  if (tasdcalib_bank==NULL){
    fprintf(stderr,
	    "tasdcalib_bank_init: fail to assign memory to bank."
	    " Abort.\n");
    exit(0);
  }
}

int tasdcalib_common_to_bank_(void)
{
  static int id = TASDCALIB_BANKID, ver = TASDCALIB_BANKVERSION;
  int rc, nobj, ii, jj;

  if (tasdcalib_bank == NULL) tasdcalib_bank_init();

  /* Initialize tasdcalib_blen, and pack the id and version to
     bank */
  if ( (rc=dst_initbank_(&id,&ver,&tasdcalib_blen,&tasdcalib_maxlen,tasdcalib_bank)) )
    return rc;

  nobj = 1;
  rc+=dst_packi4_(&tasdcalib_.num_host, &nobj,tasdcalib_bank,&tasdcalib_blen,&tasdcalib_maxlen);
  rc+=dst_packi4_(&tasdcalib_.num_det,  &nobj,tasdcalib_bank,&tasdcalib_blen,&tasdcalib_maxlen);
  rc+=dst_packi4_(&tasdcalib_.num_weather,
		  &nobj,tasdcalib_bank,&tasdcalib_blen,&tasdcalib_maxlen);
  rc+=dst_packi4_(&tasdcalib_.date,     &nobj,tasdcalib_bank,&tasdcalib_blen,&tasdcalib_maxlen);
  rc+=dst_packi4_(&tasdcalib_.time,     &nobj,tasdcalib_bank,&tasdcalib_blen,&tasdcalib_maxlen);
  nobj = 600;
  rc+=dst_packi1_(&tasdcalib_.trgMode[0],
		  &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);

  for(ii=0;ii<tasdcalib_.num_host;ii++){
    nobj = 1;
    rc+=dst_packi4_(&tasdcalib_.host[ii].site,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.host[ii].numTrg,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    for(jj=0;jj<tasdcalib_.host[ii].numTrg;jj++){
      rc+=dst_packi4_(&tasdcalib_.host[ii].trgBank[jj],
		      &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
      rc+=dst_packi4_(&tasdcalib_.host[ii].trgSec[jj],
		      &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
      rc+=dst_packi2_(&tasdcalib_.host[ii].trgPos[jj],
		      &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
      rc+=dst_packi4_(&tasdcalib_.host[ii].daqMode[jj],
		      &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    }
    nobj = 600;
    rc+=dst_packi1_(&tasdcalib_.host[ii].miss[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi2_(&tasdcalib_.host[ii].run_id[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
  }

  for(ii=0;ii<tasdcalib_.num_det;ii++){
    nobj = 1;
    rc+=dst_packi4_(&tasdcalib_.sub[ii].site,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.sub[ii].lid,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.sub[ii].livetime,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.sub[ii].warning,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi1_(&tasdcalib_.sub[ii].dontUse,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi1_(&tasdcalib_.sub[ii].dataQuality,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi1_(&tasdcalib_.sub[ii].gpsRunMode,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    nobj = 75;
    rc+=dst_packi1_(&tasdcalib_.sub[ii].miss[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    nobj = 1;
    rc+=dst_packr4_(&tasdcalib_.sub[ii].clockFreq,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].clockChirp,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].clockError,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].upedAvr,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].lpedAvr,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].upedStdev,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].lpedStdev,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].upedChisq,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].lpedChisq,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].umipNonuni,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].lmipNonuni,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].umipMev2cnt,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].lmipMev2cnt,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].umipMev2pe,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].lmipMev2pe,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].umipChisq,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].lmipChisq,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].lvl0Rate,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].lvl1Rate,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].scinti_temp,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);

    nobj = 2;
    rc+=dst_packi4_(&tasdcalib_.sub[ii].pchmip[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.sub[ii].pchped[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.sub[ii].lhpchmip[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.sub[ii].lhpchped[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.sub[ii].rhpchmip[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.sub[ii].rhpchped[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packi4_(&tasdcalib_.sub[ii].mftndof[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].mip[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].mftchi2[0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    nobj = 4;
    rc+=dst_packr4_(&tasdcalib_.sub[ii].mftp[0][0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].mftp[1][0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].mftpe[0][0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.sub[ii].mftpe[1][0],
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);

  }

  nobj = 1;
  for(ii=0;ii<tasdcalib_.num_weather;ii++){
    rc+=dst_packi4_(&tasdcalib_.weather[ii].site,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.weather[ii].averageWindSpeed,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.weather[ii].maximumWindSpeed,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.weather[ii].windDirection,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.weather[ii].atmosphericPressure,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.weather[ii].temperature,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.weather[ii].humidity,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.weather[ii].rainfall,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_packr4_(&tasdcalib_.weather[ii].numberOfHails,
		    &nobj,tasdcalib_bank,&tasdcalib_blen, &tasdcalib_maxlen);
  }

  nobj = 1;
  rc+=dst_packi4_(&tasdcalib_.footer,&nobj,tasdcalib_bank,&tasdcalib_blen,&tasdcalib_maxlen);

  return rc;
}


int tasdcalib_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdcalib_blen, tasdcalib_bank );
}


int tasdcalib_common_to_dst_(int *NumUnit)
{
  int rc;
  if ( (rc = tasdcalib_common_to_bank_()) )
    {
      fprintf (stderr,"tasdcalib_common_to_bank_ ERROR : %ld\n",
	       (long) rc);
      exit(0);
    }
  if ( (rc = tasdcalib_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdcalib_bank_to_dst_ ERROR : %ld\n",
	       (long) rc);
      exit(0);
    }
  return SUCCESS;
}


int tasdcalib_bank_to_common_(char *bank)
{
  int rc = 0;
  int nobj, ii, jj;
  tasdcalib_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rc+=dst_unpacki4_(&tasdcalib_.num_host,&nobj,bank,&tasdcalib_blen,&tasdcalib_maxlen);
  rc+=dst_unpacki4_(&tasdcalib_.num_det, &nobj,bank,&tasdcalib_blen,&tasdcalib_maxlen);
  rc+=dst_unpacki4_(&tasdcalib_.num_weather,
		    &nobj,bank,&tasdcalib_blen,&tasdcalib_maxlen);
  rc+=dst_unpacki4_(&tasdcalib_.date,    &nobj,bank,&tasdcalib_blen,&tasdcalib_maxlen);
  rc+=dst_unpacki4_(&tasdcalib_.time,    &nobj,bank,&tasdcalib_blen,&tasdcalib_maxlen);
  nobj = 600;
  rc+=dst_unpacki1_(&tasdcalib_.trgMode[0],
		    &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);

  for(ii=0;ii<tasdcalib_.num_host;ii++){
    nobj = 1;
    rc+=dst_unpacki4_(&tasdcalib_.host[ii].site,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.host[ii].numTrg,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    for(jj=0;jj<tasdcalib_.host[ii].numTrg;jj++){
      rc+=dst_unpacki4_(&tasdcalib_.host[ii].trgBank[jj],
			&nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
      rc+=dst_unpacki4_(&tasdcalib_.host[ii].trgSec[jj],
			&nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
      rc+=dst_unpacki2_(&tasdcalib_.host[ii].trgPos[jj],
			&nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
      rc+=dst_unpacki4_(&tasdcalib_.host[ii].daqMode[jj],
			&nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    }
    nobj = 600;
    rc+=dst_unpacki1_(&tasdcalib_.host[ii].miss[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki2_(&tasdcalib_.host[ii].run_id[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
  }

  for(ii=0;ii<tasdcalib_.num_det;ii++){
    nobj = 1;
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].site,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].lid,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].livetime,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].warning,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki1_(&tasdcalib_.sub[ii].dontUse,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki1_(&tasdcalib_.sub[ii].dataQuality,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki1_(&tasdcalib_.sub[ii].gpsRunMode,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    nobj = 75;
    rc+=dst_unpacki1_(&tasdcalib_.sub[ii].miss[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    nobj = 1;
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].clockFreq,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].clockChirp,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].clockError,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].upedAvr,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].lpedAvr,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].upedStdev,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].lpedStdev,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].upedChisq,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].lpedChisq,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].umipNonuni,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].lmipNonuni,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].umipMev2cnt,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].lmipMev2cnt,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].umipMev2pe,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].lmipMev2pe,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].umipChisq,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].lmipChisq,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].lvl0Rate,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].lvl1Rate,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].scinti_temp,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);


    nobj = 2;
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].pchmip[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].pchped[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].lhpchmip[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].lhpchped[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].rhpchmip[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].rhpchped[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpacki4_(&tasdcalib_.sub[ii].mftndof[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].mip[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].mftchi2[0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    nobj = 4;
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].mftp[0][0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].mftp[1][0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].mftpe[0][0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.sub[ii].mftpe[1][0],
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);

  }

  nobj = 1;
  for(ii=0;ii<tasdcalib_.num_weather;ii++){
    rc+=dst_unpacki4_(&tasdcalib_.weather[ii].site,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.weather[ii].averageWindSpeed,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.weather[ii].maximumWindSpeed,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.weather[ii].windDirection,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.weather[ii].atmosphericPressure,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.weather[ii].temperature,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.weather[ii].humidity,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.weather[ii].rainfall,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
    rc+=dst_unpackr4_(&tasdcalib_.weather[ii].numberOfHails,
		      &nobj,bank,&tasdcalib_blen, &tasdcalib_maxlen);
  }

  nobj = 1;
  rc+=dst_unpacki4_(&tasdcalib_.footer,&nobj,bank,&tasdcalib_blen,&tasdcalib_maxlen);

  return rc;
}


int tasdcalib_common_to_dump_(int *long_output)
{
  return tasdcalib_common_to_dumpf_(stdout, long_output);
}


int tasdcalib_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
