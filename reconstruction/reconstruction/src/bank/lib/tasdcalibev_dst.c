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
#include "tasdcond_dst.h"
#include "tasdinfo_dst.h"
#include "tasdpedmip_dst.h"
#include "tasdtemp_dst.h"
#include "tasdcalibev_dst.h"

tasdcalibev_dst_common tasdcalibev_;  /* allocate memory to
					 tasdcalibev_common */

static integer4 tasdcalibev_blen = 0;
static int tasdcalibev_maxlen = sizeof(int)*2+sizeof(tasdcalibev_dst_common);
static char *tasdcalibev_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdcalibev_bank_buffer_ (integer4* tasdcalibev_bank_buffer_size)
{
  (*tasdcalibev_bank_buffer_size) = tasdcalibev_blen;
  return tasdcalibev_bank;
}



static void tasdcalibev_bank_init(void)
{
  tasdcalibev_bank = (char *)calloc(tasdcalibev_maxlen, sizeof(char));
  if (tasdcalibev_bank==NULL){
    fprintf(stderr,
	    "tasdcalibev_bankInit: fail to assign memory to "
	    "bank. Abort.\n");
    exit(0);
  }
}

int tasdcalibev_common_to_bank_(void)
{
  static int id=TASDCALIBEV_BANKID, ver=TASDCALIBEV_BANKVERSION;
  int rc, nobj, ii;

  if (tasdcalibev_bank == NULL) tasdcalibev_bank_init();

  /* Initialize tasdcalibev_blen, and pack the id and version
     to bank */
  if ((rc=dst_initbank_(&id,&ver,&tasdcalibev_blen,&tasdcalibev_maxlen,tasdcalibev_bank)))
    return rc;

  nobj = 1;
  rc+=dst_packi4_(&tasdcalibev_.eventCode,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.date,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.time,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.usec,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.trgBank,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.trgPos,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.trgMode,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.daqMode,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.numTrgwf,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.numWf,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  nobj = tasdcalibev_nhmax;
  rc+=dst_packi4_(&tasdcalibev_.runId[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi4_(&tasdcalibev_.daqMiss[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);

  if(tasdcalibev_.eventCode==0){
    nobj = 24;
    rc+=dst_packi1_(&tasdcalibev_.sim.interactionModel[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = 8;
    rc+=dst_packi1_(&tasdcalibev_.sim.primaryParticleType[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = 1;
    rc+=dst_packr4_(&tasdcalibev_.sim.primaryEnergy,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sim.primaryCosZenith,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sim.primaryAzimuth,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sim.primaryFirstIntDepth,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sim.primaryArrivalTimeFromPps,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sim.primaryCorePosX,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sim.primaryCorePosY,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sim.primaryCorePosZ,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sim.thinRatio,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sim.maxWeight,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sim.trgCode,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sim.userInfo,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = 10;
    rc+=dst_packr4_(&tasdcalibev_.sim.detailUserInfo[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  }

  for(ii=0;ii<tasdcalibev_.numTrgwf;ii++){
    nobj = 1;
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].site,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].lid,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].clock,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].maxClock,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi1_(&tasdcalibev_.sub[ii].wfId,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi1_(&tasdcalibev_.sub[ii].numTrgwf,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi1_(&tasdcalibev_.sub[ii].trgCode,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi1_(&tasdcalibev_.sub[ii].wfError,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = tasdcalibev_nfadc;
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].uwf[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].lwf[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = 1;
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].clockError,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].upedAvr,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].lpedAvr,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].upedStdev,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].lpedStdev,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].umipNonuni,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].lmipNonuni,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].umipMev2cnt,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].lmipMev2cnt,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].umipMev2pe,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].lmipMev2pe,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].lvl0Rate,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].lvl1Rate,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].scintiTemp,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].warning,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi1_(&tasdcalibev_.sub[ii].dontUse,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi1_(&tasdcalibev_.sub[ii].dataQuality,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi1_(&tasdcalibev_.sub[ii].trgMode0,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi1_(&tasdcalibev_.sub[ii].trgMode1,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi1_(&tasdcalibev_.sub[ii].gpsRunMode,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].uthreLvl0,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].lthreLvl0,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].uthreLvl1,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].lthreLvl1,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].posX,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].posY,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].posZ,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].delayns,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].ppsofs,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].ppsflu,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].lonmas,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].latmas,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].heicm,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].udec5pled,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].ldec5pled,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].udec5pmip,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packi2_(&tasdcalibev_.sub[ii].ldec5pmip,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);

    nobj = 2;
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].pchmip[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].pchped[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].lhpchmip[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].lhpchped[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].rhpchmip[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].rhpchped[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packi4_(&tasdcalibev_.sub[ii].mftndof[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].mip[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].mftchi2[0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    nobj = 4;
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].mftp[0][0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].mftp[1][0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].mftpe[0][0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.sub[ii].mftpe[1][0],
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen, &tasdcalibev_maxlen);


  }

  nobj = 1;
  rc+=dst_packi4_(&tasdcalibev_.numAlive,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  nobj = tasdcalibev_.numAlive;
  rc+=dst_packi2_(&tasdcalibev_.aliveDetLid[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi2_(&tasdcalibev_.aliveDetSite[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packr4_(&tasdcalibev_.aliveDetPosX[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packr4_(&tasdcalibev_.aliveDetPosY[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packr4_(&tasdcalibev_.aliveDetPosZ[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);

  nobj = 1;
  rc+=dst_packi4_(&tasdcalibev_.numDead,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  nobj = tasdcalibev_.numDead;
  rc+=dst_packi2_(&tasdcalibev_.deadDetLid[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packi2_(&tasdcalibev_.deadDetSite[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packr4_(&tasdcalibev_.deadDetPosX[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packr4_(&tasdcalibev_.deadDetPosY[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_packr4_(&tasdcalibev_.deadDetPosZ[0],
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);

  nobj = 1;
  rc+=dst_packi4_(&tasdcalibev_.numWeather,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  for(ii=0;ii<tasdcalibev_.numWeather;ii++){
    nobj = 1;
    rc+=dst_packi4_(&tasdcalibev_.weather[ii].site,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.weather[ii].atmosphericPressure,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.weather[ii].temperature,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.weather[ii].humidity,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.weather[ii].rainfall,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_packr4_(&tasdcalibev_.weather[ii].numberOfHails,
		    &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  }

  nobj = 1;
  rc+=dst_packi4_(&tasdcalibev_.footer,
		  &nobj,tasdcalibev_bank,&tasdcalibev_blen,&tasdcalibev_maxlen);


  return rc;
}


int tasdcalibev_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit,&tasdcalibev_blen,
			 tasdcalibev_bank );
}

int tasdcalibev_common_to_dst_(int *NumUnit)
{
  int rc;
  if ( (rc = tasdcalibev_common_to_bank_()) )
    {
      fprintf (stderr,"tasdcalibev_common_to_bank_ ERROR : %ld\n",
	       (long) rc);
      exit(0);
    }
  if ( (rc = tasdcalibev_bank_to_dst_(NumUnit)) )
    {
      fprintf (stderr,"tasdcalibev_bank_to_dst_ ERROR : %ld\n",
	       (long) rc);
      exit(0);
    }
  return SUCCESS;
}

int tasdcalibev_bank_to_common_(char *bank)
{
  int rc = 0;
  int nobj, ii;

  tasdcalibev_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rc+=dst_unpacki4_(&tasdcalibev_.eventCode,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.date,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.time,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.usec,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.trgBank,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.trgPos,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.trgMode,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.daqMode,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.numTrgwf,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.numWf,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  nobj = tasdcalibev_nhmax;
  rc+=dst_unpacki4_(&tasdcalibev_.runId[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki4_(&tasdcalibev_.daqMiss[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);

  if(tasdcalibev_.eventCode==0){
    nobj = 24;
    rc+=dst_unpacki1_(&tasdcalibev_.sim.interactionModel[0],
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = 8;
    rc+=dst_unpacki1_(&tasdcalibev_.sim.primaryParticleType[0],
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = 1;
    rc+=dst_unpackr4_(&tasdcalibev_.sim.primaryEnergy,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sim.primaryCosZenith,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sim.primaryAzimuth,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sim.primaryFirstIntDepth,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sim.primaryArrivalTimeFromPps,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sim.primaryCorePosX,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sim.primaryCorePosY,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sim.primaryCorePosZ,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sim.thinRatio,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sim.maxWeight,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sim.trgCode,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sim.userInfo,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = 10;
    rc+=dst_unpackr4_(&tasdcalibev_.sim.detailUserInfo[0],
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  }

  for(ii=0;ii<tasdcalibev_.numTrgwf;ii++){
    nobj = 1;
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].site,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].lid,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].clock,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].maxClock,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki1_(&tasdcalibev_.sub[ii].wfId,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki1_(&tasdcalibev_.sub[ii].numTrgwf,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki1_(&tasdcalibev_.sub[ii].trgCode,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki1_(&tasdcalibev_.sub[ii].wfError,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = tasdcalibev_nfadc;
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].uwf[0],
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].lwf[0],
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    nobj = 1;
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].clockError,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].upedAvr,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].lpedAvr,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].upedStdev,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].lpedStdev,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].umipNonuni,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].lmipNonuni,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].umipMev2cnt,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].lmipMev2cnt,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].umipMev2pe,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].lmipMev2pe,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].lvl0Rate,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].lvl1Rate,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].scintiTemp,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].warning,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki1_(&tasdcalibev_.sub[ii].dontUse,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki1_(&tasdcalibev_.sub[ii].dataQuality,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki1_(&tasdcalibev_.sub[ii].trgMode0,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki1_(&tasdcalibev_.sub[ii].trgMode1,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki1_(&tasdcalibev_.sub[ii].gpsRunMode,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].uthreLvl0,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].lthreLvl0,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].uthreLvl1,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].lthreLvl1,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].posX,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].posY,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].posZ,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].delayns,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].ppsofs,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].ppsflu,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].lonmas,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].latmas,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].heicm,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].udec5pled,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].ldec5pled,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].udec5pmip,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpacki2_(&tasdcalibev_.sub[ii].ldec5pmip,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);


    nobj = 2;
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].pchmip[0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].pchped[0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].lhpchmip[0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].lhpchped[0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].rhpchmip[0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].rhpchped[0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpacki4_(&tasdcalibev_.sub[ii].mftndof[0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].mip[0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].mftchi2[0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    nobj = 4;
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].mftp[0][0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].mftp[1][0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].mftpe[0][0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.sub[ii].mftpe[1][0],
		      &nobj,bank,&tasdcalibev_blen, &tasdcalibev_maxlen);


  }

  nobj = 1;
  rc+=dst_unpacki4_(&tasdcalibev_.numAlive,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  nobj = tasdcalibev_.numAlive;
  rc+=dst_unpacki2_(&tasdcalibev_.aliveDetLid[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki2_(&tasdcalibev_.aliveDetSite[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpackr4_(&tasdcalibev_.aliveDetPosX[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpackr4_(&tasdcalibev_.aliveDetPosY[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpackr4_(&tasdcalibev_.aliveDetPosZ[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);

  nobj = 1;
  rc+=dst_unpacki4_(&tasdcalibev_.numDead,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  nobj = tasdcalibev_.numDead;
  rc+=dst_unpacki2_(&tasdcalibev_.deadDetLid[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpacki2_(&tasdcalibev_.deadDetSite[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpackr4_(&tasdcalibev_.deadDetPosX[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpackr4_(&tasdcalibev_.deadDetPosY[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  rc+=dst_unpackr4_(&tasdcalibev_.deadDetPosZ[0],
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);

  nobj = 1;
  rc+=dst_unpacki4_(&tasdcalibev_.numWeather,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  for(ii=0;ii<tasdcalibev_.numWeather;ii++){
    nobj = 1;
    rc+=dst_unpacki4_(&tasdcalibev_.weather[ii].site,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.weather[ii].atmosphericPressure,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.weather[ii].temperature,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.weather[ii].humidity,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.weather[ii].rainfall,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
    rc+=dst_unpackr4_(&tasdcalibev_.weather[ii].numberOfHails,
		      &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);
  }

  nobj = 1;
  rc+=dst_unpacki4_(&tasdcalibev_.footer,
		    &nobj,bank,&tasdcalibev_blen,&tasdcalibev_maxlen);

  return rc;

}

int tasdcalibev_common_to_dump_(int *long_output)
{
  return tasdcalibev_common_to_dumpf_(stdout, long_output);
}

int tasdcalibev_common_to_dumpf_(FILE* fp, int *long_output)
{
  fprintf (fp, "%s :\n","tasdcalibev");
  fprintf(fp,"runID BR  %d  DAQ MISS %d\n", tasdcalibev_.runId[0], tasdcalibev_.daqMiss[0]);
  fprintf(fp,"runID LR  %d  DAQ MISS %d\n", tasdcalibev_.runId[1], tasdcalibev_.daqMiss[1]);
  fprintf(fp,"runID SK  %d  DAQ MISS %d\n", tasdcalibev_.runId[2], tasdcalibev_.daqMiss[2]);
  fprintf(fp,"Date        %06d\n", tasdcalibev_.date);
  fprintf(fp,"Time        %06d\n", tasdcalibev_.time);
  fprintf(fp,"usec        %d\n", tasdcalibev_.usec);
  if (tasdcalibev_.eventCode == 0) 
    {
      char interactionModel[sizeof(tasdcalibev_.sim.interactionModel)+1];
      char primaryParticleType[sizeof(tasdcalibev_.sim.primaryParticleType)+1];
      memcpy(interactionModel,tasdcalibev_.sim.interactionModel,sizeof(tasdcalibev_.sim.interactionModel));
      memcpy(primaryParticleType,tasdcalibev_.sim.primaryParticleType,sizeof(tasdcalibev_.sim.primaryParticleType));      
      interactionModel[(int)sizeof(tasdcalibev_.sim.interactionModel)] = '\0';       /* ensure null termination */
      primaryParticleType[(int)sizeof(tasdcalibev_.sim.primaryParticleType)] = '\0'; /* ensure null termination */
      fprintf(fp,"%s  %s\n",interactionModel,primaryParticleType);
      fprintf(fp,"Ene= %e , Zen= %f , Azi= %f )\n", 
	      tasdcalibev_.sim.primaryEnergy, tasdcalibev_.sim.primaryCosZenith, 
	      tasdcalibev_.sim.primaryAzimuth);
      fprintf(fp,"Core pos[cm]  ( %f , %f , %f )\n", 
	      tasdcalibev_.sim.primaryCorePosX * 100., 
	      tasdcalibev_.sim.primaryCorePosY * 100., 
	      tasdcalibev_.sim.primaryCorePosZ * 100.);
    }
  if((*long_output) >= 0)
    {
      int i = 0;
      for (i = 0; i < tasdcalibev_.numTrgwf; i++) 
	{
	  fprintf(fp,"NAME: SD%04d  POS(m):( %10.3f , %10.3f , %10.3f ) Clock: %10d MaxClock: %10d PED Up: %5.2f Low: %5.2f\n", 
		  tasdcalibev_.sub[i].lid, 
		  tasdcalibev_.sub[i].posX, tasdcalibev_.sub[i].posY, tasdcalibev_.sub[i].posZ, 
		  tasdcalibev_.sub[i].clock, tasdcalibev_.sub[i].maxClock, 
		  tasdcalibev_.sub[i].upedAvr, tasdcalibev_.sub[i].lpedAvr);
	  if ((*long_output)) 
	    {
	      int j = 0, k = 0;
	      fprintf(fp,"CH1:\n");
	      for (j = 0; j < tasdcalibev_nfadc; j++)
		{
		  if(k==12)
		    {
		      fprintf(fp,"\n");
		      k = 0;
		    }
		  fprintf(fp,"%6d ", tasdcalibev_.sub[i].uwf[j]);
		  k++;
		}
	      fprintf(fp,"\nCH2:\n");
	      k = 0;
	      for (j = 0; j < tasdcalibev_nfadc; j++)
		{
		  if(k==12)
		    {
		      fprintf(fp,"\n");
		      k = 0;
		    }
		  fprintf(fp,"%6d ", tasdcalibev_.sub[i].lwf[j]);
		  k++;
		}
	      fprintf(fp,"\n");
	    }
	}
    }
  return 0;
}
