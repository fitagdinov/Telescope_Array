/*
 * tasdinfo_dst.c
 *
 * C functions for TASDINFO bank
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
#include "tasdinfo_dst.h"

tasdinfo_dst_common tasdinfo_;/* allocate memory to
				 tasdinfo_common */

static integer4 tasdinfo_blen = 0;
static int tasdinfo_maxlen =
  sizeof(int)*2+sizeof(tasdinfo_dst_common);
static char *tasdinfo_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdinfo_bank_buffer_ (integer4* tasdinfo_bank_buffer_size)
{
  (*tasdinfo_bank_buffer_size) = tasdinfo_blen;
  return tasdinfo_bank;
}



static void tasdinfo_bank_init(void)
{
  tasdinfo_bank = (char *)calloc(tasdinfo_maxlen, sizeof(char));
  if (tasdinfo_bank==NULL){
    fprintf(stderr,
	    "tasdinfo_bank_init: "
	    "fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

int tasdinfo_common_to_bank_(void)
{
  static int id = TASDINFO_BANKID, ver = TASDINFO_BANKVERSION;
  int rc, nobj, ii;

  if (tasdinfo_bank == NULL) tasdinfo_bank_init();

  /*Initialize tasdinfo_blen, and pack the id and version to bank*/
  if ( (rc=dst_initbank_(&id,&ver,&tasdinfo_blen,
			 &tasdinfo_maxlen,tasdinfo_bank)) )
    return rc;

  nobj = 1;
  rc+=dst_packi4_( &tasdinfo_.ndet,   &nobj, tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);
  rc+=dst_packi4_( &tasdinfo_.site,   &nobj, tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);
  rc+=dst_packi4_( &tasdinfo_.dateFrom,&nobj,tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);
  rc+=dst_packi4_( &tasdinfo_.dateTo,&nobj,tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);
  rc+=dst_packi4_( &tasdinfo_.timeFrom,&nobj,tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);
  rc+=dst_packi4_( &tasdinfo_.timeTo,&nobj,tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);
  rc+=dst_packi4_( &tasdinfo_.first_run_id,&nobj,tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);
  rc+=dst_packi4_( &tasdinfo_.last_run_id, &nobj, tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);
  rc+=dst_packi4_( &tasdinfo_.year, &nobj, tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);

  for(ii=0;ii<tasdinfo_.ndet;ii++){
    nobj = 1;
    rc+=dst_packi4_(&tasdinfo_.sub[ii].lid,&nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_((int*)&tasdinfo_.sub[ii].wlanidmsb,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_((int*)&tasdinfo_.sub[ii].wlanidlsb,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].upmt_id,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].lpmt_id,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].ccid,&nobj,tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 20;
    rc+=dst_packi1_(&tasdinfo_.sub[ii].box_id[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 8;
    rc+=dst_packi1_(&tasdinfo_.sub[ii].elecid[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi1_(&tasdinfo_.sub[ii].gpsid[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi1_(&tasdinfo_.sub[ii].cpldver[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi1_(&tasdinfo_.sub[ii].ccver[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 32;
    rc+=dst_packi1_(&tasdinfo_.sub[ii].firm_version[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);

    nobj = 1;
    rc+=dst_packi2_(&tasdinfo_.sub[ii].trig_mode0,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi2_(&tasdinfo_.sub[ii].trig_mode1,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi2_(&tasdinfo_.sub[ii].uthre_lvl0,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi2_(&tasdinfo_.sub[ii].lthre_lvl0,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi2_(&tasdinfo_.sub[ii].uthre_lvl1,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi2_(&tasdinfo_.sub[ii].lthre_lvl1,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);

    rc+=dst_packr4_(&tasdinfo_.sub[ii].uhv,&nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].lhv,&nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].upmtgain,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].lpmtgain,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].upmtgainError,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].lpmtgainError,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);


    rc+=dst_packi4_(&tasdinfo_.sub[ii].lonmas,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].latmas,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].heicm,&nobj,tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].lonmasSet,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].latmasSet,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].heicmSet,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].lonmasError,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].latmasError,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].heicmError,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].delayns,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].ppsofs,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].ppsfluPH,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].ppsflu3D,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);

    nobj = tasdinfo_npmax;
    rc+=dst_packr4_(&tasdinfo_.sub[ii].ucrrx[0],
		    &nobj,tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].ucrry[0],
		    &nobj,tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].ucrrsig[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 1;
    rc+=dst_packr4_(&tasdinfo_.sub[ii].uhv_led,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].udec5p,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);

    nobj = tasdinfo_npmax;
    rc+=dst_packr4_(&tasdinfo_.sub[ii].lcrrx[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].lcrry[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packr4_(&tasdinfo_.sub[ii].lcrrsig[0],
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 1;
    rc+=dst_packr4_(&tasdinfo_.sub[ii].lhv_led,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_packi4_(&tasdinfo_.sub[ii].ldec5p,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);

    rc+=dst_packi4_(&tasdinfo_.sub[ii].error_flag,
		    &nobj, tasdinfo_bank,
		    &tasdinfo_blen, &tasdinfo_maxlen);
  }

  nobj = 1;
  rc+=dst_packi4_( &tasdinfo_.footer, &nobj, tasdinfo_bank,
		   &tasdinfo_blen, &tasdinfo_maxlen);

  return rc;
}


int tasdinfo_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdinfo_blen, tasdinfo_bank );
}

int tasdinfo_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdinfo_common_to_bank_()) ){
    fprintf (stderr,"tasdinfo_common_to_bank_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  if ( (rcode = tasdinfo_bank_to_dst_(NumUnit)) ) {
    fprintf (stderr,"tasdinfo_bank_to_dst_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  return SUCCESS;
}

int tasdinfo_bank_to_common_(char *bank)
{
  int rc = 0;
  int nobj, ii;


  tasdinfo_blen = 2 * sizeof(int); /* skip id and version  */

  nobj = 1;
  rc += dst_unpacki4_( &tasdinfo_.ndet,   &nobj, bank,
		       &tasdinfo_blen, &tasdinfo_maxlen);
  rc += dst_unpacki4_( &tasdinfo_.site,   &nobj, bank,
		       &tasdinfo_blen, &tasdinfo_maxlen);
  rc += dst_unpacki4_( &tasdinfo_.dateFrom, &nobj, bank,
		       &tasdinfo_blen, &tasdinfo_maxlen);
  rc += dst_unpacki4_( &tasdinfo_.dateTo, &nobj, bank,
		       &tasdinfo_blen, &tasdinfo_maxlen);
  rc += dst_unpacki4_( &tasdinfo_.timeFrom, &nobj, bank,
		       &tasdinfo_blen, &tasdinfo_maxlen);
  rc += dst_unpacki4_( &tasdinfo_.timeTo, &nobj, bank,
		       &tasdinfo_blen, &tasdinfo_maxlen);
  rc += dst_unpacki4_( &tasdinfo_.first_run_id, &nobj, bank,
		       &tasdinfo_blen, &tasdinfo_maxlen);
  rc += dst_unpacki4_( &tasdinfo_.last_run_id, &nobj, bank,
		       &tasdinfo_blen, &tasdinfo_maxlen);
  rc += dst_unpacki4_( &tasdinfo_.year, &nobj, bank,
		       &tasdinfo_blen, &tasdinfo_maxlen);

  for(ii=0;ii<tasdinfo_.ndet;ii++){
    nobj = 1;
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].lid,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_((int*)&tasdinfo_.sub[ii].wlanidmsb,
		      &nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_((int*)&tasdinfo_.sub[ii].wlanidlsb,
		      &nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].upmt_id,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].lpmt_id,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].ccid,   &nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 20;
    rc+=dst_unpacki1_(&tasdinfo_.sub[ii].box_id[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 8;
    rc+=dst_unpacki1_(&tasdinfo_.sub[ii].elecid[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki1_(&tasdinfo_.sub[ii].gpsid[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki1_(&tasdinfo_.sub[ii].cpldver[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki1_(&tasdinfo_.sub[ii].ccver[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 32;
    rc+=dst_unpacki1_(&tasdinfo_.sub[ii].firm_version[0],
		      &nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);

    nobj = 1;
    rc+=dst_unpacki2_(&tasdinfo_.sub[ii].trig_mode0,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki2_(&tasdinfo_.sub[ii].trig_mode1,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki2_(&tasdinfo_.sub[ii].uthre_lvl0,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki2_(&tasdinfo_.sub[ii].lthre_lvl0,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki2_(&tasdinfo_.sub[ii].uthre_lvl1,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki2_(&tasdinfo_.sub[ii].lthre_lvl1,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);

    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].uhv,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].lhv,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].upmtgain,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].lpmtgain,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].upmtgainError,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].lpmtgainError,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);


    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].lonmas,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].latmas,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].heicm,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].lonmasSet,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].latmasSet,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].heicmSet,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].lonmasError,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].latmasError,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].heicmError,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].delayns,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].ppsofs,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].ppsfluPH,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].ppsflu3D,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);

    nobj = tasdinfo_npmax;
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].ucrrx[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].ucrry[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].ucrrsig[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 1;
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].uhv_led,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].udec5p,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);

    nobj = tasdinfo_npmax;
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].lcrrx[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].lcrry[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].lcrrsig[0],&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    nobj = 1;
    rc+=dst_unpackr4_(&tasdinfo_.sub[ii].lhv_led,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].ldec5p,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);

    rc+=dst_unpacki4_(&tasdinfo_.sub[ii].error_flag,&nobj, bank,
		      &tasdinfo_blen, &tasdinfo_maxlen);
  }

  nobj = 1;
  rc += dst_unpacki4_( &tasdinfo_.footer, &nobj, bank,
			  &tasdinfo_blen, &tasdinfo_maxlen);

  return rc;

}

int tasdinfo_common_to_dump_(int *long_output)
{
  return tasdinfo_common_to_dumpf_(stdout, long_output);
}

int tasdinfo_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
