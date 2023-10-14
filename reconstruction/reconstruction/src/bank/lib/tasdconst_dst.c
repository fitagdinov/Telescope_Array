/*
 * tasdconst_dst.c
 *
 * C functions for TASDCONST bank
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
#include "tasdconst_dst.h"

tasdconst_dst_common tasdconst_;/* allocate memory to
				 tasdconst_common */

static int tasdconst_blen = 0;
static int tasdconst_maxlen = sizeof(int)*2+sizeof(tasdconst_dst_common);
static char *tasdconst_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdconst_bank_buffer_ (integer4* tasdconst_bank_buffer_size)
{
  (*tasdconst_bank_buffer_size) = tasdconst_blen;
  return tasdconst_bank;
}



static void tasdconst_bank_init(void)
{
  tasdconst_bank = (char *)calloc(tasdconst_maxlen, sizeof(char));
  if (tasdconst_bank==NULL){
    fprintf(stderr,
	    "tasdconst_bank_init: "
	    "fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

int tasdconst_common_to_bank_(void)
{
  static int id = TASDCONST_BANKID, ver = TASDCONST_BANKVERSION;
  int rc, nobj;

  if (tasdconst_bank == NULL) tasdconst_bank_init();

  /*Initialize tasdconst_blen,and pack the id and version to bank*/
  if ( (rc=dst_initbank_(&id,&ver,&tasdconst_blen,
			 &tasdconst_maxlen,tasdconst_bank)) )
    return rc;

  nobj = 1;
  rc+=dst_packi4_(&tasdconst_.lid,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.site,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.dateFrom,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.dateTo,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.timeFrom,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.timeTo,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.first_run_id,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.last_run_id,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);


  rc+=dst_packi4_((int*)&tasdconst_.wlanidmsb,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_((int*)&tasdconst_.wlanidlsb,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.upmt_id,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.lpmt_id,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);


  rc+=dst_packi4_(&tasdconst_.trig_mode0,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.trig_mode1,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.uthre_lvl0,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.lthre_lvl0,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.uthre_lvl1,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.lthre_lvl1,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);


  rc+=dst_packr4_(&tasdconst_.uhv,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lhv,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.upmtgain,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lpmtgain,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);

  rc+=dst_packr4_(&tasdconst_.posX,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.posY,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.posZ,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.lonmas,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.latmas,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.heicm,&nobj,tasdconst_bank,
		  &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.lonmasSet,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.latmasSet,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.heicmSet,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lonmasError,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.latmasError,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.heicmError,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.delayns,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.ppsofs,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.ppsfluPH,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.ppsflu3D,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);


  rc+=dst_packi4_(&tasdconst_.udec5pled,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.ldec5pled,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.uhv_led,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lhv_led,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  nobj = tasdconst_nledp;
  rc+=dst_packr4_(&tasdconst_.ucrrx[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.ucrry[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.ucrrs[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lcrrx[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lcrry[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lcrrs[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);


  nobj = 1;
  rc+=dst_packi4_(&tasdconst_.livetime,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.uavr,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lavr,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.upltot,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lpltot,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.ucltot,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lcltot,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.nuplx,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.nlplx,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.nuply,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.nlply,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.udec5pmip,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.ldec5pmip,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  nobj = tasdconst_nmipl;
  rc+=dst_packr4_(&tasdconst_.uphx[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lphx[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.uphy[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lphy[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.uphs[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lphs[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.uchx[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lchx[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.uchy[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lchy[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.uchs[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lchs[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.uplx[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lplx[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.uply[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lply[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.upls[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packr4_(&tasdconst_.lpls[0],
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);

  nobj = 1;
  rc+=dst_packi4_(&tasdconst_.error_flag,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_packi4_(&tasdconst_.footer,
		  &nobj, tasdconst_bank, &tasdconst_blen, &tasdconst_maxlen);

  return rc;
}


int tasdconst_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdconst_blen, tasdconst_bank);
}

int tasdconst_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdconst_common_to_bank_()) ){
    fprintf (stderr,"tasdconst_common_to_bank_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  if ( (rcode = tasdconst_bank_to_dst_(NumUnit)) ) {
    fprintf (stderr,"tasdconst_bank_to_dst_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  return SUCCESS;
}

int tasdconst_bank_to_common_(char *bank)
{
  int rc = 0;
  int nobj;

  tasdconst_blen = 2 * sizeof(int); /* skip id and version  */


  nobj = 1;
  rc+=dst_unpacki4_(&tasdconst_.lid,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.site,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.dateFrom,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.dateTo,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.timeFrom,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.timeTo,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.first_run_id,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.last_run_id,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);


  rc+=dst_unpacki4_((int*)&tasdconst_.wlanidmsb,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_((int*)&tasdconst_.wlanidlsb,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.upmt_id,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.lpmt_id,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);


  rc+=dst_unpacki4_(&tasdconst_.trig_mode0,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.trig_mode1,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.uthre_lvl0,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.lthre_lvl0,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.uthre_lvl1,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.lthre_lvl1,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);


  rc+=dst_unpackr4_(&tasdconst_.uhv,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lhv,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.upmtgain,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lpmtgain,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);

  rc+=dst_unpackr4_(&tasdconst_.posX,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.posY,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.posZ,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.lonmas,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.latmas,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.heicm,&nobj,bank,
		    &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.lonmasSet,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.latmasSet,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.heicmSet,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lonmasError,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.latmasError,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.heicmError,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.delayns,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.ppsofs,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.ppsfluPH,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.ppsflu3D,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);


  rc+=dst_unpacki4_(&tasdconst_.udec5pled,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.ldec5pled,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.uhv_led,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lhv_led,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  nobj = tasdconst_nledp;
  rc+=dst_unpackr4_(&tasdconst_.ucrrx[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.ucrry[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.ucrrs[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lcrrx[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lcrry[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lcrrs[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);


  nobj = 1;
  rc+=dst_unpacki4_(&tasdconst_.livetime,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.uavr,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lavr,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.upltot,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lpltot,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.ucltot,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lcltot,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.nuplx,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.nlplx,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.nuply,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.nlply,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.udec5pmip,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.ldec5pmip,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  nobj = tasdconst_nmipl;
  rc+=dst_unpackr4_(&tasdconst_.uphx[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lphx[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.uphy[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lphy[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.uphs[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lphs[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.uchx[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lchx[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.uchy[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lchy[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.uchs[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lchs[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.uplx[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lplx[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.uply[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lply[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.upls[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpackr4_(&tasdconst_.lpls[0],
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);

  nobj = 1;
  rc+=dst_unpacki4_(&tasdconst_.error_flag,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);
  rc+=dst_unpacki4_(&tasdconst_.footer,
		    &nobj, bank, &tasdconst_blen, &tasdconst_maxlen);

  return rc;

}

int tasdconst_common_to_dump_(int *long_output)
{
  return tasdconst_common_to_dumpf_(stdout, long_output);
}

int tasdconst_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}
