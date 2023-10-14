/*
 * tasdmonitor_dst.c 
 *
 * C functions for TASDMONITOR bank
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
#include "tasdmonitor_dst.h"

tasdmonitor_dst_common tasdmonitor_;  /* allocate memory to
					 tasdmonitor_common */

static integer4 tasdmonitor_blen = 0;
static int tasdmonitor_maxlen = sizeof(int)*2+sizeof(tasdmonitor_dst_common);
static char *tasdmonitor_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tasdmonitor_bank_buffer_ (integer4* tasdmonitor_bank_buffer_size)
{
  (*tasdmonitor_bank_buffer_size) = tasdmonitor_blen;
  return tasdmonitor_bank;
}



static void tasdmonitor_bank_init(void)
{
  tasdmonitor_bank=(char*)calloc(tasdmonitor_maxlen, sizeof(char));
  if (tasdmonitor_bank==NULL)
    {
      fprintf(stderr,"tasdmonitor_bank_init: "
	      "fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

int tasdmonitor_common_to_bank_(void)
{
  static int id=TASDMONITOR_BANKID, ver=TASDMONITOR_BANKVERSION;
  int rc, nobj ,ii, jj;

  if (tasdmonitor_bank == NULL) tasdmonitor_bank_init();

  /* Initialize tasdmonitor_blen,
     and pack the id and version to bank */
  if ((rc=dst_initbank_(&id,&ver,&tasdmonitor_blen,
			   &tasdmonitor_maxlen,tasdmonitor_bank)) )
    return rc;

  /* integers */

  nobj = 1;
  rc+=dst_packi4_(&tasdmonitor_.date,
		  &nobj, tasdmonitor_bank,
		  &tasdmonitor_blen, &tasdmonitor_maxlen);
  rc+=dst_packi4_(&tasdmonitor_.time,
		  &nobj, tasdmonitor_bank,
		  &tasdmonitor_blen, &tasdmonitor_maxlen);
  rc+=dst_packi4_(&tasdmonitor_.num_det,
		  &nobj, tasdmonitor_bank,
		  &tasdmonitor_blen, &tasdmonitor_maxlen);

  nobj = tasdmonitor_ndmax;
  rc+=dst_packi2_(&tasdmonitor_.lid[0],
		  &nobj, tasdmonitor_bank,
		  &tasdmonitor_blen, &tasdmonitor_maxlen);

  for(ii=0;ii<tasdmonitor_nhmax;ii++){
    nobj = 1;
    rc+=dst_packi4_(&tasdmonitor_.host[ii].run_id,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi2_(&tasdmonitor_.host[ii].runtime,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi2_(&tasdmonitor_.host[ii].deadtime,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi2_(&tasdmonitor_.host[ii].num_det,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi2_(&tasdmonitor_.host[ii].error_flag,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi2_(&tasdmonitor_.host[ii].num_retry,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi2_(&tasdmonitor_.host[ii].num_error,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    for(jj=0;jj<600;jj++){
      nobj = 1;
      rc+=dst_packi4_(&tasdmonitor_.host[ii].pps[jj].run_id,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi4_(&tasdmonitor_.host[ii].pps[jj].trial,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi4_(&tasdmonitor_.host[ii].pps[jj].date,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi4_(&tasdmonitor_.host[ii].pps[jj].time,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi4_(&tasdmonitor_.host[ii].pps[jj].date_org,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi4_(&tasdmonitor_.host[ii].pps[jj].time_org,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      nobj = 10;
      rc+=dst_packi4_(&tasdmonitor_.host[ii].pps[jj].bank[0],
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      nobj = 1;
      rc+=dst_packi4_(&tasdmonitor_.host[ii].pps[jj].eventInfoCode,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].num_tbl,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].num_trigger,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].num_bank,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].cur_time,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].num_sat,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].num_retry,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].num_error,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].num_debug,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      nobj = 4;
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].vol[0],
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].cc[0],
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      nobj = 1;
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].gps_error,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.host[ii].pps[jj].error_flag,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    }
  }

  for(ii=0;ii<tasdmonitor_ndmax;ii++){
    nobj = 1;
    rc+=dst_packi2_(&tasdmonitor_.sub[ii].site,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi2_(&tasdmonitor_.sub[ii].lid,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);

    for(jj=0;jj<600;jj++){
      rc+=dst_packi4_(&tasdmonitor_.sub[ii].pps[jj].max_clock,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi4_(&tasdmonitor_.sub[ii].pps[jj].wlan_health,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.sub[ii].pps[jj].num_retry,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.sub[ii].pps[jj].cur_time,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.sub[ii].pps[jj].num_wf,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_packi2_(&tasdmonitor_.sub[ii].pps[jj].num_tbl,
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    }

    nobj = 512;
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.mip1[0],
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.mip2[0],
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    nobj = 256;
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.ped1[0],
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.ped2[0],
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    nobj = 128;
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.phl1[0],
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.phl2[0],
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.pcl1[0],
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.pcl2[0],
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    nobj = 8;
    for(jj=0;jj<10;jj++){
      rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.cc_adc[jj][0],
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    }
    for(jj=0;jj<10;jj++){
      rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.sd_adc[jj][0],
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    }
    nobj = 2;
    for(jj=0;jj<10;jj++){
      rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.rate[jj][0],
		      &nobj, tasdmonitor_bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    }
    nobj = 1;
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.date,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.time,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.gps_flag,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.cur_rate2,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.num_packet,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.num_sat,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.gps_lat,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.gps_lon,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.gps_hei,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    nobj = 10;
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].mon.dummy[0],
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);

    nobj = 1;
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].num_error,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].num_retry,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_packi4_(&tasdmonitor_.sub[ii].livetime,
		    &nobj, tasdmonitor_bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
  }

  rc+=dst_packi4_(&tasdmonitor_.footer,
		  &nobj, tasdmonitor_bank,
		  &tasdmonitor_blen, &tasdmonitor_maxlen);

  return rc;
}


int tasdmonitor_bank_to_dst_(int *NumUnit)
{
  return dst_write_bank_(NumUnit, &tasdmonitor_blen,
			 tasdmonitor_bank );
}

int tasdmonitor_common_to_dst_(int *NumUnit)
{
  int rcode;
  if ( (rcode = tasdmonitor_common_to_bank_()) ){
    fprintf (stderr,"tasdmonitor_common_to_bank_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  if ( (rcode = tasdmonitor_bank_to_dst_(NumUnit)) ){
    fprintf (stderr,"tasdmonitor_bank_to_dst_ ERROR : %ld\n",
	     (long) rcode);
    exit(0);
  }
  return SUCCESS;
}

int tasdmonitor_bank_to_common_(char *bank)
{
  int rc = 0;
  int nobj, ii, jj;

  tasdmonitor_blen = 2 * sizeof(int); /* skip id and version  */

  /* integers */

  nobj = 1;
  rc+=dst_unpacki4_(&tasdmonitor_.date,
		    &nobj, bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
  rc+=dst_unpacki4_(&tasdmonitor_.time,
		    &nobj, bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
  rc+=dst_unpacki4_(&tasdmonitor_.num_det,
		    &nobj, bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);

  nobj = tasdmonitor_ndmax;
  rc+=dst_unpacki2_(&tasdmonitor_.lid[0],
		    &nobj, bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);

  for(ii=0;ii<tasdmonitor_nhmax;ii++){
    nobj = 1;
    rc+=dst_unpacki4_(&tasdmonitor_.host[ii].run_id,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki2_(&tasdmonitor_.host[ii].runtime,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki2_(&tasdmonitor_.host[ii].deadtime,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki2_(&tasdmonitor_.host[ii].num_det,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki2_(&tasdmonitor_.host[ii].error_flag,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki2_(&tasdmonitor_.host[ii].num_retry,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki2_(&tasdmonitor_.host[ii].num_error,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    for(jj=0;jj<600;jj++){
      nobj = 1;
      rc+=dst_unpacki4_(&tasdmonitor_.host[ii].pps[jj].run_id,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki4_(&tasdmonitor_.host[ii].pps[jj].trial,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki4_(&tasdmonitor_.host[ii].pps[jj].date,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki4_(&tasdmonitor_.host[ii].pps[jj].time,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki4_(&tasdmonitor_.host[ii].pps[jj].date_org,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki4_(&tasdmonitor_.host[ii].pps[jj].time_org,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      nobj = 10;
      rc+=dst_unpacki4_(&tasdmonitor_.host[ii].pps[jj].bank[0],
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      nobj = 1;
      rc+=dst_unpacki4_(&tasdmonitor_.host[ii].pps[jj].eventInfoCode,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].num_tbl,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].num_trigger,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].num_bank,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].cur_time,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].num_sat,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].num_retry,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].num_error,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].num_debug,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      nobj = 4;
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].vol[0],
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].cc[0],
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      nobj = 1;
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].gps_error,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.host[ii].pps[jj].error_flag,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
    }
  }

  for(ii=0;ii<tasdmonitor_ndmax;ii++){
    nobj = 1;
    rc+=dst_unpacki2_(&tasdmonitor_.sub[ii].site,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki2_(&tasdmonitor_.sub[ii].lid,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);

    for(jj=0;jj<600;jj++){
      rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].pps[jj].max_clock,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].pps[jj].wlan_health,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.sub[ii].pps[jj].num_retry,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.sub[ii].pps[jj].cur_time,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.sub[ii].pps[jj].num_wf,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
      rc+=dst_unpacki2_(&tasdmonitor_.sub[ii].pps[jj].num_tbl,
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
    }

    nobj = 512;
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.mip1[0],
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.mip2[0],
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    nobj = 256;
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.ped1[0],
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.ped2[0],
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    nobj = 128;
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.phl1[0],
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.phl2[0],
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.pcl1[0],
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.pcl2[0],
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    nobj = 8;
    for(jj=0;jj<10;jj++){
      rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.cc_adc[jj][0],
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
    }
    for(jj=0;jj<10;jj++){
      rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.sd_adc[jj][0],
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
    }
    nobj = 2;
    for(jj=0;jj<10;jj++){
      rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.rate[jj][0],
			&nobj, bank,
			&tasdmonitor_blen, &tasdmonitor_maxlen);
    }
    nobj = 1;
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.date,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.time,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.gps_flag,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.cur_rate2,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.num_packet,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.num_sat,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.gps_lat,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.gps_lon,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.gps_hei,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    nobj = 10;
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].mon.dummy[0],
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);

    nobj = 1;
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].num_error,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].num_retry,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
    rc+=dst_unpacki4_(&tasdmonitor_.sub[ii].livetime,
		      &nobj, bank,
		      &tasdmonitor_blen, &tasdmonitor_maxlen);
  }

  rc+=dst_unpacki4_(&tasdmonitor_.footer,
		    &nobj, bank,
		    &tasdmonitor_blen, &tasdmonitor_maxlen);
  return rc;

}

int tasdmonitor_common_to_dump_(int *long_output)
{
  return tasdmonitor_common_to_dumpf_(stdout, long_output);
}

int tasdmonitor_common_to_dumpf_(FILE* fp, int *long_output)
{
  (void)(long_output);
  fprintf(fp,"Good-bye world !\n");
  return 0;
}

