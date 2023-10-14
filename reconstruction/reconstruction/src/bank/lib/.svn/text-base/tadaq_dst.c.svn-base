/*
 * C functions for fadc raw 1
 * MRM July 18
 * Modified Oct 4 1995 by JHB: pack/unpack only filled quantities
*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "tadaq_dst.h"  

tadaq_dst_common tadaq_;  /* allocate memory to tadaq_common */

static integer4 tadaq_blen = 0; 
static integer4 tadaq_maxlen = sizeof(integer4) * 2 + sizeof(tadaq_dst_common);
static integer1 *tadaq_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tadaq_bank_buffer_ (integer4* tadaq_bank_buffer_size)
{
  (*tadaq_bank_buffer_size) = tadaq_blen;
  return tadaq_bank;
}



static void tadaq_bank_init() {
  tadaq_bank = (integer1 *)calloc(tadaq_maxlen, sizeof(integer1));
  if (tadaq_bank==NULL) {
    fprintf (stderr,"tadaq_bank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  } /* else fprintf ( stderr,"tadaq_bank allocated memory %d\n",tadaq_maxlen); */
}    

integer4 tadaq_common_to_bank_() {
  static integer4 id = TADAQ_BANKID, ver = TADAQ_BANKVERSION;
  integer4 rcode, nobj, i, j;

  if (tadaq_bank == NULL) tadaq_bank_init();

  rcode = dst_initbank_(&id, &ver, &tadaq_blen, &tadaq_maxlen, tadaq_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj=1;
  rcode += dst_packi2_(&tadaq_.event_code, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi2_(&tadaq_.site, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.runid, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.trigid, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi2_(&tadaq_.ncameras, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

  rcode += dst_packi4_(&tadaq_.year, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.month, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.day, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.hour, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.minute, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.second, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.nanosecond, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

  /*
   * pack CTD data
   */
  /* pack pc data */
  nobj = 24;
  rcode += dst_packi1_(&tadaq_.ctd.pc.byteStream[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

  /* pack the rest */
  nobj = 1;
  rcode += dst_packi4_(&tadaq_.ctd.frame_ID, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.trg_ID, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.sync_err, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.storefull, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.code1, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.code2, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.code3, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.code4, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.trg_mask, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.swadr, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.tf_mask, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.store, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.frm_mdyy, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.frm_hms, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.frm_time_ID, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.CTD_version, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.frm_PPS_counter, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.rst_counter, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.trg_counter, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.extrg_counter, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.dt_counter, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

  nobj = 256;
  rcode += dst_packi4_(&tadaq_.ctd.info_gps1pps[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

  nobj = 1024;
  rcode += dst_packi4_(&tadaq_.ctd.triggered_frame[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.triggered_tf_mask[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

  nobj = 12;
  rcode += dst_packi4_(&tadaq_.ctd.rate[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

  nobj = 1;
  rcode += dst_packi4_(&tadaq_.ctd.latitude, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.longitude, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.hight, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.initial_hms, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.deadtime_st, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.deadPPScnt_st, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.deadfrm_st, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.deadtime_end, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.deadPPScnt_end, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_packi4_(&tadaq_.ctd.deadfrm_end, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);


  /*
   * pack TF data
   */
  for ( i=0; i<tadaq_.ncameras; i++ ) {
    /* pack pc data */
    nobj = 24;
    rcode += dst_packi1_(&tadaq_.tf[i].pc.byteStream[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

    nobj = 1;

    rcode += dst_packi2_(&tadaq_.tf[i].id, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi2_(&tadaq_.tf[i].ntube, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

    rcode += dst_packi4_(&tadaq_.tf[i].frame_ID, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].sync_monitor, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].trg_ID, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].trg_mask, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].store, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].sec_trig_code, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].tftrg_ID, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].swadr, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].board_mask, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].mode, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].mode2, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

    nobj = 256;
    rcode += dst_packi4_(&tadaq_.tf[i].hit_pt[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

    nobj = 16;
    rcode += dst_packi4_(&tadaq_.tf[i].nc_bit[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

    nobj = 1;
    rcode += dst_packi4_(&tadaq_.tf[i].version, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].reset_counter, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].trigger_counter, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_packi4_(&tadaq_.tf[i].extrigger_counter, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

    /* pack sdf data */
    for ( j=0; j<tadaq_.tf[i].ntube; j++ ) {
      nobj = tadaq_nfadc;
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].waveform[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 16;
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].hit[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 1;
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].pmt_id, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].version, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 4;
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].mean[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].disp[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 1;
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].trgmnt, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].peak, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].frmmnt, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].tmphit, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].trgcnt, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].mode, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].ctrl, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].thre, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].rstmnt, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 7;
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].rate[0], &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 1;
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].state, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packr4_(&tadaq_.tf[i].sdf[j].hitthre, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packr4_(&tadaq_.tf[i].sdf[j].ncthre, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packr4_(&tadaq_.tf[i].sdf[j].average, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packr4_(&tadaq_.tf[i].sdf[j].stdev, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_packi2_(&tadaq_.tf[i].sdf[j].tf_hit, &nobj, tadaq_bank, &tadaq_blen, &tadaq_maxlen);
    }
  }
  
  return rcode;
}

integer4 tadaq_bank_to_dst_ (integer4 *unit)
{
  integer4 rcode;
  rcode = dst_write_bank_(unit, &tadaq_blen, tadaq_bank);
  free(tadaq_bank);
  tadaq_bank = NULL; 
  
  return rcode;
}

integer4 tadaq_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
  if ( (rcode = tadaq_common_to_bank_()) )
    {
      fprintf(stderr, "tadaq_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
  if ( (rcode = tadaq_bank_to_dst_(unit)) )
    {
      fprintf(stderr, "tadaq_bank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
  return 0;
}

integer4 tadaq_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0 ;
  integer4 nobj ,i ,j;
  tadaq_blen = 2 * sizeof(integer4);	/* skip id and version  */

  nobj=1;
  rcode += dst_unpacki2_(&tadaq_.event_code, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki2_(&tadaq_.site, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.runid, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.trigid, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki2_(&tadaq_.ncameras, &nobj, bank, &tadaq_blen, &tadaq_maxlen);

  rcode += dst_unpacki4_(&tadaq_.year, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.month, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.day, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.hour, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.minute, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.second, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.nanosecond, &nobj, bank, &tadaq_blen, &tadaq_maxlen);

  /*
   * unpack CTD data
   */
  /* unpack pc data */
  nobj = 24;
  rcode += dst_unpacki1_(&tadaq_.ctd.pc.byteStream[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

  /* unpack the rest */
  nobj = 1;
  rcode += dst_unpacki4_(&tadaq_.ctd.frame_ID, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.trg_ID, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.sync_err, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.storefull, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.code1, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.code2, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.code3, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.code4, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.trg_mask, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.swadr, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.tf_mask, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.store, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.frm_mdyy, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.frm_hms, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.frm_time_ID, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.CTD_version, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.frm_PPS_counter, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.rst_counter, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.trg_counter, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.extrg_counter, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.dt_counter, &nobj, bank, &tadaq_blen, &tadaq_maxlen);

  nobj = 256;
  rcode += dst_unpacki4_(&tadaq_.ctd.info_gps1pps[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

  nobj = 1024;
  rcode += dst_unpacki4_(&tadaq_.ctd.triggered_frame[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.triggered_tf_mask[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

  nobj = 12;
  rcode += dst_unpacki4_(&tadaq_.ctd.rate[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

  nobj = 1;
  rcode += dst_unpacki4_(&tadaq_.ctd.latitude, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.longitude, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.hight, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.initial_hms, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.deadtime_st, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.deadPPScnt_st, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.deadfrm_st, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.deadtime_end, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.deadPPScnt_end, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
  rcode += dst_unpacki4_(&tadaq_.ctd.deadfrm_end, &nobj, bank, &tadaq_blen, &tadaq_maxlen);


  /*
   * unpack TF data
   */
  for ( i=0; i<tadaq_.ncameras; i++ ) {
    /* unpack pc data */
    nobj = 24;
    rcode += dst_unpacki1_(&tadaq_.tf[i].pc.byteStream[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

    nobj = 1;

    rcode += dst_unpacki2_(&tadaq_.tf[i].id, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki2_(&tadaq_.tf[i].ntube, &nobj, bank, &tadaq_blen, &tadaq_maxlen);

    rcode += dst_unpacki4_(&tadaq_.tf[i].frame_ID, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].sync_monitor, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].trg_ID, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].trg_mask, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].store, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].sec_trig_code, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].tftrg_ID, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].swadr, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].board_mask, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].mode, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].mode2, &nobj, bank, &tadaq_blen, &tadaq_maxlen);

    nobj = 256;
    rcode += dst_unpacki4_(&tadaq_.tf[i].hit_pt[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

    nobj = 16;
    rcode += dst_unpacki4_(&tadaq_.tf[i].nc_bit[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

    nobj = 1;
    rcode += dst_unpacki4_(&tadaq_.tf[i].version, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].reset_counter, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].trigger_counter, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    rcode += dst_unpacki4_(&tadaq_.tf[i].extrigger_counter, &nobj, bank, &tadaq_blen, &tadaq_maxlen);

    /* unpack sdf data */
    for ( j=0; j<tadaq_.tf[i].ntube; j++ ) {
      nobj = tadaq_nfadc;
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].waveform[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 16;
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].hit[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 1;
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].pmt_id, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].version, &nobj, bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 4;
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].mean[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].disp[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 1;
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].trgmnt, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].peak, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].frmmnt, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].tmphit, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].trgcnt, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].mode, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].ctrl, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].thre, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].rstmnt, &nobj, bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 7;
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].rate[0], &nobj, bank, &tadaq_blen, &tadaq_maxlen);

      nobj = 1;
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].state, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpackr4_(&tadaq_.tf[i].sdf[j].hitthre, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpackr4_(&tadaq_.tf[i].sdf[j].ncthre, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpackr4_(&tadaq_.tf[i].sdf[j].average, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpackr4_(&tadaq_.tf[i].sdf[j].stdev, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
      rcode += dst_unpacki2_(&tadaq_.tf[i].sdf[j].tf_hit, &nobj, bank, &tadaq_blen, &tadaq_maxlen);
    }
  }

  return rcode ;
}

integer4 tadaq_common_to_dump_(integer4 *long_output)
{
  return tadaq_common_to_dumpf_(stdout,long_output);
}

integer4 tadaq_common_to_dumpf_(FILE* fp,integer4 *long_output) {
  integer4 i;
  integer1 *sitename[2] = { "BLACK_ROCK", "LONG_RIDGE" };
  (void)(long_output);
  fprintf(fp, "TADAQ :\n");

  fprintf(fp, "  site: %s  runid: %07d  trigid: %07d  code: %d\n", 
	  sitename[tadaq_.site], tadaq_.runid, tadaq_.trigid, 
	  tadaq_.event_code);
  fprintf(fp, "  %2d/%02d/%4d  %02d:%02d:%02d.%09d\n", tadaq_.month, 
	  tadaq_.day, tadaq_.year, tadaq_.hour, tadaq_.minute, 
	  tadaq_.second, tadaq_.nanosecond);
  fprintf(fp, "  no. participating cameras: %d\n", tadaq_.ncameras);

  for ( i=0; i<tadaq_.ncameras; i++ ) {

  }
  
  return 0;
} 

