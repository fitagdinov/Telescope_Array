/*
 * mc04_dst.c 
 *
 * $Source: /hires_soft/cvsroot/bank/mc04_dst.c,v $
 * $Log: mc04_dst.c,v $
 * Revision 1.2  2008/11/13 22:41:21  doug
 * Updated to include current parameters of GH fit.
 *
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
#include "mc04_dst.h"  

#ifndef RADDEG
#define RADDEG 57.2957795131
#endif

mc04_dst_common mc04_;  /* allocate memory to mc04_common */

static integer4 mc04_blen = 0; 
static integer4 mc04_maxlen = sizeof(integer4)*2 + sizeof(mc04_dst_common);
static integer1 *mc04_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* mc04_bank_buffer_ (integer4* mc04_bank_buffer_size)
{
  (*mc04_bank_buffer_size) = mc04_blen;
  return mc04_bank;
}



static void 
mc04_bank_init(void)
{
  mc04_bank = (integer1 *)calloc(mc04_maxlen, sizeof(integer1));
  if (mc04_bank==NULL)
    {
      fprintf(stderr, 
	      "mc04_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4
mc04_common_to_bank_(void)
{  
  static integer4 id = MC04_BANKID;
  static integer4 ver = MC04_BANKVERSION;

  integer4 rcode;
  integer4 nobj;

  integer4 ieye;
  integer4 imir;
  integer4 i;

  if (mc04_bank == NULL) mc04_bank_init();

  /* Initialize mc04_blen, and pack the id and version to bank */

  if (( rcode = dst_initbank_(&id, &ver, &mc04_blen, &mc04_maxlen, mc04_bank))) return rcode;

  // energy, csmax, x0, xmax, xfin, rini[], rfin[], uthat[],  theta

  if ((rcode = dst_packr8_( &mc04_.energy, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.csmax,  (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.x0,     (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.x1,     (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.xmax,   (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.lambda, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.xfin,   (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( mc04_.rini,    (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( mc04_.rfin,    (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( mc04_.uthat,   (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.theta,  (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( mc04_.Rpvec,   (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( mc04_.Rcore,   (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.Rp,     (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;

  // detid, maxeye, neye, nmir, ntube, if_eye
  if ((rcode = dst_packi4_( &mc04_.detid,  (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &mc04_.maxeye, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &mc04_.neye,   (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &mc04_.nmir,   (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &mc04_.ntube,  (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packi4_( mc04_.if_eye,  (nobj=mc04_.maxeye, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;

  // rpvec, rcore, shwn, rp, psi
  for ( ieye=0; ieye<mc04_.maxeye; ++ieye ) {
    if ( mc04_.if_eye[ieye] != 1 ) continue;

    if ((rcode = dst_packr8_( mc04_.rsite[ieye],  (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packr8_( mc04_.rpvec[ieye],  (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packr8_( mc04_.rcore[ieye],  (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packr8_( mc04_.shwn [ieye],  (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packr8_( &(mc04_.rp [ieye]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packr8_( &(mc04_.psi[ieye]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  }

  // aero_vod, aero_hal, aero_vsh, aero_mlh
  if ((rcode = dst_packr8_( &mc04_.aero_vod, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.aero_hal, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.aero_vsh, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.aero_mlh, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;

  // la_site[3], la_wavlen, fl_totpho, fl_twidth
  if ((rcode = dst_packr8_( mc04_.la_site,    (nobj=3, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.la_wavlen, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.fl_totpho, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packr8_( &mc04_.fl_twidth, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;

  // iprim, eventNr, setNr, iseed1, iseed2
  if ((rcode = dst_packi4_( &mc04_.iprim,   (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &mc04_.eventNr, (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &mc04_.setNr,   (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &mc04_.iseed1,  (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_packi4_( &mc04_.iseed2,  (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;

  for ( ieye=0; ieye<mc04_.maxeye; ++ieye ) {
    if ( mc04_.if_eye[ieye] != 1 ) continue;

    if ((rcode = dst_packi4_( &(mc04_.eyeid    [ieye]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packi4_( &(mc04_.eye_nmir [ieye]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packi4_( &(mc04_.eye_ntube[ieye]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  }

  for ( imir=0; imir<mc04_.nmir; ++imir ) {
    if ((rcode = dst_packi4_( &(mc04_.mirid  [imir]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packi4_( &(mc04_.mir_eye[imir]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packi4_( &(mc04_.thresh [imir]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
  }

  for ( i=0; i<mc04_.ntube; ++i ) {
    if ((rcode = dst_packi4_( &(mc04_.tubeid  [i]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packi4_( &(mc04_.tube_mir[i]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packi4_( &(mc04_.tube_eye[i]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_packi4_( &(mc04_.pe      [i]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ( ver > 1 ) {
      if ((rcode = dst_packi4_( &(mc04_.triggered[i]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    }

    if ( ver > 0 ) {
      if ((rcode = dst_packr4_( &(mc04_.t_tmean[i]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
      if ((rcode = dst_packr4_( &(mc04_.t_trms [i]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
      if ((rcode = dst_packr4_( &(mc04_.t_tmin [i]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
      if ((rcode = dst_packr4_( &(mc04_.t_tmax [i]), (nobj=1, &nobj), mc04_bank, &mc04_blen, &mc04_maxlen))) return rcode;
    }
  }

  return SUCCESS;
}


integer4
mc04_bank_to_dst_(integer4 *NumUnit)
{  
  return dst_write_bank_(NumUnit, &mc04_blen, mc04_bank );
}

integer4
mc04_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if (( rcode = mc04_common_to_bank_() ))
    {
      fprintf (stderr,"mc04_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);         
    }             
  if ((rcode = mc04_bank_to_dst_(NumUnit) ))
    {
      fprintf (stderr,"mc04_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);         
    }
  return SUCCESS;
}

integer4
mc04_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  integer4 ieye;
  integer4 imir;
  integer4 i;

  integer4 bankid, bankversion;

  mc04_blen = 0;

  if ((rcode = dst_unpacki4_( &bankid,      (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &bankversion, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;

  //  mc04_blen = 2 * sizeof(integer4); /* skip id and version  */

  // energy, csmax, x0, xmax, xfin, rini[], rfin[], uthat[], theta
  if ((rcode = dst_unpackr8_( &mc04_.energy, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.csmax,  (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.x0,     (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.x1,     (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.xmax,   (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.lambda, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.xfin,   (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( mc04_.rini,    (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( mc04_.rfin,    (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( mc04_.uthat,   (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.theta,  (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( mc04_.Rpvec,   (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( mc04_.Rcore,   (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.Rp,     (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;


  // detid, maxeye, neye, nmir, ntube, if_eye
  if ((rcode = dst_unpacki4_( &mc04_.detid,  (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &mc04_.maxeye, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &mc04_.neye,   (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &mc04_.nmir,   (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &mc04_.ntube,  (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( mc04_.if_eye,  (nobj=mc04_.maxeye, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;

  // rsite, rpvec, rcore, shwn, rp, psi
  for ( ieye=0; ieye<mc04_.maxeye; ++ieye ) {
    if ( mc04_.if_eye[ieye] != 1 ) continue;

    if ((rcode = dst_unpackr8_( mc04_.rsite[ieye],  (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpackr8_( mc04_.rpvec[ieye],  (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpackr8_( mc04_.rcore[ieye],  (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpackr8_( mc04_.shwn [ieye],  (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpackr8_( &(mc04_.rp [ieye]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpackr8_( &(mc04_.psi[ieye]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  }

  // aero_vod, aero_hal, aero_vsh, aero_mlh
  if ((rcode = dst_unpackr8_( &mc04_.aero_vod, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.aero_hal, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.aero_vsh, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.aero_mlh, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;

  // la_site[3], la_wavlen, fl_totpho, fl_twidth
  if ((rcode = dst_unpackr8_( mc04_.la_site,    (nobj=3, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.la_wavlen, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.fl_totpho, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_( &mc04_.fl_twidth, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;

  // iprim, eventNr, setNr, iseed1, iseed2
  if ((rcode = dst_unpacki4_( &mc04_.iprim, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &mc04_.eventNr, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &mc04_.setNr, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &mc04_.iseed1, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &mc04_.iseed2, (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;


  for ( ieye=0; ieye<mc04_.maxeye; ++ieye ) {
    if ( mc04_.if_eye[ieye] != 1 ) continue;

    if ((rcode = dst_unpacki4_( &(mc04_.eyeid    [ieye]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpacki4_( &(mc04_.eye_nmir [ieye]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpacki4_( &(mc04_.eye_ntube[ieye]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  }

  for ( imir=0; imir<mc04_.nmir; ++imir ) {
    if ((rcode = dst_unpacki4_( &(mc04_.mirid  [imir]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpacki4_( &(mc04_.mir_eye[imir]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpacki4_( &(mc04_.thresh [imir]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
  }

  for ( i=0; i<mc04_.ntube; ++i ) {
    if ((rcode = dst_unpacki4_( &(mc04_.tubeid  [i]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpacki4_( &(mc04_.tube_mir[i]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpacki4_( &(mc04_.tube_eye[i]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    if ((rcode = dst_unpacki4_( &(mc04_.pe      [i]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;

    if (bankversion > 1) {
      if ((rcode = dst_unpacki4_( &(mc04_.triggered[i]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    }
    else {
      mc04_.triggered[i] = 1;
    }
    if (bankversion > 0) {
      if ((rcode = dst_unpackr4_( &(mc04_.t_tmean[i]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
      if ((rcode = dst_unpackr4_( &(mc04_.t_trms [i]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
      if ((rcode = dst_unpackr4_( &(mc04_.t_tmin [i]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
      if ((rcode = dst_unpackr4_( &(mc04_.t_tmax [i]), (nobj=1, &nobj), bank, &mc04_blen, &mc04_maxlen))) return rcode;
    }
    else {
      mc04_.t_tmean[i] = 0.0f;
      mc04_.t_tmean[i] = 0.0f;
      mc04_.t_tmean[i] = 0.0f;
      mc04_.t_tmean[i] = 0.0f;
    }
  }
  return SUCCESS;
}

integer4
mc04_common_to_dump_(integer4 *long_output)
{
  return mc04_common_to_dumpf_(stdout, long_output);
}

integer4
mc04_common_to_dumpf_(FILE* fp, integer4 *long_output) {

/*   integer4 ieye, imir; */
  integer4 i;

  fprintf(fp, "\nMC04______________________________________\n");
  fprintf(fp, "setNr/eventNr:  %d/%d\t",   mc04_.setNr, mc04_.eventNr);
  fprintf(fp, "triggered neye/nmir/ntube:  %d / %d / %d\n",
	  mc04_.neye, mc04_.nmir, mc04_.ntube);

  if( mc04_.iprim == MC04_LASER) {
    fprintf(fp, "event type:  LASER\n");
    fprintf(fp, "      site:  %8lg  %8lg  %8lg\n", mc04_.la_site[0],
              mc04_.la_site[1], mc04_.la_site[2]);
    fprintf(fp, " direction:  %8lg  %8lg  %8lg\n", mc04_.uthat[0],
              mc04_.uthat[1], mc04_.uthat[2]);
    fprintf(fp, "        Rp:  %lg m\n\n", mc04_.rp[0]);
  }
  else if( mc04_.iprim == MC04_FLASHER) {
    fprintf(fp, "event type:  FLASHER\n");
    fprintf(fp, "      site:  %8lg  %8lg  %8lg\n", mc04_.la_site[0],
              mc04_.la_site[1], mc04_.la_site[2]);
    fprintf(fp, " direction:  %8lg  %8lg  %8lg\n", mc04_.uthat[0],
              mc04_.uthat[1], mc04_.uthat[2]);
    fprintf(fp, "        Rp:  %lg m\n\n", mc04_.rp[0]);
  }
  else {
    fprintf(fp, "event type:  SHOWER\n");
    fprintf(fp, "    energy:  %3.2le eV\t", mc04_.energy);
    if(mc04_.iprim == MC04_PROTON) fprintf(fp, "   primary:  proton\n");
    if(mc04_.iprim == MC04_IRON)   fprintf(fp, "   primary:  iron\n");
    if(mc04_.iprim == MC04_GAMMA)  fprintf(fp, "   primary:  gamma\n");
    if(mc04_.iprim == MC04_He)     fprintf(fp, "   primary:  helium\n");
    if(mc04_.iprim == MC04_CNO)    fprintf(fp, "   primary:  CNO\n");
    if(mc04_.iprim == MC04_MgSi)   fprintf(fp, "   primary:  MgSi\n");
    fprintf(fp, "        x0:  %lg gm/cm^2\t", mc04_.x0);
    fprintf(fp, "        x1:  %lg gm/cm^2\n", mc04_.x1);
    fprintf(fp, "      xmax:  %lg gm/cm^2\t", mc04_.xmax);
    fprintf(fp, "    lambda:  %lg gm/cm^2\n", mc04_.lambda);
    fprintf(fp, "     csmax:  %lg\n\n", mc04_.csmax);

    fprintf(fp, "      theta: %12.4f,  uthat:  %14.10f  %14.10f  %14.10f\n", mc04_.theta*RADDEG, mc04_.uthat[0], mc04_.uthat[1], mc04_.uthat[2]);
    fprintf(fp, "     CLF Rp: %12.2f,  Rpvec:  %14.3f  %14.3f  %14.3f (m)\n\n", mc04_.Rp, mc04_.Rpvec[0], mc04_.Rpvec[1], mc04_.Rpvec[2]);

    fprintf(fp, "     BR  Rp/psi: %12.2f/%12.4f,  Rpvec:  %12.3f  %12.3f  %12.3f (m)\n",   mc04_.rp[0], mc04_.psi[0]*RADDEG, mc04_.rpvec[0][0], mc04_.rpvec[0][1], mc04_.rpvec[0][2]);
    fprintf(fp, "     LR  Rp/psi: %12.2f/%12.4f,  Rpvec:  %12.3f  %12.3f  %12.3f (m)\n",   mc04_.rp[1], mc04_.psi[1]*RADDEG, mc04_.rpvec[1][0], mc04_.rpvec[1][1], mc04_.rpvec[1][2]);
    fprintf(fp, "     MD  Rp/psi: %12.2f/%12.4f,  Rpvec:  %12.3f  %12.3f  %12.3f (m)\n\n", mc04_.rp[2], mc04_.psi[2]*RADDEG, mc04_.rpvec[2][0], mc04_.rpvec[2][1], mc04_.rpvec[2][2]);
  }

  fprintf(fp, "aerosols:  vaod %8.3f  hal %8.3f (km) vsh %8.3f (km) mlh  %8.3f (km)\n", mc04_.aero_vod, 1e-3*mc04_.aero_hal, 1e-3*mc04_.aero_vsh, 1e-3*mc04_.aero_mlh );

  if ( *long_output == 1 ) {
    fprintf(fp,"\n");
    for(i=0;i<mc04_.ntube;i++) {
      fprintf(fp, "e:%2d m:%3d  t:%4d  pe:%8d  t_mean: %8.2f  rms: %8.2f  min: %8.2f  max: %8.2f  trig: %1d\n", 
	      mc04_.tube_eye[i], mc04_.tube_mir[i],
	      mc04_.tubeid  [i], mc04_.pe      [i], mc04_.t_tmean[i], mc04_.t_trms[i], mc04_.t_tmin[i], mc04_.t_tmax[i],
	      mc04_.triggered[i]);
    }
    fprintf(fp, "\n");
  }

  return SUCCESS;
}




