/*
 * sttrk_dst.c 
 *
 * $Source: /hires_soft/cvsroot/bank/sttrk_dst.c,v $
 * $Log: sttrk_dst.c,v $
 * Revision 1.1  1999/06/28 21:06:14  tareq
 * Initial revision
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
#include "sttrk_dst.h"  

#ifndef DEGRAD
#define DEGRAD 57.2958
#endif

sttrk_dst_common sttrk_;  /* allocate memory to sttrk_common */

static integer4 sttrk_blen = 0; 
/* static integer4 sttrk_maxlen = sizeof(integer4) * 2 + sizeof(sttrk_dst_common); */
static integer1 *sttrk_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* sttrk_bank_buffer_ (integer4* sttrk_bank_buffer_size)
{
  (*sttrk_bank_buffer_size) = sttrk_blen;
  return sttrk_bank;
}



/* static void */
/* sttrk_bank_init(void) { */

/*   sttrk_bank = (integer1 *)calloc(sttrk_maxlen, sizeof(integer1)); */

/*   if(sttrk_bank==NULL) { */
/*     fprintf(stderr, "sttrk_bank_init: fail to assign memory to bank. Abort.\n"); */
/*     exit(0); */
/*   } */
/* } */

integer4 sttrk_common_to_bank_(void) {
	
/*   static integer4 id = STTRK_BANKID, ver = STTRK_BANKVERSION; */
/*   integer4 rcode, nobj; */
/*   integer4 i, nmir, ntube, timinfo=0; */

/*   if (sttrk_bank == NULL) sttrk_bank_init(); */

/*   /\* Initialize sttrk_blen, and pack the id and version to bank *\/ */

/*   if(rcode = dst_initbank_(&id, &ver, &sttrk_blen, &sttrk_maxlen, sttrk_bank)) */
/*     return rcode; */

/*   for(i=0; i<STTRK_MAXFIT; i++) { */
/*     timinfo = timinfo << 1; */
/*     if ( sttrk_.timinfo[i] == STTRK_TIMINFO_USED )  timinfo++; */
/*   } */
/*   if (rcode = dst_packi4asi2_( &timinfo, (nobj=1, &nobj), sttrk_bank,  */
/*                                &sttrk_blen, &sttrk_maxlen)) return rcode;  */

/*   for ( i=0; i<STTRK_MAXFIT; i++ ) { */

/*     if ( sttrk_.timinfo[i] == STTRK_TIMINFO_UNUSED ) continue; */


/*     if (rcode = dst_packi4_(&sttrk_.failmode[i], (nobj=1, &nobj), sttrk_bank,  */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode;  */

/*     if (sttrk_.failmode[i] != SUCCESS) continue; */

/*     if (rcode = dst_packr8_(&sttrk_.mchi2[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.rchi2[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.lchi2[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_packr8_(&sttrk_.mrp[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.rrp[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.lrp[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_packr8_(&sttrk_.mpsi[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.rpsi[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.lpsi[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_packr8_(&sttrk_.mthe[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.rthe[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.lthe[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_packr8_(&sttrk_.mphi[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.rphi[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(&sttrk_.lphi[i], (nobj=1, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_packr8_(sttrk_.mtkv[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.rtkv[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.ltkv[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_packr8_(sttrk_.mrpv[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.rrpv[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.lrpv[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_packr8_(sttrk_.mrpuv[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.rrpuv[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.lrpuv[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_packr8_(sttrk_.mshwn[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.rshwn[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.lshwn[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_packr8_(sttrk_.mcore[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.rcore[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.lcore[i], (nobj=3, &nobj), sttrk_bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     nmir = sttrk_.nmir[i]; */
/*     if (rcode = dst_packi4asi2_(&nmir, (nobj=1, &nobj), sttrk_bank,  */
/*                                 &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     if (rcode = dst_packi4asi2_(sttrk_.mir[i], (nobj=nmir, &nobj), sttrk_bank,  */
/*                                 &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     if (rcode = dst_packi4asi2_(sttrk_.mirntube[i], (nobj=nmir, &nobj), */
/*                         sttrk_bank, &sttrk_blen, &sttrk_maxlen)) return rcode;  */

/*     ntube = sttrk_.ntube[i]; */

/*     if (rcode = dst_packi4asi2_(&ntube, (nobj=1, &nobj), sttrk_bank,  */
/*                                 &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     if (rcode = dst_packi4asi2_(sttrk_.tube[i], (nobj=ntube, &nobj), */
/*                         sttrk_bank, &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     if (rcode = dst_packi4asi2_(sttrk_.tubemir[i], (nobj=ntube, &nobj), */
/*                         sttrk_bank, &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     if (rcode = dst_packi4asi2_(sttrk_.ig[i], (nobj=ntube, &nobj), sttrk_bank,  */
/*                                &sttrk_blen, &sttrk_maxlen)) return rcode;  */

/*     if (rcode = dst_packr8_(sttrk_.time[i], &nobj, sttrk_bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.timefit[i], &nobj, sttrk_bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.thetb[i], &nobj, sttrk_bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.sgmt[i], &nobj, sttrk_bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.asx[i], &nobj, sttrk_bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.asy[i], &nobj, sttrk_bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_packr8_(sttrk_.asz[i], &nobj, sttrk_bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*   } */
  return SUCCESS;
}


integer4 sttrk_bank_to_dst_(integer4 *NumUnit) {
	
  return dst_write_bank_(NumUnit, &sttrk_blen, sttrk_bank );
}

integer4 sttrk_common_to_dst_(integer4 *NumUnit) {

  integer4 rcode;
  
  if((rcode = sttrk_common_to_bank_())) {
    fprintf (stderr,"sttrk_common_to_bank_ ERROR : %ld\n", (long) rcode);
    exit(0);			 	
  }             
  
  if((rcode = sttrk_bank_to_dst_(NumUnit))) {
    fprintf (stderr,"sttrk_bank_to_dst_ ERROR : %ld\n", (long) rcode);
    exit(0);
  }

  return SUCCESS;
}

integer4 sttrk_bank_to_common_(integer1 *bank) {
  (void)(bank);
/*   integer4 rcode = 0; */
/*   integer4 nobj; */
/*   integer4 i, nmir, ntube, timinfo; */

/*   sttrk_blen = 2 * sizeof(integer4); /\* skip id and version  *\/ */

/*   if (rcode = dst_unpacki2asi4_(&timinfo, (nobj=1, &nobj), bank, */
/*                                 &sttrk_blen, &sttrk_maxlen)) return rcode;  */

/*   for(i=0; i<STTRK_MAXFIT; i++) { */
/*     if ( timinfo & 0x8000 )  sttrk_.timinfo[i] = STTRK_TIMINFO_USED; */
/*     else                     sttrk_.timinfo[i] = STTRK_TIMINFO_UNUSED; */
/*     timinfo = timinfo << 1; */
/*   } */

/*   for ( i=0; i<STTRK_MAXFIT; i++ ) { */

/*     if ( sttrk_.timinfo[i] == STTRK_TIMINFO_UNUSED ) continue; */

/*     if (rcode = dst_unpacki4_(&sttrk_.jday[i], (nobj=1, &nobj), bank, */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpacki4_(&sttrk_.jsec[i], (nobj=1, &nobj), bank, */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpacki4_(&sttrk_.msec[i], (nobj=1, &nobj), bank, */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpacki4_(&sttrk_.failmode[i], (nobj=1, &nobj), bank,  */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode;  */

/*     if (sttrk_.failmode[i] != SUCCESS) continue; */

/*     if (rcode = dst_unpackr8_(&sttrk_.mchi2[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.rchi2[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.lchi2[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpackr8_(&sttrk_.mrp[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.rrp[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.lrp[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpackr8_(&sttrk_.mpsi[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.rpsi[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.lpsi[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpackr8_(&sttrk_.mthe[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.rthe[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.lthe[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpackr8_(&sttrk_.mphi[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.rphi[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(&sttrk_.lphi[i], (nobj=1, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpackr8_(sttrk_.mtkv[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.rtkv[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.ltkv[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpackr8_(sttrk_.mrpv[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.rrpv[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.lrpv[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpackr8_(sttrk_.mrpuv[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.rrpuv[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.lrpuv[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpackr8_(sttrk_.mshwn[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.rshwn[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.lshwn[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpackr8_(sttrk_.mcore[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.rcore[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.lcore[i], (nobj=3, &nobj), bank,  */
/*                           &sttrk_blen, &sttrk_maxlen)) return rcode; */

/*     if (rcode = dst_unpacki2asi4_(&nmir, (nobj=1, &nobj), bank,  */
/*                                 &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     sttrk_.nmir[i] = nmir; */

/*     if (rcode = dst_unpacki2asi4_(sttrk_.mir[i], (nobj=nmir, &nobj), bank,  */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     if (rcode = dst_unpacki2asi4_(sttrk_.mirntube[i], (nobj=nmir, &nobj), bank,  */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode;  */

/*     if (rcode = dst_unpacki2asi4_(&ntube, (nobj=1, &nobj), bank,  */
/*                                 &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     sttrk_.ntube[i] = ntube; */

/*     if (rcode = dst_unpacki2asi4_(sttrk_.tube[i], (nobj=ntube, &nobj), bank,  */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     if (rcode = dst_unpacki2asi4_(sttrk_.tubemir[i], (nobj=ntube, &nobj), bank,  */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode;  */
/*     if (rcode = dst_unpacki2asi4_(sttrk_.ig[i], (nobj=ntube, &nobj), bank,  */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode;  */

/*     if (rcode = dst_unpackr8_(sttrk_.time[i], &nobj, bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.timefit[i], &nobj, bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.thetb[i], &nobj, bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.sgmt[i], &nobj, bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.asx[i], &nobj, bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.asy[i], &nobj, bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*     if (rcode = dst_unpackr8_(sttrk_.asz[i], &nobj, bank, */
/*                             &sttrk_blen, &sttrk_maxlen)) return rcode; */
/*   } */

  return SUCCESS;
}

integer4 sttrk_common_to_dump_(integer4 *long_output) {

  return sttrk_common_to_dumpf_(stdout, long_output);
}

integer4 sttrk_common_to_dumpf_(FILE* fp, integer4 *long_output) {
  (void)(fp);
  (void)(long_output);
/*   integer4 i, j, k; */
/*   static struct { */
/*     integer4  code; */
/*     integer1 *mess; */
/*   } trans2[] = { */
/*       STTRK_FIT_NOT_REQUESTED,       "Fit not requested", */
/*       STTRK_NOT_IMPLEMENTED,         "Fit not implemented", */
/*       STTRK_REQUIRED_BANKS_MISSING,   */
/*         "Bank(s) required for fit are missing or have failed", */
/*       STTRK_MISSING_TRAJECTORY_INFO, */
/*         "Bank(s) required for desired trajectory source are missing/failed", */
/*       STTRK_UPWARD_GOING_TRACK,      "Upward going track", */
/*       STTRK_TOO_FEW_GOOD_TUBES,      "Too few good tubes", */
/*       STTRK_FITTER_FAILURE,          "Fitter failed",     */
/*       STTRK_INSANE_TRAJECTORY, */
/*         "Trajectory (direction and/or core) unreasonable", */
/*       999,                          "Unknown failmode" /\* marks end of list *\/  */
/*   }; */

/*   fprintf(fp,"\nSTTRK bank. tubes: "); */
/*   for ( i=0; i<STTRK_MAXFIT; i++ ) { */
/*     if ( sttrk_.timinfo[i] == STTRK_TIMINFO_USED ) */
/*       fprintf(fp," %03d", sttrk_.ntube[i]); */
/*     else */
/*       fprintf(fp," -- "); */
/*   } */
/*   fprintf(fp,"\n"); */

/*   for(i=0; i<STTRK_MAXFIT; i++ ) { */
/*     if ( sttrk_.timinfo[i] == STTRK_TIMINFO_UNUSED ) continue; */

/*     fprintf(fp,"\n    -> Fit %2d jDay/Sec: %d/%5.5d %02d:%02d:%02d.%03d\n", */
/*              i+1, sttrk_.jday[i], sttrk_.jsec[i], sttrk_.msec[i] / 3600000, */
/*             (sttrk_.msec[i] / 60000) % 60, (sttrk_.msec[i] / 1000) % 60, */
/*              sttrk_.msec[i] % 1000); */

/*     if ( sttrk_.failmode[i] != SUCCESS ) { */
/*       for ( k=0; trans2[k].code!=999 ; k++ ) */
/*         if ( sttrk_.failmode[i] == trans2[k].code ) break; */
/*       fprintf(fp,"    %s\n", trans2[k].mess ); */
/*       continue;    /\* Nothing else to show for this fit *\/ */
/*     } */

/*     fprintf(fp,"chi2 :%10.2f   range= [%10.2f, %10.2f]\n", */
/*             sttrk_.mchi2[i], sttrk_.rchi2[i], sttrk_.lchi2[i]); */
/*     fprintf(fp,"rp   :%10.2f   range= [%10.2f, %10.2f]  meters\n", */
/*             sttrk_.mrp[i], sttrk_.rrp[i], sttrk_.lrp[i]); */
/*     fprintf(fp,"psi  :%10.2f   range= [%10.2f, %10.2f]  degrees\n", */
/*           sttrk_.mpsi[i]*DEGRAD, sttrk_.rpsi[i]*DEGRAD, sttrk_.lpsi[i]*DEGRAD); */
/*     fprintf(fp,"theta:%10.2f   range= [%10.2f, %10.2f]  degrees\n", */
/*           sttrk_.mthe[i]*DEGRAD, sttrk_.rthe[i]*DEGRAD, sttrk_.lthe[i]*DEGRAD); */
/*     fprintf(fp,"phi  :%10.2f   range= [%10.2f, %10.2f]  degrees\n", */
/*           sttrk_.mphi[i]*DEGRAD, sttrk_.rphi[i]*DEGRAD, sttrk_.lphi[i]*DEGRAD); */
/*     fprintf(fp,"\n"); */

/*     fprintf(fp,"core location x: %10.2f,  range = [ %10.2f, %10.2f]  meters\n", */
/*                sttrk_.mcore[i][0], sttrk_.rcore[i][0], sttrk_.lcore[i][0] ); */
/*     fprintf(fp,"core location y: %10.2f,  range = [ %10.2f, %10.2f]  meters\n", */
/*                sttrk_.mcore[i][1], sttrk_.rcore[i][1], sttrk_.lcore[i][1] ); */
/*     fprintf(fp,"\n"); */
/*     fprintf(fp, "shower direction: ( %7.4f, %7.4f, %7.4f )\n", */
/*                sttrk_.mtkv[i][0], sttrk_.mtkv[i][1], sttrk_.mtkv[i][2] ); */
/*     fprintf(fp, "      dir bound1: ( %7.4f, %7.4f, %7.4f )\n", */
/*                sttrk_.rtkv[i][0], sttrk_.rtkv[i][1], sttrk_.rtkv[i][2] ); */
/*     fprintf(fp, "      dir bound2: ( %7.4f, %7.4f, %7.4f )\n", */
/*                sttrk_.ltkv[i][0], sttrk_.ltkv[i][1], sttrk_.ltkv[i][2] ); */

/*   } */
/*   if( *long_output == 1 ) { */
/*     for(i=0; i<STTRK_MAXFIT; i++ ) { */
/*       if( sttrk_.timinfo[i] == STTRK_TIMINFO_UNUSED ) continue; */
/*       if( sttrk_.failmode[i] != SUCCESS ) continue; */

/*       fprintf(fp,"\n    -> Fit %2d Tubes\n", i+1); */

/*       fprintf(fp," mir tube     time (ns)  timefit    sgmt view_ang     asx     asy     asz   ig\n"); */

/*       for(j=0; j<sttrk_.ntube[i]; j++) { */
/*         fprintf(fp,"  %2d  %03d %10.1lf %10.1lf %8.1lf  %7.2lf %7.4lf %7.4lf %7.4lf %3d\n",  */
/*                    sttrk_.tubemir[i][j], sttrk_.tube[i][j],  */
/*                    sttrk_.time[i][j], sttrk_.timefit[i][j], sttrk_.sgmt[i][j],  */
/*                    sttrk_.thetb[i][j]*DEGRAD,  */
/*                    sttrk_.asx[i][j], sttrk_.asy[i][j], sttrk_.asz[i][j], */
/*                    sttrk_.ig[i][j] ); */
/*       } */
/*     } */
/*   } */
  return SUCCESS;
}
