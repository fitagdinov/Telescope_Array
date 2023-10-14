/*
 * $Source:$
 * $Log:$
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
#include "stnpe_dst.h"  

stnpe_dst_common stnpe_;  /* allocate memory to stnpe_common */

static integer4 stnpe_blen = 0; 
static integer4 stnpe_maxlen = sizeof(integer4) * 2 + sizeof(stnpe_dst_common);
static integer1 *stnpe_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* stnpe_bank_buffer_ (integer4* stnpe_bank_buffer_size)
{
  (*stnpe_bank_buffer_size) = stnpe_blen;
  return stnpe_bank;
}



#ifndef HIRES_FADC_MIR
#define HIRES_FADC_MIR 5
#endif

static void stnpe_bank_init()
{
  stnpe_bank = (integer1 *)calloc(stnpe_maxlen, sizeof(integer1));
  if (stnpe_bank==NULL)
    {
      fprintf(stderr, "stnpe_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 stnpe_common_to_bank_()
{	
  static integer4 id = STNPE_BANKID, ver = STNPE_BANKVERSION;
  integer4 rcode;
  integer4 nobj;

  if (stnpe_bank == NULL) stnpe_bank_init();

  /* Initialize stnpe_blen, and pack the id and version to bank */
  if ((rcode = dst_initbank_(&id, &ver, &stnpe_blen, &stnpe_maxlen, stnpe_bank))) return rcode;

  // jday, jsec, msec
  if ((rcode = dst_packi4_    (&stnpe_.jday, (nobj=3, &nobj), stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  // julian, jsecond, jclkcnt
  if ((rcode = dst_packi4_    (&stnpe_.julian, (nobj=3, &nobj), stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  // calib_source, neye, nmir, ntube
  if ((rcode = dst_packi4asi2_(&stnpe_.calib_source, (nobj=4, &nobj), stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  nobj = stnpe_.neye; 
  if ((rcode = dst_packi4asi2_(stnpe_.eyeid, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(stnpe_.eye_nmir, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(stnpe_.eye_ntube, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  nobj = stnpe_.nmir; 
  if ((rcode = dst_packi4asi2_(stnpe_.mirid, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(stnpe_.mir_eye, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(stnpe_.mir_rev, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4_    (stnpe_.mirevtno, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(stnpe_.mir_ntube, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4_    (stnpe_.mirtime_ns, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4_    (stnpe_.second, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4_    (stnpe_.clkcnt, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  nobj = stnpe_.ntube;
  if ((rcode = dst_packi4asi2_(stnpe_.tube_eye, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(stnpe_.tube_mir, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(stnpe_.tubeid, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(stnpe_.thresh, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode ;
  if ((rcode = dst_packr4_    (stnpe_.npe, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode ;
  if ((rcode = dst_packr4_    (stnpe_.time, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode ;
  if ((rcode = dst_packi4_    (stnpe_.dtime, &nobj, stnpe_bank, &stnpe_blen, &stnpe_maxlen))) return rcode ;

  return SUCCESS;
}

integer4 stnpe_bank_to_dst_(integer4 *NumUnit)
{	
  return dst_write_bank_(NumUnit, &stnpe_blen, stnpe_bank );
}

integer4 stnpe_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = stnpe_common_to_bank_()))
    {
      fprintf (stderr,"stnpe_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }             
  if ((rcode = stnpe_bank_to_dst_(NumUnit)))
    {
      fprintf (stderr,"stnpe_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }
  return SUCCESS;
}

integer4 stnpe_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  stnpe_blen = 2 * sizeof(integer4); /* skip id and version  */

  // jday, jsec, msec
  if ((rcode = dst_unpacki4_(    &(stnpe_.jday), (nobj = 3, &nobj) ,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  // julian, jsecond, jclkcnt
  if ((rcode = dst_unpacki4_(    &(stnpe_.julian), (nobj = 3, &nobj) ,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  // calib_source, neye, nmir, ntube
  if ((rcode = dst_unpacki2asi4_(&(stnpe_.calib_source), (nobj = 4, &nobj), bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  nobj = stnpe_.neye; 
  if ((rcode = dst_unpacki2asi4_(stnpe_.eyeid, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(stnpe_.eye_nmir, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(stnpe_.eye_ntube, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  nobj = stnpe_.nmir; 
  if ((rcode = dst_unpacki2asi4_(stnpe_.mirid, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(stnpe_.mir_eye, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(stnpe_.mir_rev, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_    (stnpe_.mirevtno, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(stnpe_.mir_ntube, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_    (stnpe_.mirtime_ns, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_    (stnpe_.second, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_    (stnpe_.clkcnt, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  nobj = stnpe_.ntube; 
  if ((rcode = dst_unpacki2asi4_(stnpe_.tube_eye, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(stnpe_.tube_mir, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(stnpe_.tubeid, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(stnpe_.thresh, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_(stnpe_.npe, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_(stnpe_.time, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_(stnpe_.dtime, &nobj,bank, &stnpe_blen, &stnpe_maxlen))) return rcode;

  return SUCCESS;
}

integer4 stnpe_time_fprint_(FILE* fp,integer4 *second,integer4 *clkcnt)
{
  // copy of fraw1_time_fprint_ 
  integer4 hh,mm,ss;
  double cc;
  ss = stnpe_.jsecond + *second;
  cc =  (((double) 50) * ((double) *clkcnt))/((double) 3);
  cc += (double) stnpe_.jclkcnt;
  if (cc > (double) 999999999) {
    cc -= (double) 1000000000;
    ss ++;
  }
  hh = ss/3600;
  mm = (ss/60)%60;
  ss = ss%60;
  /*
  fprintf (fp," second %d clkcnt %d\n",*second,*clkcnt);
  fprintf (fp," jsecond %d jclkcnt %d\n",stnpe_.jsecond,stnpe_.jclkcnt);
  */
  fprintf (fp," event store start time -- %d:%02d:%02d.%09.0f\n",hh,mm,ss,cc);
  return 0;
}

integer4 stnpe_time_print_(integer4 *second,integer4 *clkcnt)
{
  // copy of fraw1_time_print_ 
  return stnpe_time_fprint_(stdout,second,clkcnt);
}

integer4 stnpe_common_to_dump_(integer4 *long_output)
{
  return stnpe_common_to_dumpf_(stdout, long_output);
}

integer4 stnpe_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i;
  fprintf(fp, "\nSTNPE jDay/Sec: %d/%5.5d %02d:%02d:%02d.%03d ",
          stnpe_.jday, stnpe_.jsec, stnpe_.msec / 3600000,
          (stnpe_.msec / 60000) % 60, (stnpe_.msec / 1000) % 60,
          stnpe_.msec % 1000) ;
  fprintf(fp, "eyes: %2d  mirs: %2d  tubes: %4d \n",
          stnpe_.neye, stnpe_.nmir, stnpe_.ntube);

  /* -------------- eye info --------------------------*/
  for (i = 0; i < stnpe_.neye; i++) {
    fprintf(fp, " eye %2d nmir %2d ntube %4d \n",
	    stnpe_.eyeid[i], stnpe_.eye_nmir[i], stnpe_.eye_ntube[i]);
  }
  /* -------------- mir info --------------------------*/
  for (i = 0; i < stnpe_.nmir; i++) {
    fprintf(fp, " eye %2d mir %2d Rev%d evt: %9d  tubes: %3d  ",
	    stnpe_.mir_eye[i], stnpe_.mirid[i], stnpe_.mir_rev[i],
	    stnpe_.mirevtno[i], stnpe_.mir_ntube[i]);
    if ( stnpe_.mir_rev[i] != HIRES_FADC_MIR ) {
      fprintf(fp, "time: %10dnS\n", stnpe_.mirtime_ns[i] );
    }
    else {
      stnpe_time_fprint_(fp,&stnpe_.second[i],&stnpe_.clkcnt[i]);
    }
  }
  
  /* If long output is desired, show tube information */

  if ( *long_output == 1 ) {
    for (i = 0; i < stnpe_.ntube; i++) {
      fprintf(fp, "e %2d m %2d t %3d th:%4d ",
	      stnpe_.tube_eye[i],  stnpe_.tube_mir[i],
	      stnpe_.tubeid[i], stnpe_.thresh[i]);
      fprintf(fp, "npe:%9.1f time:%10.1f dtime: %8d\n",
	      stnpe_.npe[i], stnpe_.time[i], stnpe_.dtime[i]);
    }
  }

  fprintf(fp,"\n");

  return SUCCESS;
}

