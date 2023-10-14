/*
 * $Source: /hires_soft/cvsroot/bank/mcraw_dst.c,v $
 * $Log: mcraw_dst.c,v $
 * Revision 1.1  2006/05/18 17:40:11  tareq
 * *** empty log message ***
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
#include "mcraw_dst.h"  

mcraw_dst_common mcraw_;  /* allocate memory to mcraw_common */

static integer4 mcraw_blen = 0; 
static integer4 mcraw_maxlen = sizeof(integer4) * 2 + sizeof(mcraw_dst_common);
static integer1 *mcraw_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* mcraw_bank_buffer_ (integer4* mcraw_bank_buffer_size)
{
  (*mcraw_bank_buffer_size) = mcraw_blen;
  return mcraw_bank;
}



static void mcraw_bank_init()
{
  mcraw_bank = (integer1 *)calloc(mcraw_maxlen, sizeof(integer1));
  if (mcraw_bank==NULL)
    {
      fprintf(stderr, "mcraw_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 mcraw_common_to_bank_()
{	
  static integer4 id = MCRAW_BANKID, ver = MCRAW_BANKVERSION;
  integer4 rcode;
  integer4 nobj;

  if (mcraw_bank == NULL) mcraw_bank_init();

  /* Initialize mcraw_blen, and pack the id and version to bank */
  if ((rcode = dst_initbank_(&id, &ver, &mcraw_blen, &mcraw_maxlen, mcraw_bank))) return rcode;

  // jday, jsec, msec
  if ((rcode = dst_packi4_    (&mcraw_.jday, (nobj=3, &nobj), mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;

  // neye, nmir, ntube
  if ((rcode = dst_packi4asi2_(&mcraw_.neye, (nobj=3, &nobj), mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;

  nobj = mcraw_.neye; 
  if ((rcode = dst_packi4asi2_(mcraw_.eyeid, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;

  nobj = mcraw_.nmir; 
  if ((rcode = dst_packi4asi2_(mcraw_.mirid, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.mir_eye, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.mir_rev, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4_    (mcraw_.mirevtno, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.mir_ntube, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4_    (mcraw_.mirtime_ns, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;

  nobj = mcraw_.ntube;
  if ((rcode = dst_packi4asi2_(mcraw_.tube_eye, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.tube_mir, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.tubeid, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.qdca, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.qdcb, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.tdc, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.tha, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(mcraw_.thb, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode ;
  if ((rcode = dst_packr4_    (mcraw_.prxf, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode ;
  if ((rcode = dst_packr4_    (mcraw_.thcal1, &nobj, mcraw_bank, &mcraw_blen, &mcraw_maxlen))) return rcode ;

  return SUCCESS;
}

integer4 mcraw_bank_to_dst_(integer4 *NumUnit)
{	
  return dst_write_bank_(NumUnit, &mcraw_blen, mcraw_bank );
}

integer4 mcraw_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = mcraw_common_to_bank_()))
    {
      fprintf (stderr,"mcraw_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }             
  if ((rcode = mcraw_bank_to_dst_(NumUnit)))
    {
      fprintf (stderr,"mcraw_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }
  return SUCCESS;
}

integer4 mcraw_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  mcraw_blen = 2 * sizeof(integer4); /* skip id and version  */

  // jday, jsec, msec
  if ((rcode = dst_unpacki4_(    &(mcraw_.jday), (nobj = 3, &nobj) ,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;

  // neye, nmir, ntube
  if ((rcode = dst_unpacki2asi4_(&(mcraw_.neye), (nobj = 3, &nobj), bank, &mcraw_blen, &mcraw_maxlen))) return rcode;

  nobj = mcraw_.neye; 
  if ((rcode = dst_unpacki2asi4_(mcraw_.eyeid, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;

  nobj = mcraw_.nmir; 
  if ((rcode = dst_unpacki2asi4_(mcraw_.mirid, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.mir_eye, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.mir_rev, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_    (mcraw_.mirevtno, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.mir_ntube, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_    (mcraw_.mirtime_ns, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;

  nobj = mcraw_.ntube; 
  if ((rcode = dst_unpacki2asi4_(mcraw_.tube_eye, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.tube_mir, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.tubeid, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.qdca, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.qdcb, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.tdc, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.tha, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(mcraw_.thb, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_(mcraw_.prxf, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_(mcraw_.thcal1, &nobj,bank, &mcraw_blen, &mcraw_maxlen))) return rcode;

  return SUCCESS;
}

integer4 mcraw_common_to_dump_(integer4 *long_output)
{
  return mcraw_common_to_dumpf_(stdout, long_output);
}

integer4 mcraw_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i;
  fprintf(fp, "\nMCRAW jDay/Sec: %d/%5.5d %02d:%02d:%02d.%03d ",
          mcraw_.jday, mcraw_.jsec, mcraw_.msec / 3600000,
          (mcraw_.msec / 60000) % 60, (mcraw_.msec / 1000) % 60,
          mcraw_.msec % 1000) ;
  fprintf(fp, "eyes: %2d  mirs: %2d  tubes: %4d \n",
          mcraw_.neye, mcraw_.nmir, mcraw_.ntube);

  /* -------------- mir info --------------------------*/
  for (i = 0; i < mcraw_.nmir; i++) {
    fprintf(fp, " eye %2d mir %2d Rev%d evt: %9d  tubes: %3d  ",
	    mcraw_.mir_eye[i], mcraw_.mirid[i], mcraw_.mir_rev[i],
	    mcraw_.mirevtno[i], mcraw_.mir_ntube[i]);
    fprintf(fp, "time: %10dnS\n",
	    mcraw_.mirtime_ns[i] );
  }

  
  /* If long output is desired, show tube information */

  if ( *long_output == 1 ) {
    for (i = 0; i < mcraw_.ntube; i++) {
      fprintf(fp, "e %2d m %2d t %3d qA:%6d qB:%6d ",
	      mcraw_.tube_eye[i],  mcraw_.tube_mir[i],
	      mcraw_.tubeid[i], mcraw_.qdca[i], mcraw_.qdcb[i]);
      fprintf(fp, "t:%6d th:%4d/%4d ",
	      mcraw_.tdc[i], mcraw_.tha[i], mcraw_.thb[i]);
      fprintf(fp, "pX:%9.1f t1:%9.3f\n", mcraw_.prxf[i], mcraw_.thcal1[i]);
    }
  }

  fprintf(fp,"\n");

  return SUCCESS;
}

