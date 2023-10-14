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
#include "tslew_dst.h"  

tslew_dst_common tslew_;  /* allocate memory to tslew_common */

static integer4 tslew_blen = 0; 
static integer4 tslew_maxlen = sizeof(integer4) * 2 + sizeof(tslew_dst_common);
static integer1 *tslew_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tslew_bank_buffer_ (integer4* tslew_bank_buffer_size)
{
  (*tslew_bank_buffer_size) = tslew_blen;
  return tslew_bank;
}



static void tslew_bank_init()
{
  tslew_bank = (integer1 *)calloc(tslew_maxlen, sizeof(integer1));
  if (tslew_bank==NULL)
    {
      fprintf(stderr, "tslew_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 tslew_common_to_bank_()
{	
  static integer4 id = TSLEW_BANKID, ver = TSLEW_BANKVERSION;
  integer4 rcode;
  integer4 nobj;

  if (tslew_bank == NULL) tslew_bank_init();

  /* Initialize tslew_blen, and pack the id and version to bank */
  if ((rcode = dst_initbank_(&id, &ver, &tslew_blen, &tslew_maxlen, tslew_bank))) return rcode;

  // neye, nmir, ntube
  if ((rcode = dst_packi4asi2_(&tslew_.neye, (nobj=3, &nobj), tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode;

  nobj = tslew_.neye; 
  if ((rcode = dst_packi4asi2_(tslew_.eyeid, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode;

  nobj = tslew_.nmir; 
  if ((rcode = dst_packi4asi2_(tslew_.mirid, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(tslew_.mir_eye, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(tslew_.mir_rev, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(tslew_.mir_ntube, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode;

  nobj = tslew_.ntube;
  if ((rcode = dst_packi4asi2_(tslew_.tube_eye, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(tslew_.tube_mir, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(tslew_.tubeid, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_packi4asi2_(tslew_.thb, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode ;
  if ((rcode = dst_packr4_    (tslew_.thcal1, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode ;
  if ((rcode = dst_packr4_    (tslew_.tcorr, &nobj, tslew_bank, &tslew_blen, &tslew_maxlen))) return rcode ;

  return SUCCESS;
}

integer4 tslew_bank_to_dst_(integer4 *NumUnit)
{	
  return dst_write_bank_(NumUnit, &tslew_blen, tslew_bank );
}

integer4 tslew_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = tslew_common_to_bank_()))
    {
      fprintf (stderr,"tslew_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }             
  if ((rcode = tslew_bank_to_dst_(NumUnit)))
    {
      fprintf (stderr,"tslew_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }
  return SUCCESS;
}

integer4 tslew_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  tslew_blen = 2 * sizeof(integer4); /* skip id and version  */

  // neye, nmir, ntube
  if ((rcode = dst_unpacki2asi4_(&(tslew_.neye), (nobj = 3, &nobj), bank, &tslew_blen, &tslew_maxlen))) return rcode;

  nobj = tslew_.neye; 
  if ((rcode = dst_unpacki2asi4_(tslew_.eyeid, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;

  nobj = tslew_.nmir; 
  if ((rcode = dst_unpacki2asi4_(tslew_.mirid, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(tslew_.mir_eye, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(tslew_.mir_rev, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(tslew_.mir_ntube, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;

  nobj = tslew_.ntube; 
  if ((rcode = dst_unpacki2asi4_(tslew_.tube_eye, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(tslew_.tube_mir, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(tslew_.tubeid, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_unpacki2asi4_(tslew_.thb, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_(tslew_.thcal1, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;
  if ((rcode = dst_unpackr4_(tslew_.tcorr, &nobj,bank, &tslew_blen, &tslew_maxlen))) return rcode;

  return SUCCESS;
}

integer4 tslew_common_to_dump_(integer4 *long_output)
{
  return tslew_common_to_dumpf_(stdout, long_output);
}

integer4 tslew_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i;
  fprintf(fp, "\nTSLEW  ") ;
  fprintf(fp, "eyes: %2d  mirs: %2d  tubes: %4d \n",
          tslew_.neye, tslew_.nmir, tslew_.ntube);

  /* -------------- mir info --------------------------*/
  for (i = 0; i < tslew_.nmir; i++) {
    fprintf(fp, " eye %2d mir %2d Rev%d  tubes: %3d\n",
	    tslew_.mir_eye[i], tslew_.mirid[i], tslew_.mir_rev[i],
	    tslew_.mir_ntube[i]);
  }

  
  /* If long output is desired, show tube information */

  if ( *long_output == 1 ) {
    for (i = 0; i < tslew_.ntube; i++) {
      fprintf(fp, "e %2d m %2d t %3d ",
	      tslew_.tube_eye[i],  tslew_.tube_mir[i],
	      tslew_.tubeid[i] );
      fprintf(fp, " th:%4d ", tslew_.thb[i]);

      fprintf(fp, "t1:%9.3f tc:%9.3f\n", tslew_.thcal1[i], tslew_.tcorr[i]);
    }
  }

  fprintf(fp,"\n");

  return SUCCESS;
}

