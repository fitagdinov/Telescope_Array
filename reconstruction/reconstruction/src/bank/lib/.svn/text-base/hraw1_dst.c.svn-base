/*
 * $Source: /hires_soft/uvm2k/bank/hraw1_dst.c,v $
 * $Log: hraw1_dst.c,v $
 * Revision 1.7  2001/01/24 17:46:48  jeremy
 * Dump routine now displays status bits in hex.
 *
 * Revision 1.6  1997/08/28 16:37:00  jui
 * corrected bug: actually now put prxf AND thcal1 into the DST itself
 *
 * Revision 1.5  1997/08/17  22:52:33  jui
 * added prxf (roving xenon flasher calibration calculated photon count
 * and thcal1 (time from hcal1 calibration)
 *
 * Revision 1.4  1997/06/28  00:01:31  jui
 * changed jday from r*8 to jday(i*4) and jsec(i*4)
 *
 * Revision 1.3  1997/05/19  16:39:21  jui
 * removed mirscaler...no longer supported under Big-H
 * added mir_rev. Big-H is a mixed detector.
 *
 * Revision 1.2  1997/04/29  23:31:59  tareq
 * removed idth field from tube info
 * changed %5u to %5d in printf format of tha
 *
 * Revision 1.1  1997/04/28  21:52:30  tareq
 * Initial revision
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
#include "hraw1_dst.h"  

hraw1_dst_common hraw1_;  /* allocate memory to hraw1_common */

static integer4 hraw1_blen = 0; 
static integer4 hraw1_maxlen = sizeof(integer4) * 2 + sizeof(hraw1_dst_common);
static integer1 *hraw1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hraw1_bank_buffer_ (integer4* hraw1_bank_buffer_size)
{
  (*hraw1_bank_buffer_size) = hraw1_blen;
  return hraw1_bank;
}



static void hraw1_bank_init()
{
  hraw1_bank = (integer1 *)calloc(hraw1_maxlen, sizeof(integer1));
  if (hraw1_bank==NULL)
    {
      fprintf(stderr, "hraw1_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 hraw1_common_to_bank_()
{	
  static integer4 id = HRAW1_BANKID, ver = HRAW1_BANKVERSION;
  integer4 rcode;
  integer4 nobj, i;
  integer2 j;

  if (hraw1_bank == NULL) hraw1_bank_init();

  /* Initialize hraw1_blen, and pack the id and version to bank */
  if ( (rcode = dst_initbank_(&id, &ver, &hraw1_blen, &hraw1_maxlen, hraw1_bank)) )
    return rcode;
  if ( (rcode = dst_packi4_(&hraw1_.jday, (nobj=1, &nobj), hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_(&hraw1_.jsec, (nobj=1, &nobj), hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_(&hraw1_.msec, (nobj=3, &nobj), hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4asi2_(&hraw1_.nmir, (nobj=2, &nobj), hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;

  nobj = hraw1_.nmir; 
  if ( (rcode = dst_packi4asi2_(hraw1_.mir, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_(hraw1_.mir_rev, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_(hraw1_.mirevtno, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4asi2_(hraw1_.mirntube, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_(hraw1_.miraccuracy_ns, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_(hraw1_.mirtime_ns, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;

  nobj = 1; 
  for (i = 0; i < hraw1_.ntube; ++i)
      {
      j = hraw1_.tubemir[i] * 1000 + hraw1_.tube[i];
      if ( (rcode = dst_packi2_(&j, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
	      return rcode;
  }
  nobj = hraw1_.ntube;
  if ( (rcode = dst_packi4asi2_(hraw1_.qdca, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4asi2_(hraw1_.qdcb, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4asi2_(hraw1_.tdc, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4asi2_(hraw1_.tha, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4asi2_(hraw1_.thb, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode ;
  if ( (rcode = dst_packr4_(hraw1_.prxf, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode ;
  if ( (rcode = dst_packr4_(hraw1_.thcal1, &nobj, hraw1_bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode ;

  return SUCCESS;
}

integer4 hraw1_bank_to_dst_(integer4 *unit)
{	
  return dst_write_bank_(unit, &hraw1_blen, hraw1_bank );
}

integer4 hraw1_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
    if ( (rcode = hraw1_common_to_bank_()) )
    {
      fprintf (stderr,"hraw1_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }             
    if ( (rcode = hraw1_bank_to_dst_(unit) ))
    {
      fprintf (stderr,"hraw1_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }
  return SUCCESS;
}

integer4 hraw1_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj, i;
  integer2 j;

  hraw1_blen = 2 * sizeof(integer4); /* skip id and version  */
  if ( (rcode = dst_unpacki4_(&(hraw1_.jday), (nobj = 1, &nobj) ,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_(&(hraw1_.jsec), (nobj = 1, &nobj) ,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_(&(hraw1_.msec), (nobj = 3, &nobj), bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki2asi4_(&(hraw1_.nmir), (nobj = 2, &nobj), bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;

  nobj = hraw1_.nmir; 
  if ( (rcode = dst_unpacki2asi4_(hraw1_.mir, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_(hraw1_.mir_rev, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_(hraw1_.mirevtno, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki2asi4_(hraw1_.mirntube, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_(hraw1_.miraccuracy_ns, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_(hraw1_.mirtime_ns, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;

  for (i = 0; i < hraw1_.ntube; ++i)
    {
      if ( (rcode = dst_unpacki2_(&j, (nobj = 1, &nobj), bank, &hraw1_blen, &hraw1_maxlen)) )
	      return rcode;
      hraw1_.tubemir[i] = ((int)j & 0xFFFF) / 1000;
      hraw1_.tube[i] = ((int)j & 0xFFFF) % 1000;
  }
  nobj = hraw1_.ntube; 
  if ( (rcode = dst_unpacki2asi4_(hraw1_.qdca, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki2asi4_(hraw1_.qdcb, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki2asi4_(hraw1_.tdc, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki2asi4_(hraw1_.tha, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki2asi4_(hraw1_.thb, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpackr4_(hraw1_.prxf, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpackr4_(hraw1_.thcal1, &nobj,bank, &hraw1_blen, &hraw1_maxlen)) )
    return rcode;

  return SUCCESS;
}

integer4 hraw1_common_to_dump_(integer4 *long_output)
{
  return hraw1_common_to_dumpf_(stdout, long_output);
}

integer4 hraw1_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i;
  fprintf(fp, "\nHRAW1 jDay/Sec: %d/%5.5d %02d:%02d:%02d.%03d stat: %08X ",
          hraw1_.jday, hraw1_.jsec, hraw1_.msec / 3600000,
          (hraw1_.msec / 60000) % 60, (hraw1_.msec / 1000) % 60,
          hraw1_.msec % 1000, hraw1_.status) ;
  fprintf(fp, "mirs: %2d  tubes: %4d \n",
          hraw1_.nmir, hraw1_.ntube);

  /* -------------- mir info --------------------------*/
  for (i = 0; i < hraw1_.nmir; i++)
    {
      fprintf(fp, " mir %2d Rev%d evt: %9d  tubes: %3d  ",
              hraw1_.mir[i], hraw1_.mir_rev[i],
              hraw1_.mirevtno[i], hraw1_.mirntube[i]);
      fprintf(fp, "time: %10dnS +/- %dnS\n",
              hraw1_.mirtime_ns[i], hraw1_.miraccuracy_ns[i]);
    }

  
  /* If long output is desired, show tube information */

  if ( *long_output == 1 ) {
    for (i = 0; i < hraw1_.ntube; i++) {
      fprintf(fp, " m %2d t %3d qA:%6d qB:%6d ",
	      hraw1_.tubemir[i], hraw1_.tube[i], hraw1_.qdca[i], hraw1_.qdcb[i]);
      fprintf(fp, "t:%6d th:%4d/%4d ", hraw1_.tdc[i], 
	      hraw1_.tha[i], hraw1_.thb[i]);
	   fprintf(fp, "pX:%9.1f t1:%9.3f\n", hraw1_.prxf[i], hraw1_.thcal1[i]);
    }
  }

  fprintf(fp,"\n");

  return SUCCESS;
}

