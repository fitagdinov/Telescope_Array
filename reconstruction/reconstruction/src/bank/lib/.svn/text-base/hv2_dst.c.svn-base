/*
 * $Source: /hires_soft/cvsroot/bank/hv2_dst.c,v $
 * $Log: hv2_dst.c,v $
 * Revision 1.2  1995/11/03 23:56:24  jeremy
 * fixed bugs in common_to_dumpf routine.
 *
 * Revision 1.1  1995/11/02  23:37:41  jeremy
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
#include "hv2_dst.h"  

hv2_dst_common hv2_;  /* allocate memory to hv2_common */

static integer4 hv2_blen = 0; 
static integer4 hv2_maxlen = sizeof(integer4) * 2 + sizeof(hv2_dst_common);
static integer1 *hv2_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hv2_bank_buffer_ (integer4* hv2_bank_buffer_size)
{
  (*hv2_bank_buffer_size) = hv2_blen;
  return hv2_bank;
}



static void hv2_bank_init()
{
  hv2_bank = (integer1 *)calloc(hv2_maxlen, sizeof(integer1));
  if (hv2_bank==NULL)
    {
      fprintf(stderr, "hv2_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 hv2_common_to_bank_()
{	
  static integer4 id = HV2_BANKID, ver = HV2_BANKVERSION;
  integer4 rcode;
  integer4 nobj;

  if (hv2_bank == NULL) hv2_bank_init();

  /* Initialize hv2_blen, and pack the id and version to bank */
  if ((rcode = dst_initbank_(&id, &ver, &hv2_blen, &hv2_maxlen, hv2_bank)))
    return rcode;

  if ((rcode = dst_packi4_(&hv2_.nmeas, (nobj=1, &nobj), hv2_bank, &hv2_blen, &hv2_maxlen)))
    return rcode;

  if ((nobj = hv2_.nmeas) > HV2_MAXMEAS)
    { fprintf(stderr, "hr2_common_to_bank: nmeas too large\n"); exit(1); }
  
  if ((rcode = dst_packi4asi2_(hv2_.mir, &nobj, hv2_bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_(hv2_.stat, &nobj, hv2_bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4asi2_(hv2_.sclu, &nobj, hv2_bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_packr4_(hv2_.volts, &nobj, hv2_bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_(hv2_.fdate, &nobj, hv2_bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_(hv2_.ldate, &nobj, hv2_bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  return SUCCESS;
}

integer4 hv2_bank_to_dst_(integer4 *NumUnit)
{	
  return dst_write_bank_(NumUnit, &hv2_blen, hv2_bank );
}

integer4 hv2_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = hv2_common_to_bank_()))
    {
      fprintf (stderr,"hv2_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }             
  if ((rcode = hv2_bank_to_dst_(NumUnit)))
    {
      fprintf (stderr,"hv2_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }
  return SUCCESS;
}

integer4 hv2_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;
  hv2_blen = 2 * sizeof(integer4); /* skip id and version  */
  if ((rcode = dst_unpacki4_(&hv2_.nmeas, (nobj = 1, &nobj), bank, &hv2_blen, &hv2_maxlen)))
    return rcode;

  if ((nobj = hv2_.nmeas) > HV2_MAXMEAS)
    { fprintf(stderr, "hr2_bank_to_common: nmeas too large\n"); exit(1); }
    
  if ((rcode = dst_unpacki2asi4_(hv2_.mir, &nobj, bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_(hv2_.stat, &nobj, bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki2asi4_(hv2_.sclu, &nobj, bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr4_(hv2_.volts, &nobj, bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_(hv2_.fdate, &nobj, bank, &hv2_blen, &hv2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_(hv2_.ldate, &nobj, bank, &hv2_blen, &hv2_maxlen)))
    return rcode;

  return SUCCESS;
}

integer4 hv2_common_to_dump_(integer4 *long_form)
{
  return hv2_common_to_dumpf_(stdout, long_form);
}

integer4 hv2_common_to_dumpf_(FILE* fp, integer4 *long_form)
{
  int i;
  fprintf(fp, "HV2 nMeas: %d\n", hv2_.nmeas);
  if (*long_form)
    {
      fputs("    mir   stat   sclu   volts  f-date l-date\n", fp);
      for (i = 0; i < hv2_.nmeas; i++)
	fprintf(fp, "    %3d %08X %4d %7.2f  %06d %06d\n", hv2_.mir[i], hv2_.stat[i],
		hv2_.sclu[i], hv2_.volts[i], hv2_.fdate[i], hv2_.ldate[i]);
    }
  return SUCCESS;
}
