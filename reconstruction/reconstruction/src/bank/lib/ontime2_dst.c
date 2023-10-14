/*
 * ontime2_dst.c 
 * ontime2 is a monthly mirror by mirror statisics bank
 * 
 * $Source: /hires_soft/cvsroot/bank/ontime2_dst.c,v $
 * $Log: ontime2_dst.c,v $
 * Revision 1.2  2006/04/18 20:22:01  thomas
 *  ** HR1 25 MIRROR MOD **
 *
 * Revision 1.1  1999/07/21 22:06:14  stokes
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
#include "hsum_dst.h"
#include "ontime2_dst.h"

ontime2_dst_common ontime2_;	/* allocate memory to ontime2_common */

static integer4 ontime2_blen = 0;
static integer4 ontime2_maxlen = sizeof (integer4) * 2 + sizeof (ontime2_dst_common);
static integer1 *ontime2_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* ontime2_bank_buffer_ (integer4* ontime2_bank_buffer_size)
{
  (*ontime2_bank_buffer_size) = ontime2_blen;
  return ontime2_bank;
}



static void 
ontime2_bank_init (void)
{
  ontime2_bank = (integer1 *) calloc (ontime2_maxlen, sizeof (integer1));
  if (ontime2_bank == NULL)
    {
      fprintf (stderr,
	       "ontime2_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }
}

integer4 
ontime2_common_to_bank_ (void)
{
  static integer4 id = ONTIME2_BANKID, ver = ONTIME2_BANKVERSION;
  integer4 rcode, nobj;

  if (ontime2_bank == NULL)
    ontime2_bank_init ();

  if ((rcode = dst_initbank_ (&id, &ver, &ontime2_blen, &ontime2_maxlen,
			     ontime2_bank)))
    return rcode;
  if ((rcode = dst_packi1_ (*ontime2_.weat, (nobj = ONTIME2_MAX_WEAT *
					    ONTIME2_MAX_TXT_LEN, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi1_ (ontime2_.stat, (nobj = ONTIME2_MAX_TXT_LEN, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.yweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.moweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.dweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.hweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.mweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.sweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&ontime2_.nweat, (nobj = 1, &nobj), ontime2_bank,
			   &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&ontime2_.nab, (nobj = 1, &nobj), ontime2_bank,
			   &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&ontime2_.hdur, (nobj = 1, &nobj), ontime2_bank,
			   &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&ontime2_.mdur, (nobj = 1, &nobj), ontime2_bank,
			   &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&ontime2_.sdur, (nobj = 1, &nobj), ontime2_bank,
			   &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.hduro, (nobj = HR_MAX_MIR, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.mduro, (nobj = HR_MAX_MIR, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.sduro, (nobj = HR_MAX_MIR, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.ntrig, (nobj = HR_MAX_MIR, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.ytrimab, (nobj = ONTIME2_MAX_AB, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.motrimab, (nobj = ONTIME2_MAX_AB, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.dtrimab, (nobj = ONTIME2_MAX_AB, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (ontime2_.mirab, (nobj = ONTIME2_MAX_AB, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (ontime2_.ntrim, (nobj = HR_MAX_MIR, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (ontime2_.ntrimab, (nobj = ONTIME2_MAX_AB, &nobj),
			   ontime2_bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;




  return SUCCESS;
}


integer4 
ontime2_bank_to_dst_ (integer4 * NumUnit)
{
  return dst_write_bank_ (NumUnit, &ontime2_blen, ontime2_bank);
}

integer4 
ontime2_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = ontime2_common_to_bank_ ()))
    {
      fprintf (stderr, "ontime2_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = ontime2_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "ontime2_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return SUCCESS;
}

integer4 
ontime2_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj;
  /* integer2 i; */


  ontime2_blen = 2 * sizeof (integer4);		/* skip id and version  */

  if ((rcode = dst_unpacki1_ (*ontime2_.weat, (nobj = ONTIME2_MAX_WEAT *
					      ONTIME2_MAX_TXT_LEN, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki1_ (ontime2_.stat, (nobj = ONTIME2_MAX_TXT_LEN, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.yweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.moweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.dweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.hweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.mweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.sweat, (nobj = ONTIME2_MAX_WEAT, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&ontime2_.nweat, (nobj = 1, &nobj), bank,
			     &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&ontime2_.nab, (nobj = 1, &nobj), bank,
			     &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&ontime2_.hdur, (nobj = 1, &nobj), bank,
			     &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&ontime2_.mdur, (nobj = 1, &nobj), bank,
			     &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&ontime2_.sdur, (nobj = 1, &nobj), bank,
			     &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.hduro, (nobj = HR_MAX_MIR, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.mduro, (nobj = HR_MAX_MIR, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.sduro, (nobj = HR_MAX_MIR, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.ntrig, (nobj = HR_MAX_MIR, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.ytrimab, (nobj = ONTIME2_MAX_AB, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.motrimab, (nobj = ONTIME2_MAX_AB, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.dtrimab, (nobj = ONTIME2_MAX_AB, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (ontime2_.mirab, (nobj = ONTIME2_MAX_AB, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (ontime2_.ntrim, (nobj = HR_MAX_MIR, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (ontime2_.ntrimab, (nobj = ONTIME2_MAX_AB, &nobj),
			     bank, &ontime2_blen, &ontime2_maxlen)))
    return rcode;

  return SUCCESS;
}

integer4 
ontime2_common_to_dump_ (integer4 * long_output)
{
  return ontime2_common_to_dumpf_ (stdout, long_output);
}

integer4 
ontime2_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  integer4 i;

  fprintf (fp, "\n\nONTIME2 bank. \n\n\n");

  fprintf (fp, "First part in summary:  %s\n\n", ontime2_.stat);
  fprintf (fp, "Total time of global permits:  %3.3i:%2.2i:%2.2i\n\n",
	   ontime2_.hdur, ontime2_.mdur, ontime2_.sdur);
  fprintf (fp, "Mirror by mirror summary:\n\n");
  fprintf (fp, "            Triggers     Averaged Trigger Rate   Total permit time\n");
  for (i = 1; i < HR_MAX_MIR; i++)
    {
      fprintf (fp, "Mirror #%2.2i  %6i     %8.3f triggers/min     %3.3i:%2.2i:%2.2i\n",
	       i, ontime2_.ntrig[i], ontime2_.ntrim[i], ontime2_.hduro[i],
	       ontime2_.mduro[i], ontime2_.sduro[i]);
    }
  if (*long_output == 1)
    {
      fprintf (fp, "\n\nWeather Codes and Operator Comments\n\n");
      for (i = 0; i < ontime2_.nweat; i++)
	{
	  fprintf (fp, "%4.4i/%2.2i/%2.2i  %2.2i:%2.2i:%2.2i UT  %s\n",
		   ontime2_.yweat[i], ontime2_.moweat[i], ontime2_.dweat[i],
		   ontime2_.hweat[i], ontime2_.mweat[i], ontime2_.sweat[i],
		   ontime2_.weat[i]);
	}
      fprintf (fp, "\n\nAbnormal Triggering\n\n");
      for (i = 0; i < ontime2_.nab; i++)
	{
	  fprintf (fp, "%4.4i/%2.2i/%2.2i  Mirror #%2.2i:  %8.3f triggers/min\n",
		   ontime2_.ytrimab[i], ontime2_.motrimab[i],
		   ontime2_.dtrimab[i], ontime2_.mirab[i], 
		   ontime2_.ntrimab[i]);
	}
    }


  return SUCCESS;
}










