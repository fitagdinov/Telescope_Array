/*
 * hsum_dst.c 
 *
 * Revision 2.0 2013/04/01 DI fixed all GCC warnings, no variables added or replaced or removed. 
 * 
 * $Source: /hires_soft/cvsroot/bank/hsum_dst.c,v $: /hires_soft/dst95/uvm/bank/hsum_dst.h,v $
 * $Log: hsum_dst.c,v $
 * Revision 1.1  1999/07/06 21:32:50  stokes
 * Initial revision
 *: hsum_dst.h,v $
 * Revision 1.0 1999/03/05 bts
 *
 * Summary for individual pkt1 parts
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

hsum_dst_common hsum_;		/* allocate memory to hsum_common */

static integer4 hsum_blen = 0;
static integer4 hsum_maxlen = sizeof (integer4) * 2 + sizeof (hsum_dst_common);
static integer1 *hsum_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hsum_bank_buffer_ (integer4* hsum_bank_buffer_size)
{
  (*hsum_bank_buffer_size) = hsum_blen;
  return hsum_bank;
}



static void
hsum_bank_init (void)
{
  hsum_bank = (integer1 *) calloc (hsum_maxlen, sizeof (integer1));
  if (hsum_bank == NULL)
    {
      fprintf (stderr,
	       "hsum_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }
}

integer4
hsum_common_to_bank_ (void)
{
  static integer4 id = HSUM_BANKID, ver = HSUM_BANKVERSION;
  integer4 rcode, nobj;

  if (hsum_bank == NULL)
    hsum_bank_init ();

  if ((rcode = dst_initbank_ (&id, &ver, &hsum_blen, &hsum_maxlen, hsum_bank)))
    return rcode;
  if ((rcode = dst_packi1_ (*hsum_.weat, (nobj = HSUM_MAXPERMIT *
					  HSUM_MAX_TXT_LEN, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi1_ (hsum_.stat, (nobj = HSUM_MAX_TXT_LEN, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (&hsum_.jdsta, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (&hsum_.jdsto, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (hsum_.jdper, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (hsum_.jdpero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (hsum_.jdinh, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (hsum_.jdinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (hsum_.jddur, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (hsum_.jdwea, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.hsta, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.msta, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.ssta, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.hsto, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.msto, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.ssto, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.nperm, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.npermo, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.ninho, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.hper, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.mper, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.sper, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.msper, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.mirpero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.hpero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.mpero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.spero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.mspero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.hinh, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.minh, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.sinh, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.msinh, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.mirinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.hinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.minho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.sinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.msinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.hdur, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.mdur, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.sdur, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.ntrig, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.ntri, (nobj = 23, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_ (hsum_.ntrim, (nobj = 23, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.hoper, (nobj = 23, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.moper, (nobj = 23, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.soper, (nobj = 23, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (&hsum_.nweat, (nobj = 1, &nobj), hsum_bank,
			    &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.hwea, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.mwea, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_ (hsum_.swea, (nobj = HSUM_MAXPERMIT, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi1_ (&hsum_.staflag, (nobj = 1, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_packi1_ (&hsum_.permflag, (nobj = 1, &nobj),
			    hsum_bank, &hsum_blen, &hsum_maxlen)))
    return rcode;

  return SUCCESS;
}

integer4
hsum_bank_to_dst_ (integer4 * NumUnit)
{
  return dst_write_bank_ (NumUnit, &hsum_blen, hsum_bank);
}

integer4
hsum_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = hsum_common_to_bank_ ()))
    {
      fprintf (stderr, "hsum_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = hsum_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "hsum_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return SUCCESS;
}

integer4
hsum_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj;


  hsum_blen = 2 * sizeof (integer4);	/* skip id and version  */

  if ((rcode = dst_unpacki1_ (*hsum_.weat, (nobj = HSUM_MAXPERMIT *
					    HSUM_MAX_TXT_LEN, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki1_ (hsum_.stat, (nobj = HSUM_MAX_TXT_LEN, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (&hsum_.jdsta, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (&hsum_.jdsto, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (hsum_.jdper, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (hsum_.jdpero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (hsum_.jdinh, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (hsum_.jdinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (hsum_.jddur, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (hsum_.jdwea, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.hsta, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.msta, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.ssta, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.hsto, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.msto, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.ssto, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.nperm, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.npermo, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.ninho, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.hper, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.mper, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.sper, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.msper, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.mirpero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.hpero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.mpero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.spero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.mspero, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.hinh, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.minh, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.sinh, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.msinh, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.mirinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.hinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.minho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.sinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.msinho, (nobj = HSUM_MAXPERMIT_INDV, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.hdur, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.mdur, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.sdur, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.ntrig, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.ntri, (nobj = 23, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_ (hsum_.ntrim, (nobj = 23, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.hoper, (nobj = 23, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.moper, (nobj = 23, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.soper, (nobj = 23, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (&hsum_.nweat, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.hwea, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.mwea, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_ (hsum_.swea, (nobj = HSUM_MAXPERMIT, &nobj),
			      bank, &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki1_ (&hsum_.staflag, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki1_ (&hsum_.permflag, (nobj = 1, &nobj), bank,
			      &hsum_blen, &hsum_maxlen)))
    return rcode;

  return SUCCESS;
}

integer4
hsum_common_to_dump_ (integer4 * long_output)
{
  return hsum_common_to_dumpf_ (stdout, long_output);
}

integer4
hsum_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  integer4 i;
  integer1 *ptr_text_buf;
  (void)(long_output);
  fprintf (fp, "\n\nHSUM bank. \n\n\n");
  ptr_text_buf = hsum_.stat;
  fprintf (fp, "%s\n", ptr_text_buf);

  fprintf (fp, "Start time: %2.2i:%2.2i:%2.2i UT %lf JD\n", hsum_.hsta,
	   hsum_.msta, hsum_.ssta, hsum_.jdsta);
  fprintf (fp, "Stop time:  %2.2i:%2.2i:%2.2i UT %lf JD\n", hsum_.hsto,
	   hsum_.msto, hsum_.ssto, hsum_.jdsto);
  if (hsum_.staflag == 1)
    {
      fprintf (fp, "PART ENDED ABRUPTLY WITH NO STOP NOTICE PACKET\n");
    }
  if (hsum_.permflag == 1)
    {
      hsum_.nperm++;
    }
  fprintf (fp, "\n\nNumber of Global Permits: %2i\n\n", hsum_.nperm);
  if (hsum_.nperm > 0)
    {
      for (i = 0; i < hsum_.nperm; i++)
	{
	  fprintf (fp, "Permit time #%i:        %2.2i:%2.2i:%2.2i.%3.3i UT %lf JD\n",
		   (i + 1), hsum_.hper[i], hsum_.mper[i], hsum_.sper[i], hsum_.msper[i],
		   hsum_.jdper[i]);
	  fprintf (fp, "Inhibit time #%i:       %2.2i:%2.2i:%2.2i.%3.3i UT %lf JD\n",
		   (i + 1), hsum_.hinh[i], hsum_.minh[i], hsum_.sinh[i], hsum_.msinh[i],
		   hsum_.jdinh[i]);
	  fprintf (fp, "Duration of permit #%i: %2.2i:%2.2i:%2.2i     UT       %lf JD\n\n",
		   (i + 1), hsum_.hdur[i], hsum_.mdur[i], hsum_.sdur[i],
		   hsum_.jddur[i]);
	}
    }
  if (hsum_.permflag == 1)
    {
      fprintf (fp, "PERMIT ENDED ABRUPTLY WITH NO INHIBIT NOTICE PACKET\n\n");
    }
  if ((hsum_.npermo + hsum_.ninho) > 0)
    {
      fprintf (fp, "\nNumber of Individual Permits: %2i\n\n", hsum_.npermo);
      for (i = 0; i < hsum_.npermo; i++)
	{
	  fprintf (fp, "Mirror %2i:  Permitted  %2.2i:%2.2i:%2.2i.%3.3i UT %lf JD\n",
		   hsum_.mirpero[i], hsum_.hpero[i], hsum_.mpero[i], hsum_.spero[i],
		   hsum_.mspero[i], hsum_.jdpero[i]);
	}
      fprintf (fp, "\nNumber of Individual Inhibits: %2i\n\n", hsum_.ninho);
      for (i = 0; i < hsum_.ninho; i++)
	{
	  fprintf (fp, "Mirror %2i:  Inhibited  %2.2i:%2.2i:%2.2i.%3.3i UT %lf JD\n",
		   hsum_.mirinho[i], hsum_.hinho[i], hsum_.minho[i], hsum_.sinho[i],
		   hsum_.msinho[i], hsum_.jdinho[i]);
	}
    }

  fprintf (fp, "\nNumber of Triggers: %8i\n\n\n", hsum_.ntrig);
  if (hsum_.ntrig > 0)
    {
      fprintf (fp, "Mirror by Mirror Summary\n\n");
      fprintf (fp, "Mirror           Triggers   Triggers/Minute  Permit Duration\n\n");

      for (i = 1; i < 23; i++)
	{
	  fprintf (fp, "Mirror  #%2.2i    %6i    %8.3f/min           %2.2i:%2.2i:%2.2i\n",
		   i, hsum_.ntri[i], hsum_.ntrim[i], hsum_.hoper[i],
		   hsum_.moper[i], hsum_.soper[i]);
	}
    }
  fprintf (fp, "\n\n\nOperator Comments and Weather Codes\n\n");
  for (i = 0; i < hsum_.nweat; i++)
    {
      fprintf (fp, "%2.2i:%2.2i:%2.2i UT %lf JD\t", hsum_.hwea[i],
	       hsum_.mwea[i], hsum_.swea[i], hsum_.jdwea[i]);
      ptr_text_buf = hsum_.weat[i];
      fprintf (fp, "%s\n", ptr_text_buf);
    }
  if (hsum_.nweat == 0)
    {
      fprintf (fp, "(None)\n");
    }
  return SUCCESS;
}











