/*
 * hcal1_dst.c 
 *
 * $Source: /hires_soft/uvm2k/bank/hcal1_dst.c,v $
 * $Log: hcal1_dst.c,v $
 * Revision 1.2  1997/08/24 22:03:45  jui
 * fixed bug with long_output format IF construct
 *
 * Revision 1.1  1997/08/17  18:25:21  jui
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
#include "hcal1_dst.h"

hcal1_dst_common hcal1_;        /* allocate memory to hcal1_common */

static integer4 hcal1_blen = 0;
static integer4 hcal1_maxlen = sizeof (integer4) * 2 + sizeof (hcal1_dst_common);
static integer1 *hcal1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hcal1_bank_buffer_ (integer4* hcal1_bank_buffer_size)
{
  (*hcal1_bank_buffer_size) = hcal1_blen;
  return hcal1_bank;
}



static void 
hcal1_bank_init (void)
{
  hcal1_bank = (integer1 *) calloc (hcal1_maxlen, sizeof (integer1));
  if (hcal1_bank == NULL) {
    fprintf (stderr,
             "hcal1_bank_init: fail to assign memory to bank. Abort.\n");
    exit (0);
  }
}

integer4 
hcal1_common_to_bank_ (void)
{
  static integer4 id = HCAL1_BANKID, ver = HCAL1_BANKVERSION;
  integer4 rcode, nobj;

  if (hcal1_bank == NULL)
    hcal1_bank_init ();

  /* Initialize hcal1_blen, and pack the id and version to bank */
  if ( (rcode = dst_initbank_ (&id, &ver,
			       &hcal1_blen, &hcal1_maxlen, hcal1_bank)) )
    return rcode;

  /* unpack validity range Julian day and second values */
  if ( (rcode = dst_packi4_ (&hcal1_.jday1, (nobj = 1, &nobj), hcal1_bank,
			     &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_ (&hcal1_.jsec1, (nobj = 1, &nobj), hcal1_bank,
			     &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_ (&hcal1_.jday2, (nobj = 1, &nobj), hcal1_bank,
			     &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_ (&hcal1_.jsec2, (nobj = 1, &nobj), hcal1_bank,
			     &hcal1_blen, &hcal1_maxlen)) )
    return rcode;


  if ( (rcode = dst_packr4_ (&hcal1_.tg[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			     hcal1_bank, &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packr4_ (&hcal1_.tp[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			     hcal1_bank, &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packr4_ (&hcal1_.qg[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			     hcal1_bank, &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packr4_ (&hcal1_.qp[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			     hcal1_bank, &hcal1_blen, &hcal1_maxlen)) )
    return rcode;

  return SUCCESS;
}


integer4 
hcal1_bank_to_dst_ (integer4 * unit)
{
  return dst_write_bank_ (unit, &hcal1_blen, hcal1_bank);
}

integer4 
hcal1_common_to_dst_ (integer4 * unit)
{
  integer4 rcode;
    if ( (rcode = hcal1_common_to_bank_()) ) {
    fprintf (stderr, "hcal1_common_to_bank_ ERROR : %ld\n", (long) rcode);
    exit (0);
  }
    if ( (rcode = hcal1_bank_to_dst_(unit) )) {
    fprintf (stderr, "hcal1_bank_to_dst_ ERROR : %ld\n", (long) rcode);
    exit (0);
  }
  return SUCCESS;
}

integer4 
hcal1_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  hcal1_blen = 2 * sizeof (integer4);   /* skip id and version  */

  if ( (rcode = dst_unpacki4_ (&hcal1_.jday1, (nobj = 1, &nobj), bank,
			       &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_ (&hcal1_.jsec1, (nobj = 1, &nobj), bank,
			       &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_ (&hcal1_.jday2, (nobj = 1, &nobj), bank,
			       &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_ (&hcal1_.jsec2, (nobj = 1, &nobj), bank,
			       &hcal1_blen, &hcal1_maxlen)) )
    return rcode;


  if ( (rcode = dst_unpackr4_ (&hcal1_.tg[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			       bank, &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpackr4_ (&hcal1_.tp[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			       bank, &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpackr4_ (&hcal1_.qg[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			       bank, &hcal1_blen, &hcal1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpackr4_ (&hcal1_.qp[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			       bank, &hcal1_blen, &hcal1_maxlen)) )
    return rcode;

  return SUCCESS;
}

integer4 
hcal1_common_to_dump_ (integer4 * long_output)
{
  return hcal1_common_to_dumpf_ (stdout, long_output);
}

integer4 
hcal1_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{

  integer4 mirror, tube;

  fprintf (fp, "\nHCAL1 jDay/Sec(1): %d/%5.5d jDay/Sec(2): %d/%5.5d \n\n",
           hcal1_.jday1, hcal1_.jsec1, hcal1_.jday2, hcal1_.jsec2);

  if (*long_output == 1) {
    for (mirror = 0; mirror < HR_UNIV_MAXMIR; mirror++) {
      for (tube = 0; tube < HR_UNIV_MIRTUBE; tube++) {

        fprintf (fp,
             " m %2d t %3d tP: %+7.4E tG: %+7.4E qbP: %+7.4E qbG: %+7.4E\n",
               mirror + 1, tube + 1,
               hcal1_.tp[mirror][tube], hcal1_.tg[mirror][tube],
               hcal1_.qp[mirror][tube], hcal1_.qg[mirror][tube]);
      }
    }
  }


  return SUCCESS;
}
