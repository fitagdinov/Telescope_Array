/*
 * hped1_dst.c 
 *
 * $Source: /hires_soft/uvm2k/bank/hped1_dst.c,v $
 * $Log: hped1_dst.c,v $
 * Revision 1.1  1997/08/25 16:47:12  jui
 * Initial revision
 *
 * Revision 1.2  1997/08/24  22:03:45  jui
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
#include "hped1_dst.h"

hped1_dst_common hped1_;        /* allocate memory to hped1_common */

static integer4 hped1_blen = 0;
static integer4 hped1_maxlen = sizeof (integer4) * 2 + sizeof (hped1_dst_common);
static integer1 *hped1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hped1_bank_buffer_ (integer4* hped1_bank_buffer_size)
{
  (*hped1_bank_buffer_size) = hped1_blen;
  return hped1_bank;
}



static void 
hped1_bank_init (void)
{
  hped1_bank = (integer1 *) calloc (hped1_maxlen, sizeof (integer1));
  if (hped1_bank == NULL) {
    fprintf (stderr,
             "hped1_bank_init: fail to assign memory to bank. Abort.\n");
    exit (0);
  }
}

integer4 
hped1_common_to_bank_ (void)
{
  static integer4 id = HPED1_BANKID, ver = HPED1_BANKVERSION;
  integer4 rcode, nobj;

  if (hped1_bank == NULL)
    hped1_bank_init ();

  /* Initialize hped1_blen, and pack the id and version to bank */
  if ( (rcode = dst_initbank_ (&id, &ver,
			       &hped1_blen, &hped1_maxlen, hped1_bank)) )
    return rcode;

  /* unpack validity range Julian day and second values */
  if ( (rcode = dst_packi4_ (&hped1_.jday1, (nobj = 1, &nobj), hped1_bank,
			     &hped1_blen, &hped1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_ (&hped1_.jsec1, (nobj = 1, &nobj), hped1_bank,
			     &hped1_blen, &hped1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_ (&hped1_.jday2, (nobj = 1, &nobj), hped1_bank,
			     &hped1_blen, &hped1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_ (&hped1_.jsec2, (nobj = 1, &nobj), hped1_bank,
			     &hped1_blen, &hped1_maxlen)) )
    return rcode;


  if ( (rcode = dst_packr4_ (&hped1_.pA[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			     hped1_bank, &hped1_blen, &hped1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packr4_ (&hped1_.pB[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			     hped1_bank, &hped1_blen, &hped1_maxlen)) )
    return rcode;

  return SUCCESS;
}


integer4 
hped1_bank_to_dst_ (integer4 * unit)
{
  return dst_write_bank_ (unit, &hped1_blen, hped1_bank);
}

integer4 
hped1_common_to_dst_ (integer4 * unit)
{
  integer4 rcode;
    if ( (rcode = hped1_common_to_bank_()) ) {
    fprintf (stderr, "hped1_common_to_bank_ ERROR : %ld\n", (long) rcode);
    exit (0);
  }
    if ( (rcode = hped1_bank_to_dst_(unit) )) {
    fprintf (stderr, "hped1_bank_to_dst_ ERROR : %ld\n", (long) rcode);
    exit (0);
  }
  return SUCCESS;
}

integer4 
hped1_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  hped1_blen = 2 * sizeof (integer4);   /* skip id and version  */

  if ( (rcode = dst_unpacki4_ (&hped1_.jday1, (nobj = 1, &nobj), bank,
			       &hped1_blen, &hped1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_ (&hped1_.jsec1, (nobj = 1, &nobj), bank,
			       &hped1_blen, &hped1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_ (&hped1_.jday2, (nobj = 1, &nobj), bank,
			       &hped1_blen, &hped1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_ (&hped1_.jsec2, (nobj = 1, &nobj), bank,
			       &hped1_blen, &hped1_maxlen)) )
    return rcode;


  if ( (rcode = dst_unpackr4_ (&hped1_.pA[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			       bank, &hped1_blen, &hped1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpackr4_ (&hped1_.pB[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			       bank, &hped1_blen, &hped1_maxlen)) )
    return rcode;

  return SUCCESS;
}

integer4 
hped1_common_to_dump_ (integer4 * long_output)
{
  return hped1_common_to_dumpf_ (stdout, long_output);
}

integer4 
hped1_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{

  integer4 mirror, tube;

  fprintf (fp, "\nHPED1 jDay/Sec(1): %d/%5.5d jDay/Sec(2): %d/%5.5d \n\n",
           hped1_.jday1, hped1_.jsec1, hped1_.jday2, hped1_.jsec2);

  if (*long_output == 1) {
    for (mirror = 0; mirror < HR_UNIV_MAXMIR; mirror++) {
      for (tube = 0; tube < HR_UNIV_MIRTUBE; tube++) {

        fprintf (fp,
             " m %2d t %3d pA: %+12.4E pB: %+12.4E\n",
               mirror + 1, tube + 1,
               hped1_.pA[mirror][tube], hped1_.pB[mirror][tube]);
      }
    }
  }


  return SUCCESS;
}
