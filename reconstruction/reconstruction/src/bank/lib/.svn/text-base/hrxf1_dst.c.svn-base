/*
 * hrxf1_dst.c 
 *
 * $Source: /hires_soft/uvm2k/bank/hrxf1_dst.c,v $
 * $Log: hrxf1_dst.c,v $
 * Revision 1.1  1997/08/24 22:06:23  jui
 * Initial revision
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
#include "hrxf1_dst.h"

hrxf1_dst_common hrxf1_;        /* allocate memory to hrxf1_common */

static integer4 hrxf1_blen = 0;
static integer4 hrxf1_maxlen = sizeof (integer4) * 2 + sizeof (hrxf1_dst_common);
static integer1 *hrxf1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hrxf1_bank_buffer_ (integer4* hrxf1_bank_buffer_size)
{
  (*hrxf1_bank_buffer_size) = hrxf1_blen;
  return hrxf1_bank;
}



static void 
hrxf1_bank_init (void)
{
  hrxf1_bank = (integer1 *) calloc (hrxf1_maxlen, sizeof (integer1));
  if (hrxf1_bank == NULL) {
    fprintf (stderr,
             "hrxf1_bank_init: fail to assign memory to bank. Abort.\n");
    exit (0);
  }
}

integer4 
hrxf1_common_to_bank_ (void)
{
  static integer4 id = HRXF1_BANKID, ver = HRXF1_BANKVERSION;
  integer4 rcode, nobj;

  if (hrxf1_bank == NULL)
    hrxf1_bank_init ();

  /* Initialize hrxf1_blen, and pack the id and version to bank */
  if ( (rcode = dst_initbank_ (&id, &ver,
			       &hrxf1_blen, &hrxf1_maxlen, hrxf1_bank)) )
    return rcode;

  /* unpack validity range Julian day and second values */
  if ( (rcode = dst_packi4_ (&hrxf1_.jday1, (nobj = 1, &nobj), hrxf1_bank,
			     &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_ (&hrxf1_.jsec1, (nobj = 1, &nobj), hrxf1_bank,
			     &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_ (&hrxf1_.jday2, (nobj = 1, &nobj), hrxf1_bank,
			     &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packi4_ (&hrxf1_.jsec2, (nobj = 1, &nobj), hrxf1_bank,
			     &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;


  if ( (rcode = dst_packr4_ (&hrxf1_.prxf,
                           (nobj = 1, &nobj),
			     hrxf1_bank, &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packr4_ (&hrxf1_.xg[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			     hrxf1_bank, &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_packr4_ (&hrxf1_.xp[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			     hrxf1_bank, &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;

  return SUCCESS;
}


integer4 
hrxf1_bank_to_dst_ (integer4 * unit)
{
  return dst_write_bank_ (unit, &hrxf1_blen, hrxf1_bank);
}

integer4 
hrxf1_common_to_dst_ (integer4 * unit)
{
  integer4 rcode;
    if ( (rcode = hrxf1_common_to_bank_()) ) {
    fprintf (stderr, "hrxf1_common_to_bank_ ERROR : %ld\n", (long) rcode);
    exit (0);
  }
    if ( (rcode = hrxf1_bank_to_dst_(unit) )) {
    fprintf (stderr, "hrxf1_bank_to_dst_ ERROR : %ld\n", (long) rcode);
    exit (0);
  }
  return SUCCESS;
}

integer4 
hrxf1_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  hrxf1_blen = 2 * sizeof (integer4);   /* skip id and version  */

  if ( (rcode = dst_unpacki4_ (&hrxf1_.jday1, (nobj = 1, &nobj), bank,
			       &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_ (&hrxf1_.jsec1, (nobj = 1, &nobj), bank,
			       &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_ (&hrxf1_.jday2, (nobj = 1, &nobj), bank,
			       &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpacki4_ (&hrxf1_.jsec2, (nobj = 1, &nobj), bank,
			       &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;


  if ( (rcode = dst_unpackr4_ (&hrxf1_.prxf,
                           (nobj = 1, &nobj),
			       bank, &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpackr4_ (&hrxf1_.xg[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			       bank, &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;
  if ( (rcode = dst_unpackr4_ (&hrxf1_.xp[0][0],
                           (nobj = HR_UNIV_MAXMIR * HR_UNIV_MIRTUBE, &nobj),
			       bank, &hrxf1_blen, &hrxf1_maxlen)) )
    return rcode;

  return SUCCESS;
}

integer4 
hrxf1_common_to_dump_ (integer4 * long_output)
{
  return hrxf1_common_to_dumpf_ (stdout, long_output);
}

integer4 
hrxf1_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{

  integer4 mirror, tube;

  fprintf (fp, "\nHRXF1 jDay/Sec(1): %d/%5.5d jDay/Sec(2): %d/%5.5d \n",
           hrxf1_.jday1, hrxf1_.jsec1, hrxf1_.jday2, hrxf1_.jsec2);
  fprintf (fp, "prxf: assumed number of photons from flasher = %f\n\n",
           hrxf1_.prxf);

  if ( *long_output == 1 ) {
    for (mirror = 0; mirror < HR_UNIV_MAXMIR; mirror++) {
      for (tube = 0; tube < HR_UNIV_MIRTUBE; tube++) {

        fprintf (fp,
             " m %2d t %3d xP: %+12.7E xG: %+12.7E\n",
               mirror + 1, tube + 1,
               hrxf1_.xp[mirror][tube], hrxf1_.xg[mirror][tube]);
      }
    }
  }


  return SUCCESS;
}
