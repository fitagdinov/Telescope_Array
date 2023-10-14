/*
 * C functions for tlmsnp
 * Dmitri Ivanov, dmiivanov@gmail.com
 * Feb 04, 2015
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "tlmsnp_dst.h"



tlmsnp_dst_common tlmsnp_;	/* allocate memory to tlmsnp_common */

static integer4 tlmsnp_blen = 0;
static integer4 tlmsnp_maxlen =
  sizeof (integer4) * 2 + sizeof (tlmsnp_dst_common);
static integer1 *tlmsnp_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tlmsnp_bank_buffer_ (integer4* tlmsnp_bank_buffer_size)
{
  (*tlmsnp_bank_buffer_size) = tlmsnp_blen;
  return tlmsnp_bank;
}



static void
tlmsnp_bank_init ()
{
  tlmsnp_bank = (integer1 *) calloc (tlmsnp_maxlen, sizeof (integer1));
  if (tlmsnp_bank == NULL)
    {
      fprintf (stderr,
	       "tlmsnp_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }				/* else fprintf ( stderr,"tlmsnp_bank allocated memory %d\n",tlmsnp_maxlen); */
}

integer4
tlmsnp_common_to_bank_ ()
{
  static integer4 id = TLMSNP_BANKID, ver = TLMSNP_BANKVERSION;
  integer4 rcode, nobj;

  if (tlmsnp_bank == NULL)
    tlmsnp_bank_init ();

  rcode =
    dst_initbank_ (&id, &ver, &tlmsnp_blen, &tlmsnp_maxlen, tlmsnp_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj = tlmsnp_nchan_mir;
  rcode +=
    dst_packr4_ (&tlmsnp_.channel_mean[0], &nobj, tlmsnp_bank, &tlmsnp_blen,
		 &tlmsnp_maxlen);
  rcode +=
    dst_packr4_ (&tlmsnp_.channel_var[0], &nobj, tlmsnp_bank, &tlmsnp_blen,
		 &tlmsnp_maxlen);
  rcode +=
    dst_packr4_ (&tlmsnp_.channel_vgain[0], &nobj, tlmsnp_bank, &tlmsnp_blen,
		 &tlmsnp_maxlen);
  rcode +=
    dst_packr4_ (&tlmsnp_.channel_hgain[0], &nobj, tlmsnp_bank, &tlmsnp_blen,
		 &tlmsnp_maxlen);
  nobj = 1;
  rcode +=
    dst_packi4_ (&tlmsnp_.yymmdd, &nobj, tlmsnp_bank, &tlmsnp_blen,
		 &tlmsnp_maxlen);
  rcode +=
    dst_packi4_ (&tlmsnp_.hhmmss, &nobj, tlmsnp_bank, &tlmsnp_blen,
		 &tlmsnp_maxlen);
  rcode +=
    dst_packi4_ (&tlmsnp_.secfrac, &nobj, tlmsnp_bank, &tlmsnp_blen,
		 &tlmsnp_maxlen);
  rcode +=
    dst_packi4_ (&tlmsnp_.mirid, &nobj, tlmsnp_bank, &tlmsnp_blen,
		 &tlmsnp_maxlen);
  rcode +=
    dst_packi4_ (&tlmsnp_.nsnp, &nobj, tlmsnp_bank, &tlmsnp_blen,
		 &tlmsnp_maxlen);
  
  return rcode;
}

integer4
tlmsnp_bank_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_ (NumUnit, &tlmsnp_blen, tlmsnp_bank);
  free (tlmsnp_bank);
  tlmsnp_bank = NULL;
  return rcode;
}

integer4
tlmsnp_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = tlmsnp_common_to_bank_ ()))
    {
      fprintf (stderr, "tlmsnp_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = tlmsnp_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "tlmsnp_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4
tlmsnp_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj;
  tlmsnp_blen = 2 * sizeof (integer4);	/* skip id and version  */
  
  nobj = tlmsnp_nchan_mir;
  rcode +=
    dst_unpackr4_ (&tlmsnp_.channel_mean[0], &nobj, bank, &tlmsnp_blen,
		   &tlmsnp_maxlen);
  rcode +=
    dst_unpackr4_ (&tlmsnp_.channel_var[0], &nobj, bank, &tlmsnp_blen,
		   &tlmsnp_maxlen);
  rcode +=
    dst_unpackr4_ (&tlmsnp_.channel_vgain[0], &nobj, bank, &tlmsnp_blen,
		   &tlmsnp_maxlen);
  rcode +=
    dst_unpackr4_ (&tlmsnp_.channel_hgain[0], &nobj, bank, &tlmsnp_blen,
		   &tlmsnp_maxlen);
  nobj = 1;
  rcode +=
    dst_unpacki4_ (&tlmsnp_.yymmdd, &nobj, bank, &tlmsnp_blen,
		   &tlmsnp_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmsnp_.hhmmss, &nobj, bank, &tlmsnp_blen,
		   &tlmsnp_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmsnp_.secfrac, &nobj, bank, &tlmsnp_blen,
		   &tlmsnp_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmsnp_.mirid, &nobj, bank, &tlmsnp_blen,
		   &tlmsnp_maxlen);
  rcode +=
    dst_unpacki4_ (&tlmsnp_.nsnp, &nobj, bank, &tlmsnp_blen,
		   &tlmsnp_maxlen);
  
  return rcode;
}

integer4
tlmsnp_common_to_dump_ (integer4 * long_output)
{
  return tlmsnp_common_to_dumpf_ (stdout, long_output);
}

integer4
tlmsnp_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  int i = 0;
  fprintf (fp, "%s :\n","tlmsnp");
  if(*long_output ==0)
    {
      fprintf(fp,"yymmdd=%06d hhmmss=%06d secfrac=%07dx100nS mirid=%d nsnp=%d\n",
	      tlmsnp_.yymmdd,tlmsnp_.hhmmss,tlmsnp_.secfrac,tlmsnp_.mirid,tlmsnp_.nsnp);
    }
  else if (*long_output == 1)
    {
      fprintf(fp,"yymmdd=%06d hhmmss=%06d secfrac=%07dx100nS mirid=%d nsnp=%d\n",
	      tlmsnp_.yymmdd,tlmsnp_.hhmmss,tlmsnp_.secfrac,tlmsnp_.mirid,tlmsnp_.nsnp);
      fprintf(fp,"col1=channel#, col2=mean, col3=var, col4=vgain, col5=hgain\n");
      for (i=0; i < tlmsnp_nchan_mir; i++)
	{
	  fprintf(fp,"%d %f %f %f %f\n",i,tlmsnp_.channel_mean[i],tlmsnp_.channel_var[i],
		  tlmsnp_.channel_vgain[i],tlmsnp_.channel_hgain[i]);
	}
    }
  return 0;
}
