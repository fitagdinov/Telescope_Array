/*
 * C functions for rusdcal
 * Dmitri Ivanov, ivanov@physics.rutgers.edu
 * Aug 17, 2009
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "rusdcal_dst.h"



rusdcal_dst_common rusdcal_;	/* allocate memory to rusdcal_common */

static integer4 rusdcal_blen = 0;
static integer4 rusdcal_maxlen =
  sizeof (integer4) * 2 + sizeof (rusdcal_dst_common);
static integer1 *rusdcal_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* rusdcal_bank_buffer_ (integer4* rusdcal_bank_buffer_size)
{
  (*rusdcal_bank_buffer_size) = rusdcal_blen;
  return rusdcal_bank;
}



static void
rusdcal_bank_init ()
{
  rusdcal_bank = (integer1 *) calloc (rusdcal_maxlen, sizeof (integer1));
  if (rusdcal_bank == NULL)
    {
      fprintf (stderr,
	       "rusdcal_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }
}

integer4
rusdcal_common_to_bank_ ()
{
  static integer4 id = RUSDCAL_BANKID, ver = RUSDCAL_BANKVERSION;
  integer4 rcode, nobj, i, j;

  if (rusdcal_bank == NULL)
    rusdcal_bank_init ();

  rcode =
    dst_initbank_ (&id, &ver, &rusdcal_blen, &rusdcal_maxlen, rusdcal_bank);
  /* Initialize test_blen, and pack the id and version to bank */
  
  nobj=1;
  rcode +=
    dst_packi4_ (&rusdcal_.nsds, &nobj, rusdcal_bank, &rusdcal_blen,
		 &rusdcal_maxlen);
  rcode +=
    dst_packi4_ (&rusdcal_.date, &nobj, rusdcal_bank, &rusdcal_blen,
		 &rusdcal_maxlen);
  rcode +=
    dst_packi4_ (&rusdcal_.time, &nobj, rusdcal_bank, &rusdcal_blen,
		 &rusdcal_maxlen);
  nobj=rusdcal_.nsds;
  rcode +=
    dst_packi4_ (&rusdcal_.xxyy[0], &nobj, rusdcal_bank, &rusdcal_blen,
		 &rusdcal_maxlen);
  
  for (i=0; i<rusdcal_.nsds; i++)
    {
      nobj=2;
      rcode +=
	dst_packi4_ (&rusdcal_.pchmip[i][0], &nobj, rusdcal_bank, &rusdcal_blen,
		     &rusdcal_maxlen);
      rcode +=
	dst_packi4_ (&rusdcal_.pchped[i][0], &nobj, rusdcal_bank, &rusdcal_blen,
		     &rusdcal_maxlen);
      rcode +=
	dst_packi4_ (&rusdcal_.lhpchmip[i][0], &nobj, rusdcal_bank, &rusdcal_blen,
		     &rusdcal_maxlen);
      rcode +=
	dst_packi4_ (&rusdcal_.lhpchped[i][0], &nobj, rusdcal_bank, &rusdcal_blen,
		     &rusdcal_maxlen);
      rcode +=
	dst_packi4_ (&rusdcal_.rhpchmip[i][0], &nobj, rusdcal_bank, &rusdcal_blen,
		     &rusdcal_maxlen);
      rcode +=
	dst_packi4_ (&rusdcal_.rhpchped[i][0], &nobj, rusdcal_bank, &rusdcal_blen,
		     &rusdcal_maxlen);
      rcode +=
	dst_packi4_ (&rusdcal_.mftndof[i][0], &nobj, rusdcal_bank, &rusdcal_blen,
		     &rusdcal_maxlen);
      rcode +=
	dst_packr8_ (&rusdcal_.mip[i][0], &nobj, rusdcal_bank, &rusdcal_blen,
		     &rusdcal_maxlen);
      rcode +=
	dst_packr8_ (&rusdcal_.mftchi2[i][0], &nobj, rusdcal_bank, &rusdcal_blen,
		     &rusdcal_maxlen);
      
      nobj=4;
      for(j=0;j<2;j++)
	{
	  rcode +=
	    dst_packr8_ (&rusdcal_.mftp[i][j][0], &nobj, rusdcal_bank, &rusdcal_blen,
			 &rusdcal_maxlen);
	  rcode +=
	    dst_packr8_ (&rusdcal_.mftpe[i][j][0], &nobj, rusdcal_bank, &rusdcal_blen,
			 &rusdcal_maxlen);
	}
    }
  return rcode;
}

integer4
rusdcal_bank_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_ (NumUnit, &rusdcal_blen, rusdcal_bank);
  free (rusdcal_bank);
  rusdcal_bank = NULL;
  return rcode;
}

integer4
rusdcal_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = rusdcal_common_to_bank_ ()))
    {
      fprintf (stderr, "rusdcal_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = rusdcal_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "rusdcal_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4
rusdcal_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj, i, j;
  rusdcal_blen = 2 * sizeof (integer4);	/* skip id and version  */
  
  nobj=1;
  rcode +=
    dst_unpacki4_ (&rusdcal_.nsds, &nobj, bank, &rusdcal_blen,
		   &rusdcal_maxlen);
  rcode +=
    dst_unpacki4_ (&rusdcal_.date, &nobj, bank, &rusdcal_blen,
		   &rusdcal_maxlen);
  rcode +=
    dst_unpacki4_ (&rusdcal_.time, &nobj, bank, &rusdcal_blen,
		   &rusdcal_maxlen);
  nobj=rusdcal_.nsds;
  rcode +=
    dst_unpacki4_ (&rusdcal_.xxyy[0], &nobj, bank, &rusdcal_blen,
		   &rusdcal_maxlen);
  
  for (i=0; i<rusdcal_.nsds; i++)
    {
      nobj=2;
      rcode +=
	dst_unpacki4_ (&rusdcal_.pchmip[i][0], &nobj, bank, &rusdcal_blen,
		       &rusdcal_maxlen);
      rcode +=
	dst_unpacki4_ (&rusdcal_.pchped[i][0], &nobj, bank, &rusdcal_blen,
		       &rusdcal_maxlen);
      rcode +=
	dst_unpacki4_ (&rusdcal_.lhpchmip[i][0], &nobj, bank, &rusdcal_blen,
		       &rusdcal_maxlen);
      rcode +=
	dst_unpacki4_ (&rusdcal_.lhpchped[i][0], &nobj, bank, &rusdcal_blen,
		       &rusdcal_maxlen);
      rcode +=
	dst_unpacki4_ (&rusdcal_.rhpchmip[i][0], &nobj, bank, &rusdcal_blen,
		       &rusdcal_maxlen);
      rcode +=
	dst_unpacki4_ (&rusdcal_.rhpchped[i][0], &nobj, bank, &rusdcal_blen,
		       &rusdcal_maxlen);
      rcode +=
	dst_unpacki4_ (&rusdcal_.mftndof[i][0], &nobj, bank, &rusdcal_blen,
		       &rusdcal_maxlen);
      rcode +=
	dst_unpackr8_ (&rusdcal_.mip[i][0], &nobj, bank, &rusdcal_blen,
		       &rusdcal_maxlen);
      rcode +=
	dst_unpackr8_ (&rusdcal_.mftchi2[i][0], &nobj, bank, &rusdcal_blen,
		       &rusdcal_maxlen);
      nobj=4;
      for(j=0;j<2;j++)
	{
	  rcode +=
	    dst_unpackr8_ (&rusdcal_.mftp[i][j][0], &nobj, bank, &rusdcal_blen,
			   &rusdcal_maxlen);
	  rcode +=
	    dst_unpackr8_ (&rusdcal_.mftpe[i][j][0], &nobj, bank, &rusdcal_blen,
			   &rusdcal_maxlen);
	}
    }
  
  return rcode;
}

integer4
rusdcal_common_to_dump_ (integer4 * long_output)
{
  return rusdcal_common_to_dumpf_ (stdout, long_output);
}

integer4 rusdcal_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  int i,j;
  double chi2pdof[2];
  (void)(long_output);
  fprintf (fp, "%s :\n","rusdcal");
  fprintf (fp, "%s",
	   "ind xxyy pchmip    pchped  lhpchmip lhpchped  rhpchmip rhpchped  nfadcc/mip   chi2/dof\n");
  for (i = 0; i < rusdcal_.nsds; i++)
    {
      fprintf (fp, "%03d %04d", i, rusdcal_.xxyy[i]);
      for (j=0; j<2; j++)
	chi2pdof[j] = (rusdcal_.mftndof[i][j] > 0 ? (rusdcal_.mftchi2[i][j] / (double)rusdcal_.mftndof[i][j]) : 1.0e3);
      fprintf (fp, "%4d,%3d%5d,%3d%5d,%3d%5d,%3d %5d,%3d%5d,%3d%7.1f,%5.1f%6.1f%5.1f\n",
	       rusdcal_.pchmip[i][0], rusdcal_.pchmip[i][1],
	       rusdcal_.pchped[i][0], rusdcal_.pchped[i][1],
	       rusdcal_.lhpchmip[i][0], rusdcal_.lhpchmip[i][1],
	       rusdcal_.lhpchped[i][0], rusdcal_.lhpchped[i][1],
	       rusdcal_.rhpchmip[i][0], rusdcal_.rhpchmip[i][1],
	       rusdcal_.rhpchped[i][0], rusdcal_.rhpchped[i][1],
	       rusdcal_.mip[i][0],rusdcal_.mip[i][1],chi2pdof[0],chi2pdof[1]);
    }
  
  
  return 0;
}
