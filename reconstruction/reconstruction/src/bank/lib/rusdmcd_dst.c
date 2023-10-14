/*
 * C functions for rusdmcd
 * Dmitri Ivanov, ivanov@physics.rutgers.edu
 * Apr 23, 2009
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "rusdmcd_dst.h"
#define RADDEG 57.2957795131


rusdmcd_dst_common rusdmcd_;	/* allocate memory to rusdmcd_common */

static integer4 rusdmcd_blen = 0;
static integer4 rusdmcd_maxlen =
  sizeof (integer4) * 2 + sizeof (rusdmcd_dst_common);
static integer1 *rusdmcd_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* rusdmcd_bank_buffer_ (integer4* rusdmcd_bank_buffer_size)
{
  (*rusdmcd_bank_buffer_size) = rusdmcd_blen;
  return rusdmcd_bank;
}



static void rusdmcd_bank_init ()
{
  rusdmcd_bank = (integer1 *) calloc (rusdmcd_maxlen, sizeof (integer1));
  if (rusdmcd_bank == NULL)
    {
      fprintf (stderr,
	       "rusdmcd_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }				/* else fprintf ( stderr,"rusdmcd_bank allocated memory %d\n",rusdmcd_maxlen); */
}

integer4 rusdmcd_common_to_bank_ ()
{
  static integer4 id = RUSDMCD_BANKID, ver = RUSDMCD_BANKVERSION;
  integer4 rcode, nobj;
  integer4 i;

  
  if (rusdmcd_bank == NULL)
    rusdmcd_bank_init ();

  rcode =
    dst_initbank_ (&id, &ver, &rusdmcd_blen, &rusdmcd_maxlen, rusdmcd_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj = 1;

  rcode +=
    dst_packi4_ (&rusdmcd_.nsds, &nobj, rusdmcd_bank, &rusdmcd_blen,
		 &rusdmcd_maxlen);
  nobj = rusdmcd_.nsds;
  rcode +=
    dst_packi4_ (&rusdmcd_.xxyy[0], &nobj, rusdmcd_bank, &rusdmcd_blen,
		 &rusdmcd_maxlen);
  rcode +=
    dst_packi4_ (&rusdmcd_.igsd[0], &nobj, rusdmcd_bank, &rusdmcd_blen,
		 &rusdmcd_maxlen);
  nobj = 2;
  
  for (i=0; i<rusdmcd_.nsds; i++)
    {
      rcode +=
	dst_packr8_ (&rusdmcd_.edep[i][0], &nobj, rusdmcd_bank, &rusdmcd_blen,
		     &rusdmcd_maxlen);
    }
  return rcode;
}

integer4 rusdmcd_bank_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_ (NumUnit, &rusdmcd_blen, rusdmcd_bank);
  free (rusdmcd_bank);
  rusdmcd_bank = NULL;
  return rcode;
}

integer4 rusdmcd_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = rusdmcd_common_to_bank_ ()))
    {
      fprintf (stderr, "rusdmcd_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = rusdmcd_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "rusdmcd_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4 rusdmcd_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj;
  integer4 i;
  rusdmcd_blen = 2 * sizeof (integer4);	/* skip id and version  */
  
  nobj = 1;
  
  rcode +=
    dst_unpacki4_ (&rusdmcd_.nsds, &nobj, bank, &rusdmcd_blen,
		   &rusdmcd_maxlen);
  nobj = rusdmcd_.nsds;
  rcode +=
    dst_unpacki4_ (&rusdmcd_.xxyy[0], &nobj, bank, &rusdmcd_blen,
		   &rusdmcd_maxlen);
  rcode +=
    dst_unpacki4_ (&rusdmcd_.igsd[0], &nobj, bank, &rusdmcd_blen,
		   &rusdmcd_maxlen);
  nobj = 2;
  
  for (i=0; i<rusdmcd_.nsds; i++)
    {
      rcode +=
	dst_unpackr8_ (&rusdmcd_.edep[i][0], &nobj, bank, &rusdmcd_blen,
		       &rusdmcd_maxlen);
    }
  return rcode;
}

integer4 rusdmcd_common_to_dump_ (integer4 * long_output)
{
  return rusdmcd_common_to_dumpf_ (stdout, long_output);
}

integer4 rusdmcd_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  integer4 i;
  (void)(long_output);
  fprintf (fp, "%s :\n","rusdmcd");
  fprintf (fp, "nsds=%d\n",rusdmcd_.nsds);
  
  fprintf (fp, "%s%7s%14s%14s\n",
	   "xxyy","igsd","edepLo","edepUp");
  for (i=0; i<rusdmcd_.nsds; i++)
      fprintf (fp, "%04d%6d%14.2f%14.2f\n",
	       rusdmcd_.xxyy[i],rusdmcd_.igsd[i],
	       rusdmcd_.edep[i][0],rusdmcd_.edep[i][1]);
  return 0;
}
