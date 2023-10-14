/*
 * C functions for tlweat
 * Dmitri Ivanov, dmiivanov@gmail.com
 * Dec 21, 2015
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "tlweat_dst.h"



tlweat_dst_common tlweat_;	/* allocate memory to tlweat_common */

static integer4 tlweat_blen = 0;
static integer4 tlweat_maxlen =
  sizeof (integer4) * 2 + sizeof (tlweat_dst_common);
static integer1 *tlweat_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tlweat_bank_buffer_ (integer4* tlweat_bank_buffer_size)
{
  (*tlweat_bank_buffer_size) = tlweat_blen;
  return tlweat_bank;
}



static void
tlweat_bank_init ()
{
  tlweat_bank = (integer1 *) calloc (tlweat_maxlen, sizeof (integer1));
  if (tlweat_bank == NULL)
    {
      fprintf (stderr,
	       "tlweat_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }				/* else fprintf ( stderr,"tlweat_bank allocated memory %d\n",tlweat_maxlen); */
}

integer4
tlweat_common_to_bank_ ()
{
  
  static integer4 id = TLWEAT_BANKID, ver = TLWEAT_BANKVERSION;
  integer4 rcode, nobj;
  
  if (tlweat_bank == NULL)
    tlweat_bank_init ();
  
  rcode =
    dst_initbank_ (&id, &ver, &tlweat_blen, &tlweat_maxlen, tlweat_bank);
  /* Initialize test_blen, and pack the id and version to bank */
  
  nobj = 1;
  rcode += dst_packi4_ (&tlweat_.part_num, &nobj, tlweat_bank, &tlweat_blen,&tlweat_maxlen); 
  rcode += dst_packi4_ (&tlweat_.yymmdd, &nobj, tlweat_bank, &tlweat_blen,&tlweat_maxlen);
  rcode += dst_packi4_ (&tlweat_.hhmmss, &nobj, tlweat_bank, &tlweat_blen,&tlweat_maxlen);
  rcode += dst_packi4_ (&tlweat_.dt, &nobj, tlweat_bank, &tlweat_blen,&tlweat_maxlen);
  rcode += dst_packi4_ (&tlweat_.code, &nobj, tlweat_bank, &tlweat_blen,&tlweat_maxlen);
  
  return rcode;
}

integer4
tlweat_bank_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode = 0;
  rcode = dst_write_bank_ (NumUnit, &tlweat_blen, tlweat_bank);
  free (tlweat_bank);
  tlweat_bank = NULL;
  return rcode;
}

integer4
tlweat_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = tlweat_common_to_bank_ ()))
    {
      fprintf (stderr, "tlweat_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = tlweat_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "tlweat_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4
tlweat_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj;
  tlweat_blen = 2 * sizeof (integer4);	/* skip id and version  */

  nobj = 1;

  rcode += dst_unpacki4_ (&tlweat_.part_num, &nobj, bank, &tlweat_blen, &tlweat_maxlen);
  rcode += dst_unpacki4_ (&tlweat_.yymmdd, &nobj, bank, &tlweat_blen, &tlweat_maxlen);
  rcode += dst_unpacki4_ (&tlweat_.hhmmss, &nobj, bank, &tlweat_blen, &tlweat_maxlen);
  rcode += dst_unpacki4_ (&tlweat_.dt, &nobj, bank, &tlweat_blen, &tlweat_maxlen);
  rcode += dst_unpacki4_ (&tlweat_.code, &nobj, bank, &tlweat_blen, &tlweat_maxlen);

  return rcode;
}

integer4
tlweat_common_to_dump_ (integer4 * long_output)
{
  return tlweat_common_to_dumpf_ (stdout, long_output);
}

integer4
tlweat_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
 
  fprintf (fp, "%s :\n","tlweat");
  
  if(*long_output == 0)
    {
      fprintf (fp,"part_num %02d date %06d time %06d duration %04d code %07d\n",
	       tlweat_.part_num,tlweat_.yymmdd,tlweat_.hhmmss,tlweat_.dt,tlweat_.code);
    }
  else
    {
      fprintf(fp,"Part number: %02d\n", tlweat_.part_num);
      fprintf(fp,"UTC Date (YYMMDD): %06d\n", tlweat_.yymmdd);
      fprintf(fp,"Part start time (HHMMSS): %06d\n", tlweat_.hhmmss);
      fprintf(fp,"Part duration time: %04d seconds\n", tlweat_.dt);
      fprintf(fp,"Part Weather code: %07d\n", tlweat_.code);
      fprintf(fp,"n e s w o t h 7-digit weather code recorder by runners\n");
      fprintf(fp,"n = 1,  0 Clouds North?\n");
      fprintf(fp,"e = 1,  0 Clouds East?\n");
      fprintf(fp,"s = 1,  0 Clouds South?\n");
      fprintf(fp,"w = 1,  0 Clouds West?\n");
      fprintf(fp,"o = 0 - 4 Overhead cloud thickness? 5 - weat code invalid\n");
      fprintf(fp,"t = 1,  0 Stars visible?\n");
      fprintf(fp,"h = 1,  0 Was it hazy? 2 - can't tell\n");
    }

  return 0;
}
