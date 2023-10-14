/*
 * C functions for etrack
 * Dmitri Ivanov, ivanov@physics.rutgers.edu
 * Apr 23, 2011
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "etrack_dst.h"



etrack_dst_common etrack_;	/* allocate memory to etrack_common */

static integer4 etrack_blen = 0;
static integer4 etrack_maxlen =
  sizeof (integer4) * 2 + sizeof (etrack_dst_common);
static integer1 *etrack_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* etrack_bank_buffer_ (integer4* etrack_bank_buffer_size)
{
  (*etrack_bank_buffer_size) = etrack_blen;
  return etrack_bank;
}



static void
etrack_bank_init ()
{
  etrack_bank = (integer1 *) calloc (etrack_maxlen, sizeof (integer1));
  if (etrack_bank == NULL)
    {
      fprintf (stderr,
	       "etrack_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }				/* else fprintf ( stderr,"etrack_bank allocated memory %d\n",etrack_maxlen); */
}

integer4 etrack_common_to_bank_ ()
{
  static integer4 id = ETRACK_BANKID, ver = ETRACK_BANKVERSION;
  integer4 rcode, nobj;

  if (etrack_bank == NULL)
    etrack_bank_init ();

  rcode =
    dst_initbank_ (&id, &ver, &etrack_blen, &etrack_maxlen, etrack_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj = 1;
  rcode += dst_packi4_(&etrack_.nudata,&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_packi4_(&etrack_.yymmdd,&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_packi4_(&etrack_.hhmmss,&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_packi4_(&etrack_.qualct,&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  
  rcode += dst_packr4_(&etrack_.energy,&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_packr4_(&etrack_.xmax,&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_packr4_(&etrack_.theta,&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_packr4_(&etrack_.phi,&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_packr8_(&etrack_.t0,&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  
  nobj = 2;
  rcode += dst_packr4_(&etrack_.xycore[0],&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  if(etrack_.nudata < 0 || etrack_.nudata > ETRACK_NUDATA)
    {
      fprintf(stderr,"^^^^^ warning: etrack_common_to_bank_: wrong nudata value %d, must be in 0-%d range\n",
	      etrack_.nudata,ETRACK_NUDATA);
      if(etrack_.nudata < 0)
	etrack_.nudata = 0;
      else
	etrack_.nudata = ETRACK_NUDATA;
    }
  nobj = etrack_.nudata;
  rcode += dst_packr4_(&etrack_.udata[0],&nobj,etrack_bank,&etrack_blen,&etrack_maxlen);
  
  return rcode;
}

integer4 etrack_bank_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_ (NumUnit, &etrack_blen, etrack_bank);
  free (etrack_bank);
  etrack_bank = NULL;
  return rcode;
}

integer4 etrack_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = etrack_common_to_bank_ ()))
    {
      fprintf (stderr, "etrack_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = etrack_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "etrack_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4 etrack_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj;
  etrack_blen = 2 * sizeof (integer4);	/* skip id and version  */
  
  nobj = 1;
  rcode += dst_unpacki4_(&etrack_.nudata,&nobj,bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_unpacki4_(&etrack_.yymmdd,&nobj,bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_unpacki4_(&etrack_.hhmmss,&nobj,bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_unpacki4_(&etrack_.qualct,&nobj,bank,&etrack_blen,&etrack_maxlen);
  
  rcode += dst_unpackr4_(&etrack_.energy,&nobj,bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_unpackr4_(&etrack_.xmax,&nobj,bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_unpackr4_(&etrack_.theta,&nobj,bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_unpackr4_(&etrack_.phi,&nobj,bank,&etrack_blen,&etrack_maxlen);
  rcode += dst_unpackr8_(&etrack_.t0,&nobj,bank,&etrack_blen,&etrack_maxlen);
  
  nobj = 2;
  rcode += dst_unpackr4_(&etrack_.xycore[0],&nobj,bank,&etrack_blen,&etrack_maxlen);
  
  if(etrack_.nudata < 0 || etrack_.nudata > ETRACK_NUDATA)
    {
      fprintf(stderr,"^^^^^ warning: etrack_bank_to_common_: wrong nudata value %d, must be in 0-%d range\n",
	      etrack_.nudata,ETRACK_NUDATA);
      if(etrack_.nudata < 0)
	etrack_.nudata = 0;
      else
	etrack_.nudata = ETRACK_NUDATA;
    }
  nobj = etrack_.nudata;
  rcode += dst_unpackr4_(&etrack_.udata[0],&nobj,bank,&etrack_blen,&etrack_maxlen);
  
  return rcode;
}

integer4 etrack_common_to_dump_ (integer4 * long_output)
{
  return etrack_common_to_dumpf_ (stdout, long_output);
}

integer4 etrack_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  fprintf (fp, "%s :\n","etrack");
  if(etrack_.nudata < 0 || etrack_.nudata > ETRACK_NUDATA)
    {
       fprintf(stderr,"^^^^^ warning: etrack_common_to_dumpf_: wrong nudata value %d, must be in 0-%d range\n",
	       etrack_.nudata,ETRACK_NUDATA);
       if(etrack_.nudata < 0)
	 etrack_.nudata = 0;
       else
	 etrack_.nudata = ETRACK_NUDATA;
    }
  if (*long_output == 1)
    {
      fprintf(fp,"date: %06d time: %06d.%06d ",etrack_.yymmdd,etrack_.hhmmss,(int)floor(etrack_.t0+0.5));
      fprintf(fp,"energy: %.2f xmax: %.0f theta: %.3f phi: %.3f xycore: %.3e %.3e qualct: %d nudata: %d",
	      etrack_.energy,etrack_.xmax,etrack_.theta,etrack_.phi,etrack_.xycore[0],etrack_.xycore[1],
	      etrack_.qualct,etrack_.nudata);
      if(etrack_.nudata > 0)
	{
	  int i;
	  fprintf(fp," udata: ");
	  for (i = 0; i < etrack_.nudata; i++)
	    {
	      fprintf(fp,"%.3e",etrack_.udata[i]);
	      if (i < (etrack_.nudata - 1))
		fprintf(fp," ");
	    }
	}
      fprintf(fp,"\n");
    }
  else
    {
      fprintf(fp,"%06d %06d.%06d ",etrack_.yymmdd,etrack_.hhmmss,(int)floor(etrack_.t0+0.5));
      fprintf(fp,"%.2f %.0f %.3f %.3f %.3e %.3e %d %d",
	      etrack_.energy,etrack_.xmax,etrack_.theta,etrack_.phi,etrack_.xycore[0],etrack_.xycore[1],
	      etrack_.qualct,etrack_.nudata);
      if(etrack_.nudata > 0)
	{
	  int i;
	  fprintf(fp," ");
	  for (i = 0; i < etrack_.nudata; i++)
	    {
	      fprintf(fp,"%.3e",etrack_.udata[i]);
	      if (i < (etrack_.nudata - 1))
		fprintf(fp," ");
	    }
	}
      fprintf(fp,"\n");
    }
  return 0;
}
