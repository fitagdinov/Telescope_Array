/*
 * C functions for atmpar
 * Dmitri Ivanov, dmiivanov@gmail.com
 * Mar 09, 2018
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "atmpar_dst.h"
#include "fdcalib_util.h"


atmpar_dst_common atmpar_;	/* allocate memory to atmpar_common */

static integer4 atmpar_blen = 0;
static integer4 atmpar_maxlen = sizeof (integer4) * 2 + sizeof (atmpar_dst_common);
static integer1 *atmpar_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* atmpar_bank_buffer_ (integer4* atmpar_bank_buffer_size)
{
  (*atmpar_bank_buffer_size) = atmpar_blen;
  return atmpar_bank;
}



static void atmpar_bank_init ()
{
  atmpar_bank = (integer1 *) calloc (atmpar_maxlen, sizeof (integer1));
  if (atmpar_bank == NULL)
    {
      fprintf (stderr,
	       "atmpar_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }				/* else fprintf ( stderr,"atmpar_bank allocated memory %d\n",atmpar_maxlen); */
}

integer4 atmpar_common_to_bank_ ()
{
  integer4 i = 0;
  static integer4 id = ATMPAR_BANKID, ver = ATMPAR_BANKVERSION;
  integer4 rcode = 0, nobj = 0;
  if (atmpar_bank == NULL)
    atmpar_bank_init ();
  rcode = dst_initbank_ (&id, &ver, &atmpar_blen, &atmpar_maxlen, atmpar_bank);
  nobj = 1;
  rcode += dst_packi4_ ((integer4*)&atmpar_.dateFrom, &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen); 
  rcode += dst_packi4_ ((integer4*)&atmpar_.dateTo, &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen);
  rcode += dst_packi4_ (&atmpar_.modelid, &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen);
  rcode += dst_packi4_ (&atmpar_.nh, &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen);
  for (i=0; i<atmpar_.nh; i++)
    {
      rcode += dst_packr8_ (&atmpar_.h[i], &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen);
      rcode += dst_packr8_ (&atmpar_.a[i], &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen);
      rcode += dst_packr8_ (&atmpar_.b[i], &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen);
      rcode += dst_packr8_ (&atmpar_.c[i], &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen);
    }
  rcode += dst_packr8_ (&atmpar_.chi2, &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen);
  rcode += dst_packi4_ (&atmpar_.ndof, &nobj, atmpar_bank, &atmpar_blen,&atmpar_maxlen);  
  return rcode;
}

integer4 atmpar_bank_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode = 0;
  rcode = dst_write_bank_ (NumUnit, &atmpar_blen, atmpar_bank);
  free (atmpar_bank);
  atmpar_bank = NULL;
  return rcode;
}

integer4 atmpar_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = atmpar_common_to_bank_ ()))
    {
      fprintf (stderr, "atmpar_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = atmpar_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "atmpar_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4 atmpar_bank_to_common_ (integer1 * bank)
{
  integer4 i = 0, rcode = 0, nobj = 0;
  atmpar_blen = 2 * sizeof (integer4);	/* skip id and version  */
  nobj = 1;
  rcode += dst_unpacki4_ ((integer4*)&atmpar_.dateFrom, &nobj, bank, &atmpar_blen,&atmpar_maxlen); 
  rcode += dst_unpacki4_ ((integer4*)&atmpar_.dateTo, &nobj, bank, &atmpar_blen,&atmpar_maxlen);
  rcode += dst_unpacki4_ (&atmpar_.modelid, &nobj, bank, &atmpar_blen,&atmpar_maxlen);
  rcode += dst_unpacki4_ (&atmpar_.nh, &nobj, bank, &atmpar_blen,&atmpar_maxlen);
  for (i=0; i<atmpar_.nh; i++)
    {
      rcode += dst_unpackr8_ (&atmpar_.h[i], &nobj, bank, &atmpar_blen,&atmpar_maxlen);
      rcode += dst_unpackr8_ (&atmpar_.a[i], &nobj, bank, &atmpar_blen,&atmpar_maxlen);
      rcode += dst_unpackr8_ (&atmpar_.b[i], &nobj, bank, &atmpar_blen,&atmpar_maxlen);
      rcode += dst_unpackr8_ (&atmpar_.c[i], &nobj, bank, &atmpar_blen,&atmpar_maxlen);
    }
  rcode += dst_unpackr8_ (&atmpar_.chi2, &nobj, bank, &atmpar_blen,&atmpar_maxlen);
  rcode += dst_unpacki4_ (&atmpar_.ndof, &nobj, bank, &atmpar_blen,&atmpar_maxlen);    
  return rcode;
}

integer4 atmpar_common_to_dump_ (integer4 * long_output)
{
  return atmpar_common_to_dumpf_ (stdout, long_output);
}

integer4 atmpar_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  integer4 i = 0;
  fprintf (fp, "%s :\n","atmpar");
  char dateFromLine[32], dateToLine[32];
  convertSec2DateLine((time_t)atmpar_.dateFrom,dateFromLine);
  convertSec2DateLine((time_t)atmpar_.dateTo,dateToLine);
  fprintf(fp,"MODELID %d",atmpar_.modelid);
  if(atmpar_.modelid==0)
    fprintf(fp," (GDAS)");
  fprintf(fp," ");
  fprintf(fp,"FROM %s TO %s\n",dateFromLine,dateToLine);
  if(*long_output < 1)
    return 0;
  fprintf(fp,"ATMOD    10   (HIGHEST LAYER SET TO THAT OF US STD. ATM. BY KEILHAUER)\n");
  fprintf(fp,"ATMA  ");
  for(i=0; i<atmpar_.nh; i++)
    fprintf(fp,"%18.9E",atmpar_.a[i]);
  fprintf(fp,"   a values [g/cm^2]\n");  
  fprintf(fp,"ATMB  ");
  for(i=0; i<atmpar_.nh; i++)
    fprintf(fp,"%18.9E",atmpar_.b[i]);
  fprintf(fp,"   b values [g/cm^2]\n");
  fprintf(fp,"ATMC  ");
  for(i=0; i<atmpar_.nh; i++)
    fprintf(fp,"%18.9E",atmpar_.c[i]);
  fprintf(fp,"   c values [cm]\n");
  fprintf(fp,"ATMLAY");
  for(i=1; i<atmpar_.nh; i++)
    fprintf(fp,"%18.9E",atmpar_.h[i]);
  fprintf(fp,"   layer boundaries [cm]\n");
  return 0;
}


real8 h2mo(real8 h)
{
  integer4 i = 0;
  while (i < atmpar_.nh-1 && h > atmpar_.h[i+1]) 
    i++;
  if (i < atmpar_.nh-1)
    return atmpar_.a[i]+atmpar_.b[i]*exp(-h/atmpar_.c[i]);
  return atmpar_.a[i]-atmpar_.b[i]*h/atmpar_.c[i];
}

real8 mo2h(real8 mo)
{
  integer4 i=0;
  real8 h=0;
  while (i < atmpar_.nh-1 && mo < h2mo(atmpar_.h[i+1])) i++;
  if (i < atmpar_.nh-1)
    h = -atmpar_.c[i]*log((mo-atmpar_.a[i])/atmpar_.b[i]);
  else
    h = (atmpar_.a[i]-mo)*atmpar_.c[i]/atmpar_.b[i];
  return h;
}

real8 h2mo_deriv(real8 h)
{
  integer4 i = 0;
  while (i < atmpar_.nh-1 && h > atmpar_.h[i+1])
    i++;
  if (i < atmpar_.nh-1)
    return -atmpar_.b[i]/atmpar_.c[i]*exp(-h/atmpar_.c[i]);
  return -atmpar_.b[i]/atmpar_.c[i];
}
