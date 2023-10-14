/*
 * Created 2000/04/22 DRB
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fscn1_dst.h"  

fscn1_dst_common fscn1_;  /* allocate memory to pho1_common */

static integer4 fscn1_blen = 0; 
static integer4 fscn1_maxlen = sizeof(integer4) * 2 + sizeof(fscn1_dst_common);
static integer1 *fscn1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fscn1_bank_buffer_ (integer4* fscn1_bank_buffer_size)
{
  (*fscn1_bank_buffer_size) = fscn1_blen;
  return fscn1_bank;
}



static void fscn1_bank_init()
{
  fscn1_bank = (integer1 *)calloc(fscn1_maxlen, sizeof(integer1));
  if (fscn1_bank==NULL)
    {
      fprintf (stderr,"fscn1_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}    

integer4 fscn1_common_to_bank_()
{
  static integer4 id = FSCN1_BANKID, ver = FSCN1_BANKVERSION;
  integer4 rcode, nobj;

  if (fscn1_bank == NULL) fscn1_bank_init();
     
  rcode = dst_initbank_(&id, &ver, &fscn1_blen, &fscn1_maxlen, fscn1_bank);
  /* Initialize fscn1_blen, and pack the id and version to bank */
  nobj = 1;
  rcode += dst_packi4_(&fscn1_.ntube,  &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  nobj=fscn1_.ntube; 
  rcode += dst_packi4_(&fscn1_.mir[0],    &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packi4_(&fscn1_.tube[0],   &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packi4_(&fscn1_.ig[0],     &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packr4_(&fscn1_.ped[0],    &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packr4_(&fscn1_.pedrms[0], &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packi4_(&fscn1_.pamp[0],   &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packi4_(&fscn1_.pmaxt[0],  &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packr4_(&fscn1_.pnpe[0],   &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packi4_(&fscn1_.pt0[0],    &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packi4_(&fscn1_.pnt[0],    &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packr4_(&fscn1_.ptav[0],   &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_packr4_(&fscn1_.pfilt[0],  &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  return rcode ;
}

integer4 fscn1_bank_to_dst_ (integer4 *NumUnit)
{
  return dst_write_bank_(NumUnit, &fscn1_blen, fscn1_bank);
}

integer4 fscn1_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = fscn1_common_to_bank_()))
    {
      fprintf(stderr, "fscn1_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
  if ((rcode= fscn1_bank_to_dst_(NumUnit)))
    {
      fprintf(stderr, "fscn1_bank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
  return 0;
}

integer4 fscn1_bank_to_common_(integer1 *fscn1_bank)
{
  integer4 rcode = 0 ;
  integer4 nobj ;
  fscn1_blen = 2 * sizeof(integer4);	/* skip id and version  */

  nobj = 1;
  rcode += dst_unpacki4_(&fscn1_.ntube,  &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  nobj=fscn1_.ntube;  
  rcode += dst_unpacki4_(&fscn1_.mir[0],    &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpacki4_(&fscn1_.tube[0],   &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpacki4_(&fscn1_.ig[0],     &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpackr4_(&fscn1_.ped[0],    &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpackr4_(&fscn1_.pedrms[0], &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpacki4_(&fscn1_.pamp[0],   &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpacki4_(&fscn1_.pmaxt[0],  &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpackr4_(&fscn1_.pnpe[0],   &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpacki4_(&fscn1_.pt0[0],    &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpacki4_(&fscn1_.pnt[0],    &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpackr4_(&fscn1_.ptav[0],   &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  rcode += dst_unpackr4_(&fscn1_.pfilt[0],  &nobj, fscn1_bank, &fscn1_blen, &fscn1_maxlen);
  return rcode ;
}

integer4 fscn1_common_to_dump_(integer4 *long_output)
{
  return fscn1_common_to_dumpf_(stdout, long_output);
}

integer4 fscn1_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  int i;

  fprintf (fp, "\n\nFSCN1 bank (FADC scan  for HR2)\n");
  fprintf (fp, "Number of tubes = %-4d\n", fscn1_.ntube);

  /* -------------- tube info --------------------------*/
  if((*long_output) == 1)
    {  
      fprintf (fp, "tube info:\n");
      fprintf (fp, "  ++I mir tube G  ped RMS Amp Tmx    NPE  T1  NT     Tav Filt\n");
    /*fprintf (fp, "__++I_mir_tube_G..__ped._RMS.._Amp_Tmx____NPE__T1__NT_____Tav_Filt.\n");*/
    /*fprintf (fp, "__%3d_%3d_%4d._%1d_%4.1f_%3.1f_%3d_%3d_%6.1f._%3d_%3d_%7.1f._%4.1f\n");*/
      for (i = 0;i < fscn1_.ntube; i++)
	  fprintf (fp, "  %3d %3d %4d %1d %4.1f %3.1f %3d %3d %6.1f %3d %3d %7.1f %4.1f\n",
		   i+1,
		   fscn1_.mir[i], fscn1_.tube[i], fscn1_.ig[i],
		   fscn1_.ped[i], fscn1_.pedrms[i],
		   fscn1_.pamp[i], fscn1_.pmaxt[i], fscn1_.pnpe[i],
		   fscn1_.pt0[i], fscn1_.pnt[i], fscn1_.ptav[i],
		   fscn1_.pfilt[i]);
    }
  else 
      fprintf (fp, "tube info: Not displayed in short output\n");
  fprintf (fp,"\n\n");
  return 0;
} 
