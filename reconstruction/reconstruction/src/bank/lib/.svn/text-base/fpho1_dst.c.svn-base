/*
 * fpho1_dst.c
 *
 * $Source: /hires_soft/uvm2k/bank/fpho1_dst.c,v $
 * $Log: fpho1_dst.c,v $
 * Revision 1.6  2003/05/28 15:27:01  hires
 * update to nevis version to eliminate tar file dependency (boyer)
 *
 * Revision 1.4  1997/02/22  08:59:30  mjk
 * Made function declarations ANSI compatible. Got rid of additive rcode
 * stuff; return SUCCESS (from dst_err_codes.h). Put in long dump format
 * for dump routines.
 *
 * 
 * FADC equivalent version of pho1_dst_c
 * Created from pho1_dst by replacing all occurences of "pho1" with "fpho1".
 *
 * Author:  J. Boyer 11 July 1995
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
#include "fpho1_dst.h"  

fpho1_dst_common fpho1_;  /* allocate memory to fpho1_common */

static integer4 fpho1_blen = 0; 
static integer4 fpho1_maxlen = sizeof(integer4) * 2 + sizeof(fpho1_dst_common);
static integer1 *fpho1_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fpho1_bank_buffer_ (integer4* fpho1_bank_buffer_size)
{
  (*fpho1_bank_buffer_size) = fpho1_blen;
  return fpho1_bank;
}




static void fpho1_bank_init(void)
{
  fpho1_bank = (integer1 *)calloc(fpho1_maxlen, sizeof(integer1));
  if (fpho1_bank==NULL)
    {
      fprintf (stderr,"fpho1_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}    

integer4 fpho1_common_to_bank_(void)
{
  static integer4 id = FPHO1_BANKID, ver = FPHO1_BANKVERSION;
  integer4 rcode, nobj;

  if (fpho1_bank == NULL) fpho1_bank_init();
    
  if ( (rcode = dst_initbank_(&id, &ver, &fpho1_blen, &fpho1_maxlen, 
			     fpho1_bank)) ) {
    printf(" fpho1_initbank error %d\n",rcode);
    printf(" length %d maxlen %d\n",fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  /* Initialize fpho1_blen, and pack the id and version to bank */
  if ( (rcode = dst_packr8_(&fpho1_.jday, (nobj=1, &nobj), fpho1_bank, 
			    &fpho1_blen, &fpho1_maxlen)) ) {
    printf(" fpho1_pack_jday error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4asi2_(&fpho1_.ntube, (nobj=1, &nobj), fpho1_bank, 
				&fpho1_blen, &fpho1_maxlen)) ) {
    printf(" fpho1_pack_ntube error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi1_(fpho1_.calfile, (nobj=32, &nobj), fpho1_bank, 
			    &fpho1_blen, &fpho1_maxlen)) ) {
    printf(" fpho1_pack_calfile error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  nobj = fpho1_.ntube; 
  if ( (rcode = dst_packi4_(fpho1_.mirtube, &nobj, fpho1_bank, &fpho1_blen, 
			    &fpho1_maxlen)) ) {
    printf(" fpho1_pack_mirtube error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4_(fpho1_.pha, &nobj, fpho1_bank, &fpho1_blen, 
			    &fpho1_maxlen)) ) {
    printf(" fpho1_pack_pha error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4_(fpho1_.phb, &nobj, fpho1_bank, &fpho1_blen, 
			   &fpho1_maxlen)) ) {
    printf(" fpho1_pack_phb error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4asi2_(fpho1_.tab, &nobj, fpho1_bank, &fpho1_blen, 
			       &fpho1_maxlen)) ) {
    printf(" fpho1_pack_tab error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4_(fpho1_.time, &nobj, fpho1_bank, &fpho1_blen, 
			   &fpho1_maxlen)) ) {
    printf(" fpho1_pack_time error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4asi2_(fpho1_.dtime, &nobj, fpho1_bank, &fpho1_blen, 
			       &fpho1_maxlen)) ) {
    printf(" fpho1_pack_dtime error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,fpho1_blen,fpho1_maxlen);
    return rcode;
  }
  if(ver >= 1)
    {
      if ((rcode = dst_packr4_(fpho1_.adc,&nobj,fpho1_bank,&fpho1_blen, &fpho1_maxlen)))
	return rcode;
      if ((rcode = dst_packr4_(fpho1_.npe,&nobj,fpho1_bank,&fpho1_blen, &fpho1_maxlen)))
	return rcode;
      if ((rcode = dst_packr4_(fpho1_.ped,&nobj,fpho1_bank,&fpho1_blen, &fpho1_maxlen)))
	return rcode;
    }

  return SUCCESS;
}

integer4 fpho1_bank_to_dst_ (integer4 *unit)
{
  return dst_write_bank_(unit, &fpho1_blen, fpho1_bank);
}

integer4 fpho1_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
    if ( (rcode = fpho1_common_to_bank_()) )
    {
      fprintf(stderr, "fpho1_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
    if ( (rcode = fpho1_bank_to_dst_(unit) ))
    {
      fprintf(stderr, "fpho1_bank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }

  return SUCCESS;
}

integer4 fpho1_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0 ;
  integer4 nobj ;
  integer4 bankid, bankversion;
  
  fpho1_blen = 0;
  
  if ((rcode = dst_unpacki4_( &bankid,      (nobj=1, &nobj), bank, &fpho1_blen, &fpho1_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &bankversion, (nobj=1, &nobj), bank, &fpho1_blen, &fpho1_maxlen))) return rcode;
  
  if ( (rcode = dst_unpackr8_(&fpho1_.jday, (nobj=1, &nobj), bank, 
			      &fpho1_blen, &fpho1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2asi4_(&fpho1_.ntube, (nobj=1, &nobj), bank, 
				  &fpho1_blen, &fpho1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki1_(fpho1_.calfile, (nobj=32, &nobj), bank, 
			      &fpho1_blen, &fpho1_maxlen)) ) return rcode;

  nobj = fpho1_.ntube; 

  if ( (rcode = dst_unpacki4_(fpho1_.mirtube, &nobj, bank, &fpho1_blen, 
			      &fpho1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(fpho1_.pha, &nobj,bank, &fpho1_blen, 
			      &fpho1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(fpho1_.phb, &nobj,bank, &fpho1_blen, 
			      &fpho1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2asi4_(fpho1_.tab, &nobj,bank, &fpho1_blen, 
				  &fpho1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(fpho1_.time, &nobj,bank, &fpho1_blen, 
			      &fpho1_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2asi4_(fpho1_.dtime, &nobj,bank, &fpho1_blen, 
				  &fpho1_maxlen)) ) return rcode;
  if(bankversion >= 1)
    {
      if ( (rcode = dst_unpackr4_(fpho1_.adc, &nobj,bank, &fpho1_blen, 
				  &fpho1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpackr4_(fpho1_.npe, &nobj,bank, &fpho1_blen, 
				  &fpho1_maxlen)) ) return rcode;
      if ( (rcode = dst_unpackr4_(fpho1_.ped, &nobj,bank, &fpho1_blen, 
				  &fpho1_maxlen)) ) return rcode;
    }
  else
    {
      int i=0;
      for (i=0; i<fpho1_.ntube; i++)
	{
	  fpho1_.adc[i] = 0.0;
	  fpho1_.npe[i] = 0.0;
	  fpho1_.ped[i] = 0.0;
	}
    }
  return SUCCESS;
}

integer4 fpho1_common_to_dump_(integer4 *long_output)
{
  return fpho1_common_to_dumpf_(stdout, long_output);
}

integer4 fpho1_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i;

  fprintf (fp, "FPHO1 jDay: %15.9f  tubes: %4d  calfile: %.31s\n",
	   fpho1_.jday, fpho1_.ntube, fpho1_.calfile);

  if ( *long_output == 1 ) {
    for (i = 0; i < fpho1_.ntube; i++) {
      fprintf (fp, "     mir %2d tube %3d  phoA: %6d phoB: %6d time: %6d dt: %6d tab: %06d",
	       fpho1_.mirtube[i] / 1000, fpho1_.mirtube[i] % 1000, 
	       fpho1_.pha[i], fpho1_.phb[i], fpho1_.time[i], fpho1_.dtime[i], 
	       fpho1_.tab[i]);
      fprintf(fp," adc: %7.3f npe: %7.3f ped: %6.3f\n",
	      fpho1_.adc[i],fpho1_.npe[i],fpho1_.ped[i]);
    }
  }

  return SUCCESS;
} 
