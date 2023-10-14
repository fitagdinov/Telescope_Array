/*
 * lrpho_dst.c
 *
 * $Source: /hires_soft/uvm2k/bank/lrpho_dst.c,v $
 * $Log: lrpho_dst.c,v $
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
 * Created from pho1_dst by replacing all occurences of "pho1" with "lrpho".
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
#include "lrpho_dst.h"  

lrpho_dst_common lrpho_;  /* allocate memory to lrpho_common */

static integer4 lrpho_blen = 0; 
static integer4 lrpho_maxlen = sizeof(integer4) * 2 + sizeof(lrpho_dst_common);
static integer1 *lrpho_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* lrpho_bank_buffer_ (integer4* lrpho_bank_buffer_size)
{
  (*lrpho_bank_buffer_size) = lrpho_blen;
  return lrpho_bank;
}




static void lrpho_bank_init(void)
{
  lrpho_bank = (integer1 *)calloc(lrpho_maxlen, sizeof(integer1));
  if (lrpho_bank==NULL)
    {
      fprintf (stderr,"lrpho_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}    

integer4 lrpho_common_to_bank_(void)
{
  static integer4 id = LRPHO_BANKID, ver = LRPHO_BANKVERSION;
  integer4 rcode, nobj;

  if (lrpho_bank == NULL) lrpho_bank_init();
    
  if ( (rcode = dst_initbank_(&id, &ver, &lrpho_blen, &lrpho_maxlen, 
			     lrpho_bank) ) ) {
    printf(" lrpho_initbank error %d\n",rcode);
    printf(" length %d maxlen %d\n",lrpho_blen,lrpho_maxlen);
    return rcode;
  }
  /* Initialize lrpho_blen, and pack the id and version to bank */
  if ( (rcode = dst_packr8_(&lrpho_.jday, (nobj=1, &nobj), lrpho_bank, 
			   &lrpho_blen, &lrpho_maxlen)) ) {
    printf(" lrpho_pack_jday error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,lrpho_blen,lrpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4asi2_(&lrpho_.ntube, (nobj=1, &nobj), lrpho_bank, 
			       &lrpho_blen, &lrpho_maxlen)) ) {
    printf(" lrpho_pack_ntube error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,lrpho_blen,lrpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi1_(lrpho_.calfile, (nobj=32, &nobj), lrpho_bank, 
			   &lrpho_blen, &lrpho_maxlen)) ) {
    printf(" lrpho_pack_calfile error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,lrpho_blen,lrpho_maxlen);
    return rcode;
  }
  nobj = lrpho_.ntube; 
  if ( (rcode = dst_packi4_(lrpho_.mirtube, &nobj, lrpho_bank, &lrpho_blen, 
			   &lrpho_maxlen)) ) {
    printf(" lrpho_pack_mirtube error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,lrpho_blen,lrpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4_(lrpho_.pha, &nobj, lrpho_bank, &lrpho_blen, 
			   &lrpho_maxlen)) ) {
    printf(" lrpho_pack_pha error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,lrpho_blen,lrpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4_(lrpho_.phb, &nobj, lrpho_bank, &lrpho_blen, 
			   &lrpho_maxlen)) ) {
    printf(" lrpho_pack_phb error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,lrpho_blen,lrpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4asi2_(lrpho_.tab, &nobj, lrpho_bank, &lrpho_blen, 
			       &lrpho_maxlen)) ) {
    printf(" lrpho_pack_tab error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,lrpho_blen,lrpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4_(lrpho_.time, &nobj, lrpho_bank, &lrpho_blen, 
			   &lrpho_maxlen)) ) {
    printf(" lrpho_pack_time error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,lrpho_blen,lrpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4asi2_(lrpho_.dtime, &nobj, lrpho_bank, &lrpho_blen, 
			       &lrpho_maxlen)) ) {
    printf(" lrpho_pack_dtime error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,lrpho_blen,lrpho_maxlen);
    return rcode;
  }

  return SUCCESS;
}

integer4 lrpho_bank_to_dst_ (integer4 *unit)
{
  return dst_write_bank_(unit, &lrpho_blen, lrpho_bank);
}

integer4 lrpho_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
    if ( (rcode = lrpho_common_to_bank_()) )
    {
      fprintf(stderr, "lrpho_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
    if ( (rcode = lrpho_bank_to_dst_(unit) ))
    {
      fprintf(stderr, "lrpho_bank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }

  return SUCCESS;
}

integer4 lrpho_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0 ;
  integer4 nobj ;
  lrpho_blen = 2 * sizeof(integer4);	/* skip id and version  */

  if ( (rcode = dst_unpackr8_(&lrpho_.jday, (nobj=1, &nobj), bank, 
			     &lrpho_blen, &lrpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2asi4_(&lrpho_.ntube, (nobj=1, &nobj), bank, 
				 &lrpho_blen, &lrpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki1_(lrpho_.calfile, (nobj=32, &nobj), bank, 
			     &lrpho_blen, &lrpho_maxlen)) ) return rcode;

  nobj = lrpho_.ntube; 

  if ( (rcode = dst_unpacki4_(lrpho_.mirtube, &nobj, bank, &lrpho_blen, 
			     &lrpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(lrpho_.pha, &nobj,bank, &lrpho_blen, 
			     &lrpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(lrpho_.phb, &nobj,bank, &lrpho_blen, 
			     &lrpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2asi4_(lrpho_.tab, &nobj,bank, &lrpho_blen, 
				 &lrpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(lrpho_.time, &nobj,bank, &lrpho_blen, 
			     &lrpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2asi4_(lrpho_.dtime, &nobj,bank, &lrpho_blen, 
				 &lrpho_maxlen)) ) return rcode;
  return SUCCESS;
}

integer4 lrpho_common_to_dump_(integer4 *long_output)
{
  return lrpho_common_to_dumpf_(stdout, long_output);
}

integer4 lrpho_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i;

  fprintf (fp, "LRPHO jDay: %15.9f  tubes: %4d  calfile: %.31s\n",
	   lrpho_.jday, lrpho_.ntube, lrpho_.calfile);

  if ( *long_output == 1 ) {
    for (i = 0; i < lrpho_.ntube; i++) {
      fprintf (fp, "     mir %2d tube %3d  phoA: %6d phoB: %6d time: %6d dt: %6d tab: %06d\n",
	       lrpho_.mirtube[i] / 1000, lrpho_.mirtube[i] % 1000, 
	       lrpho_.pha[i], lrpho_.phb[i], lrpho_.time[i], lrpho_.dtime[i], 
	       lrpho_.tab[i]);
    }
  }

  return SUCCESS;
} 
