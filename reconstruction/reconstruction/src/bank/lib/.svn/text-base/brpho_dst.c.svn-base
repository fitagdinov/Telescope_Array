/*
 * brpho_dst.c
 *
 * $Source: /hires_soft/uvm2k/bank/brpho_dst.c,v $
 * $Log: brpho_dst.c,v $
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
 * Created from pho1_dst by replacing all occurences of "pho1" with "brpho".
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
#include "brpho_dst.h"  

brpho_dst_common brpho_;  /* allocate memory to brpho_common */

static integer4 brpho_blen = 0; 
static integer4 brpho_maxlen = sizeof(integer4) * 2 + sizeof(brpho_dst_common);
static integer1 *brpho_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* brpho_bank_buffer_ (integer4* brpho_bank_buffer_size)
{
  (*brpho_bank_buffer_size) = brpho_blen;
  return brpho_bank;
}




static void brpho_bank_init(void)
{
  brpho_bank = (integer1 *)calloc(brpho_maxlen, sizeof(integer1));
  if (brpho_bank==NULL)
    {
      fprintf (stderr,"brpho_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}    

integer4 brpho_common_to_bank_(void)
{
  static integer4 id = BRPHO_BANKID, ver = BRPHO_BANKVERSION;
  integer4 rcode, nobj;

  if (brpho_bank == NULL) brpho_bank_init();
    
  if ( (rcode = dst_initbank_(&id, &ver, &brpho_blen, &brpho_maxlen, 
			      brpho_bank)) ) {
    printf(" brpho_initbank error %d\n",rcode);
    printf(" length %d maxlen %d\n",brpho_blen,brpho_maxlen);
    return rcode;
  }
  /* Initialize brpho_blen, and pack the id and version to bank */
  if ( (rcode = dst_packr8_(&brpho_.jday, (nobj=1, &nobj), brpho_bank, 
			    &brpho_blen, &brpho_maxlen)) ) {
    printf(" brpho_pack_jday error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,brpho_blen,brpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4asi2_(&brpho_.ntube, (nobj=1, &nobj), brpho_bank, 
				&brpho_blen, &brpho_maxlen)) ) {
    printf(" brpho_pack_ntube error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,brpho_blen,brpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi1_(brpho_.calfile, (nobj=32, &nobj), brpho_bank, 
			    &brpho_blen, &brpho_maxlen)) ) {
    printf(" brpho_pack_calfile error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,brpho_blen,brpho_maxlen);
    return rcode;
  }
  nobj = brpho_.ntube; 
  if ( (rcode = dst_packi4_(brpho_.mirtube, &nobj, brpho_bank, &brpho_blen, 
			    &brpho_maxlen)) ) {
    printf(" brpho_pack_mirtube error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,brpho_blen,brpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4_(brpho_.pha, &nobj, brpho_bank, &brpho_blen, 
			    &brpho_maxlen)) ) {
    printf(" brpho_pack_pha error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,brpho_blen,brpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4_(brpho_.phb, &nobj, brpho_bank, &brpho_blen, 
			    &brpho_maxlen)) ) {
    printf(" brpho_pack_phb error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,brpho_blen,brpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4asi2_(brpho_.tab, &nobj, brpho_bank, &brpho_blen, 
				&brpho_maxlen)) ) {
    printf(" brpho_pack_tab error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,brpho_blen,brpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4_(brpho_.time, &nobj, brpho_bank, &brpho_blen, 
			    &brpho_maxlen)) ) {
    printf(" brpho_pack_time error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,brpho_blen,brpho_maxlen);
    return rcode;
  }
  if ( (rcode = dst_packi4asi2_(brpho_.dtime, &nobj, brpho_bank, &brpho_blen, 
				&brpho_maxlen)) ) {
    printf(" brpho_pack_dtime error %d\n",rcode);
    printf(" nobj %d length %d maxlen %d\n",nobj,brpho_blen,brpho_maxlen);
    return rcode;
  }

  return SUCCESS;
}

integer4 brpho_bank_to_dst_ (integer4 *unit)
{
  return dst_write_bank_(unit, &brpho_blen, brpho_bank);
}

integer4 brpho_common_to_dst_(integer4 *unit)
{
  integer4 rcode;
    if ( (rcode = brpho_common_to_bank_()) )
    {
      fprintf(stderr, "brpho_common_to_bank_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }
    if ( (rcode = brpho_bank_to_dst_(unit) ))
    {
      fprintf(stderr, "brpho_bank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
    }

  return SUCCESS;
}

integer4 brpho_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0 ;
  integer4 nobj ;
  brpho_blen = 2 * sizeof(integer4);	/* skip id and version  */

  if ( (rcode = dst_unpackr8_(&brpho_.jday, (nobj=1, &nobj), bank, 
			      &brpho_blen, &brpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2asi4_(&brpho_.ntube, (nobj=1, &nobj), bank, 
				  &brpho_blen, &brpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki1_(brpho_.calfile, (nobj=32, &nobj), bank, 
			      &brpho_blen, &brpho_maxlen)) ) return rcode;

  nobj = brpho_.ntube; 

  if ( (rcode = dst_unpacki4_(brpho_.mirtube, &nobj, bank, &brpho_blen, 
			      &brpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(brpho_.pha, &nobj,bank, &brpho_blen, 
			      &brpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(brpho_.phb, &nobj,bank, &brpho_blen, 
			      &brpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2asi4_(brpho_.tab, &nobj,bank, &brpho_blen, 
				  &brpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki4_(brpho_.time, &nobj,bank, &brpho_blen, 
			      &brpho_maxlen)) ) return rcode;
  if ( (rcode = dst_unpacki2asi4_(brpho_.dtime, &nobj,bank, &brpho_blen, 
				  &brpho_maxlen)) ) return rcode;
  return SUCCESS;
}

integer4 brpho_common_to_dump_(integer4 *long_output)
{
  return brpho_common_to_dumpf_(stdout, long_output);
}

integer4 brpho_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i;

  fprintf (fp, "BRPHO jDay: %15.9f  tubes: %4d  calfile: %.31s\n",
	   brpho_.jday, brpho_.ntube, brpho_.calfile);

  if ( *long_output == 1 ) {
    for (i = 0; i < brpho_.ntube; i++) {
      fprintf (fp, "     mir %2d tube %3d  phoA: %6d phoB: %6d time: %6d dt: %6d tab: %06d\n",
	       brpho_.mirtube[i] / 1000, brpho_.mirtube[i] % 1000, 
	       brpho_.pha[i], brpho_.phb[i], brpho_.time[i], brpho_.dtime[i], 
	       brpho_.tab[i]);
    }
  }

  return SUCCESS;
} 
