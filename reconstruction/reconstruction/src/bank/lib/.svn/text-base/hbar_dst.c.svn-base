/*
 * hbar_dst.c 
 *
 * $Source: /hires_soft/uvm2k/bank/hbar_dst.c,v $
 * $Log: hbar_dst.c,v $
 * Revision 1.2  2000/05/31 20:32:58  ben
 * Added QE sigma variables.
 *
 * Revision 1.1  2000/02/04 22:39:47  ben
 * Initial revision
 *
 *
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
#include "hbar_dst.h"  

hbar_dst_common hbar_;  /* allocate memory to hbar_common */

static integer4 hbar_blen = 0; 
static integer4 hbar_maxlen = sizeof(integer4) * 2 + sizeof(hbar_dst_common);
static integer1 *hbar_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hbar_bank_buffer_ (integer4* hbar_bank_buffer_size)
{
  (*hbar_bank_buffer_size) = hbar_blen;
  return hbar_bank;
}



static void hbar_bank_init(void)
{
  hbar_bank = (integer1 *)calloc(hbar_maxlen, sizeof(integer1));
  if (hbar_bank==NULL)
    {
      fprintf(stderr, 
	      "hbar_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 hbar_common_to_bank_(void)
{	
  static integer4 id = HBAR_BANKID, ver = HBAR_BANKVERSION;
  integer4 rcode, nobj;

  if (hbar_bank == NULL) hbar_bank_init();

  /* Initialize hbar_blen, and pack the id and version to bank */
  if ((rcode = dst_initbank_(&id, &ver, &hbar_blen, &hbar_maxlen, hbar_bank)))
    return rcode;

  nobj = 1;

  if ((rcode = dst_packi4_(&hbar_.jday, &nobj, hbar_bank, 
			   &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_packi4_(&hbar_.jsec, &nobj, hbar_bank, 
			   &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_packi4_(&hbar_.msec, &nobj, hbar_bank,
			   &hbar_blen, &hbar_maxlen)))
    return rcode;
  

  /* These need to be packed early on.. */

  if ((rcode = dst_packi4_(&hbar_.nmir, &nobj, hbar_bank,
			   &hbar_blen, &hbar_maxlen)))
    return rcode;
  if ((rcode = dst_packi4_(&hbar_.ntube, &nobj, hbar_bank,
			   &hbar_blen, &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_packi4_(&hbar_.source, &nobj, hbar_bank,
			   &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  /* Now we pack mirror-wise arrays.. */

  nobj = hbar_.nmir;

  if ((rcode = dst_packr8_(hbar_.hnpe_jday, 
			   &nobj, hbar_bank,
			   &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_packi4_(hbar_.mir, &nobj, hbar_bank,
			   &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_packr8_(hbar_.mir_reflect, &nobj,
			   hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  

  /* Now we pack tube-wise arrays.. */

  nobj = hbar_.ntube;

  if ((rcode = dst_packi4_(hbar_.tubemir, &nobj, 
			   hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_packi4_(hbar_.tube, &nobj, 
			   hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_packi4_(hbar_.qdcb, &nobj, 
			   hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_packr8_(hbar_.npe, &nobj, hbar_bank,
			   &hbar_blen, &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_packr8_(hbar_.sigma_npe, &nobj,
			   hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_packi1_((integer1 *)hbar_.first_order_gain_flag, 
			   &nobj,hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  
  if ((rcode = dst_packr8_(hbar_.second_order_gain,
			   &nobj, hbar_bank, &hbar_blen,
			   &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_packr8_(hbar_.second_order_gain_sigma, 
			   &nobj, hbar_bank, &hbar_blen,
			   &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_packi1_((integer1 *)hbar_.second_order_gain_flag, 
			   &nobj,hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;


  if ((rcode = dst_packr8_(hbar_.qe_337, &nobj, 
			   hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  if ((rcode = dst_packr8_(hbar_.sigma_qe_337, &nobj, 
			   hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_packr8_(hbar_.uv_exp, &nobj,
			   hbar_bank, &hbar_blen, &hbar_maxlen)))
    return rcode;

  return SUCCESS;
}


integer4 hbar_bank_to_dst_(integer4 *NumUnit)
{	
  return dst_write_bank_(NumUnit, &hbar_blen, hbar_bank );
}

integer4 hbar_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = hbar_common_to_bank_()))
    {
      fprintf (stderr,"hbar_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }             
  if ((rcode = hbar_bank_to_dst_(NumUnit)))
    {
      fprintf (stderr,"hbar_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }
  return SUCCESS;
}

integer4 hbar_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  
  hbar_blen = 2 * sizeof(integer4); /* skip id and version  */

  nobj = 1;
  if ((rcode = dst_unpacki4_(&hbar_.jday, &nobj, bank, 
			     &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_unpacki4_(&hbar_.jsec, &nobj, bank, 
			     &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_unpacki4_(&hbar_.msec, &nobj, bank,
			     &hbar_blen, &hbar_maxlen)))
    return rcode;
  

  /* These need to be unpacked early on.. */

  if ((rcode = dst_unpacki4_(&hbar_.nmir, &nobj, bank,
			     &hbar_blen, &hbar_maxlen)))
    return rcode;
  if ((rcode = dst_unpacki4_(&hbar_.ntube, &nobj, bank,
			     &hbar_blen, &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_unpacki4_(&hbar_.source, &nobj, bank,
			     &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  /* Now we unpack mirror-wise arrays.. */

  nobj = hbar_.nmir;

  if ((rcode = dst_unpackr8_(hbar_.hnpe_jday, 
			     &nobj, bank,
			     &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_unpacki4_(hbar_.mir, &nobj, bank,
			     &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_unpackr8_(hbar_.mir_reflect, &nobj,
			     bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  

  /* Now we unpack tube-wise arrays.. */

  nobj = hbar_.ntube;

  if ((rcode = dst_unpacki4_(hbar_.tubemir, &nobj, 
			     bank, &hbar_blen, &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_unpacki4_(hbar_.tube, &nobj, 
			     bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_unpacki4_(hbar_.qdcb, &nobj, 
			     bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_unpackr8_(hbar_.npe, &nobj, bank,
			     &hbar_blen, &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_unpackr8_(hbar_.sigma_npe, &nobj,
			     bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_unpacki1_((integer1 *)hbar_.first_order_gain_flag, 
			     &nobj,bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  
  
  if ((rcode = dst_unpackr8_(hbar_.second_order_gain,
			     &nobj, bank, &hbar_blen,
			     &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_unpackr8_(hbar_.second_order_gain_sigma, 
			     &nobj, bank, &hbar_blen,
			     &hbar_maxlen)))
    return rcode;
  
  if ((rcode = dst_unpacki1_((integer1 *)hbar_.second_order_gain_flag, 
			     &nobj,bank, &hbar_blen, &hbar_maxlen)))
    return rcode;


  if ((rcode = dst_unpackr8_(hbar_.qe_337, &nobj, 
			     bank, &hbar_blen, &hbar_maxlen)))
    return rcode;
  if ((rcode = dst_unpackr8_(hbar_.sigma_qe_337, &nobj, 
			     bank, &hbar_blen, &hbar_maxlen)))
    return rcode;

  if ((rcode = dst_unpackr8_(hbar_.uv_exp, &nobj,
			     bank, &hbar_blen, &hbar_maxlen)))
    return rcode;

  return SUCCESS;
}

integer4 hbar_common_to_dump_(integer4 *long_output)
{
  return hbar_common_to_dumpf_(stdout, long_output);
}

integer4 hbar_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i, j;
  char *sourceName[HBAR_MAXSRC] = { "RXF", "YAG" };

  
  fprintf(fp, "\nHBAR bank \n\n");

  fprintf(fp, "calibration source: %s\n", sourceName[hbar_.source]);
  
  fprintf(fp, "jDay/Sec: %d/%5.5d %02d:%02d:%02d.%03d\n", hbar_.jday,
	  hbar_.jsec, hbar_.msec / 3600000, (hbar_.msec / 60000) % 60,
	  (hbar_.msec / 1000) % 60, hbar_.msec % 1000) ;

  fprintf(fp, "Number of mirrors: %2d  tubes: %4d \n\n", 
	  hbar_.nmir, hbar_.ntube);
  
  if ( *long_output == 1 ) {
  
    for (i = 0; i < hbar_.nmir; i++)
      {
	fprintf(fp,"mirror: %d\n", hbar_.mir[i]);
	fprintf(fp,"mirror reflectivity: %g\n", hbar_.mir_reflect[i]);
	fprintf(fp,"jday of HNPE bank used: %f\n", hbar_.hnpe_jday[i]);
      }
    
    fprintf(fp, "%2s %3s %5s %6s %6s %8s %6s %6s %8s %4s %4s\n", 
	    "MR", "TUB", "QDCB", "NPE", "SIGNPE", "FLAG1",
	    "GAIN2", "SIGGN2", "FLAG2", "QE", "UV");
    fprintf(fp, "---------------------------------------------------" 
	    "-----------------\n");
	


    for(j = 0; j < hbar_.ntube; j++) 
      {
	fprintf(fp,"%2d %3d %5d %6.1f %6.1f "
		"%c%c%c%c%c%c%c%c "
		"%6.1f %6.3f %c%c%c%c%c%c%c%c %.2f %.2f\n",
		
		hbar_.tubemir[j], hbar_.tube[j], 
		hbar_.qdcb[j], hbar_.npe[j], hbar_.sigma_npe[j],
		((hbar_.first_order_gain_flag[j] & HBARBIT(7)) != 0)+'0',
		((hbar_.first_order_gain_flag[j] & HBARBIT(6)) != 0)+'0',
		((hbar_.first_order_gain_flag[j] & HBARBIT(5)) != 0)+'0',
		((hbar_.first_order_gain_flag[j] & HBARBIT(4)) != 0)+'0',
		((hbar_.first_order_gain_flag[j] & HBARBIT(3)) != 0)+'0',
		((hbar_.first_order_gain_flag[j] & HBARBIT(2)) != 0)+'0',
		((hbar_.first_order_gain_flag[j] & HBARBIT(1)) != 0)+'0',
		((hbar_.first_order_gain_flag[j] & HBARBIT(0)) != 0)+'0',
		hbar_.second_order_gain[j], 
		hbar_.second_order_gain_sigma[j],
		((hbar_.second_order_gain_flag[j] & HBARBIT(7)) != 0)+'0',
		((hbar_.second_order_gain_flag[j] & HBARBIT(6)) != 0)+'0',
		((hbar_.second_order_gain_flag[j] & HBARBIT(5)) != 0)+'0',
		((hbar_.second_order_gain_flag[j] & HBARBIT(4)) != 0)+'0',
		((hbar_.second_order_gain_flag[j] & HBARBIT(3)) != 0)+'0',
		((hbar_.second_order_gain_flag[j] & HBARBIT(2)) != 0)+'0',
		((hbar_.second_order_gain_flag[j] & HBARBIT(1)) != 0)+'0',
		((hbar_.second_order_gain_flag[j] & HBARBIT(0)) != 0)+'0',
	        hbar_.qe_337[j], hbar_.uv_exp[j]);
	    
      }
  }

  return SUCCESS;
  
}






