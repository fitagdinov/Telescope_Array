/*
 * hnpe_dst.c 
 *
 * $Source: /hires_soft/cvsroot/bank/hnpe_dst.c,v $
 * $Log: hnpe_dst.c,v $
 * Revision 1.4  2000/03/21 21:16:13  ben
 * Minor change in dstdump format.
 *
 * Revision 1.2  1999/07/26  23:48:52  smoore
 * added sigma qe 337 to dstdump output
 *
 * Revision 1.1  1999/06/15  05:19:37  ben
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
#include "hnpe_dst.h"  

hnpe_dst_common hnpe_;  /* allocate memory to hnpe_common */

static integer4 hnpe_blen = 0; 
static integer4 hnpe_maxlen = sizeof(integer4) * 2 + sizeof(hnpe_dst_common);
static integer1 *hnpe_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hnpe_bank_buffer_ (integer4* hnpe_bank_buffer_size)
{
  (*hnpe_bank_buffer_size) = hnpe_blen;
  return hnpe_bank;
}



static void hnpe_bank_init(void)
{
  hnpe_bank = (integer1 *)calloc(hnpe_maxlen, sizeof(integer1));
  if (hnpe_bank==NULL)
    {
      fprintf(stderr, 
	      "hnpe_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
    }
}

integer4 hnpe_common_to_bank_(void)
{	
  static integer4 id = HNPE_BANKID, ver = HNPE_BANKVERSION;
  integer4 rcode, nobj;

  if (hnpe_bank == NULL) hnpe_bank_init();

  /* Initialize hnpe_blen, and pack the id and version to bank */
  if ((rcode = dst_initbank_(&id, &ver, &hnpe_blen, &hnpe_maxlen, hnpe_bank)))
    return rcode;

  if ((rcode = dst_packi4_(&hnpe_.mirror, (nobj=1, &nobj), hnpe_bank, 
			   &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_packr8_(&hnpe_.start_date, (nobj=1, &nobj), hnpe_bank, 
			   &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packr8_(&hnpe_.end_date, (nobj=1, &nobj), hnpe_bank, 
			   &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_packr8_(hnpe_.uv_exp, (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packi1_(hnpe_.uv_file_name, (nobj=HNPE_FNC, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_packr8_(hnpe_.mean_qe_337, (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packr8_(hnpe_.sigma_qe_337, (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packr8_((real8 *)hnpe_.qe_337, (nobj=HR_UNIV_MIRTUBE*HNPE_MAX_SRC, 
						   &nobj), hnpe_bank, &hnpe_blen, 
			   &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packi1_(hnpe_.qe_file_name, (nobj=HNPE_FNC, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_packi4_(&hnpe_.number_src, (nobj=1, &nobj), hnpe_bank, 
			   &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packi1_((integer1 *)hnpe_.source_desc, 
			   (nobj=HNPE_MAX_SRC*HNPE_DC, &nobj), hnpe_bank, 
			   &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packi1_((integer1 *)hnpe_.source_file_name, 
			   (nobj=HNPE_MAX_SRC*HNPE_FNC, &nobj), hnpe_bank, 
			   &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_packr8_((real8 *)hnpe_.mean_qdcb, 
			   (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packr8_((real8 *)hnpe_.sigma_qdcb, 
			   (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4_((integer4 *)hnpe_.valid_src_flag, 
			   (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_packr8_((real8 *)hnpe_.mean_area, 
			   (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packr8_((real8 *)hnpe_.sigma_area, 
			   (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_packr8_((real8 *)hnpe_.mean_npe, 
			   (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packr8_((real8 *)hnpe_.sigma_npe, 
			   (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_packr8_(hnpe_.first_order_gain, 
			   (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packi1_((integer1 *)hnpe_.first_order_gain_flag, 
			   (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packr8_(hnpe_.first_order_fit_goodness, 
			   (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_packr8_(hnpe_.second_order_gain, 
			   (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packi1_((integer1 *)hnpe_.second_order_gain_flag, 
			   (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packi4_(hnpe_.ecalib_flag, 
			   (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_packr8_(hnpe_.second_order_fit_goodness, 
			   (nobj=HR_UNIV_MIRTUBE, &nobj), 
			   hnpe_bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  

  return SUCCESS;
}


integer4 hnpe_bank_to_dst_(integer4 *NumUnit)
{	
  return dst_write_bank_(NumUnit, &hnpe_blen, hnpe_bank );
}

integer4 hnpe_common_to_dst_(integer4 *NumUnit)
{
  integer4 rcode;
  if ((rcode = hnpe_common_to_bank_()))
    {
      fprintf (stderr,"hnpe_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }             
  if ((rcode = hnpe_bank_to_dst_(NumUnit)))
    {
      fprintf (stderr,"hnpe_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit(0);			 	
    }
  return SUCCESS;
}

integer4 hnpe_bank_to_common_(integer1 *bank)
{
  integer4 rcode = 0;
  integer4 nobj;

  hnpe_blen = 2 * sizeof(integer4); /* skip id and version  */


  if ((rcode = dst_unpacki4_(&hnpe_.mirror, (nobj=1, &nobj), bank, 
			     &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_unpackr8_(&hnpe_.start_date, (nobj=1, &nobj), bank, 
			     &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_(&hnpe_.end_date, (nobj=1, &nobj), bank, 
			     &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_unpackr8_(hnpe_.uv_exp, (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki1_(hnpe_.uv_file_name, (nobj=HNPE_FNC, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_unpackr8_(hnpe_.mean_qe_337, (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_(hnpe_.sigma_qe_337, (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_((real8 *)hnpe_.qe_337, (nobj=HR_UNIV_MIRTUBE * HNPE_MAX_SRC,
						     &nobj), bank, &hnpe_blen, 
			     &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki1_(hnpe_.qe_file_name, (nobj=HNPE_FNC, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_unpacki4_(&hnpe_.number_src, (nobj=1, &nobj), bank, 
			     &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki1_((integer1 *)hnpe_.source_desc, 
			     (nobj=HNPE_MAX_SRC*HNPE_DC, &nobj), bank, 
			     &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki1_((integer1 *)hnpe_.source_file_name, 
			     (nobj=HNPE_MAX_SRC*HNPE_FNC, &nobj), bank, 
			     &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_unpackr8_((real8 *)hnpe_.mean_qdcb, 
			     (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_((real8 *)hnpe_.sigma_qdcb, 
			     (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_((integer4 *)hnpe_.valid_src_flag, 
			     (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_unpackr8_((real8 *)hnpe_.mean_area, 
			     (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_((real8 *)hnpe_.sigma_area, 
			     (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_unpackr8_((real8 *)hnpe_.mean_npe, 
			     (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_((real8 *)hnpe_.sigma_npe, 
			     (nobj=HNPE_MAX_SRC*HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_unpackr8_(hnpe_.first_order_gain, 
			     (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki1_((integer1 *)hnpe_.first_order_gain_flag, 
			     (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_(hnpe_.first_order_fit_goodness, 
			     (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  if ((rcode = dst_unpackr8_(hnpe_.second_order_gain, 
			     (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki1_((integer1 *)hnpe_.second_order_gain_flag, 
			     (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_(hnpe_.ecalib_flag, 
			     (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;
  if ((rcode = dst_unpackr8_(hnpe_.second_order_fit_goodness, 
			     (nobj=HR_UNIV_MIRTUBE, &nobj), 
			     bank, &hnpe_blen, &hnpe_maxlen))) return rcode;

  return SUCCESS;
}

integer4 hnpe_common_to_dump_(integer4 *long_output)
{
  return hnpe_common_to_dumpf_(stdout, long_output);
}

integer4 hnpe_common_to_dumpf_(FILE* fp, integer4 *long_output)
{
  integer4 i, tube;

  fprintf(fp, "\nHNPE bank, mirror %d. \n\n", hnpe_.mirror);

  fprintf(fp, "Calibration is valid from %lf through %lf\n", hnpe_.start_date,
	  hnpe_.end_date);

  fprintf(fp, "UV filter curve used: %s\n", hnpe_.uv_file_name);
  fprintf(fp, "QE curve used: %s\n", hnpe_.qe_file_name);

  fprintf(fp, "\nSources:\n");
  fprintf(fp, "--------\n");
  for (i = 0; i < hnpe_.number_src; i++)
    fprintf(fp, "%s\n", hnpe_.source_desc[i]);


  if ( *long_output == 1 ) {

    fprintf(fp, "\nConstants:\n----------\n");
    for (tube = 0; tube < HR_UNIV_MIRTUBE; tube++)
      fprintf(fp, "Tube %3d: UV K value = %lf, QE at 337 nm. = %lf +/- %lf\n",
	      tube, hnpe_.uv_exp[tube], hnpe_.mean_qe_337[tube],
	      hnpe_.sigma_qe_337[tube]);

    fprintf(fp, "\nSource Details:\n--------\n");
    for (tube = 0; tube < HR_UNIV_MIRTUBE; tube++) {
      fprintf(fp,"Tube %3d: gain1 = %6.3g, gain1_flag = %c%c%c%c%c%c%c%c"
	      ", gain1_good = %6.3g\n",
	      tube+1, hnpe_.first_order_gain[tube], 
	      ((hnpe_.first_order_gain_flag[tube] & HNPE_BIT(7))!=0) + '0',
	      ((hnpe_.first_order_gain_flag[tube] & HNPE_BIT(6))!=0) + '0',
	      ((hnpe_.first_order_gain_flag[tube] & HNPE_BIT(5))!=0) + '0',
	      ((hnpe_.first_order_gain_flag[tube] & HNPE_BIT(4))!=0) + '0',
	      ((hnpe_.first_order_gain_flag[tube] & HNPE_BIT(3))!=0) + '0',
	      ((hnpe_.first_order_gain_flag[tube] & HNPE_BIT(2))!=0) + '0',
	      ((hnpe_.first_order_gain_flag[tube] & HNPE_BIT(1))!=0) + '0',
	      ((hnpe_.first_order_gain_flag[tube] & HNPE_BIT(0))!=0) + '0',
	      hnpe_.first_order_fit_goodness[tube]);
      fprintf(fp,"          gain2 = %6.3g, gain2_flag = %c%c%c%c%c%c%c%c"
	      ",%3d, gain2_good = %6.3g\n",
	      hnpe_.second_order_gain[tube], 
	      ((hnpe_.second_order_gain_flag[tube] & HNPE_BIT(7))!=0) + '0',
	      ((hnpe_.second_order_gain_flag[tube] & HNPE_BIT(6))!=0) + '0',
	      ((hnpe_.second_order_gain_flag[tube] & HNPE_BIT(5))!=0) + '0',
	      ((hnpe_.second_order_gain_flag[tube] & HNPE_BIT(4))!=0) + '0',
	      ((hnpe_.second_order_gain_flag[tube] & HNPE_BIT(3))!=0) + '0',
	      ((hnpe_.second_order_gain_flag[tube] & HNPE_BIT(2))!=0) + '0',
	      ((hnpe_.second_order_gain_flag[tube] & HNPE_BIT(1))!=0) + '0',
	      ((hnpe_.second_order_gain_flag[tube] & HNPE_BIT(0))!=0) + '0',
	      hnpe_.ecalib_flag[tube],
	      hnpe_.second_order_fit_goodness[tube]);

      for (i = 0; i < hnpe_.number_src; i++) {
	fprintf(fp,"\tSource %2d: QDCB = %7.5g, SQDCB = %5.3g, %s\n",
		i, hnpe_.mean_qdcb[i][tube], hnpe_.sigma_qdcb[i][tube],
		(hnpe_.valid_src_flag[i][tube]==1?"valid":"not valid"));

	fprintf(fp,"\t           AREA = %5.3f, SAREA = %5.3f\n",
		hnpe_.mean_area[i][tube], hnpe_.sigma_area[i][tube]);

	fprintf(fp,"\t            NPE = %5.3g, SNPE  = %5.3g\n",
		hnpe_.mean_npe[i][tube], hnpe_.sigma_npe[i][tube]);
      }
    }
  }

  return SUCCESS;
}

