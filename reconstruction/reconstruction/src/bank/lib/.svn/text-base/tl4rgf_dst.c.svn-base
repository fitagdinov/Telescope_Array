/*
 * C functions for tl4rgf
 * Dmitri Ivanov, dmiivanov@gmail.com
 * Dec 07, 2016
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "tl4rgf_dst.h"



tl4rgf_dst_common tl4rgf_;	/* allocate memory to tl4rgf_common */

static integer4 tl4rgf_blen = 0;
static integer4 tl4rgf_maxlen =
  sizeof (integer4) * 2 + sizeof (tl4rgf_dst_common);
static integer1 *tl4rgf_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* tl4rgf_bank_buffer_ (integer4* tl4rgf_bank_buffer_size)
{
  (*tl4rgf_bank_buffer_size) = tl4rgf_blen;
  return tl4rgf_bank;
}



static void
tl4rgf_bank_init ()
{
  tl4rgf_bank = (integer1 *) calloc (tl4rgf_maxlen, sizeof (integer1));
  if (tl4rgf_bank == NULL)
    {
      fprintf (stderr,
	       "tl4rgf_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }				/* else fprintf ( stderr,"tl4rgf_bank allocated memory %d\n",tl4rgf_maxlen); */
}

integer4
tl4rgf_common_to_bank_ ()
{
  static integer4 id = TL4RGF_BANKID, ver = TL4RGF_BANKVERSION;
  integer4 rcode, nobj, i;

  if (tl4rgf_bank == NULL)
    tl4rgf_bank_init ();

  rcode =
    dst_initbank_ (&id, &ver, &tl4rgf_blen, &tl4rgf_maxlen, tl4rgf_bank);
  /* Initialize test_blen, and pack the id and version to bank */

  nobj = 1;
  rcode += dst_packi4_ (&tl4rgf_.yymmdd, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen); 
  rcode += dst_packi4_ (&tl4rgf_.hhmmss, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr8_ (&tl4rgf_.t0, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.theta, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.phi, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj = 2;
  rcode += dst_packr4_ (&tl4rgf_.xycore[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj = 1;
  rcode += dst_packr4_ (&tl4rgf_.bdist, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packi1_ (&tl4rgf_.nfdsite, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj = tl4rgf_.nfdsite;
  rcode += dst_packi1_ (&tl4rgf_.site_id[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packi1_ (&tl4rgf_.is_frame[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  for (i=0; i<tl4rgf_.nfdsite; i++)
    {
      nobj=3;
      rcode += dst_packr4_ (&tl4rgf_.sdp_n[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
    }
  nobj = tl4rgf_.nfdsite;
  rcode += dst_packr4_ (&tl4rgf_.sdp_n_chi2[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packi4_ (&tl4rgf_.ngt_sdp_n[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.psi[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.rp[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.trp[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.tf_chi2[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packi4_ (&tl4rgf_.ngt_tf[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.tracklength[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.crossingtime[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.npe[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.npe_edge[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj=1;
  rcode += dst_packr4_ (&tl4rgf_.d_psi, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.d_rp, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.d_trp, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj=tl4rgf_.nfdsite;
  rcode += dst_packi4_ (&tl4rgf_.ntube[0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  for (i=0; i<tl4rgf_.nfdsite; i++)
    {
      nobj = tl4rgf_.ntube[i];
      rcode += dst_packi4_ (&tl4rgf_.tube_raw_ind[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packi4_ (&tl4rgf_.tube_fraw1_ind[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packi4_ (&tl4rgf_.tube_stpln_ind[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packi1_ (&tl4rgf_.tube_ig_sdp_n[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packi1_ (&tl4rgf_.tube_ig_tf[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
#if TL4RGF_BANKVERSION >= 1
      rcode += dst_packi1_ (&tl4rgf_.tube_sat_flag[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
#endif
      rcode += dst_packi1_ (&tl4rgf_.tube_edge[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packi4_ (&tl4rgf_.tube_mir_and_tube_id[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_azm[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_ele[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_npe[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_gfc[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_sig2noise[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_ped_or_thb[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_tm[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_tslew[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_tmerr[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_tmfit[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_palt[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_packr4_ (&tl4rgf_.tube_pazm[i][0], &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
    }
  nobj = 1;
  rcode += dst_packi4_ (&tl4rgf_.sd_xxyy, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packi4_ (&tl4rgf_.sd_ind, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.sd_vem, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.sd_tm, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.sd_tmerr, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_packr4_ (&tl4rgf_.sd_tmfit, &nobj, tl4rgf_bank, &tl4rgf_blen, &tl4rgf_maxlen);
  return rcode;
}

integer4
tl4rgf_bank_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  rcode = dst_write_bank_ (NumUnit, &tl4rgf_blen, tl4rgf_bank);
  free (tl4rgf_bank);
  tl4rgf_bank = NULL;
  return rcode;
}

integer4
tl4rgf_common_to_dst_ (integer4 * NumUnit)
{
  integer4 rcode;
  if ((rcode = tl4rgf_common_to_bank_ ()))
    {
      fprintf (stderr, "tl4rgf_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ((rcode = tl4rgf_bank_to_dst_ (NumUnit)))
    {
      fprintf (stderr, "tl4rgf_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4
tl4rgf_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj, i, j;
  integer4 bankid, bankversion;
  
  tl4rgf_blen = 0;
  
  if ((rcode = dst_unpacki4_( &bankid,      (nobj=1, &nobj), bank, &tl4rgf_blen, &tl4rgf_maxlen))) return rcode;
  if ((rcode = dst_unpacki4_( &bankversion, (nobj=1, &nobj), bank, &tl4rgf_blen, &tl4rgf_maxlen))) return rcode;
  
  nobj = 1;
  rcode += dst_unpacki4_ (&tl4rgf_.yymmdd, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen); 
  rcode += dst_unpacki4_ (&tl4rgf_.hhmmss, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr8_ (&tl4rgf_.t0, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.theta, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.phi, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj = 2;
  rcode += dst_unpackr4_ (&tl4rgf_.xycore[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj = 1;
  rcode += dst_unpackr4_ (&tl4rgf_.bdist, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpacki1_ (&tl4rgf_.nfdsite, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj = tl4rgf_.nfdsite;
  rcode += dst_unpacki1_ (&tl4rgf_.site_id[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpacki1_ (&tl4rgf_.is_frame[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  for (i=0; i<tl4rgf_.nfdsite; i++)
    {
      nobj=3;
      rcode += dst_unpackr4_ (&tl4rgf_.sdp_n[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
    }
  nobj = tl4rgf_.nfdsite;
  rcode += dst_unpackr4_ (&tl4rgf_.sdp_n_chi2[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpacki4_ (&tl4rgf_.ngt_sdp_n[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.psi[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.rp[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.trp[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.tf_chi2[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpacki4_ (&tl4rgf_.ngt_tf[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.tracklength[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.crossingtime[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.npe[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.npe_edge[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj=1;
  rcode += dst_unpackr4_ (&tl4rgf_.d_psi, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.d_rp, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.d_trp, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  nobj=tl4rgf_.nfdsite;
  rcode += dst_unpacki4_ (&tl4rgf_.ntube[0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  for (i=0; i<tl4rgf_.nfdsite; i++)
    {
      nobj = tl4rgf_.ntube[i];
      rcode += dst_unpacki4_ (&tl4rgf_.tube_raw_ind[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpacki4_ (&tl4rgf_.tube_fraw1_ind[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpacki4_ (&tl4rgf_.tube_stpln_ind[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpacki1_ (&tl4rgf_.tube_ig_sdp_n[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpacki1_ (&tl4rgf_.tube_ig_tf[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      if(bankversion >= 1)
	rcode += dst_unpacki1_ (&tl4rgf_.tube_sat_flag[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      else
	{
	  for (j=0; j<tl4rgf_.ntube[i]; j++)
	    tl4rgf_.tube_sat_flag[i][j] = 0;
	}
      rcode += dst_unpacki1_ (&tl4rgf_.tube_edge[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpacki4_ (&tl4rgf_.tube_mir_and_tube_id[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_azm[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_ele[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_npe[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_gfc[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_sig2noise[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_ped_or_thb[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_tm[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_tslew[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_tmerr[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_tmfit[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_palt[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
      rcode += dst_unpackr4_ (&tl4rgf_.tube_pazm[i][0], &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
    }
  nobj = 1;
  rcode += dst_unpacki4_ (&tl4rgf_.sd_xxyy, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpacki4_ (&tl4rgf_.sd_ind, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.sd_vem, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.sd_tm, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.sd_tmerr, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  rcode += dst_unpackr4_ (&tl4rgf_.sd_tmfit, &nobj, bank, &tl4rgf_blen, &tl4rgf_maxlen);
  return rcode;
}

integer4
tl4rgf_common_to_dump_ (integer4 * long_output)
{
  return tl4rgf_common_to_dumpf_ (stdout, long_output);
}

integer4
tl4rgf_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{
  integer4 i, j, isite_frame;
  fprintf(fp,"date=%06d time=%06d.%06d",
	  tl4rgf_.yymmdd,
	  tl4rgf_.hhmmss,
	  (integer4)floor(tl4rgf_.t0+0.5));
  fprintf(fp," sd_xxyy=%d",tl4rgf_.sd_xxyy);
  fprintf(fp," nfdsite=%d",tl4rgf_.nfdsite);
  fprintf(fp,"\n");
  if(tl4rgf_.nfdsite<1)
    {
      fprintf(fp,"-- no further information\n");
      return 0;
    }
  fprintf(fp,"theta=%.1f phi=%.1f xycore=%.6e,%.6e t0=%.2f bdist=%.6e\n",
	  tl4rgf_.theta,tl4rgf_.phi,tl4rgf_.xycore[0],tl4rgf_.xycore[1],
	  tl4rgf_.t0,tl4rgf_.bdist);
  isite_frame = 0;
  for (i=0; i < tl4rgf_.nfdsite; i++)
    {
      if(tl4rgf_.is_frame[i] == 1)
	isite_frame = i;
    }
  fprintf(fp,"fdframe=%d psi=%.1f+/-%.1f rp=%.6e+/-%.6e trp=%.2f+/-%.2f\n\n",
	  (integer4)tl4rgf_.site_id[isite_frame],
	  tl4rgf_.psi[isite_frame],tl4rgf_.d_psi,
	  tl4rgf_.rp[isite_frame],tl4rgf_.d_rp,
	  tl4rgf_.trp[isite_frame],tl4rgf_.d_trp);

  for (i=0; i< tl4rgf_.nfdsite; i++)
    {
      fprintf(fp,"site_id=%d is_frame=%d ntube=%d psi=%.1f rp=%.6e trp=%.2f\n",
	      (integer4)tl4rgf_.site_id[i],
	      (integer4)tl4rgf_.is_frame[i],
	      tl4rgf_.ntube[i],
	      tl4rgf_.psi[i],
	      tl4rgf_.rp[i],
	      tl4rgf_.trp[i]);
      fprintf(fp,"sdp_n_chi2,ngt_sdp_n=%.2f,%d->sdp_n_chi2_pdof=%.1f",
	      tl4rgf_.sdp_n_chi2[i],tl4rgf_.ngt_sdp_n[i],
	      tl4rgf_.sdp_n_chi2[i]/(real4)(tl4rgf_.ngt_sdp_n[i] > 0 ? tl4rgf_.ngt_sdp_n[i] : 1)
	      );
      fprintf(fp," tf_chi2,ngt_tf=%.2f,%d->tf_chi2_pdof=%.1f\n",
	      tl4rgf_.tf_chi2[i],tl4rgf_.ngt_tf[i],
	      tl4rgf_.tf_chi2[i]/(real4)(tl4rgf_.ngt_tf[i]  > 0 ? tl4rgf_.ngt_tf[i] : 1)
	      );
      fprintf(fp,"tracklength=%.1f crossingtime=%.1f npe=%.3e npe_edge=%.3e\n",
	      tl4rgf_.tracklength[i],tl4rgf_.crossingtime[i],
	      tl4rgf_.npe[i],tl4rgf_.npe_edge[i]);
      
      if(tl4rgf_.sd_xxyy)
	{
	  fprintf(fp,"\nsd_xxyy=%04d sd_ind=%d sd_vem=%.3e sd_tm=%.2f sd_tmerr=%.2f sd_tmfit=%.2f\n",
		  tl4rgf_.sd_xxyy,
		  tl4rgf_.sd_ind,
		  tl4rgf_.sd_vem,
		  tl4rgf_.sd_tm,
		  tl4rgf_.sd_tmerr,
		  tl4rgf_.sd_tmfit);
	}
    }	  
  if(*long_output == 1)
    {
      for (i=0; i< tl4rgf_.nfdsite; i++)
	{
	  fprintf(fp,"\nsite_id=%d ntube=%d tube information:\n",
		  tl4rgf_.site_id[i],tl4rgf_.ntube[i]);
	  fprintf(fp,"1:  tube_raw_ind,    2: tube_fraw1_ind,       3:  tube_stpln_ind,\n");
	  fprintf(fp,"4:  tube_ig_sdp_n,   5: tube_ig_tf,           6:  tube_sat_flag,\n"); 
	  fprintf(fp,"7:  tube_edge,       8: tube_mir_and_tube_id, 9:  tube_azm,\n");
	  fprintf(fp,"10: tube_ele,       11: tube_npe,             12: tube_gfc,\n");
	  fprintf(fp,"13: tube_sig2noise, 14: tube_ped_or_thb,      15: tube_tm,\n");
	  fprintf(fp,"16: tube_tslew,     17: tube_tmerr,           18: tube_tmfit,\n");
	  fprintf(fp,"19: tube_palt,      20: tube_pazm\n");
	  fprintf(fp,"%2d %5d %4d %3d %1d %1d %2d %3d %7d %8d ",1,2,3,4,5,6,7,8,9,10);
	  fprintf(fp,"%9d %9d %8d %7d %6d %5d %5d %5d %5d %4d",11,12,13,14,15,16,17,18,19,20);
	  fprintf(fp,"\n");
	  for (j=0; j<tl4rgf_.ntube[i]; j++)
	    {
	      fprintf(fp,"%04d",tl4rgf_.tube_raw_ind[i][j]);
	      fprintf(fp," %05d",tl4rgf_.tube_fraw1_ind[i][j]);
	      fprintf(fp," %04d",tl4rgf_.tube_stpln_ind[i][j]);
	      fprintf(fp," %d",(integer4)tl4rgf_.tube_ig_sdp_n[i][j]);
	      fprintf(fp," %d",(integer4)tl4rgf_.tube_ig_tf[i][j]);
	      fprintf(fp," %02d",(integer4)tl4rgf_.tube_sat_flag[i][j]);
	      fprintf(fp," %d",(integer4)tl4rgf_.tube_edge[i][j]);
	      fprintf(fp," %05d",tl4rgf_.tube_mir_and_tube_id[i][j]);
	      fprintf(fp," %3.2e",tl4rgf_.tube_azm[i][j]);
	      fprintf(fp," %3.2e",tl4rgf_.tube_ele[i][j]);
	      fprintf(fp," %5.3e",tl4rgf_.tube_npe[i][j]);
	      fprintf(fp," %5.3e",tl4rgf_.tube_gfc[i][j]);
	      fprintf(fp," %3.2e",tl4rgf_.tube_sig2noise[i][j]);
	      fprintf(fp," %6.2f",tl4rgf_.tube_ped_or_thb[i][j]);
	      fprintf(fp," %5.2f",tl4rgf_.tube_tm[i][j]);
	      fprintf(fp," %5.2f",tl4rgf_.tube_tslew[i][j]);
	      fprintf(fp," %5.2f",tl4rgf_.tube_tmerr[i][j]);
	      fprintf(fp," %5.2f",tl4rgf_.tube_tmfit[i][j]);
	      fprintf(fp," %5.2f",tl4rgf_.tube_palt[i][j]);
	      fprintf(fp," %3.2f",tl4rgf_.tube_pazm[i][j]);
	      fprintf(fp,"\n");
	    }
	}
    }
  
  return 0;
}
