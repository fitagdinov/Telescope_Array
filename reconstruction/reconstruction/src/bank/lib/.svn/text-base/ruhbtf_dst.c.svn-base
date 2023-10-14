/*
 * C functions for ruhbtf
 * Dmitri Ivanov, ivanov@physics.rutgers.edu
 * Sep 16, 2009
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "ruhbtf_dst.h"

ruhbtf_dst_common ruhbtf_;	/* allocate memory to ruhbtf_common */

static integer4 ruhbtf_blen = 0;
static integer4 ruhbtf_maxlen =
  sizeof (integer4) * 2 + sizeof (ruhbtf_dst_common);
static integer1 *ruhbtf_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* ruhbtf_bank_buffer_ (integer4* ruhbtf_bank_buffer_size)
{
  (*ruhbtf_bank_buffer_size) = ruhbtf_blen;
  return ruhbtf_bank;
}



static void
ruhbtf_bank_init ()
{
  ruhbtf_bank = (integer1 *) calloc (ruhbtf_maxlen, sizeof (integer1));
  if (ruhbtf_bank == NULL)
    {
      fprintf (stderr,
	       "ruhbtf_bank_init: fail to assign memory to bank. Abort.\n");
      exit (0);
    }				/* else fprintf ( stderr,"ruhbtf_bank allocated memory %d\n",ruhbtf_maxlen); */
}

integer4
ruhbtf_common_to_bank_()
{
  static integer4 id = RUHBTF_BANKID, ver = RUHBTF_BANKVERSION;
  integer4 rcode, nobj, i;

  if (ruhbtf_bank == NULL)
    ruhbtf_bank_init ();

  rcode =
    dst_initbank_ (&id, &ver, &ruhbtf_blen, &ruhbtf_maxlen, ruhbtf_bank);
  /* Initialize test_blen, and pack the id and version to bank */
  

  nobj = 1;
  
  rcode +=
    dst_packi4_ (&ruhbtf_.ntb, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packi4_ (&ruhbtf_.nsd, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  nobj=3;
  
  for (i=0; i < ruhbtf_.ntb; i++)
    {
      rcode +=
	dst_packr8_ (&ruhbtf_.tbuv_fd[i][0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		     &ruhbtf_maxlen);
    }
  for (i=0; i<ruhbtf_.nsd; i++)
    {
      rcode +=
	dst_packr8_ (&ruhbtf_.sdpos_clf[i][0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		     &ruhbtf_maxlen);
      rcode +=
	dst_packr8_ (&ruhbtf_.sdpos_fd[i][0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		     &ruhbtf_maxlen);
      rcode +=
	dst_packr8_ (&ruhbtf_.sd_xyzcdist[i][0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		     &ruhbtf_maxlen);
      rcode +=
	dst_packr8_ (&ruhbtf_.sdsa_pos_clf[i][0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		     &ruhbtf_maxlen);
      rcode +=
	dst_packr8_ (&ruhbtf_.sdsa_pos_fd[i][0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		     &ruhbtf_maxlen);
    }
  
  nobj=ruhbtf_.ntb;
  
  rcode +=
    dst_packr8_ (&ruhbtf_.tbnpe[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.tbtime[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.tbtime_rms[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.tbtime_err[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.tbsigma[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.tb_palt[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.tb_pazm[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.tb_tvsa_texp[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  
  nobj = ruhbtf_.nsd;
  
  rcode +=
    dst_packr8_ (&ruhbtf_.sdrho[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sdtime[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sdetime[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sd_cdist[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sd_adist[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sd_sdist[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sd_ltd[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sd_lts[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sdsa_tm[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sd_timerr[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sdsa_fddist[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sdsa_fdtime[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sd_tvsa_texp[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sdsa_palt[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sdsa_pazm[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  
  nobj = 3;

  rcode +=
    dst_packr8_ (&ruhbtf_.axi_clf[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.axi_fd[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.core_fd[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.core_fduv[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sdp_n[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.sd_cog_clf[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.fd_cogcore[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);

  nobj = 2;
  
  rcode +=
    dst_packr8_ (&ruhbtf_.r_sd[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  
  nobj = 1;
  
  rcode +=
    dst_packr8_ (&ruhbtf_.theta, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.dtheta, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.phi, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.dphi, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.xcore, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.dxcore, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.ycore, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.dycore, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.t0, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.dt0, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.dt0, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.cdist_fd, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.fd_cogcore_dist, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.psi, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.rp, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.t_rp, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.chi2, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packr8_ (&ruhbtf_.tref, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  
  nobj = ruhbtf_.ntb;
  
  rcode +=
    dst_packi4_ (&ruhbtf_.ifdplane[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);

  nobj = ruhbtf_.nsd;
  
  rcode +=
    dst_packi4_ (&ruhbtf_.irusdgeom[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packi4_ (&ruhbtf_.xxyy[0], &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  
  nobj = 1;
  
  rcode +=
    dst_packi4_ (&ruhbtf_.yymmdd, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packi4_ (&ruhbtf_.fdsiteid, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  rcode +=
    dst_packi4_ (&ruhbtf_.sdsiteid, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  // ntb, nsd were packed / unpacked at the very beginning of the function,
  // otherwise wouldn't be able to unpack the arrays above
  rcode +=
    dst_packi4_ (&ruhbtf_.ndof, &nobj, ruhbtf_bank, &ruhbtf_blen,
		 &ruhbtf_maxlen);
  
  return rcode;
}

integer4
ruhbtf_bank_to_dst_ (integer4 * unit)
{
  integer4 rcode;
  rcode = dst_write_bank_ (unit, &ruhbtf_blen, ruhbtf_bank);
  free (ruhbtf_bank);
  ruhbtf_bank = NULL;
  return rcode;
}

integer4
ruhbtf_common_to_dst_ (integer4 * unit)
{
  integer4 rcode;
  if ( (rcode = ruhbtf_common_to_bank_()) )
    {
      fprintf (stderr, "ruhbtf_common_to_bank_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  if ( (rcode = ruhbtf_bank_to_dst_(unit) ))
    {
      fprintf (stderr, "ruhbtf_bank_to_dst_ ERROR : %ld\n", (long) rcode);
      exit (0);
    }
  return 0;
}

integer4
ruhbtf_bank_to_common_ (integer1 * bank)
{
  integer4 rcode = 0;
  integer4 nobj, i;
  ruhbtf_blen = 2 * sizeof (integer4);	/* skip id and version  */
  
  nobj = 1;
  
  rcode +=
    dst_unpacki4_ (&ruhbtf_.ntb, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpacki4_ (&ruhbtf_.nsd, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  nobj=3;
  for (i=0; i < ruhbtf_.ntb; i++)
    {
      rcode +=
	dst_unpackr8_ (&ruhbtf_.tbuv_fd[i][0], &nobj, bank, &ruhbtf_blen,
		       &ruhbtf_maxlen);
    }
  for (i=0; i<ruhbtf_.nsd; i++)
    {
      rcode +=
	dst_unpackr8_ (&ruhbtf_.sdpos_clf[i][0], &nobj, bank, &ruhbtf_blen,
		       &ruhbtf_maxlen);
      rcode +=
	dst_unpackr8_ (&ruhbtf_.sdpos_fd[i][0], &nobj, bank, &ruhbtf_blen,
		       &ruhbtf_maxlen);
      rcode +=
	dst_unpackr8_ (&ruhbtf_.sd_xyzcdist[i][0], &nobj, bank, &ruhbtf_blen,
		       &ruhbtf_maxlen);
      rcode +=
	dst_unpackr8_ (&ruhbtf_.sdsa_pos_clf[i][0], &nobj, bank, &ruhbtf_blen,
		       &ruhbtf_maxlen);
      rcode +=
	dst_unpackr8_ (&ruhbtf_.sdsa_pos_fd[i][0], &nobj, bank, &ruhbtf_blen,
		       &ruhbtf_maxlen);
    }
  
  nobj=ruhbtf_.ntb;
  
  rcode +=
    dst_unpackr8_ (&ruhbtf_.tbnpe[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.tbtime[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.tbtime_rms[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.tbtime_err[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.tbsigma[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.tb_palt[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.tb_pazm[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.tb_tvsa_texp[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  
  nobj = ruhbtf_.nsd;
  
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sdrho[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sdtime[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sdetime[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sd_cdist[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sd_adist[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sd_sdist[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sd_ltd[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sd_lts[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sdsa_tm[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sd_timerr[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sdsa_fddist[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sdsa_fdtime[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sd_tvsa_texp[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sdsa_palt[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sdsa_pazm[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  
  nobj = 3;
  
  rcode +=
    dst_unpackr8_ (&ruhbtf_.axi_clf[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.axi_fd[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.core_fd[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.core_fduv[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sdp_n[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.sd_cog_clf[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.fd_cogcore[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);

  nobj = 2;
  
  rcode +=
    dst_unpackr8_ (&ruhbtf_.r_sd[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  
  nobj = 1;
  
  rcode +=
    dst_unpackr8_ (&ruhbtf_.theta, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.dtheta, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.phi, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.dphi, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.xcore, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.dxcore, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.ycore, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.dycore, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.t0, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.dt0, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.dt0, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.cdist_fd, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.fd_cogcore_dist, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.psi, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.rp, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.t_rp, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.chi2, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpackr8_ (&ruhbtf_.tref, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);

  nobj = ruhbtf_.ntb;
  
  rcode +=
    dst_unpacki4_ (&ruhbtf_.ifdplane[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);

  nobj = ruhbtf_.nsd;
  
  rcode +=
    dst_unpacki4_ (&ruhbtf_.irusdgeom[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpacki4_ (&ruhbtf_.xxyy[0], &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  
  nobj = 1;
  
  rcode +=
    dst_unpacki4_ (&ruhbtf_.yymmdd, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpacki4_ (&ruhbtf_.fdsiteid, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  rcode +=
    dst_unpacki4_ (&ruhbtf_.sdsiteid, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  // ntb, nsd were packed / unpacked at the very beginning of the function,
  // otherwise wouldn't be able to unpack the arrays above
  rcode +=
    dst_unpacki4_ (&ruhbtf_.ndof, &nobj, bank, &ruhbtf_blen,
		   &ruhbtf_maxlen);
  
  
  return rcode;
}

integer4
ruhbtf_common_to_dump_ (integer4 * long_output)
{
  return ruhbtf_common_to_dumpf_ (stdout, long_output);
}

integer4
ruhbtf_common_to_dumpf_ (FILE * fp, integer4 * long_output)
{  
  fprintf (fp, "ruhbtf: theta = %.2f phi = %.2f xcore = %.4e ycore = %.4e rp = %.4e psi = %.2f ",
	   ruhbtf_.theta,ruhbtf_.phi,ruhbtf_.xcore,ruhbtf_.ycore,ruhbtf_.rp,ruhbtf_.psi);
  fprintf (fp, "chi2 = %.2f ndof = %d\n",ruhbtf_.chi2,ruhbtf_.ndof);
  if ((*long_output) == 1)
    {
    }
  return 0;
}
