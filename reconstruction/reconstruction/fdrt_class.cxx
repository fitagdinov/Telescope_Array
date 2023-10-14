#include "fdrt_class.h"
/**
   Root tree class for fdraw DST bank.
   Last modified: Oct 21, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

ClassImp (fdraw_class)

fdraw_class::fdraw_class() : dstbank_class(FDRAW_BANKID,FDRAW_BANKVERSION)
{
  used_bankid = FDRAW_BANKID;
}

fdraw_class::~fdraw_class()
{
}

void fdraw_class::loadFromDST()
{
  fdraw_dst_common *fdraw_ptr;
  if (used_bankid == BRRAW_BANKID)
    fdraw_ptr = &brraw_;
  else if (used_bankid == LRRAW_BANKID)
    fdraw_ptr = &lrraw_;
  else
    fdraw_ptr = &fdraw_;
  
  Int_t imir,ichan,j;
    
  event_code   =  fdraw_ptr->event_code;
  part         =  fdraw_ptr->part;
  num_mir      =  fdraw_ptr->num_mir;
  event_num    =  fdraw_ptr->event_num;
  julian       =  fdraw_ptr->julian;
  jsecond      =  fdraw_ptr->jsecond;
  gps1pps_tick =  fdraw_ptr->gps1pps_tick;
  ctdclock     =  fdraw_ptr->ctdclock;
  ctd_version  =  fdraw_ptr->ctd_version;
  tf_version   =  fdraw_ptr->tf_version;
  sdf_version  =  fdraw_ptr->sdf_version;  
  trig_code.resize(num_mir);
  second.resize(num_mir);
  microsec.resize(num_mir);
  clkcnt.resize(num_mir);
  mir_num.resize(num_mir);
  num_chan.resize(num_mir);
  tf_mode.resize(num_mir);
  tf_mode2.resize(num_mir);
  hit_pt.resize(num_mir);
  channel.resize(num_mir);
  sdf_peak.resize(num_mir);
  sdf_tmphit.resize(num_mir);
  sdf_mode.resize(num_mir);
  sdf_ctrl.resize(num_mir);
  sdf_thre.resize(num_mir);
  mean.resize(num_mir);
  disp.resize(num_mir);
  m_fadc.resize(num_mir);
  for (imir=0; imir<num_mir; imir++)
    {
      trig_code[imir] =  fdraw_ptr->trig_code[imir];
      second[imir]    =  fdraw_ptr->second[imir];
      microsec[imir]  =  fdraw_ptr->microsec[imir];
      clkcnt[imir]    =  fdraw_ptr->clkcnt[imir];
      mir_num[imir]   =  fdraw_ptr->mir_num[imir];
      num_chan[imir]  =  fdraw_ptr->num_chan[imir];
      tf_mode[imir]   =  fdraw_ptr->tf_mode[imir];
      tf_mode2[imir]  =  fdraw_ptr->tf_mode2[imir];
      hit_pt[imir].resize(fdraw_nchan_mir+1);
      for (ichan=0; ichan<(int)hit_pt[imir].size(); ichan++)
	hit_pt[imir][ichan] = fdraw_ptr->hit_pt[imir][ichan];     
      channel[imir].resize(num_chan[imir]);
      sdf_peak[imir].resize(num_chan[imir]);
      sdf_tmphit[imir].resize(num_chan[imir]);
      sdf_mode[imir].resize(num_chan[imir]);
      sdf_ctrl[imir].resize(num_chan[imir]);
      sdf_thre[imir].resize(num_chan[imir]);
      mean[imir].resize(num_chan[imir]);
      disp[imir].resize(num_chan[imir]);
      m_fadc[imir].resize(num_chan[imir]);      
      for (ichan=0; ichan<num_chan[imir]; ichan++)
	{
	  channel[imir][ichan]    =  fdraw_ptr->channel[imir][ichan];
	  sdf_peak[imir][ichan]   =  fdraw_ptr->sdf_peak[imir][ichan];
	  sdf_tmphit[imir][ichan] =  fdraw_ptr->sdf_tmphit[imir][ichan];
	  sdf_mode[imir][ichan]   =  fdraw_ptr->sdf_mode[imir][ichan];
	  sdf_ctrl[imir][ichan]   =  fdraw_ptr->sdf_ctrl[imir][ichan];
	  sdf_thre[imir][ichan]   =  fdraw_ptr->sdf_thre[imir][ichan];
	  mean[imir][ichan].resize(4);
	  disp[imir][ichan].resize(4);
	  for (j=0; j<(int)mean[imir][ichan].size(); j++)
	    {
	      mean[imir][ichan][j] = fdraw_ptr->mean[imir][ichan][j];
	      disp[imir][ichan][j] = fdraw_ptr->disp[imir][ichan][j];
	    }
	  m_fadc[imir][ichan].resize(fdraw_nt_chan_max);
	  for (j=0; j<(int)m_fadc[imir][ichan].size(); j++)
	    m_fadc[imir][ichan][j] = fdraw_ptr->m_fadc[imir][ichan][j];
	}
    }
}


void fdraw_class::loadToDST()
{
  fdraw_dst_common *fdraw_ptr;
  if (used_bankid == BRRAW_BANKID)
    fdraw_ptr = &brraw_;
  else if (used_bankid == LRRAW_BANKID)
    fdraw_ptr = &lrraw_;
  else
    fdraw_ptr = &fdraw_;
    
  Int_t imir,ichan,j;
  
  fdraw_ptr->event_code   =  event_code;
  fdraw_ptr->part         =  part;
  fdraw_ptr->num_mir      =  num_mir;
  fdraw_ptr->event_num    =  event_num;
  fdraw_ptr->julian       =  julian;
  fdraw_ptr->jsecond      =  jsecond;
  fdraw_ptr->gps1pps_tick =  gps1pps_tick;
  fdraw_ptr->ctdclock     =  ctdclock;
  fdraw_ptr->ctd_version  =  ctd_version;
  fdraw_ptr->tf_version   =  tf_version;
  fdraw_ptr->sdf_version  =  sdf_version;
  for (imir=0; imir<num_mir; imir++)
    {
      fdraw_ptr->trig_code[imir] =  trig_code[imir];
      fdraw_ptr->second[imir]    =  second[imir];
      fdraw_ptr->microsec[imir]  =  microsec[imir];
      fdraw_ptr->clkcnt[imir]    =  clkcnt[imir];
      fdraw_ptr->mir_num[imir]   =  mir_num[imir];
      fdraw_ptr->num_chan[imir]  =  num_chan[imir];
      fdraw_ptr->tf_mode[imir]   =  tf_mode[imir];
      fdraw_ptr->tf_mode2[imir]  =  tf_mode2[imir];
      for (ichan=0; ichan<(int)hit_pt[imir].size(); ichan++)
	fdraw_ptr->hit_pt[imir][ichan] = hit_pt[imir][ichan];
      for (ichan=0; ichan<num_chan[imir]; ichan++)
	{
	  fdraw_ptr->channel[imir][ichan]    =  channel[imir][ichan];
	  fdraw_ptr->sdf_peak[imir][ichan]   =  sdf_peak[imir][ichan];
	  fdraw_ptr->sdf_tmphit[imir][ichan] =  sdf_tmphit[imir][ichan];
	  fdraw_ptr->sdf_mode[imir][ichan]   =  sdf_mode[imir][ichan];
	  fdraw_ptr->sdf_ctrl[imir][ichan]   =  sdf_ctrl[imir][ichan];
	  fdraw_ptr->sdf_thre[imir][ichan]   =  sdf_thre[imir][ichan];
	  for (j=0; j<(int)mean[imir][ichan].size(); j++)
	    {
	      fdraw_ptr->mean[imir][ichan][j] = mean[imir][ichan][j];
	      fdraw_ptr->disp[imir][ichan][j] = disp[imir][ichan][j];
	    }
	  for (j=0; j<(int)m_fadc[imir][ichan].size(); j++)
	    fdraw_ptr->m_fadc[imir][ichan][j] = m_fadc[imir][ichan][j];
	}
    }
}
void fdraw_class::clearOutDST()
{
  fdraw_dst_common *fdraw_ptr;
  if (used_bankid == BRRAW_BANKID)
    fdraw_ptr = &brraw_;
  else if (used_bankid == LRRAW_BANKID)
    fdraw_ptr = &lrraw_;
  else
    fdraw_ptr = &fdraw_;
  memset(fdraw_ptr,0,sizeof(fdraw_dst_common));
  loadFromDST();
}
/**
   Root tree class for brraw DST bank.
   Last modified: Sep 23, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/


ClassImp(brraw_class)

brraw_class::brraw_class()
{
  used_bankid     = BRRAW_BANKID;
  dstbank_id      = BRRAW_BANKID;
  dstbank_version = BRRAW_BANKVERSION;
}

brraw_class::~brraw_class()
{
  
}
/**
   Root tree class for lrraw DST bank.
   Last modified: Sep 23, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/


ClassImp(lrraw_class)

lrraw_class::lrraw_class()
{
  used_bankid     = LRRAW_BANKID;
  dstbank_id      = LRRAW_BANKID;
  dstbank_version = LRRAW_BANKVERSION;
}

lrraw_class::~lrraw_class()
{
  
}
/**
   Root tree class for fdplane DST bank.
   Last modified: Apr 4, 2013
   Dmitri Ivanov <dmiivanov@gmail.com>
**/

ClassImp (fdplane_class)

fdplane_class::fdplane_class() : dstbank_class(FDPLANE_BANKID,FDPLANE_BANKVERSION)
{
  used_bankid = FDPLANE_BANKID;
}

fdplane_class::~fdplane_class()
{
}

void fdplane_class::loadFromDST()
{

  Int_t i, j;
  Int_t itube;
  fdplane_dst_common *fdplane_ptr;
  if (used_bankid == BRPLANE_BANKID)
    fdplane_ptr = &brplane_;
  else if (used_bankid == LRPLANE_BANKID)
    fdplane_ptr = &lrplane_;
  else
    fdplane_ptr = &fdplane_;
  
  part = fdplane_ptr->part;
  event_num = fdplane_ptr->event_num;
  julian = fdplane_ptr->julian;
  jsecond = fdplane_ptr->jsecond;
  jsecfrac = fdplane_ptr->jsecfrac;
  second = fdplane_ptr->second;
  secfrac = fdplane_ptr->secfrac;
  ntube = fdplane_ptr->ntube;

  
  // uniqID, fmode appear only in fdplane version 1 or higher
#if FDPLANE_BANKVERSION >= 1
  uniqID = fdplane_ptr->uniqID;
  fmode  = fdplane_ptr->fmode;
#else
  uniqID = 0;
  fmode  = 0;
#endif
	  
  npe.resize(ntube);
  adc.resize(ntube);
  ped.resize(ntube);
  time.resize(ntube);
  time_rms.resize(ntube);
  sigma.resize(ntube);
  for (itube=0; itube < ntube; itube ++)
    {
      npe[itube] = fdplane_ptr->npe[itube];
      // signal in adc counts and pedestal under pulse appear in fdplane
      // bank version 2 or higher
#if FDPLANE_BANKVERSION >= 2
      adc[itube] = fdplane_ptr->adc[itube];
      ped[itube]  = fdplane_ptr->ped[itube];
#else
      adc[itube] = 0;
      ped[itube]  = 0;
#endif
      time[itube] = fdplane_ptr->time[itube];
      time_rms[itube] = fdplane_ptr->time_rms[itube];
      sigma[itube] = fdplane_ptr->sigma[itube];
    }

  for (i=0; i<3; i++)
    {
      sdp_n[i] = fdplane_ptr->sdp_n[i];
      sdp_en[i] = fdplane_ptr->sdp_en[i];
      for (j=0; j<3; j++)
	sdp_n_cov[i][j] = fdplane_ptr->sdp_n_cov[i][j];
    }
  sdp_the = fdplane_ptr->sdp_the;
  sdp_phi = fdplane_ptr->sdp_phi;
  sdp_chi2 = fdplane_ptr->sdp_chi2;

  alt.resize(ntube);
  azm.resize(ntube);
  plane_alt.resize(ntube);
  plane_azm.resize(ntube);
  for (itube=0; itube < ntube; itube ++)
    {
      alt[itube] = fdplane_ptr->alt[itube];
      azm[itube] = fdplane_ptr->azm[itube];
      plane_alt[itube] = fdplane_ptr->plane_alt[itube];
      plane_azm[itube] = fdplane_ptr->plane_azm[itube];
    }

  linefit_slope = fdplane_ptr->linefit_slope;
  linefit_eslope = fdplane_ptr->linefit_eslope;
  linefit_int = fdplane_ptr->linefit_int;
  linefit_eint = fdplane_ptr->linefit_eint;
  linefit_chi2 = fdplane_ptr->linefit_chi2;

  for (i=0; i<2; i++)
    {
      for (j=0; j<2; j++)
	linefit_cov[i][j] = fdplane_ptr->linefit_cov[i][j];
    }

  linefit_res.resize(ntube);
  linefit_tchi2.resize(ntube);
  for (itube=0; itube < ntube; itube ++)
    {
      linefit_res[itube] = fdplane_ptr->linefit_res[itube];
      linefit_tchi2[itube] = fdplane_ptr->linefit_tchi2[itube];
    }

  ptanfit_rp = fdplane_ptr->ptanfit_rp;
  ptanfit_erp = fdplane_ptr->ptanfit_erp;
  ptanfit_t0 = fdplane_ptr->ptanfit_t0;
  ptanfit_et0 = fdplane_ptr->ptanfit_et0;
  ptanfit_chi2 = fdplane_ptr->ptanfit_chi2;

  for (i=0; i<2; i++)
    {
      for (j=0; j<2; j++)
	ptanfit_cov[i][j] = fdplane_ptr->ptanfit_cov[i][j];
    }

  ptanfit_res.resize(ntube);
  ptanfit_tchi2.resize(ntube);
  for (itube=0; itube < ntube; itube ++)
    {
      ptanfit_res[itube] = fdplane_ptr->ptanfit_res[itube];
      ptanfit_tchi2[itube] = fdplane_ptr->ptanfit_tchi2[itube];
    }

  rp = fdplane_ptr->rp;
  erp = fdplane_ptr->erp;
  psi = fdplane_ptr->psi;
  epsi = fdplane_ptr->epsi;
  t0 = fdplane_ptr->t0;
  et0 = fdplane_ptr->et0;
  tanfit_chi2 = fdplane_ptr->tanfit_chi2;

  for (i=0; i<3; i++)
    {
      for (j=0; j<3; j++)
	tanfit_cov[i][j] = fdplane_ptr->tanfit_cov[i][j];
    }

  tanfit_res.resize(ntube);
  tanfit_tchi2.resize(ntube);
  for(itube=0; itube < ntube; itube ++)
    {
      tanfit_res[itube] = fdplane_ptr->tanfit_res[itube];
      tanfit_tchi2[itube] = fdplane_ptr->tanfit_tchi2[itube];
    }

  azm_extent = fdplane_ptr->azm_extent;
  time_extent = fdplane_ptr->time_extent;

  shower_zen = fdplane_ptr->shower_zen;
  shower_azm = fdplane_ptr->shower_azm;


  for (i = 0; i < 3; i++ )
    {
      shower_axis[i] = fdplane_ptr->shower_axis[i];
      rpuv[i] = fdplane_ptr->rpuv[i];
      core[i] = fdplane_ptr->core[i];
    }


  camera.resize(ntube);
  tube.resize(ntube);
  it0.resize(ntube);
  it1.resize(ntube);
  knex_qual.resize(ntube);
  tube_qual.resize(ntube);
  for (itube = 0; itube < ntube; itube ++ )
    {
      camera[itube] = fdplane_ptr->camera[itube];
      tube[itube] = fdplane_ptr->tube[itube];
      it0[itube] = fdplane_ptr->it0[itube];
      it1[itube] = fdplane_ptr->it1[itube];
      knex_qual[itube] = fdplane_ptr->knex_qual[itube];
      tube_qual[itube] = fdplane_ptr->tube_qual[itube];
    }

  ngtube = fdplane_ptr->ngtube;
  seed = fdplane_ptr->seed;
  type = fdplane_ptr->type;
  status = fdplane_ptr->status;
  siteid = fdplane_ptr->siteid;

}


void fdplane_class::loadToDST()
{
  Int_t i, j;
  Int_t itube;
  fdplane_dst_common *fdplane_ptr;
  
  if (used_bankid == BRPLANE_BANKID)
    fdplane_ptr = &brplane_;
  else if (used_bankid == LRPLANE_BANKID)
    fdplane_ptr = &lrplane_;
  else
    fdplane_ptr = &fdplane_;

  fdplane_ptr->part = part;
  fdplane_ptr->event_num = event_num;
  fdplane_ptr->julian = julian;
  fdplane_ptr->jsecond = jsecond;
  fdplane_ptr->jsecfrac = jsecfrac;
  fdplane_ptr->second = second;
  fdplane_ptr->secfrac = secfrac;
  fdplane_ptr->ntube = ntube;
  
  // uniqID, fmode appear only in fdplane version 1 or higher
#if FDPLANE_BANKVERSION >= 1
  fdplane_ptr->uniqID = uniqID;
  fdplane_ptr->fmode  = fmode;
#endif
  
  for (itube=0; itube < ntube; itube ++)
    {
      fdplane_ptr->npe[itube] = npe[itube];
      // signal in adc counts and pedestal under pulse appear in fdplane
      // bank version 2 or higher
#if FDPLANE_BANKVERSION >= 2
      fdplane_ptr->adc[itube] = adc[itube];
      fdplane_ptr->ped[itube]  = ped[itube];
#endif
      fdplane_ptr->time[itube] = time[itube];
      fdplane_ptr->time_rms[itube] = time_rms[itube];
      fdplane_ptr->sigma[itube] = sigma[itube];
    }

  for (i=0; i<3; i++)
    {
      fdplane_ptr->sdp_n[i] = sdp_n[i];
      fdplane_ptr->sdp_en[i] = sdp_en[i];
      for (j=0; j<3; j++)
	fdplane_ptr->sdp_n_cov[i][j] = sdp_n_cov[i][j];
    }
  fdplane_ptr->sdp_the = sdp_the;
  fdplane_ptr->sdp_phi = sdp_phi;
  fdplane_ptr->sdp_chi2 = sdp_chi2;
  
  for (itube=0; itube < ntube; itube ++)
    {
      fdplane_ptr->alt[itube] = alt[itube];
      fdplane_ptr->azm[itube] = azm[itube];
      fdplane_ptr->plane_alt[itube] = plane_alt[itube];
      fdplane_ptr->plane_azm[itube] = plane_azm[itube];
    }

  fdplane_ptr->linefit_slope = linefit_slope;
  fdplane_ptr->linefit_eslope = linefit_eslope;
  fdplane_ptr->linefit_int = linefit_int;
  fdplane_ptr->linefit_eint = linefit_eint;
  fdplane_ptr->linefit_chi2 = linefit_chi2;

  for (i=0; i<2; i++)
    {
      for (j=0; j<2; j++)
	fdplane_ptr->linefit_cov[i][j] = linefit_cov[i][j];
    }
  
  for (itube=0; itube < ntube; itube ++)
    {
      fdplane_ptr->linefit_res[itube] = linefit_res[itube];
      fdplane_ptr->linefit_tchi2[itube] = linefit_tchi2[itube];
    }

  fdplane_ptr->ptanfit_rp = ptanfit_rp;
  fdplane_ptr->ptanfit_erp = ptanfit_erp;
  fdplane_ptr->ptanfit_t0 = ptanfit_t0;
  fdplane_ptr->ptanfit_et0 = ptanfit_et0;
  fdplane_ptr->ptanfit_chi2 = ptanfit_chi2;

  for (i=0; i<2; i++)
    {
      for (j=0; j<2; j++)
	fdplane_ptr->ptanfit_cov[i][j] = ptanfit_cov[i][j];
    }
  
  for (itube=0; itube < ntube; itube ++)
    {
      fdplane_ptr->ptanfit_res[itube] = ptanfit_res[itube];
      fdplane_ptr->ptanfit_tchi2[itube] = ptanfit_tchi2[itube];
    }

  fdplane_ptr->rp = rp;
  fdplane_ptr->erp = erp;
  fdplane_ptr->psi = psi;
  fdplane_ptr->epsi = epsi;
  fdplane_ptr->t0 = t0;
  fdplane_ptr->et0 = et0;
  fdplane_ptr->tanfit_chi2 = tanfit_chi2;

  for (i=0; i<3; i++)
    {
      for (j=0; j<3; j++)
	fdplane_ptr->tanfit_cov[i][j] = tanfit_cov[i][j];
    }
  
  for(itube=0; itube < ntube; itube ++)
    {
      fdplane_ptr->tanfit_res[itube] = tanfit_res[itube];
      fdplane_ptr->tanfit_tchi2[itube] = tanfit_tchi2[itube];
    }

  fdplane_ptr->azm_extent = azm_extent;
  fdplane_ptr->time_extent = time_extent;

  fdplane_ptr->shower_zen = shower_zen;
  fdplane_ptr->shower_azm = shower_azm;


  for (i = 0; i < 3; i++ )
    {
      fdplane_ptr->shower_axis[i] = shower_axis[i];
      fdplane_ptr->rpuv[i] = rpuv[i];
      fdplane_ptr->core[i] = core[i];
    }
  
  for (itube = 0; itube < ntube; itube ++ )
    {
      fdplane_ptr->camera[itube] = camera[itube];
      fdplane_ptr->tube[itube] = tube[itube];
      fdplane_ptr->it0[itube] = it0[itube];
      fdplane_ptr->it1[itube] = it1[itube];
      fdplane_ptr->knex_qual[itube] = knex_qual[itube];
      fdplane_ptr->tube_qual[itube] = tube_qual[itube];
    }

  fdplane_ptr->ngtube = ngtube;
  fdplane_ptr->seed = seed;
  fdplane_ptr->type = type;
  fdplane_ptr->status = status;
  fdplane_ptr->siteid = siteid;

}
void fdplane_class::clearOutDST()
{
  fdplane_dst_common *fdplane_ptr;
  if (used_bankid == BRPLANE_BANKID)
    fdplane_ptr = &brplane_;
  else if (used_bankid == LRPLANE_BANKID)
    fdplane_ptr = &lrplane_;
  else
    fdplane_ptr = &fdplane_;
  memset(fdplane_ptr,0,sizeof(fdplane_dst_common));
  loadFromDST();
}
/**
   Root tree class for brplane DST bank.
   Last modified: Sep 18, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/


ClassImp(brplane_class)

brplane_class::brplane_class()
{
  used_bankid     = BRPLANE_BANKID;
  dstbank_id      = BRPLANE_BANKID;
  dstbank_version = BRPLANE_BANKVERSION;
}

brplane_class::~brplane_class()
{
  
}
/**
   Root tree class for lrplane DST bank.
   Last modified: Sep 18, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/


ClassImp(lrplane_class)

lrplane_class::lrplane_class()
{
  used_bankid     = LRPLANE_BANKID;
  dstbank_id      = LRPLANE_BANKID;
  dstbank_version = LRPLANE_BANKVERSION;
}
lrplane_class::~lrplane_class()
{
  
}
/**
   Root tree class for fdprofile DST bank.
   Last modified: Oct 21, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

ClassImp (fdprofile_class)

fdprofile_class::fdprofile_class() : dstbank_class(FDPROFILE_BANKID,FDPROFILE_BANKVERSION)
{
  used_bankid = FDPROFILE_BANKID;
  siteid  = 0;
  ntslice = 0;
  x.resize(3);
  dtheta.resize(3);
  darea.resize(3);
  acpt.resize(3);
  eacpt.resize(3);
  flux.resize(3);
  eflux.resize(3);
  nfl.resize(3);
  ncvdir.resize(3);
  ncvmie.resize(3);
  ncvray.resize(3);
  simflux.resize(3);
  tres.resize(3);
  tchi2.resize(3);
  ne.resize(3);
  ene.resize(3);
}

fdprofile_class::~fdprofile_class()
{
}

void fdprofile_class::loadFromDST()
{
  fdprofile_dst_common *fdprofile_ptr;
  if (used_bankid == BRPROFILE_BANKID)
    fdprofile_ptr = &brprofile_;
  else if (used_bankid == LRPROFILE_BANKID)
    fdprofile_ptr = &lrprofile_;
  else
    fdprofile_ptr = &fdprofile_;
  
  int i, itslice;
  
  siteid  = fdprofile_ptr->siteid;
  ntslice = fdprofile_ptr->ntslice;
  
  for (i=0; i<3; i++)
    {
      ngtslice[i] =   fdprofile_ptr->ngtslice[i];
      status[i]   =   fdprofile_ptr->status[i];
      rp[i]       =   fdprofile_ptr->rp[i];
      psi[i]      =   fdprofile_ptr->psi[i];
      t0[i]       =   fdprofile_ptr->t0[i];
      Xmax[i]     =   fdprofile_ptr->Xmax[i];
      eXmax[i]    =   fdprofile_ptr->eXmax[i];
      Nmax[i]     =   fdprofile_ptr->Nmax[i];
      eNmax[i]    =   fdprofile_ptr->eNmax[i];
      Energy[i]   =   fdprofile_ptr->Energy[i];
      eEnergy[i]  =   fdprofile_ptr->eEnergy[i];
      chi2[i]     =   fdprofile_ptr->chi2[i];
    }
  
  timebin.resize(ntslice);
  npe.resize(ntslice);
  enpe.resize(ntslice);  
  for (i=0; i<3; i++)
    {
      x[i].resize(ntslice);
      dtheta[i].resize(ntslice);
      darea[i].resize(ntslice);
      acpt[i].resize(ntslice);
      eacpt[i].resize(ntslice);
      flux[i].resize(ntslice);
      eflux[i].resize(ntslice);
      nfl[i].resize(ntslice);
      ncvdir[i].resize(ntslice);
      ncvmie[i].resize(ntslice);
      ncvray[i].resize(ntslice);
      simflux[i].resize(ntslice);
      tres[i].resize(ntslice);
      tchi2[i].resize(ntslice);
      ne[i].resize(ntslice);
      ene[i].resize(ntslice);
    }
  
  for (itslice=0; itslice<ntslice; itslice++)
    {
      timebin[itslice]    =  fdprofile_ptr->timebin[itslice];
      npe[itslice]        =  fdprofile_ptr->npe[itslice];
      enpe[itslice]       =  fdprofile_ptr->enpe[itslice];
      for (i=0; i<3; i++)
	{
	  x[i][itslice]       =  fdprofile_ptr->x[i][itslice];
	  dtheta[i][itslice]  =  fdprofile_ptr->dtheta[i][itslice];
	  darea[i][itslice]   =  fdprofile_ptr->darea[i][itslice];
	  acpt[i][itslice]    =  fdprofile_ptr->acpt[i][itslice];
	  eacpt[i][itslice]   =  fdprofile_ptr->eacpt[i][itslice];
	  flux[i][itslice]    =  fdprofile_ptr->flux[i][itslice];
	  eflux[i][itslice]   =  fdprofile_ptr->eflux[i][itslice];
	  nfl[i][itslice]     =  fdprofile_ptr->nfl[i][itslice];
	  ncvdir[i][itslice]  =  fdprofile_ptr->ncvdir[i][itslice];
	  ncvmie[i][itslice]  =  fdprofile_ptr->ncvmie[i][itslice];
	  ncvray[i][itslice]  =  fdprofile_ptr->ncvray[i][itslice];
	  simflux[i][itslice] =  fdprofile_ptr->simflux[i][itslice];
	  tres[i][itslice]    =  fdprofile_ptr->tres[i][itslice];
	  tchi2[i][itslice]   =  fdprofile_ptr->tchi2[i][itslice];
	  ne[i][itslice]      =  fdprofile_ptr->ne[i][itslice];
	  ene[i][itslice]     =  fdprofile_ptr->ene[i][itslice];
	}
    }
  mc = fdprofile_ptr->mc;
}


void fdprofile_class::loadToDST()
{

  fdprofile_dst_common *fdprofile_ptr;
  if (used_bankid == BRPROFILE_BANKID)
    fdprofile_ptr = &brprofile_;
  else if (used_bankid == LRPROFILE_BANKID)
    fdprofile_ptr = &lrprofile_;
  else
    fdprofile_ptr = &fdprofile_;

  int i, itslice;
 
  fdprofile_ptr->siteid  = siteid;
  fdprofile_ptr->ntslice = ntslice;
  
  for (i=0; i<3; i++)
    {
      fdprofile_ptr->ngtslice[i] =   ngtslice[i];
      fdprofile_ptr->status[i]   =   status[i];
      fdprofile_ptr->rp[i]       =   rp[i];
      fdprofile_ptr->psi[i]      =   psi[i];
      fdprofile_ptr->t0[i]       =   t0[i];
      fdprofile_ptr->Xmax[i]     =   Xmax[i];
      fdprofile_ptr->eXmax[i]    =   eXmax[i];
      fdprofile_ptr->Nmax[i]     =   Nmax[i];
      fdprofile_ptr->eNmax[i]    =   eNmax[i];
      fdprofile_ptr->Energy[i]   =   Energy[i];
      fdprofile_ptr->eEnergy[i]  =   eEnergy[i];
      fdprofile_ptr->chi2[i]     =   chi2[i];
    }
  
  for (itslice=0; itslice<ntslice; itslice++)
    {
      fdprofile_ptr->timebin[itslice]    =  timebin[itslice];
      fdprofile_ptr->npe[itslice]        =  npe[itslice];
      fdprofile_ptr->enpe[itslice]       =  enpe[itslice];
      for (i=0; i<3; i++)
	{
	  fdprofile_ptr->x[i][itslice]       =  x[i][itslice];
	  fdprofile_ptr->dtheta[i][itslice]  =  dtheta[i][itslice];
	  fdprofile_ptr->darea[i][itslice]   =  darea[i][itslice];
	  fdprofile_ptr->acpt[i][itslice]    =  acpt[i][itslice];
	  fdprofile_ptr->eacpt[i][itslice]   =  eacpt[i][itslice];
	  fdprofile_ptr->flux[i][itslice]    =  flux[i][itslice];
	  fdprofile_ptr->eflux[i][itslice]   =  eflux[i][itslice];
	  fdprofile_ptr->nfl[i][itslice]     =  nfl[i][itslice];
	  fdprofile_ptr->ncvdir[i][itslice]  =  ncvdir[i][itslice];
	  fdprofile_ptr->ncvmie[i][itslice]  =  ncvmie[i][itslice];
	  fdprofile_ptr->ncvray[i][itslice]  =  ncvray[i][itslice];
	  fdprofile_ptr->simflux[i][itslice] =  simflux[i][itslice];
	  fdprofile_ptr->tres[i][itslice]    =  tres[i][itslice];
	  fdprofile_ptr->tchi2[i][itslice]   =  tchi2[i][itslice];
	  fdprofile_ptr->ne[i][itslice]      =  ne[i][itslice];
	  fdprofile_ptr->ene[i][itslice]     =  ene[i][itslice];
	}
    }
  fdprofile_ptr->mc = mc;

}
void fdprofile_class::clearOutDST()
{
  fdprofile_dst_common *fdprofile_ptr;
  if (used_bankid == BRPROFILE_BANKID)
    fdprofile_ptr = &brprofile_;
  else if (used_bankid == LRPROFILE_BANKID)
    fdprofile_ptr = &lrprofile_;
  else
    fdprofile_ptr = &fdprofile_;
  memset(fdprofile_ptr,0,sizeof(fdprofile_dst_common));
  loadFromDST();
}
/**
   Root tree class for brprofile DST bank.
   Last modified: Sep 20, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/


ClassImp(brprofile_class)

brprofile_class::brprofile_class()
{
  used_bankid     = BRPROFILE_BANKID;
  dstbank_id      = BRPROFILE_BANKID;
  dstbank_version = BRPROFILE_BANKVERSION;
}

brprofile_class::~brprofile_class()
{
  
}
/**
   Root tree class for lrprofile DST bank.
   Last modified: Sep 20, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/


ClassImp(lrprofile_class)

lrprofile_class::lrprofile_class()
{
  used_bankid     = LRPROFILE_BANKID;
  dstbank_id      = LRPROFILE_BANKID;
  dstbank_version = LRPROFILE_BANKVERSION;
}

lrprofile_class::~lrprofile_class()
{
  
}
/**
   Root tree class for fdtubeprofile DST bank.
   Last modified: May 1, 2015
   Dmitri Ivanov <dmiivanov@gmail.com>
**/

ClassImp (fdtubeprofile_class)

#ifndef FDTUBEPROFILE_BANKID
_dstbank_not_implemented_(fdtubeprofile);
#else

#ifndef BRTUBEPROFILE_BANKID
#error Internal inconsistency of dst2k-ta: FDTUBEPROFILE_BANKID but BRTUBEPROFILE_BANKID is not
#endif

#ifndef LRTUBEPROFILE_BANKID
#error Internal inconsistency of dst2k-ta: FDTUBEPROFILE_BANKID but LRTUBEPROFILE_BANKID is not
#endif

fdtubeprofile_class::fdtubeprofile_class() : dstbank_class(FDTUBEPROFILE_BANKID,FDTUBEPROFILE_BANKVERSION)
{
  used_bankid = FDTUBEPROFILE_BANKID;
}

fdtubeprofile_class::~fdtubeprofile_class()
{
  ;
}
void fdtubeprofile_class::loadFromDST()
{
  fdtubeprofile_dst_common *fdtubeprofile_ptr;
  if (used_bankid == BRTUBEPROFILE_BANKID)
    fdtubeprofile_ptr = &brtubeprofile_;
  else if (used_bankid == LRTUBEPROFILE_BANKID)
    fdtubeprofile_ptr = &lrtubeprofile_;
  else
    fdtubeprofile_ptr = &fdtubeprofile_;
  
  ntube  = fdtubeprofile_ptr->ntube;
  siteid = fdtubeprofile_ptr->siteid;
  mc     = fdtubeprofile_ptr->mc;
  x.resize(fdtubeprofile_maxfit);
  npe.resize(fdtubeprofile_maxfit);
  enpe.resize(fdtubeprofile_maxfit);
  eacptfrac.resize(fdtubeprofile_maxfit);
  acpt.resize(fdtubeprofile_maxfit);
  eacpt.resize(fdtubeprofile_maxfit);
  flux.resize(fdtubeprofile_maxfit);
  eflux.resize(fdtubeprofile_maxfit);
  simnpe.resize(fdtubeprofile_maxfit);
  nfl.resize(fdtubeprofile_maxfit);
  ncvdir.resize(fdtubeprofile_maxfit);
  ncvmie.resize(fdtubeprofile_maxfit);
  ncvray.resize(fdtubeprofile_maxfit);
  simflux.resize(fdtubeprofile_maxfit);
  ne.resize(fdtubeprofile_maxfit);
  ene.resize(fdtubeprofile_maxfit);
  tres.resize(fdtubeprofile_maxfit);
  tchi2.resize(fdtubeprofile_maxfit);
  tube_qual.resize(fdtubeprofile_maxfit);
  for (Int_t ifit=0; ifit<fdtubeprofile_maxfit; ifit++)
    {
      ngtube[ifit]  = fdtubeprofile_ptr->ngtube[ifit];  
      rp[ifit]      = fdtubeprofile_ptr->rp[ifit];
      psi[ifit]     = fdtubeprofile_ptr->psi[ifit];
      t0[ifit]      = fdtubeprofile_ptr->t0[ifit];
      Xmax[ifit]    = fdtubeprofile_ptr->Xmax[ifit];
      eXmax[ifit]   = fdtubeprofile_ptr->eXmax[ifit];
      Nmax[ifit]    = fdtubeprofile_ptr->Nmax[ifit];
      eNmax[ifit]   = fdtubeprofile_ptr->eNmax[ifit];
      Energy[ifit]  = fdtubeprofile_ptr->Energy[ifit];
      eEnergy[ifit] = fdtubeprofile_ptr->eEnergy[ifit];
      chi2[ifit]    = fdtubeprofile_ptr->chi2[ifit];
      status[ifit]  = fdtubeprofile_ptr->status[ifit];
#if FDTUBEPROFILE_BANKVERSION >= 2
      X0[ifit]      = fdtubeprofile_ptr->X0[ifit];
      eX0[ifit]     = fdtubeprofile_ptr->eX0[ifit];
      Lambda[ifit]  = fdtubeprofile_ptr->Lambda[ifit];
      eLambda[ifit] = fdtubeprofile_ptr->eLambda[ifit];
#else
      X0[ifit]      = 0;
      eX0[ifit]     = 0;
      Lambda[ifit]  = 0;
      eLambda[ifit] = 0;
#endif
      x[ifit].resize(ntube);
      npe[ifit].resize(ntube);
      enpe[ifit].resize(ntube);
      eacptfrac[ifit].resize(ntube);
      acpt[ifit].resize(ntube);
      eacpt[ifit].resize(ntube);
      flux[ifit].resize(ntube);
      eflux[ifit].resize(ntube);
      simnpe[ifit].resize(ntube);
      nfl[ifit].resize(ntube);
      ncvdir[ifit].resize(ntube);
      ncvmie[ifit].resize(ntube);
      ncvray[ifit].resize(ntube);
      simflux[ifit].resize(ntube);
      ne[ifit].resize(ntube);
      ene[ifit].resize(ntube);
      tres[ifit].resize(ntube);
      tchi2[ifit].resize(ntube);
      tube_qual[ifit].resize(ntube);
      for (Int_t itube=0; itube<ntube; itube++)
	{
	  x[ifit][itube]         = fdtubeprofile_ptr->x[ifit][itube];
	  npe[ifit][itube]       = fdtubeprofile_ptr->npe[ifit][itube];
	  enpe[ifit][itube]      = fdtubeprofile_ptr->enpe[ifit][itube];
	  eacptfrac[ifit][itube] = fdtubeprofile_ptr->eacptfrac[ifit][itube];
	  acpt[ifit][itube]      = fdtubeprofile_ptr->acpt[ifit][itube];
	  eacpt[ifit][itube]     = fdtubeprofile_ptr->eacpt[ifit][itube];
	  flux[ifit][itube]      = fdtubeprofile_ptr->flux[ifit][itube];
	  eflux[ifit][itube]     = fdtubeprofile_ptr->eflux[ifit][itube];
	  simnpe[ifit][itube]    = fdtubeprofile_ptr->simnpe[ifit][itube];
	  nfl[ifit][itube]       = fdtubeprofile_ptr->nfl[ifit][itube];
	  ncvdir[ifit][itube]    = fdtubeprofile_ptr->ncvdir[ifit][itube];
	  ncvmie[ifit][itube]    = fdtubeprofile_ptr->ncvmie[ifit][itube];
	  ncvray[ifit][itube]    = fdtubeprofile_ptr->ncvray[ifit][itube];
	  simflux[ifit][itube]   = fdtubeprofile_ptr->simflux[ifit][itube];
	  ne[ifit][itube]        = fdtubeprofile_ptr->ne[ifit][itube];
	  ene[ifit][itube]       = fdtubeprofile_ptr->ene[ifit][itube];
	  tres[ifit][itube]      = fdtubeprofile_ptr->tres[ifit][itube];
	  tchi2[ifit][itube]     = fdtubeprofile_ptr->tchi2[ifit][itube];
	  tube_qual[ifit][itube] = fdtubeprofile_ptr->tube_qual[ifit][itube];
	}
    }
  camera.resize(ntube);
  tube.resize(ntube);
  for (Int_t itube=0; itube<ntube; itube++)
    {
      camera[itube] = fdtubeprofile_ptr->camera[itube];
      tube[itube]   = fdtubeprofile_ptr->tube[itube];
    }
  simtime.resize(3);
  simtrms.resize(3);
  simtres.resize(3);
  timechi2.resize(3);
  for (Int_t i=0; i<(Int_t)simtime.size(); i++)
    {
      simtime[i].resize(ntube);
      simtrms[i].resize(ntube);
      simtres[i].resize(ntube);
      timechi2[i].resize(ntube);
      for (Int_t itube=0; itube<ntube;itube++)
	{
#if FDTUBEPROFILE_BANKVERSION >= 3
	  simtime[i][itube] = fdtubeprofile_ptr->simtime[i][itube];
	  simtrms[i][itube] = fdtubeprofile_ptr->simtrms[i][itube];
	  simtres[i][itube] = fdtubeprofile_ptr->simtres[i][itube];
	  timechi2[i][itube] = fdtubeprofile_ptr->timechi2[i][itube];
#else
	  simtime[i][itube] = 0;
	  simtrms[i][itube] = 0;
	  simtres[i][itube] = 0;
	  timechi2[i][itube] = 0;
#endif
	}
    }
}

void fdtubeprofile_class::loadToDST()
{  
  fdtubeprofile_dst_common *fdtubeprofile_ptr;  
  if (used_bankid == BRTUBEPROFILE_BANKID)
    fdtubeprofile_ptr = &brtubeprofile_;
  else if (used_bankid == LRTUBEPROFILE_BANKID)
    fdtubeprofile_ptr = &lrtubeprofile_;
  else
    fdtubeprofile_ptr = &fdtubeprofile_;
  
  fdtubeprofile_ptr->ntube  = ntube;
  fdtubeprofile_ptr->siteid = siteid;
  fdtubeprofile_ptr->mc     = mc;
  
  for (Int_t ifit=0; ifit<(Int_t)x.size(); ifit++)
    {
      fdtubeprofile_ptr->ngtube[ifit]  = ngtube[ifit];  
      fdtubeprofile_ptr->rp[ifit]      = rp[ifit];
      fdtubeprofile_ptr->psi[ifit]     = psi[ifit];
      fdtubeprofile_ptr->t0[ifit]      = t0[ifit];
      fdtubeprofile_ptr->Xmax[ifit]    = Xmax[ifit];
      fdtubeprofile_ptr->eXmax[ifit]   = eXmax[ifit];
      fdtubeprofile_ptr->Nmax[ifit]    = Nmax[ifit];
      fdtubeprofile_ptr->eNmax[ifit]   = eNmax[ifit];
      fdtubeprofile_ptr->Energy[ifit]  = Energy[ifit];
      fdtubeprofile_ptr->eEnergy[ifit] = eEnergy[ifit];
      fdtubeprofile_ptr->chi2[ifit]    = chi2[ifit];
      fdtubeprofile_ptr->status[ifit]  = status[ifit];
#if FDTUBEPROFILE_BANKVERSION >= 2
      fdtubeprofile_ptr->X0[ifit]      = X0[ifit];
      fdtubeprofile_ptr->eX0[ifit]     = eX0[ifit];
      fdtubeprofile_ptr->Lambda[ifit]  = Lambda[ifit];
      fdtubeprofile_ptr->eLambda[ifit] = eLambda[ifit];
#endif
      for (Int_t itube=0; itube<ntube; itube++)
	{
	  fdtubeprofile_ptr->x[ifit][itube]         = x[ifit][itube];
	  fdtubeprofile_ptr->npe[ifit][itube]       = npe[ifit][itube];
	  fdtubeprofile_ptr->enpe[ifit][itube]      = enpe[ifit][itube] ;
	  fdtubeprofile_ptr->eacptfrac[ifit][itube] = eacptfrac[ifit][itube];
	  fdtubeprofile_ptr->acpt[ifit][itube]      = acpt[ifit][itube];
	  fdtubeprofile_ptr->eacpt[ifit][itube]     = eacpt[ifit][itube];
	  fdtubeprofile_ptr->flux[ifit][itube]      = flux[ifit][itube];
	  fdtubeprofile_ptr->eflux[ifit][itube]     = eflux[ifit][itube];
	  fdtubeprofile_ptr->simnpe[ifit][itube]    = simnpe[ifit][itube];
	  fdtubeprofile_ptr->nfl[ifit][itube]       = nfl[ifit][itube];
	  fdtubeprofile_ptr->ncvdir[ifit][itube]    = ncvdir[ifit][itube];
	  fdtubeprofile_ptr->ncvmie[ifit][itube]    = ncvmie[ifit][itube];
	  fdtubeprofile_ptr->ncvray[ifit][itube]    = ncvray[ifit][itube];
	  fdtubeprofile_ptr->simflux[ifit][itube]   = simflux[ifit][itube];
	  fdtubeprofile_ptr->ne[ifit][itube]        = ne[ifit][itube];
	  fdtubeprofile_ptr->ene[ifit][itube]       = ene[ifit][itube];
	  fdtubeprofile_ptr->tres[ifit][itube]      = tres[ifit][itube];
	  fdtubeprofile_ptr->tchi2[ifit][itube]     = tchi2[ifit][itube];
	  fdtubeprofile_ptr->tube_qual[ifit][itube] = tube_qual[ifit][itube];
	}
    }
  for (Int_t itube=0; itube<ntube; itube++)
    {
      fdtubeprofile_ptr->camera[itube] = camera[itube];
      fdtubeprofile_ptr->tube[itube]   = tube[itube];
    }
  for (Int_t i=0; i<(Int_t)simtime.size(); i++)
    {
      for (Int_t itube=0; itube<ntube;itube++)
	{
#if FDTUBEPROFILE_BANKVERSION >= 3
	  fdtubeprofile_ptr->simtime[i][itube] = simtime[i][itube];
	  fdtubeprofile_ptr->simtrms[i][itube] = simtrms[i][itube];
	  fdtubeprofile_ptr->simtres[i][itube] = simtres[i][itube];
	  fdtubeprofile_ptr->timechi2[i][itube] = timechi2[i][itube];
#endif
	}
    }
}
void fdtubeprofile_class::clearOutDST()
{
  fdtubeprofile_dst_common *fdtubeprofile_ptr;  
  if (used_bankid == BRTUBEPROFILE_BANKID)
    fdtubeprofile_ptr = &brtubeprofile_;
  else if (used_bankid == LRTUBEPROFILE_BANKID)
    fdtubeprofile_ptr = &lrtubeprofile_;
  else
    fdtubeprofile_ptr = &fdtubeprofile_;
  memset(fdtubeprofile_ptr,0,sizeof(fdtubeprofile_dst_common));
  loadFromDST();
}
#endif
/**
   Root tree class for brtubeprofile DST bank.
   Last modified: Sep 25, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/


ClassImp(brtubeprofile_class)

#ifdef FDTUBEPROFILE_BANKID
brtubeprofile_class::brtubeprofile_class()
{
#ifndef BRTUBEPROFILE_BANKID
#error Internal inconsistency of dst2k-ta: FDTUBEPROFILE_BANKID but BRTUBEPROFILE_BANKID is not
#endif
  used_bankid     = BRTUBEPROFILE_BANKID;
  dstbank_id      = BRTUBEPROFILE_BANKID;
  dstbank_version = BRTUBEPROFILE_BANKVERSION;
}

brtubeprofile_class::~brtubeprofile_class()
{
  
}
#else
_dstbank_empty_constructor_destructor_(brtubeprofile);
#endif
/**
   Root tree class for lrtubeprofile DST bank.
   Last modified: Sep 25, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/


ClassImp(lrtubeprofile_class)

#ifdef FDTUBEPROFILE_BANKID
lrtubeprofile_class::lrtubeprofile_class()
{
#ifndef LRTUBEPROFILE_BANKID
#error Internal inconsistency of dst2k-ta: FDTUBEPROFILE_BANKID but LRTUBEPROFILE_BANKID is not
#endif
  used_bankid     = LRTUBEPROFILE_BANKID;
  dstbank_id      = LRTUBEPROFILE_BANKID;
  dstbank_version = LRTUBEPROFILE_BANKVERSION;
}

lrtubeprofile_class::~lrtubeprofile_class()
{
  
}
#else
_dstbank_empty_constructor_destructor_(lrtubeprofile);
#endif

ClassImp (hbar_class)

#ifdef HBAR_BANKID
hbar_class::hbar_class() : dstbank_class(HBAR_BANKID,HBAR_BANKVERSION) {;}
hbar_class::~hbar_class() {;}
void hbar_class::loadFromDST()
{
  Int_t imir, itube;
  
  jday    =  hbar_.jday;
  jsec    =  hbar_.jsec;
  msec    =  hbar_.msec;
  source  =  hbar_.source;
  nmir    =  hbar_.nmir;
  ntube   =  hbar_.ntube;
  
  hnpe_jday.resize(nmir);
  mir.resize(nmir);
  mir_reflect.resize(nmir);
  for (imir = 0; imir < nmir; imir ++ )
    {
      hnpe_jday[imir]    =   hbar_.hnpe_jday[imir];
      mir[imir]          =   hbar_.mir[imir];
      mir_reflect[imir]  =   hbar_.mir_reflect[imir];
    }
  tubemir.resize(ntube);
  tube.resize(ntube);
  qdcb.resize(ntube);
  npe.resize(ntube);
  sigma_npe.resize(ntube);
  first_order_gain_flag.resize(ntube);
  second_order_gain.resize(ntube);
  second_order_gain_sigma.resize(ntube);
  second_order_gain_flag.resize(ntube);
  qe_337.resize(ntube);
  sigma_qe_337.resize(ntube);
  uv_exp.resize(ntube);
  for (itube = 0; itube < ntube; itube ++)
    {
      tubemir[itube]                 =   hbar_.tubemir[itube];
      tube[itube]                    =   hbar_.tube[itube];
      qdcb[itube]                    =   hbar_.qdcb[itube];
      npe[itube]                     =   hbar_.npe[itube];
      sigma_npe[itube]               =   hbar_.sigma_npe[itube];
      first_order_gain_flag[itube]   =   hbar_.first_order_gain_flag[itube];
      second_order_gain[itube]       =   hbar_.second_order_gain[itube]; 
      second_order_gain_sigma[itube] =   hbar_.second_order_gain_sigma[itube];
      second_order_gain_flag[itube]  =   hbar_.second_order_gain_flag[itube];
      qe_337[itube]                  =   hbar_.qe_337[itube];
      sigma_qe_337[itube]            =   hbar_.sigma_qe_337[itube];
      uv_exp[itube]                  =   hbar_.uv_exp[itube];
    }
}
void hbar_class::loadToDST()
{
  Int_t imir, itube;
  hbar_.jday    =  jday;
  hbar_.jsec    =  jsec;
  hbar_.msec    =  msec;
  hbar_.source  =  source;
  hbar_.nmir    =  nmir;
  hbar_.ntube   =  ntube;
   
  for (imir = 0; imir < nmir; imir ++ )
    {
      hbar_.hnpe_jday[imir]    =   hnpe_jday[imir];
      hbar_.mir[imir]          =   mir[imir];
      hbar_.mir_reflect[imir]  =   mir_reflect[imir];
    }
  for (itube = 0; itube < ntube; itube ++)
    {
      hbar_.tubemir[itube]                 =   tubemir[itube];
      hbar_.tube[itube]                    =   tube[itube];
      hbar_.qdcb[itube]                    =   qdcb[itube];
      hbar_.npe[itube]                     =   npe[itube];
      hbar_.sigma_npe[itube]               =   sigma_npe[itube];
      hbar_.first_order_gain_flag[itube]   =   first_order_gain_flag[itube];
      hbar_.second_order_gain[itube]       =   second_order_gain[itube]; 
      hbar_.second_order_gain_sigma[itube] =   second_order_gain_sigma[itube];
      hbar_.second_order_gain_flag[itube]  =   second_order_gain_flag[itube];
      hbar_.qe_337[itube]                  =   qe_337[itube];
      hbar_.sigma_qe_337[itube]            =   sigma_qe_337[itube];
      hbar_.uv_exp[itube]                  =   uv_exp[itube];
    }
}
void hbar_class::clearOutDST()
{
  memset(&hbar_,0,sizeof(hbar_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(hbar);
#endif

ClassImp (hraw1_class)

hraw1_class::hraw1_class() : dstbank_class(HRAW1_BANKID,HRAW1_BANKVERSION)
{  
}

hraw1_class::~hraw1_class()
{ 
}

void hraw1_class::loadFromDST()
{
  Int_t imir, itube;
  jday     =   hraw1_.jday;
  jsec     =   hraw1_.jsec;
  msec     =   hraw1_.msec;
  status   =   hraw1_.status;
  nmir     =   hraw1_.nmir;
  ntube    =   hraw1_.ntube;  
  mir.resize(nmir);
  mir_rev.resize(nmir);
  mirevtno.resize(nmir);
  mirntube.resize(nmir);
  miraccuracy_ns.resize(nmir);
  mirtime_ns.resize(nmir);
  for (imir = 0; imir < nmir; imir ++ )
    {
      mir[imir]             =    hraw1_.mir[imir];
      mir_rev[imir]         =    hraw1_.mir_rev[imir];
      mirevtno[imir]        =    hraw1_.mirevtno[imir];
      mirntube[imir]        =    hraw1_.mirntube[imir];
      miraccuracy_ns[imir]  =    hraw1_.miraccuracy_ns[imir];
      mirtime_ns[imir]      =    hraw1_.mirtime_ns[imir];
    }  
  tubemir.resize(ntube);
  tube.resize(ntube);
  qdca.resize(ntube);
  qdcb.resize(ntube);
  tdc.resize(ntube);
  tha.resize(ntube);
  thb.resize(ntube);
  prxf.resize(ntube);
  thcal1.resize(ntube);
  for (itube = 0; itube < ntube; itube++)
    {
      tubemir[itube]  =    hraw1_.tubemir[itube];
      tube[itube]     =    hraw1_.tube[itube];
      qdca[itube]     =    hraw1_.qdca[itube];
      qdcb[itube]     =    hraw1_.qdcb[itube];
      tdc[itube]      =    hraw1_.tdc[itube];
      tha[itube]      =    hraw1_.tha[itube];
      thb[itube]      =    hraw1_.thb[itube];
      prxf[itube]     =    hraw1_.prxf[itube];
      thcal1[itube]   =    hraw1_.thcal1[itube];
    }
}
void hraw1_class::loadToDST()
{
  Int_t imir, itube;  
  hraw1_.jday     =   jday;
  hraw1_.jsec     =   jsec;
  hraw1_.msec     =   msec;
  hraw1_.status   =   status;
  hraw1_.nmir     =   nmir;
  hraw1_.ntube    =   ntube;  
  for (imir = 0; imir < nmir; imir ++ )
    {
      hraw1_.mir[imir]             =    mir[imir] ;
      hraw1_.mir_rev[imir]         =    mir_rev[imir] ;
      hraw1_.mirevtno[imir]        =    mirevtno[imir] ;
      hraw1_.mirntube[imir]        =    mirntube[imir] ;
      hraw1_.miraccuracy_ns[imir]  =    miraccuracy_ns[imir] ;
      hraw1_.mirtime_ns[imir]      =    mirtime_ns[imir] ;
    }  
  for (itube = 0; itube < ntube; itube++)
    {
      hraw1_.tubemir[itube]  =    tubemir[itube];
      hraw1_.tube[itube]     =    tube[itube];
      hraw1_.qdca[itube]     =    qdca[itube];
      hraw1_.qdcb[itube]     =    qdcb[itube];
      hraw1_.tdc[itube]      =    tdc[itube];
      hraw1_.tha[itube]      =    tha[itube];
      hraw1_.thb[itube]      =    thb[itube];
      hraw1_.prxf[itube]     =    prxf[itube];
      hraw1_.thcal1[itube]   =    thcal1[itube];
    }  
}
void hraw1_class::clearOutDST()
{
  memset(&hraw1_,0,sizeof(hraw1_dst_common));
  loadFromDST();
}

ClassImp (mc04_class)

mc04_class::mc04_class() : dstbank_class(MC04_BANKID,MC04_BANKVERSION)
{
  
}

mc04_class::~mc04_class()
{
  
}

void mc04_class::loadFromDST()
{
  energy    = mc04_.energy;
  csmax     = mc04_.csmax;
  x0        = mc04_.x0;
  x1        = mc04_.x1;
  xmax      = mc04_.xmax;
  lambda    = mc04_.lambda;
  xfin      = mc04_.xfin;
  theta     = mc04_.theta;
  Rp        = mc04_.Rp;
  aero_vod  = mc04_.aero_vod;
  aero_hal  = mc04_.aero_hal;
  aero_vsh  = mc04_.aero_vsh;
  aero_mlh  = mc04_.aero_mlh;   
  la_wavlen = mc04_.la_wavlen;  
  fl_totpho = mc04_.fl_totpho;   
  fl_twidth = mc04_.fl_twidth; 
  iprim     = mc04_.iprim;
  eventNr   = mc04_.eventNr; 
  setNr     = mc04_.setNr;
  iseed1    = mc04_.iseed1;   
  iseed2    = mc04_.iseed2; 
  detid     = mc04_.detid; 
  maxeye    = mc04_.maxeye; 
  neye      = mc04_.neye;
  nmir      = mc04_.nmir;   
  ntube     = mc04_.ntube;
  for  (Int_t i=0; i < 3; i ++)
    {
      rini[i]   = mc04_.rini[i];  
      rfin[i]   = mc04_.rfin[i];
      uthat[i]  = mc04_.uthat[i];
      Rpvec[i]   = mc04_.Rpvec[i]; 
      Rcore[i]   = mc04_.Rcore[i];
      la_site[i] = mc04_.la_site[i];
    }
  rsite.resize(maxeye);
  rpvec.resize(maxeye);
  rcore.resize(maxeye);
  shwn.resize(maxeye);
  rp.resize(maxeye);
  psi.resize(maxeye);
  if_eye.resize(maxeye);
  eyeid.resize(maxeye);
  eye_nmir.resize(maxeye);
  eye_ntube.resize(maxeye);
  for (Int_t ieye = 0; ieye < (Int_t)eye_ntube.size(); ieye ++ )
    {
      rp[ieye]     = mc04_.rp[ieye];
      psi[ieye]    = mc04_.psi[ieye];
      if_eye[ieye] = mc04_.if_eye[ieye];
      eyeid[ieye]     = mc04_.eyeid[ieye];
      eye_nmir[ieye]  = mc04_.eye_nmir[ieye];
      eye_ntube[ieye] = mc04_.eye_ntube[ieye];
      rsite[ieye].resize(3);
      rpvec[ieye].resize(3);
      rcore[ieye].resize(3);
      shwn[ieye].resize(3);
      for  (Int_t i=0; i < (Int_t)shwn[ieye].size(); i ++)
	{
	  rsite[ieye][i] = mc04_.rsite[ieye][i];
	  rpvec[ieye][i] = mc04_.rpvec[ieye][i];
	  rcore[ieye][i] = mc04_.rcore[ieye][i];
	  shwn[ieye][i]  = mc04_.shwn[ieye][i];
	}
    }  
  mirid.resize(nmir);
  mir_eye.resize(nmir);
  thresh.resize(nmir);
  for (Int_t imir = 0; imir < (Int_t)thresh.size(); imir ++ )
    {
      mirid[imir]   = mc04_.mirid[imir];
      mir_eye[imir] = mc04_.mir_eye[imir];
      thresh[imir]  = mc04_.thresh[imir];
    }
  tubeid.resize(ntube);
  tube_mir.resize(ntube);
  tube_eye.resize(ntube);
  pe.resize(ntube);
  triggered.resize(ntube);
  t_tmean.resize(ntube);
  t_trms.resize(ntube);
  t_tmin.resize(ntube);
  t_tmax.resize(ntube);
  for (Int_t itube = 0; itube < (Int_t)pe.size(); itube ++ )
    {
      tubeid[itube]   = mc04_.tubeid[itube];
      tube_mir[itube] = mc04_.tube_mir[itube];
      tube_eye[itube] = mc04_.tube_eye[itube];
      pe[itube]       = mc04_.pe[itube];
#if MC04_BANKVERSION >= 2
      triggered[itube] = mc04_.triggered[itube];
      t_tmean[itube]   = mc04_.t_tmean[itube];
      t_trms[itube]    = mc04_.t_trms[itube];
      t_tmin[itube]    = mc04_.t_tmin[itube];
      t_tmax[itube]    = mc04_.t_tmax[itube];
#else
      triggered[itube] = 0;
      t_tmean[itube]   = 0;
      t_trms[itube]    = 0;
      t_tmin[itube]    = 0;
      t_tmax[itube]    = 0;
#endif
    }
  
}


void mc04_class::loadToDST()
{
  mc04_.energy = energy;
  mc04_.csmax = csmax;
  mc04_.x0 = x0;
  mc04_.x1 = x1;
  mc04_.xmax = xmax;
  mc04_.lambda = lambda;
  mc04_.xfin = xfin;
  mc04_.theta = theta;
  mc04_.Rp = Rp;
  mc04_.aero_vod = aero_vod;
  mc04_.aero_hal = aero_hal;
  mc04_.aero_vsh = aero_vsh;
  mc04_.aero_mlh = aero_mlh;   
  mc04_.la_wavlen = la_wavlen;  
  mc04_.fl_totpho = fl_totpho;   
  mc04_.fl_twidth = fl_twidth; 
  mc04_.iprim = iprim;
  mc04_.eventNr = eventNr; 
  mc04_.setNr = setNr;
  mc04_.iseed1 = iseed1;   
  mc04_.iseed2 = iseed2; 
  mc04_.detid = detid; 
  mc04_.maxeye = maxeye; 
  mc04_.neye = neye;
  mc04_.nmir = nmir;   
  mc04_.ntube = ntube;
  for  (Int_t i=0; i < 3; i ++)
    {
      mc04_.rini[i] = rini[i];  
      mc04_.rfin[i] = rfin[i];
      mc04_.uthat[i] = uthat[i];
      mc04_.Rpvec[i] = Rpvec[i]; 
      mc04_.Rcore[i] = Rcore[i];
      mc04_.la_site[i] = la_site[i];
    }
  for (Int_t ieye = 0; ieye < (Int_t)eye_ntube.size(); ieye ++ )
    {
      mc04_.rp[ieye] = rp[ieye];
      mc04_.psi[ieye] = psi[ieye];
      mc04_.if_eye[ieye] = if_eye[ieye];
      mc04_.eyeid[ieye] = eyeid[ieye];
      mc04_.eye_nmir[ieye] = eye_nmir[ieye];
      mc04_.eye_ntube[ieye] = eye_ntube[ieye];
      for  (Int_t i=0; i < (Int_t)shwn[ieye].size(); i ++)
	{
	  mc04_.rsite[ieye][i] = rsite[ieye][i];
	  mc04_.rpvec[ieye][i] = rpvec[ieye][i];
	  mc04_.rcore[ieye][i] = rcore[ieye][i];
	  mc04_.shwn[ieye][i] = shwn[ieye][i];
	}
    }
  for (Int_t imir = 0; imir < (Int_t)thresh.size(); imir ++ )
    {
      mc04_.mirid[imir] = mirid[imir];
      mc04_.mir_eye[imir] = mir_eye[imir];
      mc04_.thresh[imir] = thresh[imir];
    }
  for (Int_t itube = 0; itube < (Int_t)pe.size(); itube ++ )
    {
      mc04_.tubeid[itube] = tubeid[itube];
      mc04_.tube_mir[itube] = tube_mir[itube];
      mc04_.tube_eye[itube] = tube_eye[itube];
      mc04_.pe[itube] = pe[itube];
#if MC04_BANKVERSION >= 2
      mc04_.triggered[itube] = triggered[itube];
      mc04_.t_tmean[itube] = t_tmean[itube];
      mc04_.t_trms[itube] = t_trms[itube];
      mc04_.t_tmin[itube] = t_tmin[itube];
      mc04_.t_tmax[itube] = t_tmax[itube];
#endif
    }
}

void mc04_class::clearOutDST()
{
  memset(&mc04_,0,sizeof(mc04_dst_common));
  loadFromDST();
}

ClassImp (mcraw_class)

mcraw_class::mcraw_class() : dstbank_class(MCRAW_BANKID,MCRAW_BANKVERSION)
{
  
}

mcraw_class::~mcraw_class()
{
  
}

void mcraw_class::loadFromDST()
{
  Int_t ieye, imir, itube;
  
  jday     =   mcraw_.jday;
  jsec     =   mcraw_.jsec;
  msec     =   mcraw_.msec;
  neye     =   mcraw_.neye;
  nmir     =   mcraw_.nmir;
  ntube    =   mcraw_.ntube;
  
  eyeid.resize(neye);
  for (ieye = 0; ieye < neye; ieye++)
    {
      eyeid[ieye] = mcraw_.eyeid[ieye];
    }
  
  mirid.resize(nmir);
  mir_eye.resize(nmir);
  mir_rev.resize(nmir);
  mirevtno.resize(nmir);
  mir_ntube.resize(nmir);
  mirtime_ns.resize(nmir);
  for (imir = 0; imir < nmir; imir++)
    {
      mirid[imir]       =  mcraw_.mirid[imir];
      mir_eye[imir]     =  mcraw_.mir_eye[imir];
      mir_rev[imir]     =  mcraw_.mir_rev[imir];
      mirevtno[imir]    =  mcraw_.mirevtno[imir];
      mir_ntube[imir]   =  mcraw_.mir_ntube[imir];
      mirtime_ns[imir]  =  mcraw_.mirtime_ns[imir];
    }
  
  tube_eye.resize(ntube);
  tube_mir.resize(ntube);
  tubeid.resize(ntube);
  qdca.resize(ntube);
  qdcb.resize(ntube);
  tdc.resize(ntube);
  tha.resize(ntube);
  thb.resize(ntube);
  prxf.resize(ntube);
  thcal1.resize(ntube);
  for (itube = 0; itube < ntube; itube ++)
    {
      tube_eye[itube]  =   mcraw_.tube_eye[itube];
      tube_mir[itube]  =   mcraw_.tube_mir[itube];
      tubeid[itube]    =   mcraw_.tubeid[itube];
      qdca[itube]      =   mcraw_.qdca[itube];
      qdcb[itube]      =   mcraw_.qdcb[itube];
      tdc[itube]       =   mcraw_.tdc[itube];
      tha[itube]       =   mcraw_.tha[itube];
      thb[itube]       =   mcraw_.thb[itube];
      prxf[itube]      =   mcraw_.prxf[itube];
      thcal1[itube]    =   mcraw_.thcal1[itube];
    } 
}

void mcraw_class::loadToDST()
{
  Int_t ieye, imir, itube;
  
  mcraw_.jday     =   jday;
  mcraw_.jsec     =   jsec;
  mcraw_.msec     =   msec;
  mcraw_.neye     =   neye;
  mcraw_.nmir     =   nmir;
  mcraw_.ntube    =   ntube;
  
  for (ieye = 0; ieye < neye; ieye++)
    {
      mcraw_.eyeid[ieye] = eyeid[ieye];
    }
  
  for (imir = 0; imir < nmir; imir++)
    {
      mcraw_.mirid[imir]       =  mirid[imir];
      mcraw_.mir_eye[imir]     =  mir_eye[imir];
      mcraw_.mir_rev[imir]     =  mir_rev[imir];
      mcraw_.mirevtno[imir]    =  mirevtno[imir];
      mcraw_.mir_ntube[imir]   =  mir_ntube[imir];
      mcraw_.mirtime_ns[imir]  =  mirtime_ns[imir];
    }
  
  for (itube = 0; itube < ntube; itube ++)
    {
      mcraw_.tube_eye[itube]  =   tube_eye[itube];
      mcraw_.tube_mir[itube]  =   tube_mir[itube];
      mcraw_.tubeid[itube]    =   tubeid[itube];
      mcraw_.qdca[itube]      =   qdca[itube];
      mcraw_.qdcb[itube]      =   qdcb[itube];
      mcraw_.tdc[itube]       =   tdc[itube];
      mcraw_.tha[itube]       =   tha[itube];
      mcraw_.thb[itube]       =   thb[itube];
      mcraw_.prxf[itube]      =   prxf[itube];
      mcraw_.thcal1[itube]    =   thcal1[itube];
    }
}

void mcraw_class::clearOutDST()
{
  memset(&mcraw_,0,sizeof(mcraw_dst_common));
  loadFromDST();
}


ClassImp (stps2_class)

stps2_class::stps2_class() : dstbank_class(STPS2_BANKID,STPS2_BANKVERSION)
{
  
}

stps2_class::~stps2_class()
{
  
}

void stps2_class::loadFromDST()
{
  
  Int_t ieye;
  
  maxeye = stps2_.maxeye;

  
  plog.resize(maxeye);
  rvec.resize(maxeye);
  rwalk.resize(maxeye);
  ang.resize(maxeye);
  aveTime.resize(maxeye);
  sigmaTime.resize(maxeye);
  avePhot.resize(maxeye);
  sigmaPhot.resize(maxeye);
  lifetime.resize(maxeye);
  totalLifetime.resize(maxeye);
  inTimeTubes.resize(maxeye);
  if_eye.resize(maxeye);
  upward.resize(maxeye);
  for (ieye = 0; ieye < maxeye; ieye ++)
    {
      plog[ieye]           =    stps2_.plog[ieye];
      rvec[ieye]           =    stps2_.rvec[ieye];
      rwalk[ieye]          =    stps2_.rwalk[ieye];
      ang[ieye]            =    stps2_.ang[ieye];
      aveTime[ieye]        =    stps2_.aveTime[ieye];
      sigmaTime[ieye]      =    stps2_.sigmaTime[ieye];
      avePhot[ieye]        =    stps2_.avePhot[ieye];
      sigmaPhot[ieye]      =    stps2_.sigmaPhot[ieye];
      lifetime[ieye]       =    stps2_.lifetime[ieye];
      totalLifetime[ieye]  =    stps2_.totalLifetime[ieye];
      inTimeTubes[ieye]    =    stps2_.inTimeTubes[ieye];
      if_eye[ieye]         =    stps2_.if_eye[ieye];
      upward[ieye]         =    stps2_.upward[ieye];
    }
  
  
}

void stps2_class::loadToDST()
{
  Int_t ieye;
  
  stps2_.maxeye = maxeye;
  
  for (ieye = 0; ieye < maxeye; ieye ++)
    {
      stps2_.plog[ieye]           =    plog[ieye];
      stps2_.rvec[ieye]           =    rvec[ieye];
      stps2_.rwalk[ieye]          =    rwalk[ieye];
      stps2_.ang[ieye]            =    ang[ieye];
      stps2_.aveTime[ieye]        =    aveTime[ieye];
      stps2_.sigmaTime[ieye]      =    sigmaTime[ieye];
      stps2_.avePhot[ieye]        =    avePhot[ieye];
      stps2_.sigmaPhot[ieye]      =    sigmaPhot[ieye];
      stps2_.lifetime[ieye]       =    lifetime[ieye];
      stps2_.totalLifetime[ieye]  =    totalLifetime[ieye];
      stps2_.inTimeTubes[ieye]    =    inTimeTubes[ieye];
      stps2_.if_eye[ieye]         =    if_eye[ieye];
      stps2_.upward[ieye]         =    upward[ieye];
    }
  
}
void stps2_class::clearOutDST()
{
  memset(&stps2_,0,sizeof(stps2_dst_common));
  loadFromDST();
}

ClassImp (stpln_class)

stpln_class::stpln_class() : dstbank_class(STPLN_BANKID,STPLN_BANKVERSION)
{
  
}
stpln_class::~stpln_class()
{
  
}
void stpln_class::loadFromDST()
{  
  Int_t i, ieye, imir, itube;  
  jday     =   stpln_.jday;
  jsec     =   stpln_.jsec;
  msec     =   stpln_.msec;
  neye     =   stpln_.neye;
  nmir     =   stpln_.nmir;
  ntube    =   stpln_.ntube;
  maxeye   =   stpln_.maxeye;
  if_eye.resize(maxeye);
  eyeid.resize(maxeye);
  eye_nmir.resize(maxeye);
  eye_ngmir.resize(maxeye);
  eye_ntube.resize(maxeye);
  eye_ngtube.resize(maxeye);
  n_ampwt.resize(maxeye);
  errn_ampwt.resize(maxeye);
  rmsdevpln.resize(maxeye);
  rmsdevtim.resize(maxeye);
  tracklength.resize(maxeye);
  crossingtime.resize(maxeye);
  ph_per_gtube.resize(maxeye);  
  for (ieye = 0; ieye < maxeye; ieye++)
    {
      if_eye[ieye]        =  stpln_.if_eye[ieye];
      eyeid[ieye]         =  stpln_.eyeid[ieye];
      eye_nmir[ieye]      =  stpln_.eye_nmir[ieye];
      eye_ngmir[ieye]     =  stpln_.eye_ngmir[ieye];
      eye_ntube[ieye]     =  stpln_.eye_ntube[ieye];
      eye_ngtube[ieye]    =  stpln_.eye_ngtube[ieye];      
      n_ampwt[ieye].resize(3);
      errn_ampwt[ieye].resize(6);
      for (i=0; i<3; i++)
	n_ampwt[ieye][i]       =  stpln_.n_ampwt[ieye][i];
      for (i=0; i<6; i++)
	errn_ampwt[ieye][i]    =  stpln_.errn_ampwt[ieye][i];
      rmsdevpln[ieye]     =  stpln_.rmsdevpln[ieye];
      rmsdevtim[ieye]     =  stpln_.rmsdevtim[ieye];
      tracklength[ieye]   =  stpln_.tracklength[ieye];
      crossingtime[ieye]  =  stpln_.crossingtime[ieye];
      ph_per_gtube[ieye]  =  stpln_.ph_per_gtube[ieye];
    }  
  mirid.resize(nmir);
  mir_eye.resize(nmir);
  mir_type.resize(nmir);
  mir_ngtube.resize(nmir);
  mirtime_ns.resize(nmir);
  for (imir = 0; imir < nmir; imir ++)
    {
      mirid[imir]       =   stpln_.mirid[imir];
      mir_eye[imir]     =   stpln_.mir_eye[imir];
      mir_type[imir]    =   stpln_.mir_type[imir];
      mir_ngtube[imir]  =   stpln_.mir_ngtube[imir];
      mirtime_ns[imir]  =   stpln_.mirtime_ns[imir];
    }
  ig.resize(ntube);
  tube_eye.resize(ntube);
  saturated.resize(ntube);
  mir_tube_id.resize(ntube);
  for (itube = 0; itube < ntube; itube++)
    {
      ig[itube]          =   stpln_.ig[itube];
      tube_eye[itube]    =   stpln_.tube_eye [itube];
#if STPLN_BANKVERSION >= 2
      saturated[itube]   =   stpln_.saturated[itube];
      mir_tube_id[itube] =   stpln_.mir_tube_id[itube];
#else
      saturated[itube]   =   0;
      mir_tube_id[itube] =   0;
#endif
    }
}
void stpln_class::loadToDST()
{
  Int_t i, ieye, imir, itube; 
  stpln_.jday     =   jday;
  stpln_.jsec     =   jsec;
  stpln_.msec     =   msec;
  stpln_.neye     =   neye;
  stpln_.nmir     =   nmir;
  stpln_.ntube    =   ntube;
  stpln_.maxeye   =   maxeye;  
  for (ieye = 0; ieye < maxeye; ieye++)
    {
      stpln_.if_eye[ieye]        =  if_eye[ieye];
      stpln_.eyeid[ieye]         =  eyeid[ieye];
      stpln_.eye_nmir[ieye]      =  eye_nmir[ieye];
      stpln_.eye_ngmir[ieye]     =  eye_ngmir[ieye];
      stpln_.eye_ntube[ieye]     =  eye_ntube[ieye];
      stpln_.eye_ngtube[ieye]    =  eye_ngtube[ieye];      
      for (i=0; i<(int)n_ampwt[ieye].size(); i++)
	stpln_.n_ampwt[ieye][i]       =  n_ampwt[ieye][i];
      for (i=0; i<(int)errn_ampwt[ieye].size(); i++)	    
	stpln_.errn_ampwt[ieye][i]    =  errn_ampwt[ieye][i];
      stpln_.rmsdevpln[ieye]     =  rmsdevpln[ieye];
      stpln_.rmsdevtim[ieye]     =  rmsdevtim[ieye];
      stpln_.tracklength[ieye]   =  tracklength[ieye];
      stpln_.crossingtime[ieye]  =  crossingtime[ieye];
      stpln_.ph_per_gtube[ieye]  =  ph_per_gtube[ieye];
    } 
  for (imir = 0; imir < nmir; imir ++)
    {
      stpln_.mirid[imir]       =   mirid[imir];
      stpln_.mir_eye[imir]     =   mir_eye[imir];
      stpln_.mir_type[imir]    =   mir_type[imir];
      stpln_.mir_ngtube[imir]  =   mir_ngtube[imir];
      stpln_.mirtime_ns[imir]  =   mirtime_ns[imir];
    }  
  for (itube = 0; itube < ntube; itube++)
    {
      stpln_.ig[itube]         =   ig[itube];
      stpln_.tube_eye[itube]   =   tube_eye [itube];
#if STPLN_BANKVERSION >= 2
      stpln_.saturated[itube]   = saturated[itube];
      stpln_.mir_tube_id[itube] = mir_tube_id[itube];
#endif
    }
}

void stpln_class::clearOutDST()
{
  memset(&stpln_,0,sizeof(stpln_dst_common));
  loadFromDST();
}


ClassImp (hctim_class)

#ifdef HCTIM_BANKID
hctim_class::hctim_class() : dstbank_class(HCTIM_BANKID,HCTIM_BANKVERSION) {;}
hctim_class::~hctim_class() {;}
void hctim_class::loadFromDST()
{
  mchi2.resize(HCTIM_MAXFIT);
  rchi2.resize(HCTIM_MAXFIT);
  lchi2.resize(HCTIM_MAXFIT);
  mrp.resize(HCTIM_MAXFIT);
  rrp.resize(HCTIM_MAXFIT);
  lrp.resize(HCTIM_MAXFIT);
  mpsi.resize(HCTIM_MAXFIT);
  rpsi.resize(HCTIM_MAXFIT);
  lpsi.resize(HCTIM_MAXFIT);
  mthe.resize(HCTIM_MAXFIT);
  rthe.resize(HCTIM_MAXFIT);
  lthe.resize(HCTIM_MAXFIT);
  mphi.resize(HCTIM_MAXFIT);
  rphi.resize(HCTIM_MAXFIT);
  lphi.resize(HCTIM_MAXFIT);
  failmode.resize(HCTIM_MAXFIT);
  timinfo.resize(HCTIM_MAXFIT);
  jday.resize(HCTIM_MAXFIT);
  jsec.resize(HCTIM_MAXFIT);
  msec.resize(HCTIM_MAXFIT);
  ntube.resize(HCTIM_MAXFIT);
  nmir.resize(HCTIM_MAXFIT);
  mtkv.resize(HCTIM_MAXFIT);
  rtkv.resize(HCTIM_MAXFIT);
  ltkv.resize(HCTIM_MAXFIT);
  mrpv.resize(HCTIM_MAXFIT);
  rrpv.resize(HCTIM_MAXFIT);
  lrpv.resize(HCTIM_MAXFIT);
  mrpuv.resize(HCTIM_MAXFIT);
  rrpuv.resize(HCTIM_MAXFIT);
  lrpuv.resize(HCTIM_MAXFIT);
  mshwn.resize(HCTIM_MAXFIT);
  rshwn.resize(HCTIM_MAXFIT);
  lshwn.resize(HCTIM_MAXFIT);
  mcore.resize(HCTIM_MAXFIT);
  rcore.resize(HCTIM_MAXFIT);
  lcore.resize(HCTIM_MAXFIT);
  tubemir.resize(HCTIM_MAXFIT);
  tube.resize(HCTIM_MAXFIT);	
  ig.resize(HCTIM_MAXFIT);
  time.resize(HCTIM_MAXFIT);
  timefit.resize(HCTIM_MAXFIT);
  thetb.resize(HCTIM_MAXFIT);
  sgmt.resize(HCTIM_MAXFIT);
  asx.resize(HCTIM_MAXFIT);
  asy.resize(HCTIM_MAXFIT);
  asz.resize(HCTIM_MAXFIT);
  mir.resize(HCTIM_MAXFIT);
  mirntube.resize(HCTIM_MAXFIT);
  for (Int_t ifit = 0; ifit < (Int_t)mchi2.size(); ifit ++ )
    {   
      mchi2[ifit] = hctim_.mchi2[ifit]; 
      rchi2[ifit] = hctim_.rchi2[ifit]; 
      lchi2[ifit] = hctim_.lchi2[ifit];  
      mrp[ifit] = hctim_.mrp[ifit]; 
      rrp[ifit] = hctim_.rrp[ifit]; 
      lrp[ifit] = hctim_.lrp[ifit];
      mpsi[ifit] = hctim_.mpsi[ifit]; 
      rpsi[ifit] = hctim_.rpsi[ifit]; 
      lpsi[ifit] = hctim_.lpsi[ifit];
      mthe[ifit] = hctim_.mthe[ifit]; 
      rthe[ifit] = hctim_.rthe[ifit]; 
      lthe[ifit] = hctim_.lthe[ifit];
      mphi[ifit] = hctim_.mphi[ifit]; 
      rphi[ifit] = hctim_.rphi[ifit]; 
      lphi[ifit] = hctim_.lphi[ifit];
      failmode[ifit] = hctim_.failmode[ifit];  
      timinfo[ifit] = hctim_.timinfo[ifit];      
      jday[ifit] = hctim_.jday[ifit];
      jsec[ifit] = hctim_.jsec[ifit];
      msec[ifit] = hctim_.msec[ifit]; 
      ntube[ifit] = hctim_.ntube[ifit];
      nmir[ifit] = hctim_.nmir[ifit];
      mtkv[ifit].resize(3);
      rtkv[ifit].resize(3);
      ltkv[ifit].resize(3);
      mrpv[ifit].resize(3);
      rrpv[ifit].resize(3);
      lrpv[ifit].resize(3);
      mrpuv[ifit].resize(3);
      rrpuv[ifit].resize(3);
      lrpuv[ifit].resize(3);
      mshwn[ifit].resize(3);
      rshwn[ifit].resize(3);
      lshwn[ifit].resize(3);
      mcore[ifit].resize(3);
      rcore[ifit].resize(3);
      lcore[ifit].resize(3);
      for (Int_t k = 0; k < (Int_t)mtkv[ifit].size(); k++)
	{
	  mtkv[ifit][k] = hctim_.mtkv[ifit][k]; 
	  rtkv[ifit][k] = hctim_.rtkv[ifit][k]; 
	  ltkv[ifit][k] = hctim_.ltkv[ifit][k];
	  mrpv[ifit][k] = hctim_.mrpv[ifit][k]; 
	  rrpv[ifit][k] = hctim_.rrpv[ifit][k]; 
	  lrpv[ifit][k] = hctim_.lrpv[ifit][k];
	  mrpuv[ifit][k] = hctim_.mrpuv[ifit][k]; 
	  rrpuv[ifit][k] = hctim_.rrpuv[ifit][k]; 
	  lrpuv[ifit][k] = hctim_.lrpuv[ifit][k];
	  mshwn[ifit][k] = hctim_.mshwn[ifit][k]; 
	  rshwn[ifit][k] = hctim_.rshwn[ifit][k]; 
	  lshwn[ifit][k] = hctim_.lshwn[ifit][k];
	  mcore[ifit][k] = hctim_.mcore[ifit][k]; 
	  rcore[ifit][k] = hctim_.rcore[ifit][k]; 
	  lcore[ifit][k] = hctim_.lcore[ifit][k];
	}
      tubemir[ifit].resize(ntube[ifit]);     
      tube[ifit].resize(ntube[ifit]);	
      ig[ifit].resize(ntube[ifit]);
      time[ifit].resize(ntube[ifit]);
      timefit[ifit].resize(ntube[ifit]);
      thetb[ifit].resize(ntube[ifit]);
      sgmt[ifit].resize(ntube[ifit]);
      asx[ifit].resize(ntube[ifit]);
      asy[ifit].resize(ntube[ifit]);
      asz[ifit].resize(ntube[ifit]);
      for (Int_t itube = 0; itube < (Int_t)tubemir[ifit].size(); itube ++)
	{
	  tubemir[ifit][itube] = hctim_.tubemir[ifit][itube];
	  tube[ifit][itube] = hctim_.tube[ifit][itube];	
	  ig[ifit][itube] = hctim_.ig[ifit][itube];
	  time[ifit][itube] = hctim_.time[ifit][itube];
	  timefit[ifit][itube] = hctim_.timefit[ifit][itube];
	  thetb[ifit][itube] = hctim_.thetb[ifit][itube];
	  sgmt[ifit][itube] = hctim_.sgmt[ifit][itube];
	  asx[ifit][itube] = hctim_.asx[ifit][itube];
	  asy[ifit][itube] = hctim_.asy[ifit][itube];
	  asz[ifit][itube] = hctim_.asz[ifit][itube];
	}
      mir[ifit].resize(nmir[ifit]);
      mirntube[ifit].resize(nmir[ifit]);
      for (Int_t imir = 0; imir < (Int_t)mir[ifit].size(); imir ++)
	{  
	  mir[ifit][imir] = hctim_.mir[ifit][imir];
	  mirntube[ifit][imir] = hctim_.mirntube[ifit][imir];
	}
    }
}
void hctim_class::loadToDST()
{
  for (Int_t ifit = 0; ifit < (Int_t)mchi2.size(); ifit ++ )
    {   
      hctim_.mchi2[ifit] = mchi2[ifit]; 
      hctim_.rchi2[ifit] = rchi2[ifit]; 
      hctim_.lchi2[ifit] = lchi2[ifit];  
      hctim_.mrp[ifit] = mrp[ifit]; 
      hctim_.rrp[ifit] = rrp[ifit]; 
      hctim_.lrp[ifit] = lrp[ifit];
      hctim_.mpsi[ifit] = mpsi[ifit]; 
      hctim_.rpsi[ifit] = rpsi[ifit]; 
      hctim_.lpsi[ifit] = lpsi[ifit];
      hctim_.mthe[ifit] = mthe[ifit]; 
      hctim_.rthe[ifit] = rthe[ifit]; 
      hctim_.lthe[ifit] = lthe[ifit];
      hctim_.mphi[ifit] = mphi[ifit]; 
      hctim_.rphi[ifit] = rphi[ifit]; 
      hctim_.lphi[ifit] = lphi[ifit];
      hctim_.failmode[ifit] = failmode[ifit];  
      hctim_.timinfo[ifit] = timinfo[ifit];      
      hctim_.jday[ifit] = jday[ifit];
      hctim_.jsec[ifit] = jsec[ifit];
      hctim_.msec[ifit] = msec[ifit]; 
      hctim_.ntube[ifit] = ntube[ifit];
      hctim_.nmir[ifit] = nmir[ifit];
      for (Int_t k = 0; k < (Int_t)mtkv[ifit].size(); k++)
	{
	  hctim_.mtkv[ifit][k] = mtkv[ifit][k]; 
	  hctim_.rtkv[ifit][k] = rtkv[ifit][k]; 
	  hctim_.ltkv[ifit][k] = ltkv[ifit][k];
	  hctim_.mrpv[ifit][k] = mrpv[ifit][k]; 
	  hctim_.rrpv[ifit][k] = rrpv[ifit][k]; 
	  hctim_.lrpv[ifit][k] = lrpv[ifit][k];
	  hctim_.mrpuv[ifit][k] = mrpuv[ifit][k]; 
	  hctim_.rrpuv[ifit][k] = rrpuv[ifit][k]; 
	  hctim_.lrpuv[ifit][k] = lrpuv[ifit][k];
	  hctim_.mshwn[ifit][k] = mshwn[ifit][k]; 
	  hctim_.rshwn[ifit][k] = rshwn[ifit][k]; 
	  hctim_.lshwn[ifit][k] = lshwn[ifit][k];
	  hctim_.mcore[ifit][k] = mcore[ifit][k]; 
	  hctim_.rcore[ifit][k] = rcore[ifit][k]; 
	  hctim_.lcore[ifit][k] = lcore[ifit][k];
	}
      for (Int_t itube = 0; itube < (Int_t)tubemir[ifit].size(); itube ++)
	{
	  hctim_.tubemir[ifit][itube] = tubemir[ifit][itube];
	  hctim_.tube[ifit][itube] = tube[ifit][itube];	
	  hctim_.ig[ifit][itube] = ig[ifit][itube];
	  hctim_.time[ifit][itube] = time[ifit][itube];
	  hctim_.timefit[ifit][itube] = timefit[ifit][itube];
	  hctim_.thetb[ifit][itube] = thetb[ifit][itube];
	  hctim_.sgmt[ifit][itube] = sgmt[ifit][itube];
	  hctim_.asx[ifit][itube] = asx[ifit][itube];
	  hctim_.asy[ifit][itube] = asy[ifit][itube];
	  hctim_.asz[ifit][itube] = asz[ifit][itube];
	}
      for (Int_t imir = 0; imir < (Int_t)mir[ifit].size(); imir ++)
	{  
	  hctim_.mir[ifit][imir] = mir[ifit][imir];
	  hctim_.mirntube[ifit][imir] = mirntube[ifit][imir];
	}
    }
}
void hctim_class::clearOutDST()
{
  memset(&hctim_,0,sizeof(hctim_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(hctim);
#endif

ClassImp (hcbin_class)

#ifdef HCBIN_BANKID
hcbin_class::hcbin_class() : dstbank_class(HCBIN_BANKID,HCBIN_BANKVERSION) {;}
hcbin_class::~hcbin_class() {;}
void hcbin_class::loadFromDST()
{  
  bvx.resize(HCBIN_MAXFIT);
  bvy.resize(HCBIN_MAXFIT);
  bvz.resize(HCBIN_MAXFIT);
  bsz.resize(HCBIN_MAXFIT);
  sig.resize(HCBIN_MAXFIT);
  sigerr.resize(HCBIN_MAXFIT);
  cfc.resize(HCBIN_MAXFIT);
  ig.resize(HCBIN_MAXFIT);
  nbin.resize(HCBIN_MAXFIT);
  failmode.resize(HCBIN_MAXFIT);
  bininfo.resize(HCBIN_MAXFIT);
  jday.resize(HCBIN_MAXFIT);
  jsec.resize(HCBIN_MAXFIT);
  msec.resize(HCBIN_MAXFIT);
  for(Int_t ifit = 0; ifit < (Int_t)bvx.size(); ifit++)
    {
      bvx[ifit].resize(HCBIN_MAXBIN);
      bvy[ifit].resize(HCBIN_MAXBIN);
      bvz[ifit].resize(HCBIN_MAXBIN);
      bsz[ifit].resize(HCBIN_MAXBIN);
      sig[ifit].resize(HCBIN_MAXBIN);
      sigerr[ifit].resize(HCBIN_MAXBIN);
      cfc[ifit].resize(HCBIN_MAXBIN);
      ig[ifit].resize(HCBIN_MAXBIN);
      for(Int_t ibin = 0; ibin < (Int_t)bvx[ifit].size(); ibin++)
	{
	  bvx[ifit][ibin] = hcbin_.bvx[ifit][ibin];
	  bvy[ifit][ibin] = hcbin_.bvy[ifit][ibin];
	  bvz[ifit][ibin] = hcbin_.bvz[ifit][ibin];
	  bsz[ifit][ibin] = hcbin_.bsz[ifit][ibin];
	  sig[ifit][ibin] = hcbin_.sig[ifit][ibin];
	  sigerr[ifit][ibin] = hcbin_.sigerr[ifit][ibin];
	  cfc[ifit][ibin] = hcbin_.cfc[ifit][ibin];
	  ig[ifit][ibin] = hcbin_.ig[ifit][ibin];
	}
      nbin[ifit] = hcbin_.nbin[ifit];
      failmode[ifit] = hcbin_.failmode[ifit];
      bininfo[ifit] = hcbin_.bininfo[ifit];
      jday[ifit] = hcbin_.jday[ifit];
      jsec[ifit] = hcbin_.jsec[ifit];
      msec[ifit] = hcbin_.msec[ifit];
    }
}
void hcbin_class::loadToDST()
{  
  for(Int_t ifit = 0; ifit < (Int_t)bvx.size(); ifit++)
    { 
      for(Int_t ibin = 0; ibin < (Int_t)bvx[ifit].size(); ibin++)
	{
	  hcbin_.bvx[ifit][ibin] = bvx[ifit][ibin];
	  hcbin_.bvy[ifit][ibin] = bvy[ifit][ibin];
	  hcbin_.bvz[ifit][ibin] = bvz[ifit][ibin];
	  hcbin_.bsz[ifit][ibin] = bsz[ifit][ibin];
	  hcbin_.sig[ifit][ibin] = sig[ifit][ibin];
	  hcbin_.sigerr[ifit][ibin] = sigerr[ifit][ibin];
	  hcbin_.cfc[ifit][ibin] = cfc[ifit][ibin];
	  hcbin_.ig[ifit][ibin] = ig[ifit][ibin];
	}
      hcbin_.nbin[ifit] = nbin[ifit];
      hcbin_.failmode[ifit] = failmode[ifit];
      hcbin_.bininfo[ifit] = bininfo[ifit];
      hcbin_.jday[ifit] = jday[ifit];
      hcbin_.jsec[ifit] = jsec[ifit];
      hcbin_.msec[ifit] = msec[ifit];
    }
}
void hcbin_class::clearOutDST()
{
  memset(&hcbin_,0,sizeof(hcbin_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(hcbin);
#endif

ClassImp (prfc_class)

#ifdef PRFC_BANKID
prfc_class::prfc_class() : dstbank_class(PRFC_BANKID,PRFC_BANKVERSION) {;}
prfc_class::~prfc_class() {;}
void prfc_class::loadFromDST()
{
  chi2.resize(PRFC_MAXFIT);
  szmx.resize(PRFC_MAXFIT);
  dszmx.resize(PRFC_MAXFIT);
  rszmx.resize(PRFC_MAXFIT);
  lszmx.resize(PRFC_MAXFIT);
  tszmx.resize(PRFC_MAXFIT);
  xm.resize(PRFC_MAXFIT);
  dxm.resize(PRFC_MAXFIT);
  rxm.resize(PRFC_MAXFIT);
  lxm.resize(PRFC_MAXFIT);
  txm.resize(PRFC_MAXFIT);
  x0.resize(PRFC_MAXFIT);
  dx0.resize(PRFC_MAXFIT);
  rx0.resize(PRFC_MAXFIT);
  lx0.resize(PRFC_MAXFIT);
  tx0.resize(PRFC_MAXFIT);
  lambda.resize(PRFC_MAXFIT);
  dlambda.resize(PRFC_MAXFIT);
  rlambda.resize(PRFC_MAXFIT);
  llambda.resize(PRFC_MAXFIT);
  tlambda.resize(PRFC_MAXFIT);
  eng.resize(PRFC_MAXFIT);
  deng.resize(PRFC_MAXFIT);
  reng.resize(PRFC_MAXFIT);
  leng.resize(PRFC_MAXFIT);
  teng.resize(PRFC_MAXFIT);
  dep.resize(PRFC_MAXFIT);
  gm.resize(PRFC_MAXFIT);
  scin.resize(PRFC_MAXFIT);
  rayl.resize(PRFC_MAXFIT);
  aero.resize(PRFC_MAXFIT);
  crnk.resize(PRFC_MAXFIT);
  sigmc.resize(PRFC_MAXFIT);
  sig.resize(PRFC_MAXFIT);
  ig.resize(PRFC_MAXFIT);
  mxel.resize(PRFC_MAXFIT);
  nel.resize(PRFC_MAXFIT);
  mor.resize(PRFC_MAXFIT);
  pflinfo.resize(PRFC_MAXFIT);
  bininfo.resize(PRFC_MAXFIT);
  mtxinfo.resize(PRFC_MAXFIT);
  failmode.resize(PRFC_MAXFIT);
  nbin.resize(PRFC_MAXFIT);
  traj_source.resize(PRFC_MAXFIT);
  errstat.resize(PRFC_MAXFIT);
  ndf.resize(PRFC_MAXFIT);
  for(Int_t ifit = 0; ifit < (Int_t)chi2.size(); ifit++)
    {
      chi2[ifit] = prfc_.chi2[ifit];
      szmx[ifit] = prfc_.szmx[ifit];
      dszmx[ifit] = prfc_.dszmx[ifit];
      rszmx[ifit] = prfc_.rszmx[ifit];
      lszmx[ifit] = prfc_.lszmx[ifit];
      tszmx[ifit] = prfc_.tszmx[ifit];
      xm[ifit] = prfc_.xm[ifit];
      dxm[ifit] = prfc_.dxm[ifit];
      rxm[ifit] = prfc_.rxm[ifit];
      lxm[ifit] = prfc_.lxm[ifit];
      txm[ifit] = prfc_.txm[ifit];
      x0[ifit] = prfc_.x0[ifit];
      dx0[ifit] = prfc_.dx0[ifit];
      rx0[ifit] = prfc_.rx0[ifit];
      lx0[ifit] = prfc_.lx0[ifit];
      tx0[ifit] = prfc_.tx0[ifit];
      lambda[ifit] = prfc_.lambda[ifit];
      dlambda[ifit] = prfc_.dlambda[ifit];
      rlambda[ifit] = prfc_.rlambda[ifit];
      llambda[ifit] = prfc_.llambda[ifit];
      tlambda[ifit] = prfc_.tlambda[ifit];
      eng[ifit] = prfc_.eng[ifit];
      deng[ifit] = prfc_.deng[ifit];
      reng[ifit] = prfc_.reng[ifit];
      leng[ifit] = prfc_.leng[ifit];
      teng[ifit] = prfc_.teng[ifit];
      dep[ifit].resize(PRFC_MAXBIN);
      gm[ifit].resize(PRFC_MAXBIN);
      scin[ifit].resize(PRFC_MAXBIN);
      rayl[ifit].resize(PRFC_MAXBIN);
      aero[ifit].resize(PRFC_MAXBIN);
      crnk[ifit].resize(PRFC_MAXBIN);
      sigmc[ifit].resize(PRFC_MAXBIN);
      sig[ifit].resize(PRFC_MAXBIN);
      ig[ifit].resize(PRFC_MAXBIN);
      for(Int_t ibin = 0; ibin < (Int_t)dep[ifit].size(); ibin++)
	{
	  dep[ifit][ibin] = prfc_.dep[ifit][ibin];
	  gm[ifit][ibin] = prfc_.gm[ifit][ibin];
	  scin[ifit][ibin] = prfc_.scin[ifit][ibin];
	  rayl[ifit][ibin] = prfc_.rayl[ifit][ibin];
	  aero[ifit][ibin] = prfc_.aero[ifit][ibin];
	  crnk[ifit][ibin] = prfc_.crnk[ifit][ibin];
	  sigmc[ifit][ibin] = prfc_.sigmc[ifit][ibin];
	  sig[ifit][ibin] = prfc_.sig[ifit][ibin];
	  ig[ifit][ibin] = prfc_.ig[ifit][ibin];
	}
      mxel[ifit].resize(PRFC_MAXMEL);
      for(Int_t imel = 0; imel < (Int_t)mxel[ifit].size(); imel++)
	mxel[ifit][imel] = prfc_.mxel[ifit][imel];
      nel[ifit] = prfc_.nel[ifit];
      mor[ifit] = prfc_.mor[ifit];
      pflinfo[ifit] = prfc_.pflinfo[ifit];
      bininfo[ifit] = prfc_.bininfo[ifit];
      mtxinfo[ifit] = prfc_.mtxinfo[ifit];
      failmode[ifit] = prfc_.failmode[ifit];
      nbin[ifit] = prfc_.nbin[ifit];
      traj_source[ifit] = prfc_.traj_source[ifit];
      errstat[ifit] = prfc_.errstat[ifit];
      ndf[ifit] = prfc_.ndf[ifit];
    }
}
void prfc_class::loadToDST()
{
  for(Int_t ifit = 0; ifit < PRFC_MAXFIT; ifit++)
    {
      prfc_.chi2[ifit] = chi2[ifit];
      prfc_.szmx[ifit] = szmx[ifit];
      prfc_.dszmx[ifit] = dszmx[ifit];
      prfc_.rszmx[ifit] = rszmx[ifit];
      prfc_.lszmx[ifit] = lszmx[ifit];
      prfc_.tszmx[ifit] = tszmx[ifit];
      prfc_.xm[ifit] = xm[ifit];
      prfc_.dxm[ifit] = dxm[ifit];
      prfc_.rxm[ifit] = rxm[ifit];
      prfc_.lxm[ifit] = lxm[ifit];
      prfc_.txm[ifit] = txm[ifit];
      prfc_.x0[ifit] = x0[ifit];
      prfc_.dx0[ifit] = dx0[ifit];
      prfc_.rx0[ifit] = rx0[ifit];
      prfc_.lx0[ifit] = lx0[ifit];
      prfc_.tx0[ifit] = tx0[ifit];
      prfc_.lambda[ifit] = lambda[ifit];
      prfc_.dlambda[ifit] = dlambda[ifit];
      prfc_.rlambda[ifit] = rlambda[ifit];
      prfc_.llambda[ifit] = llambda[ifit];
      prfc_.tlambda[ifit] = tlambda[ifit];
      prfc_.eng[ifit] = eng[ifit];
      prfc_.deng[ifit] = deng[ifit];
      prfc_.reng[ifit] = reng[ifit];
      prfc_.leng[ifit] = leng[ifit];
      prfc_.teng[ifit] = teng[ifit];
      for(Int_t ibin = 0; ibin < (Int_t)dep[ifit].size(); ibin++)
	{
	  prfc_.dep[ifit][ibin]   = dep[ifit][ibin];
	  prfc_.gm[ifit][ibin]    = gm[ifit][ibin];
	  prfc_.scin[ifit][ibin]  = scin[ifit][ibin];
	  prfc_.rayl[ifit][ibin]  = rayl[ifit][ibin];
	  prfc_.aero[ifit][ibin]  = aero[ifit][ibin];
	  prfc_.crnk[ifit][ibin]  = crnk[ifit][ibin];
	  prfc_.sigmc[ifit][ibin] = sigmc[ifit][ibin];
	  prfc_.sig[ifit][ibin]   = sig[ifit][ibin];
	  prfc_.ig[ifit][ibin]    = ig[ifit][ibin];
	}
      for(Int_t imel = 0; imel < (int)mxel[ifit].size(); imel++)
	prfc_.mxel[ifit][imel] = mxel[ifit][imel];
      prfc_.nel[ifit] = nel[ifit];
      prfc_.mor[ifit] = mor[ifit];
      prfc_.pflinfo[ifit] = pflinfo[ifit];
      prfc_.bininfo[ifit] = bininfo[ifit];
      prfc_.mtxinfo[ifit] = mtxinfo[ifit];
      prfc_.failmode[ifit] = failmode[ifit];
      prfc_.nbin[ifit] = nbin[ifit];
      prfc_.traj_source[ifit] = traj_source[ifit];
      prfc_.errstat[ifit] = errstat[ifit];
      prfc_.ndf[ifit] = ndf[ifit];
    }
}
void prfc_class::clearOutDST()
{
  memset(&prfc_,0,sizeof(prfc_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(prfc);
#endif


///////////////////// fdatmos_param_class ////////////////////
ClassImp(fdatmos_param_class)

#ifdef FDATMOS_PARAM_BANKID
fdatmos_param_class::fdatmos_param_class() : dstbank_class(FDATMOS_PARAM_BANKID,FDATMOS_PARAM_BANKVERSION)  
{
  used_bankid = FDATMOS_PARAM_BANKID;
}
fdatmos_param_class::~fdatmos_param_class() {;}
void fdatmos_param_class::loadFromDST()
{  
  fdatmos_param_dst_common* ptr_to_dst_bank = &fdatmos_param_;
#ifdef GDAS_BANKID
  if(used_bankid == GDAS_BANKID)
    ptr_to_dst_bank = &gdas_;
#endif
  uniqID = ptr_to_dst_bank->uniqID;
  dateFrom = ptr_to_dst_bank->dateFrom;
  dateTo = ptr_to_dst_bank->dateTo;
  nItem = ptr_to_dst_bank->nItem;
  height.resize(nItem);
  pressure.resize(nItem);
  pressureError.resize(nItem);
  temperature.resize(nItem);
  temperatureError.resize(nItem);
  dewPoint.resize(nItem);
  dewPointError.resize(nItem);
  for(Int_t i=0; i<nItem; i++)
    {
      height[i] = ptr_to_dst_bank->height[i];
      pressure[i] = ptr_to_dst_bank->pressure[i];
      pressureError[i] = ptr_to_dst_bank->pressureError[i];
      temperature[i] = ptr_to_dst_bank->temperature[i];
      temperatureError[i] = ptr_to_dst_bank->temperatureError[i];
      dewPoint[i] = ptr_to_dst_bank->dewPoint[i];
      dewPointError[i] = ptr_to_dst_bank->dewPointError[i];
    }
}
void fdatmos_param_class::loadToDST()
{
  fdatmos_param_dst_common* ptr_to_dst_bank = &fdatmos_param_;
#ifdef GDAS_BANKID
  if(used_bankid == GDAS_BANKID)
    ptr_to_dst_bank = &gdas_;
#endif
  ptr_to_dst_bank->uniqID = uniqID;
  ptr_to_dst_bank->dateFrom = dateFrom;
  ptr_to_dst_bank->dateTo = dateTo;
  ptr_to_dst_bank->nItem = nItem;
  for(Int_t i=0; i<(Int_t)height.size(); i++)
    {
      ptr_to_dst_bank->height[i] = height[i];
      ptr_to_dst_bank->pressure[i] = pressure[i];
      ptr_to_dst_bank->pressureError[i] = pressureError[i];
      ptr_to_dst_bank->temperature[i] = temperature[i];
      ptr_to_dst_bank->temperatureError[i] = temperatureError[i];
      ptr_to_dst_bank->dewPoint[i] = dewPoint[i];
      ptr_to_dst_bank->dewPointError[i] = dewPointError[i];
    }
}
void fdatmos_param_class::clearOutDST()
{
  fdatmos_param_dst_common* ptr_to_dst_bank = &fdatmos_param_;
#ifdef GDAS_BANKID
  if(used_bankid == GDAS_BANKID)
    ptr_to_dst_bank = &gdas_;
#endif
  memset(ptr_to_dst_bank,0,sizeof(fdatmos_param_dst_common));
  loadFromDST();
}
#else
_dstbank_not_implemented_(fdatmos_param);
#endif




///////////////////// gdas_class ////////////////////
ClassImp(gdas_class)

#ifdef GDAS_BANKID
gdas_class::gdas_class() 
{
  used_bankid     = GDAS_BANKID;
  dstbank_id      = GDAS_BANKID;
  dstbank_version = GDAS_BANKVERSION; 
}
gdas_class::~gdas_class() 
{
  ;
}
#else
_dstbank_empty_constructor_destructor_(gdas);
#endif
