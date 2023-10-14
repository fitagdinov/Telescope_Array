#include "pass1plot.h"

using namespace std;

ClassImp(md_pcgf_class)

md_pcgf_class::md_pcgf_class() 
{
  for (Int_t i=0; i<3; i++)
    imin[i] = 0;
  for (Int_t i=0; i < HCTIM_MAXFIT; i++)
    {
      igfit[i]    = 0;
      avgcfc[i]   = 0;
      ndft_fit[i] = 0;
    }
  ig        = 0;
  tot_npe   = 0;
  nbins     = 0;
  ng        = 0;
  sum       = 0;
  FIT_START = 10;
  c2c       = 0;
  c2p       = 0;
  c2t       = 0;
  ndfp      = 0;
  ndft      = 0;
  normp     = 0;
  normt     = 0;
}

md_pcgf_class::~md_pcgf_class() { ; }

bool pass1plot::get_md_pcgf()
{
  
  // where the constrained profile fits start in the utafd DST banks
  const Int_t FIT_START = 10;

  if(hraw1->ntube < 1)
    {
      fprintf(stderr,"error: get_md_pcgf: hraw1 is missing or not filled; forgot init_md_plane_geom ?\n");
      return false;
    }
  if (!have_stpln || stpln->ntube < 1)
    {
      fprintf(stderr,"error: get_md_pcgf: stpln is missing or empty \n");
      return false;
    }
  if (!have_hctim)
    {
      fprintf(stderr,"error: get_md_pcgf: hctim is missing\n");
      return false;
    }
  if (!have_hcbin)
    {
      fprintf(stderr,"error: get_md_pcgf: hcbin is missing\n");
      return false;
    }
  if (!have_prfc)
    {
      fprintf(stderr,"error: get_md_pcgf: prfc is missing\n");
      return false;
    }
  
 
  /////// GOODNESS OF FIT/ IMINC DETERMINATION (below) //////////////////////
  
  Int_t ng = 0;          // number of good tubes as determined by plane-fit
  Double_t  tot_npe = 0; // total number of npe's for the event
  for(Int_t itube = 0; itube < hraw1->ntube; itube++)
    {
      if(stpln->ig[itube] == 1)
	{
	  tot_npe=tot_npe+hraw1->prxf[itube];
	  ng++;
	}
    }
  
  
  // ig is determined to be good if there was a triggered mirror 
  // according to the function in strz5_plug_in_data.c: strz5_fill_struct_stat
  // on line 304.
  Int_t ig = (hraw1->nmir > 0);
  
  
  
  Int_t igfit[HCTIM_MAXFIT];            // determination of a good fit (fit good)
  Double_t avgcfc[HCTIM_MAXFIT];        // determination of a good fit (avg. correction fact.)
  Double_t ndft_fit[HCTIM_MAXFIT];      // # degrees of freedom in timing fit
 
  Int_t    nbins = 0;  // number of bins in the best fit
  Double_t sum   = 0;  // adders: avgcfc in the best fit
  for(Int_t ifit=0; ifit < HCTIM_MAXFIT; ifit++)
    {

      // determine if particular fit was good
      if((hctim->timinfo[ifit] == HCTIM_TIMINFO_UNUSED) ||
	 (hctim->failmode[ifit] != SUCCESS))
	igfit[ifit] = 0;
      else
	igfit[ifit] = 1;
      
      // determine average correction factor for particular fit
      nbins = 0;
      sum = 0;
      for(Int_t ibin=0; ibin<hcbin->nbin[ifit]; ibin++)
	{
	  if(hcbin->ig[ifit][ibin] != 1) continue;
	  sum += hcbin->cfc[ifit][ibin];
	  nbins++;
	}
      if(nbins > 0) sum /= (Double_t) nbins;
      avgcfc[ifit] = sum;

      // number of degrees of freedom for the timing fit
      ndft_fit[ifit] = (Double_t)(ng - 3);

    }

  // determine imin[] values: combined, profile, timing
  Double_t c2p = 1.0e20;   // chi2 of the profile fit
  Double_t c2t = 1.0e20;   // chi2 of the time fit
  Int_t imin[3] = {0,0,0}; // the minimum chi-squares of combined, profile, and timing
  Double_t ndfp = 0;       // profile fit number of degrees of freedom
  Double_t ndft = 0;       // time fit number of degrees of freedom
  for(Int_t ifit = FIT_START; ifit < HCTIM_MAXFIT; ifit++)
    {
      if(igfit[ifit] != 1) continue;
	      
      if(prfc->chi2[ifit] < c2p)
	{
	  c2p = prfc->chi2[ifit];
	  ndfp = prfc->ndf[ifit];
	  imin[1] = ifit;
	}
      if(hctim->mchi2[ifit] < c2t)
	{
	  c2t = hctim->mchi2[ifit];
	  ndft = ndft_fit[ifit];
	  imin[2] = ifit;
	}
    }
  Double_t normp = ( c2p == 0.0 ) ? 1.0 : ndfp / c2p; // normalization for profile fit chi2
  Double_t normt = ( c2t == 0.0 ) ? 1.0 : ndft / c2t; // normalization for time fit chi2
  Double_t c2c = 1.0e20;    // combined profile and time fit chi2
  
  for(Int_t ifit = FIT_START; ifit < HCTIM_MAXFIT; ifit++)
    {
      if(igfit[ifit] != 1) continue;
	      
      c2p = normp * prfc->chi2[ifit];
      c2t = normt * hctim->mchi2[ifit];
	      
      if((c2p + c2t) < c2c)
	{
	  c2c = c2p + c2t;
	  imin[0] = ifit;
	}
    }
  
  // record the variables into md_pcgf class
  for (Int_t ifit = 0; ifit < 3; ifit++)
    md_pcgf->imin[ifit] = imin[ifit];
  for (Int_t ifit = 0; ifit < HCTIM_MAXFIT; ifit++)
    {
      md_pcgf->igfit[ifit]    = igfit[ifit];
      md_pcgf->avgcfc[ifit]   = avgcfc[ifit];
      md_pcgf->ndft_fit[ifit] = ndft_fit[ifit];
    }
  md_pcgf->ig        = ig;
  md_pcgf->tot_npe   = tot_npe;
  md_pcgf->nbins     = nbins;
  md_pcgf->ng        = ng;
  md_pcgf->sum       = sum;
  md_pcgf->FIT_START = FIT_START;
  md_pcgf->c2c       = c2c;
  md_pcgf->c2p       = c2p;
  md_pcgf->c2t       = c2t;
  md_pcgf->ndfp      = ndfp;
  md_pcgf->ndft      = ndft;
  md_pcgf->normp     = normp;
  md_pcgf->normt     = normt;
  
  /******************* GOODNESS OF FIT/ IMINC DETERMINATION (above) ****************************/
  return true;
}


// This is a routine for obtaining hraw1 class variable from 
// mcraw class.  Applicable for MD MC events.
bool pass1plot::get_hraw1()
{
  if (!have_mcraw)
    {
      fprintf(stderr,"error: pass1plot::get_hraw1: must have mcraw\n");
      return false;
    }
  mcraw->loadToDST();
  tafd10info::mcraw2hraw1();
  hraw1->loadFromDST();
  return true;
}

bool pass1plot::init_md_plane_geom(bool printWarning)
{
  if (!have_stpln || stpln->ntube < 1)
    {
      if(printWarning)
	fprintf(stderr,"warning: init_md_plane_geom: stpln is either absent or empty\n");
      return false;
    }
  if ((!have_hraw1 && !have_mcraw) || (hraw1->ntube < 1 && mcraw->ntube < 1))
    {
      if(printWarning)
	fprintf(stderr,
		"warning: init_md_plane_geom: both hraw1 and mcraw are either not present or not filled\n");
      return false;
    }
  if(!have_hraw1 || hraw1->ntube < 1)
    return get_hraw1();
  
  return true;
}

Bool_t pass1plot::set_fdplane(Int_t siteid, Bool_t chk_ntube, Bool_t printWarn)
{
  TString fdplane_name[2] = {"brplane", "lrplane"};
  if(siteid==0)
    fdplane = (fdplane_class *)brplane;
  else if (siteid==1)
    fdplane = (fdplane_class *)lrplane;
  else
    {
      fprintf(stderr,"error: set_fdplane: FD siteid must be either 0(BR) or 1(LR)\n");
      fdplane = 0;
      return false;
    }
  if(!have_fdplane[siteid])
    {
      fdplane  = 0;
      if(printWarn)
	fprintf(stderr,"warning: %s branch not in %s\n",
		fdplane_name[siteid].Data(),pass1tree->GetName());
      return false;
    }
  if(chk_ntube)
    {
      if(fdplane->ntube < 1)
	{
	  if(printWarn)
	    fprintf(stderr,"warning: %s branch doesn't have any tube information\n",fdplane_name[siteid].Data());
	  return false;
	}
    }
  return true;
}

Bool_t pass1plot::set_fdprofile(Int_t siteid, Bool_t chk_ntube, Bool_t printWarn)
{
  TString fdprofile_name[2] = {"brprofile","lrprofile"};
  if(siteid==0)
    fdprofile = (fdprofile_class *)brprofile;
  else if (siteid==1)
    fdprofile = (fdprofile_class *)lrplane;
  else
    {
      fprintf(stderr,"error: set_fdprofile: FD siteid must be either 0(BR) or 1(LR)\n");
      fdprofile = 0;
      return false;
    }
  if(!have_fdprofile[siteid])
    {
      fdprofile  = 0;
      if(printWarn)
	fprintf(stderr,"warning: %s branch not in %s\n",
		fdprofile_name[siteid].Data(),pass1tree->GetName());
      return false;
    }
  if(chk_ntube)
    {
      if(fdprofile->ntslice < 1)
	{
	  if(printWarn)
	    fprintf(stderr,"warning: %s branch doesn't have any time slice information\n",
		    fdprofile_name[siteid].Data());
	  return false;
	}
    }
  return true;
}

bool pass1plot::get_fd_tube_pd(int fdsiteid, int mir_num, int tube_num, double *v)
{
  switch(fdsiteid)
    {
    case 0:
      {
	return tafd10info::get_br_tube_pd(mir_num,tube_num,v);
	break;
      }
    case 1:
      {
	return tafd10info::get_lr_tube_pd(mir_num,tube_num,v);
	break;
      }
    case 2:
      {
	return tafd10info::get_md_tube_pd(mir_num,tube_num,v);
	break;
      }
    default:
      {
	fprintf(stderr,"error: get_fd_tube_pd: fdsiteid must be in 0-2 range\n");
	return false;
	break;
      }
    }
}

bool pass1plot::get_fd_time(int fdsiteid, int fdJday, int fdJsec, int *yymmdd, int *hhmmss)
{
  switch(fdsiteid)
    {
    case 0:
    case 1:
      {
	tafd10info::get_brlr_time(fdJday,fdJsec,yymmdd,hhmmss);
	break;
      }
    case 2:
      {
	tafd10info::get_md_time(fdJday,fdJsec,yymmdd,hhmmss);
	break;
      }
    default:
      {
	fprintf(stderr,"error: get_fd_time: fdsiteid must be in 0-2 range\n");
	return false;
	break;
      }
    }
  return true;
}
