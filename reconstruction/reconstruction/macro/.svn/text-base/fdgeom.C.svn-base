
#define TAFD_MAXTUBES   0xe00    // maximum number of tubes for any TA FD detector
#define TAFD_MAXMIR     14       // maximum number of mirrors  for any TA FD detector
#define TABRLR_MAXMIR   12       // maximum number of BR/LR mirrors
#define TAMD_MAXMIR     14       // maximum number of MD mirrors
#define TAFD_NTPMIR     0x100    // number of tubes per mirror for a TA FD detector

#define NPIX_PER_DEGREE  25      // number of pixels/degree for the FD event display
#define R_MAX_ED  1.0            // maximum cicle radius on the FD ED in elevation, degrees

using namespace TMath;

double clfevent_to_fd(Int_t fdsiteid, double *sdp_n, double *rp, double *psi)
{
  double xclf[3] = {0,0,1};
  double xfd[3];
  double core_clf[3] = {0,0,0};
  double core_fd[3];
  double vmag;
  rot_clf2fdsite(fdsiteid,xclf,xfd);
  clf2fdsite(fdsiteid,core_clf,core_fd);
  crossp(core_fd,xfd,sdp_n);
  (*rp)  = normv(sdp_n,sdp_n);
  (*psi) = 360.0 - RadToDeg() * ACos(dotp(xfd,core_fd)/sqrt(dotp(core_fd,core_fd)));
}


// sdp_n - shower detector plane normal vector, in FD coordinate system 
// x - vector in FD coordinate systme (INPUT)
// *alt_sdp - altitude in SDP frame, Degree (OUTPUT)
// *azm_sdp - azimuthal angle in SDP frame, Degree (OUTPUT)
void get_alt_azm_in_SDP(Double_t *sdp_n, Double_t *x, Double_t *alt_sdp, Double_t *azm_sdp)
{
  Int_t i;
  Double_t vmag,ex[3],ey[3],ez[3],y[3];

  // z-axis of the shower-detector plane coordinate system.
  // actually it is negative of the shower detector plane vector, because
  // the shower-detector plane vector was chosen so that 
  // (event axis) cross ( sdp normal) = (rp vector)
  for (i = 0; i < 3; i ++ )
    ez[i] = -sdp_n[i];
  
  vmag = sqrt (ez[0]*ez[0]+ez[1]*ez[1]);
  
  // x-axis of the shower-detector plane coordinate system,
  // a cross-product b/w FD z-axis and SDP z axis
  ex[0] = -ez[1]/vmag; ex[1] = ez[0]/vmag; ex[2] = 0.0;
  
  // y-axis of the shower-detector plane coordinate system, a cross-product
  // b/w SDP z - axis and SDP x - axis
  ey[0] = -ez[0]*ez[2]/vmag; ey[1] = -ez[1]*ez[2]/vmag; ey[2] = vmag;
  

  // Get the transformed vector in SDP frame
  y[0] = 0.0; y[1] = 0.0; y[2] = 0.0; vmag = 0.0;
  for (i=0; i<3; i++)
    {      
      vmag += x[i]*x[i];
      y[0] += x[i]*ex[i];
      y[1] += x[i]*ey[i];
      y[2] += x[i]*ez[i];
    }
  vmag = sqrt(vmag);
  for (i=0; i<3; i++)
    y[i] /= vmag;
  
  // altitude and azimuth in SDP frame
  (*alt_sdp) = RadToDeg() * ASin(y[2]);
  (*azm_sdp) = RadToDeg() * ATan2(y[1],y[0]);
  while ((*azm_sdp) < 0.0)
    (*azm_sdp) += 360.0;
}

// check if FD geometry variables are OK
bool chk_fd_geom(Int_t fdsiteid)
{
  Int_t i;
  Double_t dp,sdp_n[3];
  if (fdsiteid >= 0 && fdsiteid < 2)
    {
      if (!p1.set_fdplane(fdsiteid))
	return false ;
      for (i=0; i<3; i++)
	sdp_n[i] = p1.fdplane->sdp_n[i];
    }
  else if (fdsiteid == 2)
    {
      if(!p1.init_md_plane_geom())
	return false;
      for (i=0; i<3; i++)
	sdp_n[i] = p1.stpln->n_ampwt[2][i];
    }
  else
    {
      fprintf(stderr,"error: chk_fd_geom: fdsiteid must be in 0-2 range\n");
      return false;
    }
  dp = 0;
  for (i=0; i<3; i++)
    dp += sdp_n[i]*sdp_n[i];
  dp = Sqrt(dp);
  if(Abs(dp-1.0) > 1e-5)
    {
      fprintf(stderr,"error: chk_fd_geom: %s-FD SDP normal magnitude is not 1: vmag = %.6e\n",
	      fd_name[fdsiteid],dp);
      return false;
    }
  return true;
}




bool plotFDed(Int_t fdsiteid, Double_t secfrac_reference = -1.0, TCanvas *canv = 0)
{
  
  // initialize eleveation vs azimuth for SDP normal
  // function, if it hasn't been initialized yet
  if(!fEleVsAzi_sdp[fdsiteid])
    {
      TString fn = "fEleVsAzi_sdp";
      fn += fdsiteid;
      fEleVsAzi_sdp[fdsiteid] = new TF1(fn,ele_vs_azi_sdp,0.0,720.0,3);
      fEleVsAzi_sdp[fdsiteid]->SetNpx(300);
    }
  
  if(!chk_fd_geom(fdsiteid))
    return false;
  
  // set the canvas
  if(!canv)
    canv = c1;
  
  Int_t ngrid,npts;
  Double_t phi, phi_min, phi_max;
  Double_t azimuth[TAFD_MAXTUBES];
  Double_t elevation[TAFD_MAXTUBES];
  Int_t imir;
  Double_t tmin; // earliest time with respect to the reference time
  Int_t npts;
  Double_t 
    tb_azimuth[TAFD_MAXTUBES],
    tb_elevation[TAFD_MAXTUBES],
    tb_time[TAFD_MAXTUBES],
    tb_npe[TAFD_MAXTUBES];
  Int_t mirrors[TAFD_MAXMIR]; // mirror ids with good tubes in them
  Int_t nmir;  // number of mirrors (with good tubes in them)
  bool got_mir; // dummy
  Double_t npeMax; // maximum NPE
  Double_t tube_v[3]; // tube pointing direction
  Int_t itube;
  Int_t Year, Month, Day, Hour, Minute, Second,yymmdd,hhmmss;
  Double_t xlo, xup, ylo, yup; // ED XY range
  Int_t nx,ny,ix,iy,ixmin,ixmax,iymin,iymax;
  Double_t rval,xval,yval,tval,npeval,dx,dy,yxrat;
  TString hName, title;
  
  npts                   = 0;
  nmir                   = 0;
  ngrid                  = 0;
  FD_Tref[fdsiteid]      = 1.0e20;
  tmin                   = 1.0e20;
  npeMax                 = 1.0;
  phi_min                = 360.0;
  phi_max                = 0.0;
  
  if (fdsiteid < 2)
    {
      p1.get_fd_time(fdsiteid,p1.fdplane->julian,p1.fdplane->jsecond,&yymmdd,&hhmmss);
      fEleVsAzi_sdp[fdsiteid]->SetParameters(p1.fdplane->sdp_n[0],p1.fdplane->sdp_n[1],p1.fdplane->sdp_n[2]);
      for (itube=0; itube < p1.fdplane->ntube; itube++)
	{
	  if(p1.fdplane->tube_qual[itube] != 1) 
	    continue;
	  if(!p1.get_fd_tube_pd(fdsiteid,p1.fdplane->camera[itube],p1.fdplane->tube[itube],tube_v))
	    return false;
	  got_mir = false;
	  for (imir=0; imir<nmir; imir++)
	    {
	      if (mirrors[imir] == p1.fdplane->camera[itube])
		{
		  got_mir = true;
		  break;
		}
	    }
	  if(!got_mir)
	    {
	      if(nmir >= TABRLR_MAXMIR)
		{
		  fprintf(stderr,"error: number of unique mirrors for %s-FD seems larger than %s\n",
			  TABRLR_MAXMIR,fd_name[fdsiteid]);
		  return false;
		}
	      mirrors[nmir] = p1.fdplane->camera[itube];
	      nmir++;
	    }
	  phi = RadToDeg()* ATan2(tube_v[1],tube_v[0]);
	  while (phi < 0) phi += 360.0;
	  while (phi >=360.0) phi -= 360.0;
	  if (phi_min > phi) phi_min = phi;
	  if (phi_max < phi) phi_max = phi;
	  tb_azimuth[npts] = phi;
	  tb_elevation[npts] = RadToDeg() * ASin(tube_v[2]);
	  tb_time[npts] = ((double)p1.fdplane->jsecfrac)/1e3 - 25.6 + p1.fdplane->time[itube]/1.0e3;
	  if (tb_time[npts] < FD_Tref[fdsiteid] ) FD_Tref[fdsiteid] = tb_time[npts];
	  tb_npe[npts] = p1.fdplane->npe[itube];
	  if (tb_npe[npts] < 1.0) tb_npe[npts] = 1.0;
	  if(tmin > tb_time[npts]) tmin = tb_time[npts];
	  if (npeMax < tb_npe[npts]) npeMax = tb_npe[npts];
	  npts++;
	}
      ngrid = 0;
      for (imir=0; imir < nmir; imir++)
	{
	  // BR/LR FD tube IDs are from 0 to 255
	  for (itube=0; itube < TAFD_NTPMIR; itube++)
	    {
	      if(!p1.get_fd_tube_pd(fdsiteid,mirrors[imir],itube,tube_v)) return false;
	      phi = RadToDeg()* ATan2(tube_v[1],tube_v[0]);
	      while (phi < 0) phi += 360.0;
	      while (phi >=360.0) phi -= 360.0;
	      if(phi_max < phi) phi_max = phi;
	      if(phi_min > phi) phi_min = phi;
	      azimuth[ngrid]   = phi;
	      elevation[ngrid] = RadToDeg()* ASin(tube_v[2]); 
	      ngrid++;
	    }
	}
    }
  else
    {
      p1.get_fd_time(fdsiteid,p1.hraw1->jday,p1.hraw1->jsec,&yymmdd,&hhmmss);
      fEleVsAzi_sdp[fdsiteid]->SetParameters(p1.stpln->n_ampwt[2][0],p1.stpln->n_ampwt[2][1],p1.stpln->n_ampwt[2][2]);
      for (itube=0; itube < p1.hraw1->ntube; itube++)
	{
	  if(p1.stpln->ig[itube] < 1) 
	    continue;
	  if(!p1.get_fd_tube_pd(fdsiteid,p1.hraw1->tubemir[itube], p1.hraw1->tube[itube],tube_v))
	    return false;
	  got_mir = false;
	  for (imir=0; imir<nmir; imir++)
	    {
	      if (mirrors[imir] == p1.hraw1->tubemir[itube])
		{
		  got_mir = true;
		  break;
		}
	    }
	  if(!got_mir)
	    {
	      if(nmir >= TAMD_MAXMIR)
		{
		  fprintf(stderr,"error: number of unique mirrors for %s-FD seems larger than %s\n",
			  TAMD_MAXMIR,fd_name[fdsiteid]);
		  return false;
		}
	      mirrors[nmir] = p1.hraw1->tubemir[itube];
	      nmir++;
	    }
	  phi = RadToDeg()* ATan2(tube_v[1],tube_v[0]);
	  while (phi < 0) phi += 360.0;
	  while (phi >=360.0) phi -= 360.0;
	  if (phi_min > phi) phi_min = phi;
	  if (phi_max < phi) phi_max = phi;
	  tb_azimuth[npts] = phi;
	  tb_elevation[npts] = RadToDeg() * ASin(tube_v[2]);
	  tb_time[npts] = p1.hraw1->mirtime_ns[0]/1.0e3 + p1.hraw1->thcal1[itube];
	  if (tb_time[npts] < FD_Tref[fdsiteid] ) FD_Tref[fdsiteid] = tb_time[npts];
	  tb_npe[npts] = p1.MD_PMT_QE * p1.hraw1->prxf[itube];
	  if (tb_npe[npts] < 1.0) tb_npe[npts] = 1.0;
	  if(tmin > tb_time[npts]) tmin = tb_time[npts];
	  if (npeMax < tb_npe[npts]) npeMax = tb_npe[npts];
	  npts++;
	}
      ngrid = 0;
      for (imir=0; imir < nmir; imir++)
	{
	  // MD FD tube IDs are from 1 to 256
	  for (itube=1; itube <= TAFD_NTPMIR; itube++)
	    {
	      if(!p1.get_fd_tube_pd(fdsiteid,mirrors[imir],itube,tube_v)) return false;
	      phi = RadToDeg()* ATan2(tube_v[1],tube_v[0]);
	      while (phi < 0) phi += 360.0;
	      while (phi >=360.0) phi -= 360.0;
	      if(phi_max < phi) phi_max = phi;
	      if(phi_min > phi) phi_min = phi;
	      azimuth[ngrid]   = phi;
	      elevation[ngrid] = RadToDeg()* ASin(tube_v[2]); 
	      ngrid++;
	    }
	}
    }  
  
  // make sure got points for displaying the event
  if (npts < 1)
    {
      fprintf(stderr,"error(plotED): no good tubes in event for %s-FD\n",fd_name[fdsiteid]);
      return false;
    }
  if(ngrid < 1)
    {
      fprintf(stderr,"error(plotED): no grid points for %s-FD\n",fd_name[fdsiteid]);
      return false;
    }
  
  // make all displayed times smaller by using either the earliest time
  // or some specified reference time
  if (secfrac_reference > -0.999999999)
    FD_Tref[fdsiteid] = secfrac_reference * 1.0e6;
  
  for (itube=0; itube < npts; itube++)
    tb_time[itube] -= FD_Tref[fdsiteid];
  tmin -= FD_Tref[fdsiteid];
  
  // adjusting the ED azimuthal range to fit to the screen
  if (phi_max - phi_min > 330.0)
    { 
      for (itube=0; itube < ngrid; itube++)
	{
	  if (azimuth[itube] < 40.0)
	    azimuth[itube] += 360.0;
	  if (itube < npts)
	    {
	      if (tb_azimuth[itube] < 40.0)
		tb_azimuth[itube] += 360.0;
	    }
	}
    }
  

  Year   = 2000+(yymmdd/10000);
  Month  = (yymmdd%10000)/100;
  Day    = yymmdd%10000;
  Hour   = hhmmss/10000;
  Minute = (hhmmss%10000)/100;
  Second = hhmmss%10000;
  title.Form("%s-FD %04d/%02d/%02d %02d:%02d:%02d.%06d",fd_name[fdsiteid],
	     Year,Month,Day,Hour,Minute,Second,(Int_t)Floor(FD_Tref[fdsiteid] + 0.5));

  if (gEDfd[fdsiteid])
    delete gEDfd[fdsiteid];
  gEDfd[fdsiteid] = new TGraph(ngrid,azimuth,elevation);
  
  gEDfd[fdsiteid]->SetTitle(title);
  gEDfd[fdsiteid]->SetMarkerStyle(20);
  gEDfd[fdsiteid]->SetMarkerSize(0.5);
  gEDfd[fdsiteid]->GetXaxis()->SetTitle("Azimuth [Degree]");
  gEDfd[fdsiteid]->GetYaxis()->SetTitle("Elevation [Degree]");
  gEDfd[fdsiteid]->GetXaxis()->CenterTitle();
  gEDfd[fdsiteid]->GetYaxis()->CenterTitle();
  xlo = gEDfd[fdsiteid]->GetXaxis()->GetXmin();
  xup = gEDfd[fdsiteid]->GetXaxis()->GetXmax();
  ylo = gEDfd[fdsiteid]->GetYaxis()->GetXmin();
  yup = gEDfd[fdsiteid]->GetYaxis()->GetXmax();
  nx = (Int_t)Floor((xup-xlo)*(Double_t)NPIX_PER_DEGREE + 0.5);
  ny = (Int_t)Floor((yup-ylo)*(Double_t)NPIX_PER_DEGREE + 0.5);
  yxrat=(yup-ylo)/(xup-xlo);
  hName ="hEDfd";
  hName += fdsiteid;
  if (hEDfd[fdsiteid]) delete hEDfd[fdsiteid];
  hEDfd[fdsiteid] = new TH2F(hName,"",nx,xlo,xup,ny,ylo,yup);
  for ( itube = 0; itube < npts; itube++)
    {
      xval = tb_azimuth[itube];
      yval = tb_elevation[itube];
      tval = tb_time[itube];
      npeval = tb_npe[itube];
      rval = sqrt(npeval/npeMax) * R_MAX_ED;
      ixmin = Max(1,(Int_t)Floor((xval-rval/yxrat-xlo)/(xup-xlo)*(Double_t)nx));
      ixmax = Min(nx,(Int_t)Ceil((xval+rval/yxrat-xlo)/(xup-xlo)*(Double_t)nx));
      iymin = Max(1,(Int_t)Floor((yval-rval-ylo)/(yup-ylo)*(Double_t)ny));
      iymax = Min(ny,(Int_t)Ceil((yval+rval-ylo)/(yup-ylo)*(Double_t)ny));
      for (ix=ixmin; ix<=ixmax; ix++)
	{
	  for (iy=iymin; iy<=iymax; iy++)
	    {
	      dx = xval - xlo + (xlo-xup)*(ix-0.5)/(Double_t)nx;
	      dy = yval - ylo + (ylo-yup)*(iy-0.5)/(Double_t)ny;
	      if (sqrt(dx*dx*yxrat*yxrat+dy*dy) <= rval)
		hEDfd[fdsiteid]->SetBinContent(ix,iy,tval+1e-2);
	    }
	}
    }
  hEDfd[fdsiteid]->SetMinimum(tmin-(1e-9)*Abs(tmin));
  TPaveLabel *tl = new TPaveLabel(xup-2.0,yup,xup,yup+(yup-ylo)/15.0,"Time,  [#mus]");
  tl->SetTextSize(1.0);
  tl->SetBorderSize(0);
  tl->SetFillStyle(0);
  tl->SetTextFont(62);
  canv->cd();
  canv->Clear();
  gEDfd[fdsiteid]->Draw("a,p");
  hEDfd[fdsiteid]->Draw("zcol,same");
  canv->Modified(); 
  canv->Update();
  ((TPaletteAxis *)hEDfd[fdsiteid]->FindObject("palette"))->SetLabelSize(0.03);
  ((TPaletteAxis *)hEDfd[fdsiteid]->FindObject("palette"))->SetX2NDC(0.94);
  if(Abs(fEleVsAzi_sdp[fdsiteid]->GetParameter(2)) < 1e-3)
    fEleVsAzi_sdp[fdsiteid]->SetParameter(2,1e-3);
  fEleVsAzi_sdp[fdsiteid]->Draw("same");
  tl->Draw();
  canv->Modified(); 
  canv->Update();
  return true;
}



Double_t psi_epsi(Double_t psi, Double_t epsi)
{
  return epsi * (180.0-psi)*(180.0-psi)/(180.0*180.0);
}


bool fitTvsAfd(Int_t fdsiteid, Double_t secfrac_reference = -1.0, 
	       bool is_clf = false, bool fix_clf_geom=false, bool vmode = true)
{
  
  if(!chk_fd_geom(fdsiteid))
    return false;
  
  Int_t itube;
  Int_t npts;
  Double_t tube_time[0xc00];
  Double_t tube_timerr[0xc00];
  Double_t tube_sdpangle[0xc00];
  Double_t palt,pazm,tube_v[3],sdp_n[3];
  Double_t psi_fix,rp_fix;
  Int_t i;
  
  FD_Tref[fdsiteid] = 1e20.0;
  
  npts = 0;
  if (fdsiteid  < 2)
    {
      for (itube=0; itube < p1.fdplane->ntube; itube ++)
	{
	  if (p1.fdplane->tube_qual[itube] != 1)
	    continue;
	  tube_time[npts] = ((double)p1.fdplane->jsecfrac)/1e3 - 25.6 + p1.fdplane->time[itube]/1.0e3;
	  tube_timerr[npts] = 4.25 * p1.fdplane->time_rms[itube] / 1.0e3;
	  tube_sdpangle[npts] = p1.fdplane->plane_azm[itube] * RadToDeg();
	  if(tube_time[npts] < FD_Tref[fdsiteid])
	    FD_Tref[fdsiteid] = tube_time[npts];
	  npts ++ ;
	}
    }
  else
    {  
      if (p1.hraw1->ntube != p1.stpln->ntube)
	{
	  fprintf(stderr,"error: hraw1 ntube (%d) not equal to stpln ntube (%d)!\n",
		  p1.hraw1->ntube, p1.stpln->ntube);
	  return false;
	}
      if(!is_clf)
	{
	  for (i=0; i<3; i++) 
	    sdp_n[i] = p1.stpln->n_ampwt[2][i];
	}
      else
	{
	  for (i=0; i<3; i++) 
	    sdp_n[i] = -p1.stpln->n_ampwt[2][i];
	}
      for (itube=0; itube < p1.stpln->ntube; itube ++)
	{
	  if (p1.stpln->ig[itube] < 1)
	    continue;
	  tube_time[npts]   = ((double)p1.hraw1->mirtime_ns[0])/1e3 + p1.hraw1->thcal1[itube];
	  tube_timerr[npts] = 0.055 + 0.225 / sqrt(p1.hraw1->prxf[itube]*p1.MD_PMT_QE);
	  p1.get_fd_tube_pd(2,p1.hraw1->tubemir[itube],p1.hraw1->tube[itube],tube_v);
	  get_alt_azm_in_SDP(sdp_n,tube_v,&palt,&pazm);
	  tube_sdpangle[npts] = pazm; //-0.2;
	  if(tube_time[npts] < FD_Tref[fdsiteid])
	    FD_Tref[fdsiteid] = tube_time[npts];
	  npts ++ ;
	}
    }
  
  // make all displayed times smaller by using either the earliest time
  // or some specified reference time
  if (secfrac_reference > -0.999999999)
    FD_Tref[fdsiteid] = secfrac_reference * 1.0e6;
  
  for (itube=0; itube < npts; itube++)
    tube_time[itube] -= FD_Tref[fdsiteid];
  
  if (gTvsAfd[fdsiteid])
    {
      delete gTvsAfd[fdsiteid];
      gTvsAfd[fdsiteid] = 0;
    } 
  gTvsAfd[fdsiteid] = new TGraphErrors(npts,tube_sdpangle,tube_time,0,tube_timerr);
  
  
  if(!is_clf)
    {
      fTvsAfd->FixParameter(2,90.0);
      if (vmode)
	{
	  gTvsAfd[fdsiteid]->Fit(fTvsAfd);
	  fTvsAfd->ReleaseParameter(2);
	  gTvsAfd[fdsiteid]->Fit(fTvsAfd);
	}
      else
	{
	  gTvsAfd[fdsiteid]->Fit(fTvsAfd,"Q,0");
	  fTvsAfd->ReleaseParameter(2);
	  gTvsAfd[fdsiteid]->Fit(fTvsAfd,"0,Q");
	  gTvsAfd[fdsiteid]->GetFunction("fTvsAfd")->ResetBit(1<<9);
	}      
    }
  else
    {
      if(!fix_clf_geom)
	{
	  fTvsAfd_CLF->FixParameter(2,270.0);
	  if (vmode)
	    {
	      gTvsAfd[fdsiteid]->Fit(fTvsAfd_CLF);
	      fTvsAfd_CLF->ReleaseParameter(2);
	      gTvsAfd[fdsiteid]->Fit(fTvsAfd_CLF);
	    }
	  else
	    {
	      gTvsAfd[fdsiteid]->Fit(fTvsAfd_CLF,"Q,0");
	      fTvsAfd_CLF->ReleaseParameter(2);
	      gTvsAfd[fdsiteid]->Fit(fTvsAfd_CLF,"0,Q");
	      gTvsAfd[fdsiteid]->GetFunction("fTvsAfd_CLF")->ResetBit(1<<9);
	    }
	}
      else
	{
	  // clfevent_to_fd(fdsiteid,sdp_n,&rp_fix,&psi_fix);
	  rp_fix  = 20.86e3;
	  psi_fix = 270.0;
	  // if(fdsiteid<2)
	  //   for (int ix=0; ix<3; ix++) 
	  //     p1.fdplane->sdp_n[ix] = sdp_n[ix];
	  // else
	  //   for (int ix=0; ix<3; ix++) 
	  //     p1.stpln->n_ampwt[2][ix] = sdp_n[ix];
	  fTvsAfd_CLF->FixParameter(1,rp_fix);
	  fTvsAfd_CLF->FixParameter(2,psi_fix);
	  if(vmode)
	    {
	      gTvsAfd[fdsiteid]->Fit(fTvsAfd_CLF);
	      fprintf(stdout,"fdsiteid: %d rp_fix: %f psi_fix: %f\n",fdsiteid,rp_fix,psi_fix);
	    }
	  else
	    {
	      gTvsAfd[fdsiteid]->Fit(fTvsAfd_CLF,"0,Q");
	      gTvsAfd[fdsiteid]->GetFunction("fTvsAfd_CLF")->ResetBit(1<<9);
	    }
	}
    }
  if(vmode)
    fprintf(stdout,"FDSITEID: %d REFERENCE TIME: %.9f\n",fdsiteid,FD_Tref[fdsiteid]/1e6);
  return true;
}

void plotTvsAfd(Int_t fdsiteid, Double_t secfrac_reference = -1.0, 
		bool is_clf = false, bool fix_clf_geom=false, TCanvas *canv = 0 )
{
  TString title;
  if(!fitTvsAfd(fdsiteid,secfrac_reference,is_clf,fix_clf_geom,true))
    return;
  title.Form("Time vs Angle, %s-Mono",fd_name[fdsiteid]);
  gTvsAfd[fdsiteid]->SetTitle(title);
  gTvsAfd[fdsiteid]->GetXaxis()->SetTitle("Angle in SDP, [ Degree ]");
  gTvsAfd[fdsiteid]->GetYaxis()->SetTitle("Time, [ uS ]");
  gTvsAfd[fdsiteid]->SetMarkerStyle(20);
  gTvsAfd[fdsiteid]->SetMarkerSize(0.7);
  if(!canv)
    canv = c1;
  canv->cd();
  gTvsAfd[fdsiteid]->Draw("a,p");
}

bool plotTvsAfd_ALL(Double_t secfrac_reference = -1.0, bool is_clf = false, bool fix_clf_geom=false)
{
  if(curEdMode != ED_FD_MODE) 
    fd_mode();
  
  TCanvas *canv;
  Double_t tref_secfrac; // common FD reference time
  int i;
  tref_secfrac = 1e20.0;
  for (i=0; i<3; i++)
    {
      if(!fitTvsAfd(i,secfrac_reference,is_clf,fix_clf_geom,false))
	continue;
      if(FD_Tref[i]/1e6 < tref_secfrac)
	tref_secfrac = FD_Tref[i]/1e6;
    }
  if(tref_secfrac > 1.0)
    {
      fprintf(stderr,"error: unable to determine the common reference time\n");
      return false;
    }  
  // round the reference time second fraction to the nearest 0.1 second if this is CLF
  if(is_clf)
    tref_secfrac = Floor(tref_secfrac*10.0 + 0.5) / 10.0;
  
  for (fdsiteid=0; fdsiteid<3; fdsiteid++)
    {
      if(fdsiteid==0)
	canv = c4;
      else if(fdsiteid==1)
	canv = c5;
      else
	canv = c6;
      plotTvsAfd(fdsiteid,tref_secfrac,is_clf,fix_clf_geom,canv);
    }
  
   return true;
}



bool plotFD_ALL(Double_t secfrac_reference = -1.0, bool is_clf = false, bool fix_clf_geom=false)
{
  if(curEdMode != ED_FD_MODE) 
    fd_mode();
  
  int i;
  TString cn;
  TCanvas *canv;

  Double_t tref_secfrac; // common FD reference time
  
  tref_secfrac = 1e20.0;
  for (i=0; i<3; i++)
    {
      if(!fitTvsAfd(i,secfrac_reference,is_clf,fix_clf_geom,false))
	continue;
      if(FD_Tref[i]/1e6 < tref_secfrac)
	tref_secfrac = FD_Tref[i]/1e6;
    }
  if(tref_secfrac > 1.0)
    {
      fprintf(stderr,"error: unable to determine the common reference time\n");
      return false;
    }
  
  // round the reference time second fraction to the nearest 0.1 second if this is CLF
  if(is_clf)
    tref_secfrac = Floor(tref_secfrac*10.0 + 0.5) / 10.0;
  
  for (i=1; i<=6; i++)
    {
      cn="c";
      cn+=i;
      if(!(canv=(TCanvas*)gROOT->FindObject(cn)))
	continue;
      canv->SetGridx();
      canv->SetGridy();
      canv->SetTickx();
      canv->SetTicky();
    }

  plotTvsAfd(0,tref_secfrac,is_clf,fix_clf_geom,c4);
  plotTvsAfd(1,tref_secfrac,is_clf,fix_clf_geom,c5);
  plotTvsAfd(2,tref_secfrac,is_clf,fix_clf_geom,c6);
  
  plotFDed(0,tref_secfrac,c1);
  plotFDed(1,tref_secfrac,c2);
  plotFDed(2,tref_secfrac,c3);
  
  if(is_clf && fix_clf_geom)
    {
      
      fprintf(stdout, "BR = %.3f LR = %.3f MD = %.3f\n",
	      gTvsAfd[0]->GetFunction("fTvsAfd_CLF")->GetParameter(0)+12.0/300.0,
	      gTvsAfd[1]->GetFunction("fTvsAfd_CLF")->GetParameter(0)-138.0/300.0,
	      gTvsAfd[2]->GetFunction("fTvsAfd_CLF")->GetParameter(0)-184.0/300.0);
      
    }
  return true;
}

void writeCLFtimeTree(char *outfile="clftime.root")
{
  Double_t tbr,tlr,tmd;
  TFile *f = new TFile(outfile,"recreate");
  TTree *t = new TTree("tCLFtime","");  
  t->Branch("tbr",&tbr,"tbr/D");
  t->Branch("tlr",&tlr,"tlr/D");
  t->Branch("tmd",&tmd,"tmd/D");  
  int nevents = p1.GetEntries();
  for (int i=0; i<nevents; i++)
    {
      p1.GetEntry(i);
      fprintf(stdout,"Completed: %.0f%c\r", 
	      (Double_t)i/(Double_t)(nevents-1)*100.0,'%');
      fflush(stdout);
      
      // BR
      fitTvsAfd(0,0.0,true,true,false);
      tbr = gTvsAfd[0]->GetFunction("fTvsAfd_CLF")->GetParameter(0);
      tbr += (12.0/300.0);
      
      // LR
      fitTvsAfd(1,0.0,true,true,false);
      tlr = gTvsAfd[1]->GetFunction("fTvsAfd_CLF")->GetParameter(0);
      tlr -= (138.0/300.0);
      
      // MD
      fitTvsAfd(2,0.0,true,true,false);
      tmd = gTvsAfd[2]->GetFunction("fTvsAfd_CLF")->GetParameter(0); 
      tmd -= (184.0/300.0);
      
      t->Fill();
    }
  t->Write();
  f->Close();
  fprintf(stdout,"\n");
}
