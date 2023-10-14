// Make a full FADC histogram for a given counter
bool histFADC(Int_t xxyy = 0, bool subtract_pedestal = false, 
	      bool use_icrr_ped=false, bool verbose=true)
{
  Int_t i,k,j;
  Int_t isub,iwf,icrr_iwf,nwf;
  Int_t wf_ind[10];
  Int_t il,iu;
  char hTitle[125];
  Int_t nbins;
  Int_t ngaps;
  Float_t *leftedgeX = 0;
  Double_t *gapsize=0;
  Double_t bcont, ped;

  if (use_icrr_ped && !p1.have_tasdcalibev)
    {
      fprintf(stderr,"Must have tasdcalibev branch to use icrr pedestals\n");
      return false;
    }
  
  if (xxyy == 0)
    {
      fprintf (stderr, "(1): XXYY for the counter\n");
      return false;
    }
  iwf = -1;
  for (i=0; i < p1.rusdraw->nofwf; i++)
    {
      if (p1.rusdraw->xxyy[i] == xxyy)
	{
	  iwf = i;
	  break;
	}
    }
  if (iwf == -1)
    {
      fprintf (stderr, 
	       "detector XXYY = %04d is not found in the event read out\n",
	       xxyy);
      return false;
    }
  il=( (iwf-10) < 0 ? 0 : (iwf-10));
  iu=( (iwf+10) >= p1.rusdraw->nofwf ? (p1.rusdraw->nofwf-1) : (iwf+10));
  nwf=0;
  for (i=il;i<=iu;i++)
    {
      if(p1.rusdraw->xxyy[i]==p1.rusdraw->xxyy[iwf])
	{
	  wf_ind[nwf]=i;
	  nwf++;
	}
    }
  ngaps = nwf-1;
  nbins = 128;
  il=wf_ind[0];
  if (ngaps > 0) 
    {
      gapsize = new Double_t[ngaps];
      for (i=0; i<ngaps; i++)
	{
	  gapsize [i] =
	    (
	     (Double_t)p1.rusdraw->clkcnt[il+i+1] / (Double_t)p1.rusdraw->mclkcnt[il+i+1] -
	     (Double_t)p1.rusdraw->clkcnt[il+i]   / (Double_t)p1.rusdraw->mclkcnt[il+i]
	     ) * 50e6 - 128.0;
	  if (verbose)
	    fprintf (stdout,"gapsize (%d) = %e x 20 nS = %f uS\n",
		     i,gapsize[i],0.02*gapsize[i]);
	  if(gapsize[i] < 0)
	    {
	      gapsize[i]=1e-4;
	      if (verbose)
		fprintf (stdout, "gapsize %d set to %e x 20 nS = %f uS\n",
			 i,gapsize[i],0.02*gapsize[i]);
	    }
	  nbins += 129;
	}
    }
  leftedgeX = new Float_t[nbins+1];
  j = 0;
  leftedgeX[0] = -0.5;
  for (i=1; i < (nbins+1); i++)
    {
      if ( i>128 && ((i-129) % 129 == 0) && j < ngaps)
	{
	  leftedgeX[i] = leftedgeX[i-1]+gapsize[j];
	  j++;
	}
      else
	{
	  leftedgeX[i] = leftedgeX[i-1]+1.0;
	}
    }
  for (k=0; k<2; k++) 
    if(hFADC[k]) delete hFADC[k];
  sprintf(hTitle,"XXYY=%04d (LOWER)",p1.rusdraw->xxyy[iwf]);
  hFADC[0] = new TH1F ("hFADC0",hTitle, nbins, leftedgeX);
  sprintf(hTitle,"XXYY=%04d (UPPER)",p1.rusdraw->xxyy[iwf]);
  hFADC[1] = new TH1F ("hFADC1",hTitle, nbins, leftedgeX);
  il=wf_ind[0];
  j = 0;
  for (i=0; i < nbins; i++)
    { 
      if ( i>127 && ((i-128) % 129 == 0) && j < ngaps)
	{
	  j++;
	  for (k=0; k<0; k++) hFADC[k] -> SetBinContent(i+1,0.0);
	}
      else
	{
	  for (k=0; k<2; k++) 
	    {
	      bcont = p1.rusdraw->fadc[il+j][k][i-j*129];
	      if (subtract_pedestal)
		{
		  if (use_icrr_ped)
		    {
		      icrr_iwf=p1.get_tasdcalibev_iwf(il+j);
		      if (k==0)
			ped = p1.tasdcalibev->sub[icrr_iwf].lpedAvr;
		      else
			ped = p1.tasdcalibev->sub[icrr_iwf].upedAvr;
		    }
		  else
		    {
		      ped = ((double)p1.rusdraw->pchped[il+j][k]/8.0);
		    }
		  bcont -= ped;
		}
	      hFADC[k]->SetBinContent(i+1,bcont);
	    }
	}
    }
  if(leftedgeX)
    delete leftedgeX;
  if(gapsize)
    delete gapsize;
  return true;
}

bool plotFADC(Int_t xxyy=0, bool subtract_pedestal = false, bool use_icrr_ped = false)
{
  Int_t k;
  Int_t ihit, iwf,icrr_iwf;
  Double_t x1,x2;
  Double_t ped,pedmrms,pedprms;
  
  if (use_icrr_ped && !p1.have_tasdcalibev)
    {
      fprintf(stderr,"Must have tasdcalibev branch to use icrr pedestals\n");
      return false;
    }
  
  if (!histFADC(xxyy,subtract_pedestal,use_icrr_ped))
    return false;

  ihit = -1;
  for ( k = 0; k < p1.rufptn->nhits; k++)
    {
      if (xxyy==p1.rufptn->xxyy[k])
	{
	  ihit = k;
	  break;
	}
    }
  if (ihit == -1)
    {
      fprintf(stderr, "Information for XXYY = %04d is not found in rufptn\n",
	      xxyy);
      return false;
    }
  
  for (k=0; k<2; k++)
    {
      c3->cd(2-k);
      hFADC[k]->Draw();
      
      // draw the horizontal pedestal lines
      // if pedestals are not subtracted
      if (!subtract_pedestal)
	{
	  x1 = hFADC[k]->GetXaxis()->GetXmin();
	  x2 = hFADC[k]->GetXaxis()->GetXmax();

	  if (use_icrr_ped)
	    {
	      iwf=p1.rufptn->wfindex[iwf];
	      icrr_iwf=p1.get_tasdcalibev_iwf(iwf);
	      if (k==0)
		{
		  ped = p1.tasdcalibev->sub[icrr_iwf].lpedAvr;
		  pedmrms = ped - p1.tasdcalibev->sub[icrr_iwf].lpedStdev;
		  pedprms = ped + p1.tasdcalibev->sub[icrr_iwf].lpedStdev;
		}
	      else
		{
		  ped = p1.tasdcalibev->sub[icrr_iwf].upedAvr;
		  pedmrms = ped - p1.tasdcalibev->sub[icrr_iwf].upedStdev;
		  pedprms = ped + p1.tasdcalibev->sub[icrr_iwf].upedStdev; 
		}
	    }
	  else
	    {
	      ped = p1.rufptn->ped[ihit][k];
	      pedmrms = p1.rufptn->ped[ihit][k]-p1.rufptn->pederr[ihit][k];
	      pedprms = p1.rufptn->ped[ihit][k]+p1.rufptn->pederr[ihit][k];
	    }
	  pass1plot_drawLine(x1,ped,x2,ped,3,1);
	  pass1plot_drawLine(x1,pedmrms,x2,pedmrms,1,2);
	  pass1plot_drawLine(x1,pedprms,x2,pedprms,1,2);
	}
    }
  return true;
}

bool plotFADC(Int_t x, Int_t y, bool subtract_pedestal = false, bool use_icrr_ped = false)
{
  Int_t xxyy=x*100+y;
  return plotFADC(xxyy,subtract_pedestal,use_icrr_ped);
}


void stop_fadcsum()
{
  TH1F *h;
  h   = (TH1F *)gROOT->FindObject("hl");
  if(h)
    h->Delete();
  h   = (TH1F *)gROOT->FindObject("hu");
  if(h)
    h->Delete();
  h = (TH1F *)gROOT->FindObject("hbox");
  if(h)
    h->Delete();
  c1->DeleteExec("fadcSlidingSum");
  gStyle->SetOptTitle(1);
}

void start_fadcsum(Int_t x, Int_t y, bool subtract_pedestal=true)
{
  Double_t hmin,hmax;
  Int_t xxyy,k;
  
  static bool fadcsum_started = false;
  
  if (fadcsum_started)
    stop_fadcsum();
  
  xxyy=x*100+y;
  if(!histFADC(xxyy,subtract_pedestal))
    return;
  
  fadcsum_started = true;
  
  c1->cd();
  hFADC[0]->SetLineColor(4);
  hFADC[1]->SetLineColor(2);
  hmin=1e10;
  hmax=1e-10;
  for (k=0; k<2; k++)
    {
      if (hFADC[k]->GetMinimum() < hmin)
	hmin=hFADC[k]->GetMinimum();
      if (hFADC[k]->GetMaximum() > hmax)
	hmax=hFADC[k]->GetMaximum();
    }
  for (k=0; k<2; k++)
    {
      hFADC[k]->SetMinimum(hmin-0.1*hmax);
      hFADC[k]->SetMaximum(hmax+0.1*hmax);
    }
  gStyle->SetOptTitle(0);
  hFADC[0]->Draw();
  hFADC[1]->Draw("same");
  hFADC[0]->Clone("hl");
  hFADC[1]->Clone("hu");
  hFADC[1]->Clone("hbox");
  TH1F *hbox = (TH1F *) gROOT->FindObject("hbox");
  hbox->SetMinimum(hmin);
  hbox->SetMaximum(hmax);
  c1->AddExec("fadcSlidingSum","fadcSlidingSum()");
}
void fadcSlidingSum()
{
  static int binx_prev = -1;
  int nbinsx,binx,ix,ixmax;
  int k;
  TH1F *hl;
  TH1F *hu;
  TH1F *hbox;
  double hmin,hmax;
  TObject *select = gPad->GetSelected();
  if(!select) return;
  gPad->GetCanvas()->FeedbackMode(kTRUE);
  int px = gPad->GetEventX();
  int py = gPad->GetEventY();
  double xpos=gPad->AbsPixeltoX(px);
  double ypos=gPad->AbsPixeltoY(py);
  
  hl   = (TH1F *)gROOT->FindObject("hl");
  hu   = (TH1F *)gROOT->FindObject("hu");
  hbox = (TH1F *)gROOT->FindObject("hbox");
  nbinsx = hFADC[0]->GetNbinsX();
  binx = hFADC[0]->FindBin(xpos); 
  // Move the sliding window
  if (binx != binx_prev)
    {
      binx_prev = binx;
      hmin=hbox->GetMinimum();
      hmax=hbox->GetMaximum();
      hbox->Reset();
      hbox->SetMinimum(hmin);
      hbox->SetMaximum(hmax);
      hl->Reset();
      hu->Reset();
      hbox->Reset();
      ixmax=binx+7;
      if (ixmax > nbinsx)
	ixmax = nbinsx;
      for (ix=binx; ix<=ixmax; ix++)
	{
	  if (hbox->GetBinWidth(ix) < 0.5)
	    {
	      ix++;
	      ixmax++;
	      if (ixmax > nbinsx) 
		ixmax=nbinsx;
	    }
	  hbox->SetBinContent(ix,hmax);
	  hl->SetBinContent(ix,hFADC[0]->GetBinContent(ix));
	  hu->SetBinContent(ix,hFADC[1]->GetBinContent(ix));
	}
      hbox->SetFillColor(1);
      hl->SetFillColor(4);
      hu->SetFillColor(2);
      hbox->Draw("same");
      hl->Draw("same");
      hu->Draw("same");
      c1->Modified();
      c1->Update();
      fprintf(stdout, "Upper: %d  Lower: %d\n", 
	      (Int_t)TMath::Floor(hu->Integral()+0.5), 
	      (Int_t)TMath::Floor(hl->Integral()+0.5) 
	      );
    }
}







void plotVEM()
{
  Int_t k;
  for (k=1; k<=2; k++)
    {
      c3 -> cd(k);
      p1.hVEM[2-k] -> Draw();
    }
}

void plotPed()
{
  Int_t k;
  for (k=1; k<=2; k++)
    {
      c3 -> cd(k);
      p1.hPed[2-k] -> Draw();
      p1.hFadcPed[2-k] -> SetLineColor(2);
      p1.hFadcPed[2-k] -> Draw("same");
    }
}


bool pfadc(Int_t ihit = 0)
{
  Double_t x1,y1,x2,y2,dx;
  Int_t j,k;
  if(!p1.fadc_hist(ihit)) return false;
  for (k=0; k<2; k++)
    {
      c3->cd(2-k);
      p1.hfadc[k]->Draw();
      // channel where the signal started.  Draw a red line through
      // the left edge of this channel.
      dx = p1.hfadc[k]->GetBinWidth(1);
      x1 = -0.5 + dx * p1.rufptn->sstart[ihit][k];
      y1 = p1.hfadc[k]->GetMinimum(); 
      y2 = p1.hfadc[k]->GetMaximum();
      pass1plot_drawLine(x1,y1,x1,y2,1,2);

      // channel of the point of inflection of the first signal. Draw a blue line through the
      // left edge of this channel.
      x1 = -0.5 + dx * p1.rufptn->lderiv[ihit][k];
      pass1plot_drawLine(x1,y1,x1,y2,1,4);
      


      // channel where the signal has ended. Draw a red line through the right edge of
      // this channel.
      j = p1.rufptn->sstop[ihit][k]+1 + (p1.rufptn->nfold[ihit]-1)*129;  // adjusted signal stop channel
      x1=p1.hfadc[k]->GetBinLowEdge(j)+1.0;
      pass1plot_drawLine(x1,y1,x1,y2,1,2);
     


      // draw the horizontal pedestal line
      x1 = 0.0;
      j=127+(p1.rufptn->nfold[ihit]-1)*129; // largest FADC channel
      x2=p1.hfadc[k]->GetBinLowEdge(j+1) + 1.0;
      y1 = p1.rufptn->ped[ihit][k];
      pass1plot_drawLine(x1,y1,x2,y1,3,1);
      pass1plot_drawLine(x1,y1-p1.rufptn->pederr[ihit][k],x2,y1-p1.rufptn->pederr[ihit][k],1,2);
      pass1plot_drawLine(x1,y1+p1.rufptn->pederr[ihit][k],x2,y1+p1.rufptn->pederr[ihit][k],1,2);
      
    }
  return true;
}

bool pfadc (Int_t x, Int_t y)
{
  Int_t i;
  Int_t irfuptn_hold[NWFMAX];
  Int_t nrufptn_hold;
  Int_t irufptn;
  Int_t xxyy;
  xxyy = 100*x + y;
  nrufptn_hold = 0;
  for (irufptn=0; irufptn < p1.rufptn->nhits; irufptn ++ )
    {
      if (p1.rufptn->xxyy[irufptn] == xxyy)
	{
	  fprintf (stdout, "irufptn = %d matches xxyy=%04d\n",
		   irufptn,xxyy);
	  irfuptn_hold[nrufptn_hold] = irufptn;
	  nrufptn_hold++;
	}
    }
  if (nrufptn_hold > 0)
    { 
      if (nrufptn_hold > 1)
	{
	  fprintf (stdout, "Plotting the 1st signal for xxyy=%04d\n",xxyy);
	  fprintf (stdout, "To see the rest, do:\n");
	  for (i = 1; i < nrufptn_hold; i++)
	    {
	      irufptn = irfuptn_hold[i];
	      fprintf (stdout, "pfadc(%d)\n",irufptn);
	    }
	}
      irufptn = irfuptn_hold[0];
      return pfadc(irufptn);
    }
  fprintf (stdout, "Unable to find matches for x = %d y = %d\n", x, y);
  return false;
}




bool 
pnpart(Int_t ihit = 0)
{
  Int_t iwf,k;
  

  
  if(!pfadc(ihit))
    return false;
  
  if(!p1.npart_hist(ihit))
    return false;
  
  iwf = p1.rufptn->wfindex[ihit];
  for (k=0; k<2; k++)
    {
      c6->cd(2-k);
      p1.hNpart[k]->SetLineColor(2);
      p1.hNpart[k]->Draw("");
      p1.hNfadc[k]->Draw ("same");      
    }
  
  
  return true;
}





void mwf(Bool_t fsclust = false)
{
  p1.analyze_Mf_hits(fsclust);
  c3->cd(1);
  p1.hNfold->Draw();
  c3->cd(2);
  p1.hQrat->Draw();
}


void histDcore()
{
  p1.histDcore();


  
  c3->cd(1);
  p1.hDcoreR2vsRmin->Draw("box");
  p1.pDcoreR2vsRmin->SetLineColor(2);
  p1.pDcoreR2vsRmin->SetLineWidth(3);
  p1.pDcoreR2vsRmin->Draw("same");
  
  c3->cd(2);
  p1.hNremVsRmin->Draw("box");
  p1.pNremVsRmin->SetLineColor(2);
  p1.pNremVsRmin->SetLineWidth(3);
  p1.pNremVsRmin->Draw("same");
  

  
  c6->cd(1);
  p1.hDcoreR2vsOneOverQmin->Draw("box");
  p1.pDcoreR2vsOneOverQmin->SetLineColor(2);
  p1.pDcoreR2vsOneOverQmin->SetLineWidth(3);
  p1.pDcoreR2vsOneOverQmin->Draw("same");
  
  
  
  c6->cd(2);
  p1.hNremVsOneOverQmin->Draw("box");
  p1.pNremVsOneOverQmin->SetLineColor(2);
  p1.pNremVsOneOverQmin->SetLineWidth(3);
  p1.pNremVsOneOverQmin->Draw("same");


  

  
}


void histSignal()
{


  p1.histSignal();
  
  c3->cd(1);
  p1.hQscat->Draw("box");
  p1.pQscat->SetLineColor(2);
  p1.pQscat->SetLineWidth(3);
  p1.pQscat->Draw("same");


  c3->cd(2);
  p1.hTscat->Draw("box");
  p1.pTscat->SetLineColor(2);
  p1.pTscat->SetLineWidth(3);
  p1.pTscat->Draw("same");
  
  c1->cd();
  p1.hQupQloRat->Draw();

  c2->cd();
  p1.hQupQloRatScat->Draw("box");
  p1.pQupQloRatScat->SetLineColor(2);
  p1.pQupQloRatScat->SetLineWidth(2);
  p1.pQupQloRatScat->Draw("same");

}




void histSat()
{

  p1.histSat();


  // c1->cd();
//   p1.hDcoreRnoSat->Draw();
  

  c1->cd();
  
  p1.hTdSat->SetLineWidth(2);
  p1.hTdSat->Draw();
  
  p1.hTsSat->SetLineColor(4);
  p1.hTsSat->SetLineWidth(2);
  p1.hTsSat->Draw("same");
  
  p1.hTrSat->SetLineColor(2);
  p1.hTrSat->SetLineWidth(2);
  p1.hTrSat->Draw("same");
  

  
  c2->cd();
  pass1plot_plotScat(p1.hDcoreRnoSatVsR,p1.pDcoreRnoSatVsR);
  
  
 
}



void pfadcs(Double_t nsec = 1.0)
{
  Int_t ihit;
  Int_t k;

  TStopwatch t;

  for (ihit=0; ihit < p1.rufptn->nhits; ihit++)
    { 
      pfadc(ihit);
      fprintf (stdout, "ihit = %d\n",ihit);
      for (k=1; k<=2; k++)
	{
	  c3->cd(k);
	  gPad->Modified();
	  gPad->Update();
	}
      t.Start(kTRUE);
      while (t.RealTime() < nsec) t.Continue();
      
    }

  
}


// Counte charge vs Time plot
void plotQvsTime()
{
  
  Int_t isd;
  Int_t npts;
  Int_t ngpts;

  Double_t q_all[NWFMAX];
  Double_t t_all[NWFMAX];
  Double_t q_good[NWFMAX];
  Double_t t_good[NWFMAX];
  
  npts  = 0;
  ngpts = 0;
  for (isd=0; isd < p1.rusdgeom->nsds; isd++)
    {
      if (p1.rusdgeom->igsd[isd] < 1)
	continue;      
      t_all[npts] = p1.rusdgeom->sdtime[isd];
      q_all[npts] = p1.rusdgeom->pulsa[isd];
      npts ++;
      
      if (p1.rusdgeom->igsd[isd] < 4)
	continue;
      t_good[ngpts] = p1.rusdgeom->sdtime[isd];
      q_good[ngpts] = p1.rusdgeom->pulsa[isd];
      ngpts ++;
      
    }

  fprintf (stderr, "ready\n");
  TGraph *gQvsTall = new TGraph(npts,t_all,q_all);
  gQvsTall->SetMarkerStyle(20);
  gQvsTall->SetMarkerSize(1.0);
  gQvsTall->SetMarkerColor(1);
  
  fprintf (stderr, "OK\n");
  TGraph *gQvsTgood = new TGraph(ngpts,t_good,q_good);
  gQvsTgood->SetMarkerSize(1.0);
  gQvsTgood->SetMarkerColor(3);


  c1->cd();
  c1->Clear();
  gQvsTall->Draw("a,p");
  gQvsTgood->Draw("p");
  
}

void recInfo()
{

 
  Double_t t_theta;
  Double_t t_phi;
  Double_t t_xcore;
  Double_t t_ycore;


  Double_t p_theta, ep_theta;
  Double_t p_phi,   ep_phi;
  Double_t p_xcore, ep_xcore;
  Double_t p_ycore, ep_ycore;


  Double_t ls_theta, els_theta;
  Double_t ls_phi,   els_phi;
  Double_t ls_xcore, els_xcore;
  Double_t ls_ycore, els_ycore;

  
  Double_t ls1_theta, els1_theta;
  Double_t ls1_phi,   els1_phi;
  Double_t ls1_xcore, els1_xcore;
  Double_t ls1_ycore, els1_ycore;

  Double_t ldf_xcore, eldf_xcore;
  Double_t ldf_ycore, eldf_ycore;
  Double_t ldf_s800,  ldf_log10en;

  
  
  Double_t gldf_theta, egldf_theta;
  Double_t gldf_phi,   egldf_phi;
  Double_t gldf_xcore, egldf_xcore;
  Double_t gldf_ycore, egldf_ycore;
  Double_t gldf_s800,  gldf_log10en;

  Double_t fd_theta,fd_phi,fd_t_rp,fd_rp,fd_psi,fd_epsi;
  Double_t fd_core[3]; // FD core position in SD frame and in [1200m] units
  Int_t fdsiteid;
  char fdsite_name[2][3] = {"BR","LR"};

  Double_t mctheta,mcphi,mcxcore,mcycore,mc_log10en;

  t_theta = p1.rufptn->tyro_theta[2];
  t_phi   = p1.rufptn->tyro_phi[2];
  t_xcore = p1.rufptn->tyro_xymoments[2][0];
  t_ycore = p1.rufptn->tyro_xymoments[2][1];

  
  p_theta   = p1.rusdgeom->theta[0]; ep_theta = p1.rusdgeom->dtheta[0];
  p_phi     = p1.rusdgeom->phi[0];   ep_phi = p1.rusdgeom->dphi[0];
  p_xcore   = p1.rusdgeom->xcore[0]; ep_xcore = p1.rusdgeom->dxcore[0];
  p_ycore   = p1.rusdgeom->ycore[0]; ep_ycore = p1.rusdgeom->dycore[0];

  
  ls_theta   = p1.rusdgeom->theta[1]; els_theta = p1.rusdgeom->dtheta[1];
  ls_phi     = p1.rusdgeom->phi[1];   els_phi = p1.rusdgeom->dphi[1];
  ls_xcore   = p1.rusdgeom->xcore[1]; els_xcore = p1.rusdgeom->dxcore[1];
  ls_ycore   = p1.rusdgeom->ycore[1]; els_ycore = p1.rusdgeom->dycore[1];

  ls1_theta   = p1.rusdgeom->theta[2]; els1_theta = p1.rusdgeom->dtheta[2];
  ls1_phi     = p1.rusdgeom->phi[2];   els1_phi = p1.rusdgeom->dphi[2];
  ls1_xcore   = p1.rusdgeom->xcore[2]; els1_xcore = p1.rusdgeom->dxcore[2];
  ls1_ycore   = p1.rusdgeom->ycore[2]; els1_ycore = p1.rusdgeom->dycore[2];


  ldf_xcore   = p1.rufldf->xcore[0];   eldf_xcore = p1.rufldf->dxcore[0];
  ldf_ycore   = p1.rufldf->ycore[0];   eldf_ycore = p1.rufldf->dycore[0];
  ldf_s800    = p1.rufldf->s800[0];
  ldf_log10en = 18.0+TMath::Log10(p1.rufldf->energy[0]);

  
  
  gldf_theta   = p1.rufldf->theta;     egldf_theta = p1.rufldf->dtheta;
  gldf_phi     = p1.rufldf->phi;       egldf_phi = p1.rufldf->dphi;
  gldf_xcore   = p1.rufldf->xcore[1];  egldf_xcore = p1.rufldf->dxcore[1];
  gldf_ycore   = p1.rufldf->ycore[1];  egldf_ycore = p1.rufldf->dycore[1];
  gldf_s800    = p1.rufldf->s800[1];
  gldf_log10en = 18.0+TMath::Log10(p1.rufldf->energy[1]);
  
 

  fprintf (stdout, "\n\n------------------------------------------");
  fprintf (stdout," RECONSTRUCTION INFORMATION: ");
  fprintf (stdout,"--------------------------------------------------\n\n");
  fprintf (stdout, "\nEVENT NUMBER: %d\n\n",(Int_t)p1.pass1tree->GetReadEvent());
  
  fprintf (stdout,"METHOD");
  fprintf (stdout,"%12s %20s %20s %20s %22s %16s\n\n",
	   "THETA","PHI","XCORE","YCORE","S800","LOG10(E/EV)");
  fprintf (stdout, "TYRO: %12.2f %22.2f %17.2f %20.2f",
	   t_theta,t_phi,t_xcore,t_ycore);
  fprintf (stdout, "%24s %16s\n","----","-----------");

  
  fprintf (stdout, "PLANE: %11.2f +/- %5.2f %12.2f +/- %5.2f",
	   p_theta,ep_theta,p_phi,ep_phi);
  fprintf (stdout, "%8.2f +/- %5.2f %10.2f +/- %5.2f",
	   p_xcore,ep_xcore,p_ycore,ep_ycore);
  fprintf (stdout, "%14s %16s\n","----","-----------");
  

  fprintf (stdout, "LINSLEY: %9.2f +/- %5.2f %12.2f +/- %5.2f",
	   ls_theta,els_theta,ls_phi,els_phi);
  fprintf (stdout, "%8.2f +/- %5.2f %10.2f +/- %5.2f",
	    ls_xcore,els_xcore,ls_ycore,els_ycore);
  fprintf (stdout, "%14s %16s\n","----","-----------");


  
  fprintf (stdout, "LINSLEY1: %8.2f +/- %5.2f %12.2f +/- %5.2f",
	   ls1_theta,els1_theta,ls1_phi,els1_phi);
  fprintf (stdout, "%8.2f +/- %5.2f %10.2f +/- %5.2f",
	   ls1_xcore,els1_xcore,ls1_ycore,els1_ycore);
  fprintf (stdout, "%14s %16s\n","----","-----------");
  

  fprintf (stdout, "LDF: %20s %21s","------------","------------");
  fprintf (stdout, "%12.2f +/- %5.2f %10.2f +/- %5.2f",
	   ldf_xcore,eldf_xcore,ldf_ycore,eldf_ycore);
  fprintf (stdout,"%14.2f %10.2f\n",ldf_s800,ldf_log10en);
  
  
  fprintf (stdout, "GEOM-LDF: %8.2f +/- %5.2f %12.2f +/- %5.2f",
	   gldf_theta,egldf_theta,gldf_phi,egldf_phi);
  fprintf (stdout, "%8.2f +/- %5.2f %10.2f +/- %5.2f",
	   gldf_xcore,egldf_xcore,gldf_ycore,egldf_ycore);
  fprintf (stdout,"%14.2f %10.2f\n",gldf_s800,gldf_log10en);
  
  for (int ifd=0; ifd<2; ifd++)
    {
      if (p1.set_fdplane(ifd,true,false))
	{
	  fd_theta = p1.fdplane->shower_zen * TMath::RadToDeg();
	  fd_phi = p1.fdplane->shower_azm * TMath::RadToDeg() + 180.0;
	  if (fd_phi > 360.0) fd_phi -= 360.0;
	  fd_psi = p1.fdplane->psi * TMath::RadToDeg();
	  fd_epsi = p1.fdplane->epsi * TMath::RadToDeg();
	  fd_t_rp = p1.fdplane->t0/1.0e3;
	  fd_rp   = p1.fdplane->rp;
	  fdsiteid = p1.fdplane->siteid;
	  if (fdsiteid < 0 || fdsiteid > 1)
	    {
	      fprintf (stderr, "\n\n ****** FD site ID %d is not supported ****\n\n",fdsiteid);
	      return;
	    }
	  fdsite2clf(fdsiteid,p1.fdplane->core,fd_core);
	  for (Int_t i=0; i<3; i++)
	    fd_core[i] /= 1.2e3;
	  fd_core[0] -= sd_origin_x_clf;
	  fd_core[1] -= sd_origin_y_clf;
	  fprintf (stdout, "%s-FD: %11.2f %22.2f %17.2f %20.2f",
		   fdsite_name[fdsiteid],fd_theta,fd_phi,fd_core[0],fd_core[1]);
	  fprintf (stdout, "%24s %16s\n","----","-----------");
	  fprintf (stdout, 
		   "%s-FD (MORE): ZCORE = %.2f T_PR = %.2f RP = %.4e PSI = %.2f +/- %.2f\n",
		   fdsite_name[fdsiteid], fd_core[2],fd_t_rp, fd_rp,fd_psi,fd_epsi);
	}
    }

  if(p1.haveMC)
    {
      if(p1.rusdmc->energy > 1e-7)
	{
	  mctheta=TMath::RadToDeg()*p1.rusdmc->theta;
	  mcphi=TMath::RadToDeg()*p1.rusdmc->phi;
	  mcxcore=p1.rusdmc1->xcore;
	  mcycore=p1.rusdmc1->ycore;
	  mc_log10en=18.0+TMath::Log10(p1.rusdmc->energy);
	  fprintf (stdout, "MC:       %8.2f +/- %5.2f %12.2f +/- %5.2f",mctheta,0.0,mcphi,0.0);
	  fprintf (stdout, "%8.2f +/- %5.2f %10.2f +/- %5.2f",mcxcore,0.0,mcycore,0.0);
	  fprintf (stdout,"%14.2f %10.2f\n",0.0,mc_log10en);
	}
      else
	fprintf(stdout, "MC:       --------------   N/A -------------\n");
    }
  fprintf (stdout, "\n\n");
  fflush(stdout);

}





