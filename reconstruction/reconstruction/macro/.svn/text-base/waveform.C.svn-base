using namespace TMath;

// To fill the raw FADC histogram for a given counter
// hFADC[0], hFADC[1] will be allocated and filled:
// hFADC[0] - lower, hFADC[1] - upper
// time on the X-axis is uS
bool fill_fadc_one_SD_uS(Int_t xxyy=0,
			 bool ans_in_vem = false,
			 bool subtract_ped=false, 
			 bool use_icrr_ped=false, 
			 bool verbose=false)
{

  // reference time in the entire event, [uS] ( with respect to UTC second)
  Double_t tref = (1.0e6) * ((p1.rusdgeom->tearliest) - Floor(p1.rusdgeom->tearliest));
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
  Int_t iwf = -1;
  Double_t tref_wf = 0.0; // reference time for waveforms in a particular SD, [uS]
  for (Int_t i=0; i < p1.rusdraw->nofwf; i++)
    {
      if (p1.rusdraw->xxyy[i] == xxyy)
	{
	  iwf      = i;
	  tref_wf  = (1.0e6) * ((double)p1.rusdraw->clkcnt[iwf]) / ((double)p1.rusdraw->mclkcnt[iwf]);
	  break;
	}
    }
  if (iwf == -1)
    {
      fprintf (stderr,"detector XXYY = %04d is not found in the event read out\n",xxyy);
      return false;
    }
  Int_t nfold=0;
  Int_t* wf_ind = new Int_t[p1.rusdraw->nofwf];
  for (Int_t i=0;i<p1.rusdraw->nofwf;i++)
    {
      if(p1.rusdraw->xxyy[i]==p1.rusdraw->xxyy[iwf])
	{
	  wf_ind[nfold]=i;
	  nfold++;
	}
    }
  Int_t nbins = 128;
  Int_t ngaps = nfold-1;
  Int_t i0    = wf_ind[0];
  Double_t *gapsize = 0;
  if (ngaps > 0) 
    {
      gapsize = new Double_t[ngaps];
      for (Int_t i=0; i<ngaps; i++)
	{
	  gapsize [i] =
	    (
	     (Double_t)p1.rusdraw->clkcnt[i0+i+1] / (Double_t)p1.rusdraw->mclkcnt[i0+i+1] -
	     (Double_t)p1.rusdraw->clkcnt[i0+i]   / (Double_t)p1.rusdraw->mclkcnt[i0+i]
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
  
  // bins in 0.02 [uS] units.  Have to be made with respect to the reference time in the entire event.
  Float_t *leftedgeX = new Float_t[nbins+1];
  leftedgeX[0] = -0.5;
  Int_t j = 0;
  for (Int_t i=1; i < (nbins+1); i++)
    {
      if ( i>128 && ((i-129) % 129 == 0) && j < ngaps)
	{
	  leftedgeX[i] = leftedgeX[i-1]+gapsize[j];
	  j++;
	}
      else
	leftedgeX[i] = leftedgeX[i-1]+1.0;
    }
  

  // units have to be converted to uS and the reference time needs to be adjusted, so that all SDs use
  // a common refference time (earliest time in the event readout)
  for (Int_t i=0; i<(nbins+1); i++)
    {
      leftedgeX[i] *= 0.02; // now X-axis is in [uS]
      // adjusting the beginning of the histograms so that its with respect to the
      // common reference time in the event.
      leftedgeX[i] += (tref_wf-tref);
    }
  
  // clean up the previous histograms if allocated and initialize the new ones
  for (Int_t k=0; k<2; k++)
    {
      if(hFADC[k])
	delete hFADC[k];
    }
  TString hTitle;
  hTitle.Form("XXYY=%04d (LOWER)",p1.rusdraw->xxyy[iwf]);
  hFADC[0] = new TH1F ("hFADC0",hTitle, nbins, leftedgeX);
  hTitle.Form("XXYY=%04d (UPPER)",p1.rusdraw->xxyy[iwf]);
  hFADC[1] = new TH1F ("hFADC1",hTitle, nbins, leftedgeX);
  i0=wf_ind[0];
  j = 0;
  for (Int_t i=0; i < nbins; i++)
    { 
      if ( i>127 && ((i-128) % 129 == 0) && j < ngaps)
	{
	  j++;
	  for (Int_t k=0; k<0; k++) hFADC[k] -> SetBinContent(i+1,0.0);
	}
      else
	{
	  for (Int_t k=0; k<2; k++) 
	    {
	      Double_t bcont = p1.rusdraw->fadc[i0+j][k][i-j*129];
	      if (subtract_ped)
		{
		  Double_t ped = 0.0;
		  if (use_icrr_ped)
		    {
		      Int_t icrr_iwf=p1.get_tasdcalibev_iwf(i0+j);
		      if (k==0)
			ped = p1.tasdcalibev->sub[icrr_iwf].lpedAvr;
		      else
			ped = p1.tasdcalibev->sub[icrr_iwf].upedAvr;
		    }
		  else
		    ped = ((double)p1.rusdraw->pchped[i0+j][k]/8.0);
		  bcont -= ped;
		}
	      if (ans_in_vem)
	      	bcont /= (p1.rusdraw->mip[i0+j][k] * Cos(DegToRad()*35.0));
	      hFADC[k]->SetBinContent(i+1,bcont);
	    }
	}
    }
  if(leftedgeX)
    delete[] leftedgeX;
  if(gapsize)
    delete[] gapsize;
  if(wf_ind)
    delete[] wf_ind;
  return true;
}

// to fill FADC traces for multiple SDs:
// nxxyy - number of counters
// *xxyy - array with counter indices
// hfadc[][2] - TH1F histograms that will be allocated
// sf - suffix to add to histogram object names,
// which will be of the form "hFADC_xxyy_ilayer"sfx
// and filled with FADC traces
// returns: how many FADC traces have been filled
int fill_fadc_mul_SD_uS(Int_t nxxyy, 
			Int_t *xxyy,
			TH1F *hfadc[NSDMAX][2],
			TString sfx="",
			bool ans_in_vem = false,
			bool subtract_ped=false,
			bool use_icrr_ped = false,
			bool verbose = false)
{
  int n,xx,yy,ixxyy,ilayer;
  char is_xxyy_filled[SD_X_MAX][SD_Y_MAX];
  TString hfadcName;
  TH1F *hTest;
  if (use_icrr_ped && !p1.have_tasdcalibev)
    {
      fprintf(stderr,"Must have tasdcalibev branch to use icrr pedestals\n");
      return 0;
    }
  memset(is_xxyy_filled,0,sizeof(is_xxyy_filled));
  n = 0;
  for (ixxyy = 0; ixxyy < nxxyy; ixxyy++)
    { 
      xx = xxyy[ixxyy]/100;
      yy = xxyy[ixxyy]%100;
      if (xx < 1 || xx > SD_X_MAX || yy < 1 || yy > SD_Y_MAX)
	{
	  fprintf(stderr,"warning: invalid XXYY = %04d\n",xxyy[ixxyy]);
	  continue;
	}
      if (is_xxyy_filled[xx-1][yy-1])
	{
	  fprintf(stderr, "warning: XXYY = %04d is already filled. skipping.\n",xxyy[ixxyy]);
	  continue;
	}
      if(!fill_fadc_one_SD_uS(xxyy[ixxyy],ans_in_vem,subtract_ped,use_icrr_ped,verbose))
	continue;
      for (ilayer=0; ilayer<2; ilayer++)
	{
	  hfadcName.Form("hFADC_%04d_%d",xxyy[ixxyy],ilayer);
	  if (sfx.Length() > 0)
	    hfadcName += sfx;
	  if ((gROOT->FindObject(hfadcName)))
	    delete (gROOT->FindObject(hfadcName));
	  hfadc[n][ilayer] = (TH1F *)hFADC[ilayer]->Clone(hfadcName);
	}
      is_xxyy_filled[xx-1][yy-1] = 1;
      n++;
    }
  return n;
}


void plotFADC_one_SD_us(Int_t xxyy, bool ans_in_vem = false, bool subtract_ped = false)
{
  if(!fill_fadc_one_SD_uS(xxyy,ans_in_vem,subtract_ped))
    return;
  if(!hFADC[0] || !hFADC[1])
    {
      fprintf(stderr,"error: failed to fill FADC hitograms\n");
      return;
    }  
  Double_t hmin = hFADC[0]->GetMinimum();
  if(hFADC[1]->GetMinimum() < hmin)
    hmin = hFADC[1]->GetMinimum();
  Double_t hmax = hFADC[0]->GetMaximum();
  if(hFADC[1]->GetMaximum() > hmax)
    hmax = hFADC[1]->GetMaximum();  
  Double_t tmin = hFADC[0]->GetXaxis()->GetXmin();
  if(hFADC[1]->GetXaxis()->GetXmin() < tmin)
    tmin = hFADC[1]->GetXaxis()->GetXmin();
  Double_t tmax = hFADC[0]->GetXaxis()->GetXmax();
  if(hFADC[1]->GetXaxis()->GetXmax() > tmax)
    tmax = hFADC[1]->GetXaxis()->GetXmax();
  c2->cd();
  c2->Clear();
  hFADCfrm->SetMinimum(hmin);
  hFADCfrm->SetMaximum(hmax);
  hFADCfrm->GetXaxis()->SetRangeUser(tmin-0.1*tmin,tmax+0.1*tmax);
  hFADCfrm->SetStats(0);
  if(ans_in_vem)
    hFADCfrm->GetYaxis()->SetTitle("Signal [VEM]");
  else
    hFADCfrm->GetYaxis()->SetTitle("Signal [FADC Counts]");
  hFADCfrm->GetXaxis()->SetTitle("Time [#mus]");
  hFADCfrm->GetXaxis()->CenterTitle();
  hFADCfrm->GetYaxis()->CenterTitle();
  hFADCfrm->GetXaxis()->SetTitleSize(0.055);
  hFADCfrm->GetYaxis()->SetTitleSize(0.055);
  hFADCfrm->Draw();
  hFADC[0]->SetLineColor(kBlue);
  hFADC[1]->SetLineColor(kRed);
  hFADC[0]->Draw("same");
  hFADC[1]->Draw("same");
  c2->Modified();
  c2->Update();
}

void plotFADC_all(bool ans_in_vem = false, bool subtract_ped = false)
{
  Int_t nxxyy = 0;
  Int_t *xxyy = new Int_t[p1.rusdgeom->nsds];
  for (Int_t isd=0; isd<p1.rusdgeom->nsds; isd++)
    {
      // choose only working SDs
      if (p1.rusdgeom->igsd[isd] < 1)
	continue;
      xxyy[nxxyy] = p1.rusdgeom->xxyy[isd];
      nxxyy ++;
    }
  Int_t nsdfilled = fill_fadc_mul_SD_uS(nxxyy,xxyy,hFADC_all_counters,"",ans_in_vem,subtract_ped);
  if(xxyy)
    delete[] xxyy;
  Double_t hmin = 1.0e20; // min and maximum of the histograms
  Double_t hmax = -1e20;
  Double_t t1 = 1e20; // earliest and latest histogram beginning times
  Double_t t2 = -1e20;
  for (Int_t isd=0; isd<nsdfilled; isd++)
    {
      Double_t t = hFADC_all_counters[isd][0]->GetBinLowEdge(1);
      if (t < t1)
	t1 = t;
      t = hFADC_all_counters[isd][0]->GetBinLowEdge(hFADC_all_counters[isd][0]->GetNbinsX())
	+ hFADC_all_counters[isd][0]->GetBinWidth(hFADC_all_counters[isd][0]->GetNbinsX());
      if (t>t2)
	t2 = t;
      for (Int_t k=0; k<2; k++)
	{
	  if (hFADC_all_counters[isd][k]->GetMinimum() < hmin)
	    hmin = hFADC_all_counters[isd][k]->GetMinimum();
	  if (hFADC_all_counters[isd][k]->GetMaximum() > hmax)
	    hmax = hFADC_all_counters[isd][k]->GetMaximum();
	}
    }
  hmax *= 1.15;
  hmin -= 0.15 * Abs(hmin);
  
  c1->cd();
  c1->Clear();
  hFADCfrm->SetMinimum(hmin);
  hFADCfrm->SetMaximum(hmax);
  hFADCfrm->GetXaxis()->SetRangeUser(t1-10.0,t2+10.0);
  hFADCfrm->SetStats(0);
  if(ans_in_vem)
    hFADCfrm->GetYaxis()->SetTitle("Signal [VEM]");
  else
    hFADCfrm->GetYaxis()->SetTitle("Signal [FADC Counts]");
  hFADCfrm->GetXaxis()->SetTitle("Time [#mus]");
  hFADCfrm->GetXaxis()->CenterTitle();
  hFADCfrm->GetYaxis()->CenterTitle();
  hFADCfrm->GetXaxis()->SetTitleSize(0.055);
  hFADCfrm->GetYaxis()->SetTitleSize(0.055);
  hFADCfrm->Draw();
  
  // plot using colors
  for (Int_t isd=0; isd<nsdfilled; isd++)
    {
      hFADC_all_counters[isd][0]->SetLineColor(kBlue);
      hFADC_all_counters[isd][1]->SetLineColor(kRed);
      hFADC_all_counters[isd][0]->Draw("same");
      hFADC_all_counters[isd][1]->Draw("same");
    }
  c1->Modified();
  c1->Update();
  
}
