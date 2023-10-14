
void wait_nsec(Double_t n_wait_sec=1.0)
{
  TStopwatch t;
  t.Start(kTRUE);
  while (t.RealTime() < n_wait_sec) t.Continue();
}


// to print the histograms into file descriptor
bool print_data_mc(TString hName, FILE *fp = stdout)
{
  
  TH1 *hdt = 0;
  TH1 *hmc = 0;
  TH2 *h2dt = 0; 
  TH2 *h2mc = 0;
  TObject *odt = 0;
  TObject *omc = 0;

  Int_t    nx,ny,ix,iy;
  Double_t bcont;
  double a;
  
  if (fp == 0)
    fp = stdout;
  
  odt = dtfl->Get(hName);
  if (odt==0)
    {
      cerr << hName << " not found in the data file " << endl;
      return;
    }
  omc = mcfl->Get(hName);
  if (omc==0)
    {
      cerr << hName << " not found in the MC file " << endl;
      return;
    }
  
  if (odt->InheritsFrom("TH2"))
    {
      h2dt = (TH2 *)odt;
      if (!omc->InheritsFrom("TH2"))
	{
	  cerr << "mc:   " << hName << " must be inherit from TH2" << endl;
	  return;
	}
      h2mc = (TH2 *)omc;
      nx = h2dt->GetNbinsX();
      ny = h2dt->GetNbinsY();
      a = (fabs(h2mc->Integral())< 1e-5 ? 0.0 : h2dt->Integral()/h2mc->Integral());
      fprintf(fp,"BINX:\tBINY:\tDATA:\tMC:\tNMC\n");
      for (ix=1; ix<=nx; ix++)
	{
	  for (iy=1;iy<=ny; iy++)
	    {
	      fprintf(fp,"%.2f\t%.2f\t%d\t%d\t%.5e\n",
		      h2dt->GetBinCenter(ix),
		      h2dt->GetBinCenter(iy),
		      (int)h2dt->GetBinContent(ix,iy),
		      (int)h2mc->GetBinContent(ix,iy),
		      a*h2mc->GetBinContent(ix,iy));
	    }
	}
      return;
    }
  else if (odt->InheritsFrom("TH1"))
    {
      hdt=(TH1 *)odt;
      if (!omc->InheritsFrom("TH1"))
	{
	  cerr << "mc:   " << hName << " must inherit from TH1" << endl;
	  return;
	}
      hmc=(TH1 *)omc;
      
      nx = hdt->GetNbinsX();
      a = (fabs(hmc->Integral())< 1e-5 ? 0.0 : hdt->Integral()/hmc->Integral());
      fprintf(fp,"BINX:\tDATA:\tMC:\tNMC:\n");
      for (ix=1; ix<=nx; ix++)
	{
	  fprintf(fp,
		  "%.2f\t%d\t%d\t%.5e\n",
		  hdt->GetBinCenter(ix),
		  (int)hdt->GetBinContent(ix),
		  (int)hmc->GetBinContent(ix),
		  a*hmc->GetBinContent(ix));
	  
	}
      return;
    }
  else
    {
      cerr << "Type " << odt->ClassName() << " is not supported" << endl;
    }
}


bool compare_data_mc(TH1D *h1, TH1D *h2, Bool_t verbose=true, bool suppress_zero=true)
{
  TH1D *hcomp, *hcompto;
  TH1D *hrat;
  int nb1,nb2;
  double s1,s2;
  double normFactor;
  double ymax;
  TPaveStats *stats;
  Double_t x1ndc,x2ndc,y1ndc,y2ndc;
  hcomp=(TH1D *)gROOT->FindObject("hcomp");
  if(hcomp) hcomp->Delete();
  hcomp=(TH1D *)h1->Clone("hcomp");
  hcompto=(TH1D *)gROOT->FindObject("hcompto");
  if(hcompto) hcompto->Delete();
  hrat=(TH1D *)gROOT->FindObject("hrat");
  if(hrat) hrat->Delete();  
  hcompto=(TH1D *)h2->Clone("hcompto");
  hcompto->SetTitle("");
  hcomp->Reset();
  hcompto->Reset();
  hcomp->Add(h1);;
  hcompto->Add(h2);
  nb1=hcomp->GetNbinsX();
  nb2=hcompto->GetNbinsX();
  if(nb1!=nb2){
    printf("Both histograms must have the same binning!\n");
    return false;
  }
  s1=hcomp->Integral();
  s2=hcompto->Integral();
  if(verbose)
    printf("Integrals: %f %f\n",s1,s2);
  if(s1*s1<1e-5){
    printf("1st histogram sum is too small, can't normalize\n");
    return false;
  }
  // Normalize the 2nd histogram to the 1st one
  normFactor=s1/s2;
  hcompto->Sumw2();
  hcompto->Scale(normFactor);
  if(hcomp->GetMaximum() < hcompto->GetMaximum())
    hcomp->SetMaximum(1.1 * (hcompto->GetMaximum()));
  hrat=(TH1D *)hcomp->Clone("hrat");
  // Compute the ratio
  hrat->Sumw2();
  hrat->Divide(hcompto);
  hrat->SetEntries(h1->GetEntries());
  if(verbose)
    hrat->Fit("pol1");
  else
    hrat->Fit("pol1","Q");
  hrat->SetTitle("DATA/MC Ratio");
  hrat->GetFunction("pol1")->SetParNames("const","slope");
  // plot the comparison histograms
  cdtmc->cd(1);
  hcomp->SetLineColor(1);    // Data - BLACK
  hcomp->SetMarkerStyle(21);
  if(!suppress_zero)
    hcomp->SetMinimum(0.0);
  hcomp->Draw("e1p");
  gPad->Modified();
  gPad->Update();
  if((stats=(TPaveStats *)hcomp->FindObject("stats")))
    {
      stats->SetOptStat(111110);
      stats->SetOptFit(0);
      x1ndc=stats->GetX1NDC();
      y1ndc=stats->GetY1NDC();
      x2ndc=stats->GetX2NDC();
      y2ndc=stats->GetY2NDC();
    }
  hcompto->SetLineColor(2);  // MC - RED
  hcompto->SetLineWidth(2);
  hcompto->Draw("hist,same");
  // plot the ratio histogram
  cdtmc->cd(2);
  hrat->GetYaxis()->SetTitle("Ratio, DATA/MC");
  hrat->GetYaxis()->SetTitleOffset(0.7);
  hrat->GetYaxis()->SetRangeUser(0.0,2.0);
  hrat->SetMarkerStyle(21);
  hrat->SetMarkerSize(0.6);
  hrat->Draw();
  gPad->Modified();
  gPad->Update();
  if((stats=(TPaveStats *)hrat->FindObject("stats")))
    {
      stats->SetOptStat(0);
      stats->SetOptFit(0);
      stats->SetX1NDC(x1ndc-0.1);
      stats->SetY1NDC(y1ndc);
      stats->SetX2NDC(x2ndc);
      stats->SetY2NDC(y2ndc-0.05);
      stats->SetOptFit(1);
    }
  // add the titles
  hcomp->SetTitle(h1->GetTitle());
  hcomp->GetXaxis()->SetTitle(h1->GetXaxis()->GetTitle());
  hcompto->GetXaxis()->SetTitle(h1->GetXaxis()->GetTitle());
  hrat->GetXaxis()->SetTitle(h1->GetXaxis()->GetTitle());
  return true;
}

bool compare_data_mc(TProfile *p1, TProfile *p2)
{
  Double_t ymin,ymax;
  TPaveStats *stats;
  p1->SetLineColor(1);
  p1->SetLineWidth(3);
  p1->SetMarkerStyle(20);
  p1->SetMarkerSize(1.2);
  p1->SetMarkerColor(1);
  p2->SetLineColor(2);
  p2->SetLineWidth(3);
  p2->SetMarkerStyle(20);
  p2->SetMarkerSize(1.2);
  p2->SetMarkerColor(2);
  ymin = p1->GetMinimum();
  ymax = p1->GetMaximum();
  if(p2->GetMinimum() < ymin)
    ymin = p2->GetMinimum();
  if(p2->GetMaximum() > ymax)
    ymax = p2->GetMaximum();
  p1->SetMinimum(ymin-0.1*ymin);
  p1->SetMaximum(ymax+0.5*ymax);
  c1->cd();
  p1->Draw();
  c1->Modified();
  c1->Update();
  if((stats=(TPaveStats *)p1->FindObject("stats")))
    {
      stats->SetY1NDC(0.6);
      stats->SetY2NDC(0.85);
      stats->SetOptStat(111110);
    }
  c1->Modified();
  c1->Update();
  p2->Draw("e1p,x0,same");
  c1->Modified();
  c1->Update();
  return true;
}

void dtmc_cmp(const char *hName, bool verbose=true, bool suppress_zero = true)
{
  TH1D *hdt = 0;
  TH1D *hmc = 0;
  TProfile *pdt = 0;
  TProfile *pmc = 0;
  TObject *odt,*omc;
  TString TH1DtypeName;
  TString TProfileTypeName;
  TH1DtypeName="TH1D";
  TProfileTypeName="TProfile";
  odt = dtfl->Get(hName);
  if (odt==0)
    {
      cerr << hName << " not found in the data file " << endl;
      return;
    }
  omc = mcfl->Get(hName);
  if (omc==0)
    {
      cerr << hName << " not found in the MC file " << endl;
      return;
    }
  if (odt->ClassName() == TH1DtypeName)
    {
      hdt=(TH1D *)odt;
      if (omc->ClassName() != TH1DtypeName)
	{
	  cerr << "mc:   " << hName << " must be of" << TH1DtypeName 
	       << " type" << endl;
	  return;
	}
      hmc=(TH1D *)omc;
      compare_data_mc(hdt,hmc,verbose,suppress_zero);
      c1->cd();
      hdt->Draw();
    }
  else if (odt->ClassName() == TProfileTypeName)
    {
      pdt = (TProfile *)odt;
      if (omc->ClassName() != TProfileTypeName)
	{
	  cerr << "mc:   " << hName << " must be of" << TProfileTypeName
	       << " type" << endl;
	  return;
	}
      pmc = (TProfile *)omc;
      compare_data_mc(pdt,pmc,verbose);
    }
  else
    {
      cerr << "Type " << odt->ClassName() << " is not supported" << endl;
    }
}



void dtmc_ontop(const char *hName)
{
  TH1 *hdt = 0;
  TH1 *hmc = 0;

  double hmax;
  
  odt = dtfl->Get(hName);
  if (odt==0)
    {
      cerr << hName << " not found in the data file " << endl;
      return;
    }
  omc = mcfl->Get(hName);
  if (omc==0)
    {
      cerr << hName << " not found in the MC file " << endl;
      return;
    }
  if (!odt->InheritsFrom("TH1"))
    {
      cerr << odt->ClassName() << " not supported" << endl;
      return;
    }
  if (!omc->InheritsFrom("TH1"))
    {
      cerr << omc->ClassName() << " not supported" << endl;
      return;
    }
  
  hdt=(TH1 *)odt;
  hmc=(TH1 *)omc;
  
  hmax=hdt->GetMaximum();
  hmax=(hmax>=hmc->GetMaximum() ? hmax : hmc->GetMaximum());
  hdt->SetMaximum(hmax+0.1*hmax);
  hmc->SetMaximum(hmax+0.1*hmax);
  
  c1->cd();
  hdt->SetMarkerStyle(20);
  hdt->SetMarkerSize(1.0);
  hdt->SetLineWidth(2);
  hdt->Draw("e1p,x0");
  hmc->SetLineColor(2);
  hmc->SetLineWidth(2);
  hmc->Draw("hist,same");  
  c1->Modified();
  c1->Update();
}

void plotS800vsSecThetaProf(Int_t icut)
{
  Int_t ienergy;
  Double_t elo,eup;
  Int_t icol;
  TProfile *pr;
  TString hName;
  TObject *obj = 0;
  TLegend *legend;
  TString lentry;
  if (icut < 0 || icut >= NCUTLEVELS)
    {
      cerr << "icut must be in 0 to " << (NCUTLEVELS-1) 
	   << " range" << endl;
      return;
    }
  legend = new TLegend(0.71,0.68,0.95,0.995,"");
  legend->SetBorderSize(1);
  legend->SetTextSize(0.04);
  c1->cd();
  for (ienergy=(NLOG10EBINS-1); ienergy>=0; ienergy--)
    {
      icol = 51 + 
	(Int_t)TMath::Floor(((Double_t)ienergy/(Double_t)(NLOG10EBINS-1))*49.5);
      elo = LOG10EMIN + (Double_t)ienergy *
	(LOG10EMAX-LOG10EMIN)/(Double_t)NLOG10EBINS;
      eup = LOG10EMIN + (Double_t)(ienergy+1) * 
	(LOG10EMAX-LOG10EMIN)/(Double_t)NLOG10EBINS;
      lentry.Form("10^{%.1f} - 10^{%.1f} eV",elo,eup);
      hName  = pS800vsSecThetaName;
      hName += icut;
      hName += "_";
      hName += ienergy;
      obj = mcfl->Get(hName);
      if (!obj)
	{
	  cerr << hName << " is not found in the MC file" << endl;
	  return;
	}
      pr = (TProfile *) obj;
      pr->SetLineColor(icol);
      pr->SetLineWidth(2);
      pr->SetStats(0);
      if (ienergy == (NLOG10EBINS-1))
	{
	  pr->Draw();
	  c1->Modified();
	  c1->Update();
	  pr->GetYaxis()->SetRangeUser(10.0,1.0e3);
	  c1->Modified();
	  c1->Update();
	  pr->SetStats(0);
	}
      else
	{
	  pr->Draw("same");
	}
      legend->AddEntry(pr,lentry);
    }
  legend->Draw();
}



void printS800vsSecThetaProf(Int_t icut, FILE *fp = stdout)
{
  Int_t ienergy;
  Double_t elo,eup;
  TProfile *pr;
  TString hName;
  TObject *obj = 0;
  Int_t ix,nx;
  Double_t bcont;
  if (icut < 0 || icut >= NCUTLEVELS)
    {
      cerr << "icut must be in 0 to " << (NCUTLEVELS-1) 
	   << " range" << endl;
      return;
    }
  fprintf(fp,"ELO\tEUP\tSECTH\tS800\n");
  for (ienergy=(NLOG10EBINS-1); ienergy>=0; ienergy--)
    {
      elo = LOG10EMIN + (Double_t)ienergy *
	(LOG10EMAX-LOG10EMIN)/(Double_t)NLOG10EBINS;
      eup = LOG10EMIN + (Double_t)(ienergy+1) * 
	(LOG10EMAX-LOG10EMIN)/(Double_t)NLOG10EBINS;
      
      hName  = pS800vsSecThetaName;
      hName += icut;
      hName += "_";
      hName += ienergy;
      obj = mcfl->Get(hName);
      if (!obj)
	{
	  cerr << hName << " is not found in the MC file" << endl;
	  return;
	}
      pr = (TProfile *) obj;
      nx=pr->GetNbinsX();
      for (ix=1;ix<=nx;ix++)
	{
	  fprintf(fp,"%.2f\t%.2f\t%.2f\t%.2f\n",
		  elo,eup,
		  pr->GetBinCenter(ix),
		  pr->GetBinContent(ix));
	}
      
    }
  
}

void mk_data_mc_plots(Int_t ihist_start = 0, Double_t n_wait_sec = 1.0, 
		      TString outbname="", TString fExt=".png")
{
  Int_t ihist, icut;
  Int_t ilayer;
  TString hName;
  TString sdt,smc;
  TObject *omc, *odt;
  TH1D *h1dt, *h1mc;
  TH2D *h2dt, *h2mc;
  TProfile *p1dt, *p1mc;
  TString xtitle;
  TString fName;
  Bool_t writeFile;
  TString TH1DtypeName,TH2DtypeName,TProfileTypeName;
  TPaveStats *stats;
  FILE *fp;
  bool suppress_zero = true;
  TH1DtypeName="TH1D";
  TH2DtypeName="TH2D";
  TProfileTypeName="TProfile";
  
  writeFile = (outbname.Length() > 0 ? true : false);

  if (ihist_start < 0 || ihist_start >= NDTMCHISTWCUTS)
    {
      cerr << "Warning: ihist_start should be in " << 0 << " to " 
	   << NDTMCHISTWCUTS-1 << " range" << endl;
    }
  

  // Main data/mc histograms with cut levels
  for (ihist = ihist_start; ihist < NDTMCHISTWCUTS; ihist ++ )
    {
      for ( icut = 0; icut < NCUTLEVELS; icut++)
	{
	  hName = dtmchist_wcuts[ihist];
	  
	  // for some histograms, zero shouldn't be suppressed
	  suppress_zero = true;
	  if(hName.Contains("hPhi") || hName.Contains("hRa") ||
	     hName.Contains("hSid"))
	    suppress_zero = false;
	  
	  hName += icut;
	  odt = dtfl->Get(hName);
	  if (!odt)
	    {
	      cerr << " Can't find " << hName << "in data file " << endl;
	      continue;
	    }
	  omc = mcfl->Get(hName);
	  if (!omc)
	    {
	      cerr << " Can't find " << hName << "in data file " << endl;
	      continue;
	    }
	  sdt = odt->ClassName();
	  smc = omc->ClassName();
	  if (sdt != smc)
	    {
	      cerr << "Object " << hName << " has type " << sdt <<
		" in data file but " << smc << " in mc file !" << endl;
	      continue;
	    } 
	  if (sdt == TH1DtypeName)
	    {
	      h1dt = (TH1D *)odt;
	      h1mc = (TH1D *)omc;
	      compare_data_mc(h1dt,h1mc,false,suppress_zero);
	      cdtmc->Modified();
	      cdtmc->Update();
	      fprintf (stdout, "ihist = %d icut = %d hName = '%s'\n",
		       ihist,icut,hName.Data());
	      wait_nsec(n_wait_sec);
	      if(writeFile)
		{
		  fName = outbname;
		  fName += "_";
		  fName += hName;
		  fName += fExt;
		  cdtmc->SaveAs(fName);
		  fName = outbname;
		  fName += "_";
		  fName += hName;
		  fName += ".txt";
		  if(!(fp=fopen(fName.Data(),"w")))
		    {
		      cerr << "Can't start " << fName << endl;
		      return;
		    }
		  print_data_mc(hName,fp);
		  fclose(fp);
		}
	      dtmc_ontop(hName);
	      wait_nsec(n_wait_sec);
	      if(writeFile)
		{
		  fName = outbname;
		  fName += "_";
		  fName += hName;
		  fName += "_";
		  fName += "ontop";
		  fName += fExt;
		  c1->SaveAs(fName);
		}
	    }
	  if (sdt == TProfileTypeName)
	    {
	      p1dt = (TProfile *)odt;
	      p1mc = (TProfile *)omc;
	      compare_data_mc(p1dt,p1mc);
	      c1->Modified();
	      c1->Update();
	      fprintf (stdout, "ihist = %d icut = %d hName = '%s'\n",
		       ihist,icut,hName.Data());
	      wait_nsec(n_wait_sec);
	      if(writeFile)
		{
		  fName = outbname;
		  fName += "_";
		  fName += hName;
		  fName += fExt;
		  c1->SaveAs(fName);
		  fName = outbname;
		  fName += "_";
		  fName += hName;
		  fName += ".txt";
		  if(!(fp=fopen(fName.Data(),"w")))
		    {
		      cerr << "Can't start " << fName << endl;
		      return;
		    }
		  print_data_mc(hName,fp);
		  fclose(fp);
		}
	      dtmc_ontop(hName);
	      wait_nsec(n_wait_sec);
	      if(writeFile)
		{
		  fName = outbname;
		  fName += "_";
		  fName += hName;
		  fName += "_";
		  fName += "ontop";
		  fName += fExt;
		  c1->SaveAs(fName);
		}
	    } 
	}
    }
  // Calibration data/mc histograms
  for (ihist = 0; ihist < NDTMCHISTCALIB; ihist++ )
    {
      for (ilayer=0; ilayer < 2; ilayer++)
	{
	  hName = dtmchist_calib[ihist];
	  hName += ilayer;
	  odt = dtfl->Get(hName);
	  if (!odt)
	    {
	      cerr << " Can't find " << hName << "in data file " << endl;
	      continue;
	    }
	  omc = mcfl->Get(hName);
	  if (!omc)
	    {
	      cerr << " Can't find " << hName << "in data file " << endl;
	      continue;
	    }
	  sdt = odt->ClassName();
	  smc = omc->ClassName();
	  if (sdt != smc)
	    {
	      cerr << "Object " << hName << " has type " << sdt <<
		" in data file but " << smc << " in mc file !" << endl;
	      continue;
	    } 
	  if (sdt == TH1DtypeName)
	    {
	      h1dt = (TH1D *)odt;
	      h1mc = (TH1D *)omc;
	      compare_data_mc(h1dt,h1mc,false);
	      cdtmc->Modified();
	      cdtmc->Update();
	      fprintf (stdout, "ihist = %d ilayer = %d hName = '%s'\n",
		       ihist,ilayer,hName.Data());
	      wait_nsec(n_wait_sec);
	      if(writeFile)
		{
		  fName = outbname;
		  fName += "_";
		  fName += hName;
		  fName += fExt;
		  cdtmc->SaveAs(fName);
		  fName = outbname;
		  fName += "_";
		  fName += hName;
		  fName += ".txt";
		  if(!(fp=fopen(fName.Data(),"w")))
		    {
		      cerr << "Can't start " << fName << endl;
		      return;
		    }
		  print_data_mc(hName,fp);
		  fclose(fp);
		}
	      dtmc_ontop(hName);
	      wait_nsec(n_wait_sec);
	      if(writeFile)
		{
		  fName = outbname;
		  fName += "_";
		  fName += hName;
		  fName += "_";
		  fName += "ontop";
		  fName += fExt;
		  c1->SaveAs(fName);
		}
	    }
	}
    }

  ////// MC-only histograms //////////////////

  c1->cd();
  for (ihist = 0; ihist < NRESHISTWCUTS; ihist++ )
    {
      for ( icut = 0; icut < NCUTLEVELS; icut++)
	{
	  hName = reshist_wcuts[ihist];
	  hName += icut;
	  omc = mcfl->Get(hName);
	  if (!omc)
	    {
	      cerr << " Can't find " << hName << "in data file " << endl;
	      continue;
	    }
	  smc = omc->ClassName();
	  if (smc == TH1DtypeName)
	    {
	      h1mc = (TH1D *)omc;
	      h1mc->Fit("gaus","Q");
	      c1->Modified();
	      c1->Update();
	      if((stats=(TPaveStats *)h1mc->FindObject("stats")))
		{
		  stats->SetY1NDC(0.6);
		  stats->SetY2NDC(0.85);
		  stats->SetOptStat(111110);
		}
	      c1->Modified();
	      c1->Update();
	    }
	  else if (smc == TH2DtypeName)
	    {
	      h2mc = (TH2D *)omc;
	      h2mc->Draw("box");
	      c1->Modified();
	      c1->Update();
	      if((stats=(TPaveStats *)h2mc->FindObject("stats")))
		{
		  stats->SetY1NDC(0.6);
		  stats->SetY2NDC(0.85);
		  stats->SetOptStat(10);
		}
	      c1->Modified();
	      c1->Update();
	    }
	  else if (smc == TProfileTypeName)
	    {
	      p1mc = (TProfile *)omc;
	      p1mc->Draw();
	      c1->Modified();
	      c1->Update();
	      if((stats=(TPaveStats *)p1mc->FindObject("stats")))
		{
		  stats->SetY1NDC(0.6);
		  stats->SetY2NDC(0.85);
		  stats->SetOptStat(111110);
		}
	      c1->Modified();
	      c1->Update();
	    }
	  else
	    {
	      cerr << "Warning: type " << smc << " is not supported" << endl;
	      continue;
	    }
	  fprintf (stdout, "ihist = %d icut = %d hName = '%s'\n",
		   ihist,icut,hName.Data());
	  wait_nsec(n_wait_sec);
	  if(writeFile)
	    {
	      fName = outbname;
	      fName += "_";
	      fName += hName;
	      fName += fExt;
	      c1->SaveAs(fName);
	      fName = outbname;
	      fName += "_";
	      fName += hName;
	      fName += ".txt";
	      if(!(fp=fopen(fName.Data(),"w")))
		{
		  cerr << "Can't start " << fName << endl;
		  return;
		}
	      print_data_mc(hName,fp);
	      fclose(fp);
	    }
	}
    }
  // S800 vs sec(theta) profile plots
  for (icut = 0; icut < NCUTLEVELS; icut++ )
    {
      plotS800vsSecThetaProf(icut);
      c1->SetLogy();
      c1->Modified();
      c1->Update();
      if(writeFile)
	{
	  fName = outbname;
	  fName += "_";
	  fName += pS800vsSecThetaName;
	  fName += icut;
	  fName += fExt;
	  c1->SaveAs(fName);
	  fName = outbname;
	  fName += "_";
	  fName += pS800vsSecThetaName;
	  fName += icut;
	  fName += ".txt";
	  if(!(fp=fopen(fName.Data(),"w")))
	    {
	      cerr << "Can't start " << fName << endl;
	      return;
	    }
	  printS800vsSecThetaProf(icut,fp);
	  fclose(fp);
	}
      wait_nsec(n_wait_sec);
      c1->SetLogy(0);
    }
}
