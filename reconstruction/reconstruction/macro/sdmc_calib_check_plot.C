

TCanvas *c1 = 0;
TChain   *tMon                      = 0;
TProfile *pFbadSDs                  = 0;
TProfile *pFbadSDsICRR_Only         = 0;
TProfile *pFbadSDsDontUse_Only      = 0;
TProfile *pFbadSDsRutgers_Only      = 0;
TProfile *pFbadSDsICRR_And_Rutgers  = 0;
TLegend *lFbadSDs                   = 0;

void prep_hist_sty(TH1 *h, bool set_date_on_x_axis=false)
{
  h->SetLineWidth(3);
  h->GetXaxis()->CenterTitle();
  h->GetYaxis()->CenterTitle();
  h->GetXaxis()->SetTitleSize(0.055);
  h->GetYaxis()->SetTitleSize(0.055);
  h->GetYaxis()->SetTitleOffset(0.8);
  if(set_date_on_x_axis)
    {
      h->GetXaxis()->SetTitle("");
      TDatime d_offset(20080511,0);
      h->GetXaxis()->SetTimeOffset(d_offset.Convert());
      h->GetXaxis()->SetTimeFormat("%Y/%m/%d");
      h->GetXaxis()->SetTimeDisplay(1);
    }
}


void sdmc_calib_check_plot(const char* bad_counter_file="bad_counters_080511_160511.root",
			    Int_t nbins = 8,
			    Double_t xlo_days = -50.0,
			    Double_t xup_days = 3000.0,
			    Bool_t show_DontUse_Only=false)
{
  Double_t xlo = xlo_days*86400.0;
  Double_t xup = xup_days*86400.0;
  
  gROOT->SetStyle("Plain");
  gStyle->SetLineWidth(3);
  gROOT->Macro("sdmc_calib_check_bitf.C+");
  
  if(tMon)
    delete tMon;
  tMon = new TChain("tMon");
  tMon->AddFile(bad_counter_file);
  gROOT->cd();

  if(pFbadSDs)
    delete pFbadSDs;
  pFbadSDs = new TProfile("pFbadSDs",";Day since 2008/05/11;<F^{BAD SD}_{ICRR OR RUTGERS} / (10 min. mon. cycle)>",
				  nbins,xlo,xup);
  prep_hist_sty(pFbadSDs,1);
  tMon->Draw("nsd/507.0:(day_since_080511+sec_since_midnight/86400.0)*86400.0>>pFbadSDs","","goff");
 
  if(pFbadSDsRutgers_Only)
    delete pFbadSDsRutgers_Only;
  pFbadSDsRutgers_Only = 
    new TProfile("pFbadSDsRutgers_Only",";Day since 2008/05/11;<F^{BAD SD}_{RUTGERS Only} / (10 min. mon. cycle)>",
		 nbins,xlo,xup);
  prep_hist_sty(pFbadSDsRutgers_Only,1);
  tMon->Draw("Sum$((bitf[] != 0 && !is_icrr_cal_mc(bitf[])) ? 1 : 0)/507.0:(day_since_080511+sec_since_midnight/86400.0)*86400.0>>pFbadSDsRutgers_Only","","goff");
  
  if(pFbadSDsICRR_Only)
    delete pFbadSDsICRR_Only;
  pFbadSDsICRR_Only = 
    new TProfile("pFbadSDsICRR_Only",";Day since 2008/05/11;<F^{BAD SD}_{ICRR Only} / (10 min. mon. cycle)>",
		 nbins,xlo,xup);
  prep_hist_sty(pFbadSDsICRR_Only,1);
  tMon->Draw("Sum$((bitf[] != 0 && !is_ru_cal_rc(bitf[])) ? 1 : 0)/507.0:(day_since_080511+sec_since_midnight/86400.0)*86400.0>>pFbadSDsICRR_Only","","goff");
  
  if(pFbadSDsDontUse_Only)
    delete pFbadSDsDontUse_Only;
  pFbadSDsDontUse_Only = 
    new TProfile("pFbadSDsDontUse_Only",";Day since 2008/05/11;<F^{BAD SD}_{ICRR Only} / (10 min. mon. cycle)>",
		 nbins,xlo,xup);
  prep_hist_sty(pFbadSDsDontUse_Only,1);
  tMon->Draw("Sum$(bitf[] == 1 ? 1 : 0)/507.0:(day_since_080511+sec_since_midnight/86400.0)*86400.0>>pFbadSDsDontUse_Only","","goff");
  
  if(pFbadSDsICRR_And_Rutgers)
    delete pFbadSDsICRR_And_Rutgers;
  pFbadSDsICRR_And_Rutgers = 
    new TProfile("pFbadSDsICRR_And_Rutgers",";Day since 2008/05/11;<N^{BAD SD}_{RUTGERS Only} / (10 min. mon. cycle)>",
		 nbins,xlo,xup);
  prep_hist_sty(pFbadSDsICRR_And_Rutgers,1);
  tMon->Draw("Sum$(is_icrr_cal_mc(bitf[]) && is_ru_cal_rc(bitf[]) ? 1 : 0)/507.0:(day_since_080511+sec_since_midnight/86400.0)*86400.0>>pFbadSDsICRR_And_Rutgers","","goff");
 
  
  if(!c1) c1 = new TCanvas("c1","c1",1200,500); c1->SetBottomMargin(0.125); c1->SetLogy();
  pFbadSDs->SetStats(0);
  pFbadSDs->SetMinimum(2e-3);
  pFbadSDs->SetMaximum(0.1);
  pFbadSDs->Draw("hist");
  pFbadSDs->GetYaxis()->SetTitle("<F^{BAD SD} / (10 minute monitoring cycle)>");
  pFbadSDsICRR_Only->SetLineColor(kRed);
  pFbadSDsICRR_Only->Draw("hist,same");
  pFbadSDsDontUse_Only->SetLineColor(kMagenta+1);
  if(show_DontUse_Only)
    pFbadSDsDontUse_Only->Draw("hist,same");
  pFbadSDsRutgers_Only->SetLineColor(kBlue);
  pFbadSDsRutgers_Only->Draw("hist,same");
  pFbadSDsICRR_And_Rutgers->SetLineColor(kGreen+1);
  pFbadSDsICRR_And_Rutgers->Draw("hist,same");

  if(lFbadSDs)
    delete lFbadSDs;
  lFbadSDs = new TLegend(0.12,0.81,0.67,1.00);
  lFbadSDs->AddEntry(pFbadSDs,            "ICRR OR Rutgers","l");
  lFbadSDs->AddEntry(pFbadSDsICRR_Only,   "ICRR NOT Rutgers (Causes DATA/MC disagreement)","l");
  lFbadSDs->AddEntry(pFbadSDsICRR_And_Rutgers,"ICRR AND Rutgers (No problem)","l");
  if(show_DontUse_Only)
    lFbadSDs->AddEntry(pFbadSDsDontUse_Only,
		       "ICRR dontUse flag is set, everything else OK (part of ICRR NOT Rutgers)","l");
  lFbadSDs->AddEntry(pFbadSDsRutgers_Only,"Rutgers NOT ICRR (Not a problem, both DATA and MC have that)","l");
  lFbadSDs->Draw();
  
}
