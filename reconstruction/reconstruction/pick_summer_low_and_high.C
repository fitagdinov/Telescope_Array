#include "TMath.h"
#include "TTimeStamp.h"

const Int_t YYMMDD_START =  80511;
const Int_t YYMMDD_STOP  = 180511;

TCanvas *c1  = 0;
TCanvas *c2  = 0;
TCanvas *c3  = 0;

double get_hr_since_mn_cent_16(int hhmmss)
{  
  double hr_since_mn=(1.0/3600.0)*(double)(3600*(hhmmss/10000)+60*((hhmmss%10000)/100)+(hhmmss%100));
  if(hr_since_mn < 8.0)
    hr_since_mn += 24.0;
  return hr_since_mn;
}
// 0 = winter season
// 1 = summer season
int get_season(int yymmdd)
{
  int m = ((yymmdd%10000)/100);
  if(m >= 4 && m <= 9)
    return 1;
  return 0;
}


const Int_t    npbins = 101;
const Double_t p_lo  = -500;
const Double_t p_hi  = 100500;
TProfile* p_gdas_summer_low_high_density[2] = 
  {
    new TProfile("p_gdas_summer_low_high_rho_0","summer low",npbins,p_lo,p_hi,"S"),
    new TProfile("p_gdas_summer_low_high_rho_1","summer high",npbins,p_lo,p_hi,"S")
  };
TProfile* p_gdas_summer_low_high_height[2] = 
  {
    new TProfile("p_gdas_summer_low_high_h_0","summer low",npbins,p_lo,p_hi,"S"),
    new TProfile("p_gdas_summer_low_high_h_1","summer high",npbins,p_lo,p_hi,"S")
  };
TGraphErrors *g_rho_vs_h[2] = {0,0};
TGraphErrors *g_summer_low_high[2] = { 0, 0 };
TGraphErrors *g_summer_low_high_rsd[2] = { 0, 0 };


TF1 *f_summer_low_high[2] = { 0, 0 };

atmpar_class summer_low_high[2];

Double_t function_summer_low_high_ratio(Double_t h)
{
  return f_summer_low_high[1]->Eval(h)/f_summer_low_high[0]->Eval(h);
}

TF1 *f_summer_low_high_ratio = new TF1("f_summer_low_high_ratio","function_summer_low_high_ratio(x)",0.0,1e7);


void pick_summer_low_and_high()
{ 
    
  for (Int_t i=0; i<2; i++)
    {
      p_gdas_summer_low_high_density[i]->Reset();
      p_gdas_summer_low_high_height[i]->Reset();
    }
  if(!have_atmpar)
    {
      fprintf(stderr,"error: no atmpar branch in %s, use atmpar.run program first, then dst2rt_ta.run to create the input\n");
      fprintf(stderr,"root tree file that can be used by this routine\n");
      exit(2);
    }
  for (Int_t entry=0; entry < taTree->GetEntries(); entry++)
    {
      taTree->GetEntry(entry); 
      // use atmpospheric data cyles that are not corrupted and that fit 
      // well into the CORSIKA layer model
      if(atmpar->ndof < 1 || atmpar->chi2/atmpar->ndof > 15.0)
	continue;
      if(!atm.loadVariables(gdas))
	continue; // choose only good atmospheric cycles
      if(atm.yymmddFrom < YYMMDD_START)
	continue;
      if(atm.yymmddFrom > YYMMDD_STOP)
	continue;
      // Choosing summer like season only
      if(get_season(atm.yymmddFrom) != 1)
	continue;


      Int_t ibin = 0;
      if(atm.hhmmssFrom >= 73000 && atm.hhmmssTo > 13000 && atm.hhmmssTo <= 163000)
	ibin = 0;
      else
	ibin = 1;

      // const Int_t idivide        =      0;
      // const Int_t ngdas          =      8;
      // const Int_t gdas_hhmmss_lo =  13000;
      // const Int_t gdas_hhmmss_hi = 223000;
      // const Int_t gdas_len_hours = 3;
      // Int_t ibin = 0;
      // Int_t j = (atm.hhmmssFrom - gdas_hhmmss_lo) / 10000 / gdas_len_hours;
      // Int_t d = j-idivide;
      // if(d < 0)
      // 	d += ngdas;
      // if(d < ngdas/2) 
      // 	ibin = 1;
      // else
      // 	ibin = 0;
      //fprintf(stderr,"%06d %d %d %d %d\n", atm.hhmmssFrom,j, idivide, d, ibin);



      TGraph *g = atm.GetMoVsHderiv();
      TGraph *g1 = atm.GetPvsH();
      for (int i=0; i<g->GetN(); i++)
	{
	  Double_t h,rho,pres;
	  g->GetPoint(i,h,rho);
	  g1->GetPoint(i,h,pres);
	  p_gdas_summer_low_high_density[ibin]->Fill(pres,rho);
	  p_gdas_summer_low_high_height[ibin]->Fill(pres,h);
	}
    }
  
  for (Int_t i=0; i<2; i++)
    {
      if(g_rho_vs_h[i])
	delete g_rho_vs_h[i];
      g_rho_vs_h[i] = new TGraphErrors(0);
      g_rho_vs_h[i]->SetMarkerStyle(21);
      Int_t npts = 0;
      for (Int_t ix=1; ix<= p_gdas_summer_low_high_height[i]->GetNbinsX(); ix++)
	{
	  if(p_gdas_summer_low_high_height[i]->GetBinEntries(ix) < 1)
	    continue;
	  g_rho_vs_h[i]->SetPoint(npts,
				  p_gdas_summer_low_high_height[i]->GetBinContent(ix),
				  p_gdas_summer_low_high_density[i]->GetBinContent(ix));
	  g_rho_vs_h[i]->SetPointError(npts,
				       p_gdas_summer_low_high_height[i]->GetBinError(ix),
				       p_gdas_summer_low_high_density[i]->GetBinError(ix));
	  npts++;
	}
      atm.loadFromRhoVsHgraphWerr(g_rho_vs_h[i]);
      atm.Fit();
      atm.put2atmpar(&summer_low_high[i]);

      if(i == 0)
	{
	  TTimeStamp t1((UInt_t)(2000*10000+YYMMDD_START),(UInt_t)73000,(UInt_t)0,kTRUE,(Int_t)0);
	  TTimeStamp t2((UInt_t)(2000*10000+YYMMDD_STOP),(UInt_t)163000,(UInt_t)0,kTRUE,(Int_t)0);
	  summer_low_high[i].dateFrom = t1.GetTimeSpec().tv_sec;
	  summer_low_high[i].dateTo   = t2.GetTimeSpec().tv_sec;
	}
      else
	{
	  TTimeStamp t1((UInt_t)(2000*10000+YYMMDD_START),(UInt_t)163000,(UInt_t)0,kTRUE,(Int_t)0);
	  TTimeStamp t2((UInt_t)(2000*10000+YYMMDD_STOP),(UInt_t)13000,(UInt_t)0,kTRUE,(Int_t)0);
	  summer_low_high[i].dateFrom = t1.GetTimeSpec().tv_sec;
	  summer_low_high[i].dateTo   = t2.GetTimeSpec().tv_sec;
	}
      
      if(g_summer_low_high[i])
	delete g_summer_low_high[i];
      g_summer_low_high[i] = (TGraphErrors)((TGraphErrors*)atm.GetMoVsHderiv())->Clone();
      if(g_summer_low_high_rsd[i])
	delete g_summer_low_high_rsd[i];
      g_summer_low_high_rsd[i] = new TGraphErrors(0);
      g_summer_low_high_rsd[i]->SetMarkerStyle(20);
      if(f_summer_low_high[i])
	delete f_summer_low_high[i];
      for (Int_t ipoint = 0; ipoint < g_summer_low_high[i]->GetN(); ipoint++)
	{
	  Double_t x,y,ey;
	  g_summer_low_high[i]->GetPoint(ipoint,x,y);
	  ey = g_summer_low_high[i]->GetErrorY(ipoint);
	  TString s; 
	  s.Form("f_summer_low_high_%d",i);
	  f_summer_low_high[i] = (TF1*)atm.GetMoVsHderivfit()->Clone(s);
	  y = (y - f_summer_low_high[i]->Eval(x)) / f_summer_low_high[i]->Eval(x);
	  ey = ey / f_summer_low_high[i]->Eval(x);
	  g_summer_low_high_rsd[i]->SetPoint(ipoint,x,y);
	  g_summer_low_high_rsd[i]->SetPointError(ipoint,0,ey);
	}


    }
  
  if(!c1)
    {
      c1 = new TCanvas("c1","c1",800,600);
      c1->Divide(1,2);
    }
  c1->cd(1);
  g_summer_low_high[1]->Draw("a,e1p");
  c1->cd(2);
  g_summer_low_high[0]->Draw("a,e1p");
  if(!c2)
    {
      c2 = new TCanvas("c2","c2",800,600);
      c2->Divide(1,2);
    }
  c2->cd(1);
  g_summer_low_high_rsd[1]->Draw("a,e1p");
  c2->cd(2);
  g_summer_low_high_rsd[0]->Draw("a,e1p");

  if(!c3)
    c3 = new TCanvas("c3","c3",800,600);
  c3->cd();
  f_summer_low_high_ratio->Draw();
  // f_summer_low_high[1]->SetLineColor(kRed);
  // f_summer_low_high[1]->Draw();
  // f_summer_low_high[0]->SetLineColor(kBlack);
  // f_summer_low_high[0]->Draw("same");

}



