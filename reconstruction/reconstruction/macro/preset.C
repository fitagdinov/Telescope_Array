
{
  
  Bool_t *ran_start;
  Bool_t successFlag;
  
  if (ran_start != 0)
    {
      fprintf(stderr, 
              "A start script can be executed only once per session\n");
      successFlag = false;
      return;
    }
  else
    {
      ran_start = new Bool_t;
      successFlag = true;
    }
  
  const int NDTMCHISTWCUTS  = 29;
  const int NDTMCHISTCALIB = 4;
  const int NRESHISTWCUTS = 8;
  const int NCUTLEVELS = 8;
  
  // For making S800 vs sec(theta) profile plots
  const int NLOG10EBINS = 6;
  const double LOG10EMIN = 18.8;
  const double LOG10EMAX = 20.0;
  

  TString dtmchist_wcuts[NDTMCHISTWCUTS] = 
    {
      "hTheta",
      "hPhi",
      "hGfChi2Pdof",
      "hLdfChi2Pdof",
      "hXcore",
      "hYcore",
      "hS800",
      "hEnergy",
      "hNgSd",
      "hQtot",
      "hQtotNoSat",
      "hQpSd",
      "hQpSdNoSat",
      "hNsdNotClust",
      "hQpSdNotClust",
      "hPdErr",
      "hSigmaS800oS800",
      "hHa",
      "hSid",
      "hRa",
      "hDec",
      "hL",
      "hB",
      "hSgl",
      "hSgb",
      "pNgSdVsEn",
      "pNsdNotClustVsEn",
      "pQtotVsEn",
      "pQtotNoSatVsEn"
    };

  TString dtmchist_calib[NDTMCHISTCALIB]=
    {
      "hFadcPmip",
      "hFwhmMip",
      "hPchPed",
      "hFwhmPed"
    };

  TString reshist_wcuts[NRESHISTWCUTS]=
    {
      "hThetaRes",
      "hPhiRes",
      "hXcoreRes",
      "hYcoreRes",
      "hEnergyResRat",
      "hEnergyResLog",
      "hEnergyRes2D",
      "pEnergyRes"
    };

  // S800 vs sec(theta) profile plots basename
  TString pS800vsSecThetaName="pS800vsSecTheta";

  ///////////// SET THE STYLE /////////////////
  gStyle->SetCanvasColor(0);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(0);
  gStyle->SetFrameFillStyle(0);
  gStyle->SetFrameFillColor(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasBorderSize(0);
  gStyle->SetPadColor(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadBorderSize(0);
  gStyle->SetTitleFillColor(0);
  gStyle->SetStatColor(0);
  gStyle->SetLineWidth(2);
  gStyle->SetOptStat(1);
  gStyle->SetOptFit(1);
  gStyle->SetPalette(1,0);
  
}
