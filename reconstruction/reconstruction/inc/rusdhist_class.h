/*
 * rusdhist_class.h
 *
 *  Created on: Nov 30, 2009
 *      Author: ivanov
 */

#ifndef RUSDHIST_CLASS_H_
#define RUSDHIST_CLASS_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "sduti.h"
#include "TObject.h"
#include "TH1D.h"
#include "TProfile.h"
#include "TH2D.h"
#include "TTree.h"
#include "TMath.h"
#include "TFile.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TString.h"
#include "rusdhist.h"

// For making S800 vs sec(theta) profile plots
#define NLOG10EBINS 6
#define LOG10EMIN 18.8
#define LOG10EMAX 20.0

// latest energy estimation routine
extern double sden_jun2010(double s800, double theta);

#define NCUTLEVELS 8 // levels of cuts
class rusdhist_class
{
public:
  
  const listOfOpt& opt;  // reference to command line options
  
  // Reconstructed variables (1D histograms)
  TH1D *hTheta[NCUTLEVELS];
  TH1D *hPhi[NCUTLEVELS];
  TH1D *hGfChi2Pdof[NCUTLEVELS];
  TH1D *hLdfChi2Pdof[NCUTLEVELS];
  TH1D *hXcore[NCUTLEVELS];
  TH1D *hYcore[NCUTLEVELS];
  TH1D *hS800[NCUTLEVELS];
  TH1D *hEnergy[NCUTLEVELS];
  TH1D *hNgSd[NCUTLEVELS];
  TH1D *hQtot[NCUTLEVELS];
  TH1D *hQtotNoSat[NCUTLEVELS];
  TH1D *hQpSd[NCUTLEVELS];
  TH1D *hQpSdNoSat[NCUTLEVELS];
  TH1D *hNsdNotClust[NCUTLEVELS];
  TH1D *hQpSdNotClust[NCUTLEVELS];
  TH1D *hPdErr[NCUTLEVELS];
  TH1D *hSigmaS800oS800[NCUTLEVELS];
  TH1D *hHa[NCUTLEVELS];
  TH1D *hSid[NCUTLEVELS];
  TH1D *hRa[NCUTLEVELS];
  TH1D *hDec[NCUTLEVELS];
  TH1D *hL[NCUTLEVELS];
  TH1D *hB[NCUTLEVELS];
  TH1D *hSgl[NCUTLEVELS];
  TH1D *hSgb[NCUTLEVELS];
  
  // Reconstructed variables (profile plots)
  TProfile *pNgSdVsEn[NCUTLEVELS];
  TProfile *pNsdNotClustVsEn[NCUTLEVELS];
  TProfile *pQtotVsEn[NCUTLEVELS];
  TProfile *pQtotNoSatVsEn[NCUTLEVELS];


  // Calibration information
  TH1D *hFadcPmip[2];
  TH1D *hFwhmMip[2];
  TH1D *hPchPed[2];
  TH1D *hFwhmPed[2];

  // Resolution histograms (MC only)
  TH1D *hThetaRes[NCUTLEVELS];
  TH1D *hPhiRes[NCUTLEVELS];
  TH1D *hXcoreRes[NCUTLEVELS];
  TH1D *hYcoreRes[NCUTLEVELS];
  TH1D *hEnergyResRat[NCUTLEVELS];
  TH1D *hEnergyResLog[NCUTLEVELS];
  TH2D *hEnergyRes2D[NCUTLEVELS];
  TProfile *pEnergyRes[NCUTLEVELS];

  // S800 vs sec(theta) profile plots (MC only)
  TProfile *pS800vsSecTheta[NCUTLEVELS][NLOG10EBINS];



  TString cutName[NCUTLEVELS];
  
  rusdhist_class(listOfOpt& passed_opt);
  virtual ~rusdhist_class();
  void Fill(bool have_mc_banks = false);

private:
  TFile *rootfile;
  void book_hist(const char* rootfilename);
  void end_hist();
  void getNameAndTitle(const char *hNameBare, const char *hTitleBare,
		       Int_t icut, TString *hName, TString *hTitle);
  void setXtitle(TH1 *h, const char *xtitle);
  void printErr(const char *form, ...);
  // weights the E^-3 MC thrown energies so that it follows HiRes spectral
  // indices ( ankle either 18.65 or 18.75 depending on the option )
  // position at the ankle has weight = 1
  double ankle_weight_for_e3(double energyEeV);
};

#endif /* RUSDHIST_CLASS_H_ */
