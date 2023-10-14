#ifndef _p2ldffitter_h_
#define _p2ldffitter_h_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TMinuit.h"
#include "TMath.h"
#include "event.h"


#define NGSDS RUFPTNMH      // max. number of good SDs

#define NFPARS          3  // max. number of fit parameters
 
// error on core location
#define P2LDF_TEMP_dr 0.15


// Max. number of iterations for minuit
#define P2LDF_MITR 50000


class p2ldffitter_class
{
 public:
  
  Int_t sds_hit[SDMON_X_MAX][SDMON_Y_MAX];
   
  TMinuit *gMinuit;        // Minuit fitter
  Double_t sdorigin_xy[2]; // Position of SD origin with respect to CLF in CLF frame
  
  
  
  // Event direction
  Double_t theta,phi;
  
  // Fit parameters
  Double_t 
    R[2],dR[2];                // core position, in counter separation units
  Double_t S,dS;               // LDF scale
  
  Double_t S600;               // Signal size at r=600m
  Double_t S600_0;             // Signal size at r=600m, accounting attenuation
  
  Double_t S800;               // Signal size at r=800m
  Double_t S800_0;             // Signal size at r=800m, accoutnting attenuation
  
  Double_t energy;             // Energy in EeV
  Double_t log10en;            // log10 (energy in eV)
 
  Double_t chi2;              // FCN at its minimum
  Int_t ndof;                 // number of fit pts minus number of fit params
  
  
  
  // Points used in fitting, so that they can be retreived outide of class
  Int_t nfpts; // number of fit points
  Int_t napts; // number of actual SDs that had non-zero charge 
  Double_t fX[NGSDS][3];
  Double_t frho[NGSDS];
  Double_t fdrho[NGSDS];
  
  // loads variables needed in the fitter
  bool loadVariables();
  bool doFit(bool fixCore);
  bool hasConverged();
  Int_t clean(Double_t deltaChi2, bool fixCore);
  
  p2ldffitter_class();
  virtual ~p2ldffitter_class();

 };

#endif
