#ifndef _sdgeomfitter_h_
#define _sdgeomfitter_h_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TObject.h"
#include "TMinuit.h"
#include "TMath.h"
#include "event.h"
#include "sdrt_class.h"
#include "TGraphErrors.h"
#include "TProfile.h"
#include "TH2F.h"


#define NGSDS RUFPTNMH     // max. number of good SDs

#define NFPARS          6  // max. number of fit parameters
#define NFPARS_PLANE    5
#define NFPARS_AGASA    5
#define NFPARS_LINSLEY  5 
#define NFPARS_SPHERE   6
#define NFPARS_PARABOLA 6
#define NFPARS_LINSLEY1 6

#define SDGEOM_TEMP_dt 0.2
#define nsecTo1200m  2.49827048333e-4
#define SDGEOM_TEMP_dr 0.15 // error on core location


// Used in cleaning the space-time cluster.  If chi2 of the fit without the
// i'th point improves by this value or better, then the i'th point is removed
// from space-time cluster
#define SDGEOM_DCHI2 4.0

// Max. number of iterations for minuit
#define SDGEOM_MITR 50000

//
//   Example of a program to fit non-equidistant data points
//   =======================================================
//
//   The fitting function fcn is a simple chisquare function
//   The data consists of 5 data points (arrays x,y,z) + the errors in errorsz
//   More details on the various functions or parameters for these functions
//   can be obtained in an interactive ROOT session with:
//    Root > TMinuit *minuit = new TMinuit(10);
//    Root > minuit->mnhelp("*")  to see the list of possible keywords
//    Root > minuit->mnhelp("SET") explains most parameters
//


class sdgeomfitter : public TObject
{
 public:
  TMinuit *gMinuit;        // Minuit fitter
  Double_t sdorigin_xy[2]; // Position of SD origin with respect to CLF in CLF frame
  TGraphErrors *gTrsdVsR;  // Time fit residual vs distance from core in ground plane
  TGraphErrors *gTrsdVsS;  // Time fit residual vs distance from shower axis
  TGraphErrors *gTvsU;     // Time vs distance along the shower axis in ground plane
  TGraphErrors *gQvsS;     // Charge [VEM] vs Distance from Shower axis
  
  Int_t ngpts;            // rufptn indices of good points after cleaning
  Int_t goodpts[NGSDS];
  
  TH2F *hRsdS;            // Residuals vs S scatter plot
  TProfile *pRsdS;        // Residuals vs S profile plot
  
  
  TH2F *hRsdRho;            // Residuals vs rho scatter plot
  TProfile *pRsdRho;        // Residuals vs rho profile plot
  
  
  // These are the fit parameters
  Double_t 
    theta, phi, dtheta, dphi,  // shower direction (from where it came), degree
    R[2],dR[2],                // core position, in counter separation units
    T0,dT0,                    // Time when the core hits the ground, relative to earliest hit
    a,da;                      // Curvature parameter
  
  
  

  Double_t linsleyA;          // factor that goes into Linley's Time Delay function,
                              // averaged over counters in event
  Double_t chi2;              // FCN at its minimum
  Int_t ndof;                 // number of fit pts minus number of fit params
  
  // loads variables from the space-time cluster
  bool loadVariables_stclust(rufptn_class *rufptn1, rusdgeom_class *rusdgeom1);
  
  void xycoor (Int_t xxyy, Int_t *xy) {xy[0]=xxyy/100; xy[1]=xxyy%100;}
  
  // To remove counters which increase the overall chi2 by at least
  // deltaChi2. First, a worst counter in the cluster is identified. If it increases
  // the chi2 by more than deltaChi2, this counter is removed. If a bad counter
  // was removed, then the routing will search for another worst counter and so on.
  int cleanClust(Double_t deltaChi2 = SDGEOM_DCHI2, bool verbose = true);
  
  

  // Calculates the time residual of i'th data point
  // Output:
  // trst - residual on time
  // terr - time fluctuation
  // chrg - charge of i'th point
  // rcode - distance from core of i'th point
  void calcRsd(Int_t ipoint, Double_t *trsd, Double_t *terr, Double_t *chrg,
      Double_t *rcore, Double_t *score);
  
  
  
  // To clean the profile plots of residuals.
  void cleanRsd();
  
  // This function fills the residuals of each data point into a profile histogram
  // versus R
  void fillRsd();
  
  
  // Fits to various time delay functions. 
  // whatFCN = 0: Fitting to a plane
  // whatFCN = 1: Fitting to AGASA function
  // whatFCN = 2: Fitting to Modified Linsley's function
  // whatFCN = 3: Fitting to a sphere
  // whatFCN = 4: Fitting to a parabola
  bool Ifit(Int_t whatFCN = 0, bool verbose = true);
  sdgeomfitter();
  
  // Calculates the starting values for core using square root of charge
  void calcNewCore (rufptn_class *pass1, Double_t *newCore);
  
  virtual ~sdgeomfitter();
  void compVars(Int_t whatFCN = 0);
  
  
  
 private:
   
   // To be able to restore the old fit parameters and re-use them as starting values.
   // Useful when cleaning out the bad counters
   real8 
     theta_old, phi_old,
     R_old[2],                
     T0_old;                    
   
   void save_fpars();    // to save fit parameters in a buffer
   void restore_fpars(); // to restore the fit parameters, so that they can be used as starting values
  
  
  
  ClassDef(sdgeomfitter,1)
 };

#endif
