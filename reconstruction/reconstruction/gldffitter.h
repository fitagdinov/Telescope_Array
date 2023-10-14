#ifndef _gldffitter_h_
#define _gldffitter_h_

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
#include "TF1.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TH2F.h"

#define NGSDS RUFPTNMH    // max. number of good SDs
#define GLDF_NFPARS  6    // max. number of fit parameters
#define GLDF_TEMP_dt 0.2
#define nsecTo1200m  2.49827048333e-4
#define GLDF_TEMP_dr 0.15  // error on core location

// Determines by how much one needs to scale the corresponing LDF and Time fit errors to
// have the chi2 in the right place
#define GLDF_ERR_SCALE 0.82


// Used in cleaning the space-time cluster.  If chi2 of the fit without the
// i'th point improves by this value or better, then the i'th point is removed
// from space-time cluster
#define GLDF_DCHI2 3.0

// Max. number of iterations for minuit
#define GLDF_MITR 50000

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


class gldffitter : public TObject
  {
public:

  Int_t sds_hit[SDMON_X_MAX][SDMON_Y_MAX];

  TMinuit *gMinuit; // Minuit fitter
  Double_t sdorigin_xy[2]; // Position of SD origin with respect to CLF in CLF frame
  TGraphErrors *gTrsdVsR; // Time fit residual vs distance from core in ground plane
  TGraphErrors *gTrsdVsS; // Time fit residual vs distance from shower axis
  TGraphErrors *gTvsU; // Time vs distance along the shower axis in ground plane
  TGraphErrors *gRhoVsS; // Charge Density [VEM/m^2] vs Distance from Shower axis
  TGraphErrors *gRhoRsdVsS; // Charge [VEM/m^2] vs Distance from Shower axis

  Int_t nfitsds;    // Number SDs used in fitting (including zero charge) 
  Int_t nldffitsds; // Number of non-zero charge SDs used in LDF
  Int_t ntfitsds;   // Number of SDs used in time fitting
  Int_t nfitpts;    // Number of effective fit points which goes into calculation of ndof
  Int_t    fxxyy[NGSDS];  // lid of counters in the fit
  // 0 - zero charge counters that was put in
  // 1 - time fit only counters
  // 2 - LDF and time fit counter
  Int_t    fpflag[NGSDS];
  Double_t fX[NGSDS][3]; // Position used in fitting
  Double_t ft[NGSDS];    // Time used in fitting
  Double_t fdt[NGSDS];   // Time resolution used in fitting
  Double_t frho[NGSDS];  // Charge density used in fitting
  Double_t fdrho[NGSDS]; // Charge density fluctuation used in fitting

  TH2F *hTrsdS; // Residuals vs s scatter plot
  TProfile *pTrsdS; // Residuals vs s profile plot


  TH2F *hTrsdRho; // Residuals vs rho scatter plot
  TProfile *pTrsdRho; // Residuals vs rho profile plot


  // These are the fit parameters
  Double_t theta, phi, dtheta, dphi, // shower direction degree
      R[2], dR[2], // core position, in counter separation units
      T0, dT0, // Time when the core hits the ground, relative to earliest hit
      S, dS; // LDF scaling constant


  TF1 *ldfFun; // LDF curve, to show on the plots
  
  Double_t s600;
  Double_t s600_0;
  Double_t s800;
  Double_t s800_0;
  Double_t aenergy;
  Double_t energy;
  Double_t log10en;
  
  Double_t chi2; // FCN at its minimum
  Int_t ndof; // number of fit pts minus number of fit params

  // loads variables from the space-time cluster
  bool loadVariables(rusdraw_class *rusdraw1, rufptn_class *rufptn1,
      rusdgeom_class *rusdgeom1, rufldf_class *rufldf1);

  void xycoor(Int_t xxyy, Int_t *xy)
    {
      xy[0]=xxyy/100;
      xy[1]=xxyy%100;
    }

  // To remove counters which increase the overall chi2 by at least
  // deltaChi2. First, a worst counter in the cluster is identified. If it increases
  // the chi2 by more than deltaChi2, this counter is removed. If a bad counter
  // was removed, then the routing will search for another worst counter and so on.
  int clean(Double_t deltaChi2 = GLDF_DCHI2, bool verbose = true);

  // Calculates the time residual of i'th data point
  // Output:
  // trst     - residual on time
  // terr     - time fluctuation (properly scaled version of that from time alone fit)
  // chrg     - charge of i'th point
  // chrgdens - corresponding error on charge density (properly scaled version of that for LDF alone fit)
  // rcode - distance from core of i'th point
  void calcRsd(Int_t isds, Double_t *trsd, Double_t *terr,
      Double_t *chrgdens, Double_t *chrgdenserr, Double_t *rcore, Double_t *score);

  // To clean the profile plots of residuals.
  void cleanRsd();

  // This function fills the residuals of each data point into a profile histogram
  // versus R
  void fillRsd();

  // Fits to various time delay functions. 
  // whatFCN = 0: Fitting to a plane
  // whatFCN = 1: Fitting to Modified Linsley's function
  bool Ifit(bool verbose = false);
  gldffitter();

  virtual ~gldffitter();
  void compVars();

private:

  // To be able to restore the old fit parameters and re-use them as starting values.
  // Useful when cleaning out the bad counters
  real8 theta_old, phi_old, R_old[2], T0_old,S_old;

  void save_fpars(); // to save fit parameters in a buffer
  void restore_fpars(); // to restore the fit parameters, so that they can be used as starting values

  void save_fitinfo(); // saves fitting info (data points corresponding to non-zero SDs in LDF & time fitting parts)

  void calc_results();
  void print_results();
  
ClassDef(gldffitter,1)
  };

#endif
