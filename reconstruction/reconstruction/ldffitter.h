#ifndef _ldffitter_h_
#define _ldffitter_h_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TObject.h"
#include "TMinuit.h"
#include "TMath.h"
#include "event.h"
#include "sdrt_class.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TProfile.h"
#include "TH2F.h"


#define NGSDS  RUFPTNMH   // max. number of good SDs
#define NFPARS 3          // number of fit parameters
 
// error on center of gravity core location
#define LDF_COG_dr 0.15

// Max. number of iterations for minuit
#define LDF_MITR 50000

class ldffitter : public TObject
{
public:
  
  Int_t sds_hit[SDMON_X_MAX][SDMON_Y_MAX];
   
  TMinuit *gMinuit;        // Minuit fitter
  Double_t sdorigin_xy[2]; // SD origin with respect to CLF in CLF frame
  TGraphErrors *gqvsr;     // Charge [VEM] vs Distance from Shower axis
  TGraphErrors *grsdvsr;   // Residuals vs Distance from Shower axis
  TF1 *ldfFun;             // Fitted Charge[VEM] vs Distance from Shower axis
  
  // Event geometry
  Double_t theta; // Zenith angle [Degree]
  Double_t phi;   // Azimuthal angle [Degree]
  Double_t coreXY[2]; // Cenger of gravity core XY position, [1200m] units 
  
  Int_t iexclude;           // if set, then the corresponding point doesn't participate in chi2
  Int_t nfpts;              // number of points in the fit
  Int_t napts;              // number of actual SDs that were hit
  Int_t pflag[NGSDS];       // Flag for the point: 0 - zero charge counter put in, 1 - actual counter
  Int_t xxyy_posid[NGSDS];  // counter position ID
  Double_t X[NGSDS][3];     // hit positions, [1200m] units
  Double_t grn_dist[NGSDS]; // distance from the shower core on the ground [1200m]
  Double_t ltr_dist[NGSDS]; // lateral distance from the shower axis for each point [1200m]
  Double_t rho[NGSDS];      // charge density (vem/m^2)
  Double_t drho[NGSDS];     // error on charge density [vem/m^2]
  Double_t rhorsd[NGSDS];   // fit residual [VEM/m^2] (calculated while fitting)
  
  // Fit parameters
  Double_t R[2], dR[2];     // fitted core position [1200m] units, CLF, with respect to SD origin
  Double_t S, dS;           // scaling factor in front of LDF
  
  Double_t S600;            // Signal size at r=600m
  Double_t S600_0;          // Signal size at r=600m, accounting attenuation
  
  Double_t S800;            // Signal size at r=800m
  Double_t S800_0;          // Signal size at r=800m, accoutnting attenuation
  
  Double_t energy;          // energy in EeV
  Double_t log10en;         // log10 (energy in eV)
  
  Double_t chi2;            // FCN at its minimum
  Int_t    ndof;            // number of fit pts minus number of fit params
  
  // loads variables needed in the fitter
  bool loadVariables(rusdraw_class *rusdraw1, 
		     rufptn_class *rufptn1, 
		     rusdgeom_class *rusdgeom1);
  void prepPlotVars(); // To compute various plots
  
  void xycoor (Int_t xxyy, Int_t *xy) {xy[0]=xxyy/100; xy[1]=xxyy%100;}
  
  void remove_point(Int_t ipoint); // remove the point from the variable structure
  void calc_var();                 // calculate the chi2 and other variables given the fit parameters
  bool get_param(TMinuit* g);      // get the best fit parameters from Minuit and re-calculate the chi2
  
  // This function fills the residuals of each data point into a 
  // profile histogram versus R
  void fillRsd();
  
  bool Ifit(bool fixCore, bool verbose);
  Int_t clean(Double_t deltaChi2, bool fixCore, bool verbose);
  ldffitter();
  bool add_takenout_SD(Int_t xxyy); // add SD to the list of taken out SDs
  void reset_takenout_SD();  // reset the list of taken out SDS
  int print_takenout_SD(); // show which SDs are being taken out
  int print_points(Bool_t sort_by_r=true); // print the LDF points
  virtual ~ldffitter();
  
private:
  
  Int_t n_takenout_sd;     // non-zero if any SDs are taken out from the fit
  Int_t takenout_sd[NGSDS];
  
  ClassDef(ldffitter,1)
};

#endif
