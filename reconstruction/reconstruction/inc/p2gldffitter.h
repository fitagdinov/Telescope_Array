#ifndef _p2gldffitter_h_
#define _p2gldffitter_h_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TMinuit.h"
#include "TMath.h"
#include "event.h"


#define NGSDS RUFPTNMH     // max. number of good SDs
#define P2GLDF_NFPARS  6   // max. number of fit parameters
#define P2GLDF_TEMP_dt 0.2
#define nsecTo1200m  2.49827048333e-4
#define P2GLDF_TEMP_dr 0.15 // error on core location


// Determines by how much one needs to scale the corresponing LDF and Time fit errors to
// have the chi2 in the right place
#define P2GLDF_ERR_SCALE 0.82

// Used in cleaning the space-time cluster.  If chi2 of the fit without the
// i'th point improves by this value or better, then the i'th point is removed
// from space-time cluster
#define P2GLDF_DCHI2 4.0

// Max. number of iterations for minuit
#define P2GLDF_MITR 50000


class p2gldffitter_class : public TObject
  {
public:

  Int_t sds_hit[SDMON_X_MAX][SDMON_Y_MAX];

  TMinuit *gMinuit; // Minuit fitter
  Double_t sdorigin_xy[2]; // Position of SD origin with respect to CLF in CLF frame

  Int_t nfitsds; // Number SDs used in fitting (including zero charge) 
  Int_t nldffitsds; // Number of non-zero charge SDs used in LDF
  Int_t ntfitsds; // Number of SDs used in time fitting
  Int_t nfitpts; // Number of effective fit points which goes into calculation of ndof
  Int_t fxxyy[NGSDS]; // lid of counters in the fit
  // 0 - zero charge counters that was put in
  // 1 - time fit only counters
  // 2 - LDF and time fit counter
  Int_t fpflag[NGSDS];
  Double_t fX[NGSDS][3]; // Position used in fitting
  Double_t ft[NGSDS]; // Time used in fitting
  Double_t fdt[NGSDS]; // Time resolution used in fitting
  Double_t frho[NGSDS]; // Charge density used in fitting
  Double_t fdrho[NGSDS]; // Charge density fluctuation used in fitting


  // These are the fit parameters
  Double_t theta, phi, dtheta, dphi, // shower direction degree
      R[2], dR[2], // core position, in counter separation units
      T0, dT0, // Time when the core hits the ground, relative to earliest hit
      S, dS; // LDF scaling constant

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
  bool loadVariables();

  void xycoor(Int_t xxyy, Int_t *xy)
    {
      xy[0]=xxyy/100;
      xy[1]=xxyy%100;
    }

  // To remove counters which increase the overall chi2 by at least
  // deltaChi2. First, a worst counter in the cluster is identified. If it increases
  // the chi2 by more than deltaChi2, this counter is removed. If a bad counter
  // was removed, then the routing will search for another worst counter and so on.
  int clean(Double_t deltaChi2 = P2GLDF_DCHI2);

  bool doFit();
  bool hasConverged();
  p2gldffitter_class();

  virtual ~p2gldffitter_class();

private:

  // To be able to restore the old fit parameters and re-use them as starting values.
  // Useful when cleaning out the bad counters
  real8 theta_old, phi_old, R_old[2], T0_old, S_old;

  void save_fpars(); // to save fit parameters in a buffer
  void restore_fpars(); // to restore the fit parameters, so that they can be used as starting values

  void save_fitinfo(); // saves fitting info (data points corresponding to non-zero SDs in LDF & time fitting parts)

  void calc_results();
  };

#endif
