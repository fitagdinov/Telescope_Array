#ifndef P1GEOMFITTER_H_
#define P1GEOMFITTER_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "TObject.h"
#include "TMinuit.h"
#include "TMath.h"
#include "event.h"

#define P1GEOMFITTER_MNFPARS 6 // max. # of fit parameters
#define NFPARS_PLANE    5
#define NFPARS_LINSLEY  5
#define NFPARS_LINSLEY1 6
#define nsecTo1200m  2.49827048333e-4 // converts nS to [1200m] units
#define P1GEOM_TEMP_dr 0.15 // error on barycentric core location

// Used in cleaning the space-time cluster.  If chi2 of the fit without the
// i'th point improves by this value or better, then the i'th point is removed
// from space-time cluster
#define P1GEOM_DCHI2 10.0

// Max. number of iterations for minuit
#define P1GEOM_MITR 50000


class p1geomfitter_class
{
 public:
  TMinuit *gMinuit;        // Minuit fitter
  real8 sdorigin_xy[2]; // Position of SD origin with respect to CLF in CLF frame
  
  integer4 ngpts;            // rufptn indices of good points after cleaning
  integer4 goodpts[RUFPTNMH];
  
  // These are the fit parameters
  real8 
    theta, phi, dtheta, dphi,  // shower direction (from where it came), degree
    R[2],dR[2],                // core position, in counter separation units
    T0,dT0,                    // Time when the core hits the ground, relative to earliest hit
    a,da;                      // Curvature parameter
  

  real8 chi2;              // FCN at its minimum
  integer4 ndof;                 // number of fit pts minus number of fit params
  
  // loads variables into the fitter.
  bool loadVariables();

  
  // To remove counters which increase the overall chi2 by at least
  // deltaChi2. First, a worst counter in the cluster is identified. If it increases
  // the chi2 by more than deltaChi2, this counter is removed. If a bad counter
  // was removed, then the routing will search for another worst counter and so on.
  integer4 cleanClust(real8 deltaChi2 = P1GEOM_DCHI2);
  integer4 doFit(integer4 whatFCN);
  
  // Checks if the last call to fitter converged
  bool hasConverged();
  p1geomfitter_class();
  virtual ~p1geomfitter_class();
  
  
 private:
   
   // To be able to restore the old fit parameters and re-use them as starting values.
   // Useful when cleaning out the bad counters
   real8 
     theta_old, phi_old,
     R_old[2],T0_old;                      
   
   void save_fpars();    // to save fit parameters in a buffer
   void restore_fpars(); // to restore the fit parameters, so that they can be used as starting values
  
 };


#endif /*P1GEOMFITTER_H_*/
