#include "sduti.h"
#include "p1geomfitter.h"
#include "tacoortrans.h"

using namespace TMath;

// Fit variables
static integer4 nfitpts; // number of points in the fit
static real8 X[RUFPTNMH][3], // hit positions, counter separation units
    t[RUFPTNMH], // hit time (relative to earliest hit, counter separation units)
    dt[RUFPTNMH], // time resolution for each counter
    rho[RUFPTNMH], // charge (VEM/m^2)
    drho[RUFPTNMH]; // error on charge density (VEM/m^2)

static integer4 rufptnindex[RUFPTNMH]; // rufptn indices of each point

// if set to non-negative value, then i'th data point is excluded in fcn_cleaner
static integer4 iexclude;

// To keep the barycentric core calculation
static real8 cogXY[2];

/* 
 Modified Linsley's Time Delay Function
 
 INPUTS:
 rho:   charge density (VEM / square meter )
 R:     distance from the shower axis, meters
 theta: zenith angle (degree)
 
 OUTPUTS:
 td: time delay (nS) 
 ts: time delay fluctuation (nS) 
 */

static void ltdts(real8 Rho, real8 R, real8 theta, real8 *td, real8 *ts)
  {
    real8 a;

    if (theta < 25.0)
      {
        a = 3.3836 - 0.01848 * theta;
      }
    else if ((theta >= 25.0) && (theta <35.0))
      {
        a=(0.6511268210e-4*(theta-.2614963683))*(theta*theta-134.7902422*theta
            +4558.524091);
      }
    else
      {
        a = exp( -3.2e-2 * theta + 2.0);
      }

    // >=6 counters, DCHI2 = 10.0
    (*td) = 0.80 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
    (*ts) = 0.60 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);

    /////////// Works only for >= 7 counters, DCHI2=3.0 ...    
    //        if (theta < 29.8)
    //          {
    //            a = 3.3836 - 0.01848 * theta;
    //          }
    //        else if((theta >= 29.8) && (theta <30.2))
    //          {
    //            a = 0.1271306187e-2*(theta-24.30256467)*(theta-35.53822851)*(theta-100.4386680);
    //          }
    //        else
    //          {
    //            a = exp( -3.2e-2 * theta + 2.0);
    //          }
    //        *td = 0.67 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
    //        *ts = 0.29 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3); 


  }

// Alternative time delay / fluctuation function based on Linsley's but with variable
// curvature parameter and shower development factor.
static void ltdts1(Double_t Rho, Double_t R, Double_t a, Double_t usintheta,
		   Double_t *td, Double_t *ts)
  {
    // >= 4 counters

    (*td) = a * Power((1.-0.1 * usintheta/1200.), 1.05) * Power(
        (1 + R / 30.0), 1.35) * Power(Rho, -0.5);

    (*ts) = 1.56 * Power((1.-0.1 * usintheta/1200.), 1.05) * Power( (1 + R
        / 30.0), 1.5) * Power(Rho, -0.3);

  }

// To remove i'th data point from the fitting
static void remove_point(integer4 ipoint)
  {
    integer4 i, j;
    nfitpts -= 1;
    for (i=ipoint; i<nfitpts; i++)
      {
        for (j=0; j<3; j++)
          X[i][j]=X[i+1][j];
        t[i]=t[i+1];
        dt[i]=dt[i+1];
        rho[i]=rho[i+1];
        drho[i]=drho[i+1];
        rufptnindex[i]=rufptnindex[i+1];
      }
  }

// FIT FUNCTION (plane):

/* ARGUMENTS: x,y is counter location in counter separation units in CLF frame with respect
 * to SD origin.  z is the counter height above CLF plane (in counter separation units)
 (PARAMETERS): 
 par[0] - zenith angle, degrees
 par[1] - azimuthal angle, degrees
 par[2] - x-position of the core, counter separation units
 par[3] - y-position of the core, counter separation units
 par[4] - time of the core hit, relative to earliest hit time, counter separation units
 RETURNS: time for a given position in counter separation units
 */
static real8 tvsx_plane(real8 x, real8 y, real8 z, real8 *par)
  {
    real8 degrad=DegToRad(); // to convert from degrees to radians
    real8 d[2] =
      { (x-par[2]), (y-par[3]) }; /* vector distance from the core in xy - plane */
    // Dot product of distance from core and shower axis vector
    real8 dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0] + sin(par[1]
        *degrad)*d[1]);
    return par[4]+dotp - z * cos(par[0]*degrad);
  }

//______________________________________________________________________________
static void fcn_plane(integer4 &npar, real8 *gin, real8 &f, real8 *par,
    integer4 iflag)
  {
    integer4 i;
    real8 chisq = 0;
    real8 delta;
    real8 denom;
    real8 degrad = DegToRad();
    real8 dotp, s;
    real8 d[2];
    real8 ltd, lts;
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {
        d[0] = X[i][0]-par[2];
        d[1] = X[i][1]-par[3];

        // Dot product of distance from core and shower axis vector
        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
            + sin(par[1] *degrad)*d[1]);
        // Distance in shower front plane
        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        // To get error bars from modified Linsley's
        ltdts(rho[i], s*1.2e3, par[0], &ltd, &lts);
        lts *= nsecTo1200m;
        denom = sqrt(lts*lts+dt[i]*dt[i]);
        delta = t[i] - tvsx_plane(X[i][0], X[i][1], X[i][2], par);
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3])*(cogXY[1]
        -par[3])) / (P1GEOM_TEMP_dr*P1GEOM_TEMP_dr);
    f = chisq;
  }

//______________________________________________________________________________
static void fcn_linsley(integer4 &npar, real8 *gin, real8 &f, real8 *par,
    integer4 iflag)
  {
    integer4 i;
    real8 ltd, lts; // Modified Linsley's time delay and time delay fluctuation
    real8 chisq = 0;
    real8 delta, denom;
    real8 degrad = DegToRad();
    real8 dotp, s;
    real8 d[2];
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {

        d[0] = X[i][0]-par[2];
        d[1] = X[i][1]-par[3];
        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
            + sin(par[1] *degrad)*d[1]); // Dot product of distance from core and shower axis vector

        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        ltdts(rho[i], s*1.2e3, par[0], &ltd, &lts);
        ltd *= nsecTo1200m;
        lts *= nsecTo1200m;
        denom = lts*lts + dt[i]*dt[i];
        delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - ltd;
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3]) *(cogXY[1]
        -par[3])) / (P1GEOM_TEMP_dr*P1GEOM_TEMP_dr);
    f = chisq;
  }

//______________________________________________________________________________
static void fcn_linsley1(integer4 &npar, real8 *gin, real8 &f, real8 *par,
    integer4 iflag)
  {
    integer4 i;
    real8 ltd, lts; // Modified Linsley's time delay and time delay fluctuation
    real8 chisq = 0;
    real8 delta, denom;
    real8 degrad = DegToRad();
    real8 dotp, s;
    real8 d[2];
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {

        d[0] = X[i][0]-par[2];
        d[1] = X[i][1]-par[3];
        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
            + sin(par[1] *degrad)*d[1]); // Dot product of distance from core and shower axis vector

        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        ltdts1(rho[i], s*1.2e3, par[5], dotp*1.2e3, &ltd, &lts);
        ltd *= nsecTo1200m;
        lts *= nsecTo1200m;
        denom = lts*lts + dt[i]*dt[i];
        delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - ltd;
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3]) *(cogXY[1]
        -par[3])) / (P1GEOM_TEMP_dr*P1GEOM_TEMP_dr);
    f = chisq;
  }

//______________________________________________________________________________
static void fcn_cleaner(integer4 &npar, real8 *gin, real8 &f, real8 *par,
    integer4 iflag)
  {
    integer4 i;
    real8 ltd, lts; // Modified Linsley's time delay and time delay fluctuation
    real8 degrad = DegToRad();
    real8 dotp, s;
    real8 d[2];

    real8 chisq = 0;
    real8 delta, denom;
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {
        if (i==iexclude)
          continue;

        d[0] = X[i][0]-par[2];
        d[1] = X[i][1]-par[3];
        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
            + sin(par[1] *degrad)*d[1]);
        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        ltdts(rho[i], s*1.2e3, par[0], &ltd, &lts);
        ltd *= nsecTo1200m;
        lts *= nsecTo1200m;
        denom = lts*lts+dt[i]*dt[i];
        delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - ltd;
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3]) *(cogXY[1]
        -par[3])) / (P1GEOM_TEMP_dr*P1GEOM_TEMP_dr);
    f = chisq;
  }

//______________________________________________________________________________
static void fcn_cleaner1(integer4 &npar, real8 *gin, real8 &f, real8 *par,
    integer4 iflag)
  {
    integer4 i;
    real8 ltd, lts; // Modified Linsley's time delay and time delay fluctuation
    real8 degrad = DegToRad();
    real8 dotp, s;
    real8 d[2];

    real8 chisq = 0;
    real8 delta, denom;
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {
        if (i==iexclude)
          continue;

        d[0] = X[i][0]-par[2];
        d[1] = X[i][1]-par[3];
        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
            + sin(par[1] *degrad)*d[1]);
        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        ltdts1(rho[i], s*1.2e3, par[5], dotp*1.2e3, &ltd, &lts);
        ltd *= nsecTo1200m;
        lts *= nsecTo1200m;
        denom = lts*lts+dt[i]*dt[i];
        delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - ltd;
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3]) *(cogXY[1]
        -par[3])) / (P1GEOM_TEMP_dr*P1GEOM_TEMP_dr);
    f = chisq;
  }

p1geomfitter_class::p1geomfitter_class()
  {
    gMinuit = 0;
    sdorigin_xy[0] = RUSDGEOM_ORIGIN_X_CLF;
    sdorigin_xy[1] = RUSDGEOM_ORIGIN_Y_CLF;
  }

p1geomfitter_class::~p1geomfitter_class()
  {
  }

bool p1geomfitter_class::loadVariables()
  {
    integer4 i;
    real8 xy[2];

    /* These are initial values of the fit parameters */
    T0 = rufptn_.tyro_tfitpars[2][0];

    // Barycentric core calculation from tyro analysis
    memcpy(R, rufptn_.tyro_xymoments[2], (integer4)(2*sizeof(real8)));
    memcpy(cogXY, rufptn_.tyro_xymoments[2], (integer4)(2*sizeof(real8)));

    // tyro theta and phi, using upper and lower layers
    theta = rufptn_.tyro_theta[2];
    phi = rufptn_.tyro_phi[2];

    a = 2.6; // When fitting with variable curvature, this is the starting value.

    nfitpts = 0;
    for (i=0; i < rufptn_.nhits; i++)
      {
        // Use hits that are in space-time cluster only
        if (rufptn_.isgood[i] < 3)
          continue;

        // Counter X,Y coordinates in 1200m units in CLF plane with respect to convenent
        // sd origin.
        xy[0] = rufptn_.xyzclf[i][0] - sdorigin_xy[0];
        xy[1] = rufptn_.xyzclf[i][1] - sdorigin_xy[1];
        X[nfitpts][0] = xy[0];
        X[nfitpts][1] = xy[1];
        // Height of the counter above the CLF plane, in [1200m] units
        X[nfitpts][2] = rufptn_.xyzclf[i][2];

        // Time and time resolution for each counter, in [1200m] units.  Time
        // relative to the earliest signal used (in some cases, this earliest time is
        // not a part of any clusters)
        t [nfitpts]= (rufptn_.reltime[i][0]+rufptn_.reltime[i][1]) / 2.0;

        // Time resolution of each SD
        dt[nfitpts] = 0.5 * sqrt(rufptn_.timeerr[i][0]*rufptn_.timeerr[i][0]
            + rufptn_.timeerr[i][1]*rufptn_.timeerr[i][1]);

        // Charge density in VEM/m^2
        rho[nfitpts]= ((rufptn_.pulsa[i][0]+rufptn_.pulsa[i][1])/2.0) / 3.0;
        drho[nfitpts]= (sqrt(rufptn_.pulsaerr[i][0]*rufptn_.pulsaerr[i][0]
            + rufptn_.pulsaerr[i][1]*rufptn_.pulsaerr[i][1])/2.0) / 3.0;
        // Keep rufptn indeces of good points (will be changed in cleaning)
        rufptnindex[nfitpts] = i;
        nfitpts++;
      }

    ngpts = nfitpts;
    memcpy(goodpts, rufptnindex, ngpts*sizeof(integer4));

    ndof = nfitpts - NFPARS_LINSLEY;
    return true;
  }

//______________________________________________________________________________


// To recalculate the center of gravity when SDs are cut out
static bool recalcCog()
  {
    integer4 i, j;
    real8 w;
    // Recalculate the COG core position with cuts (above) applied.
    for (j=0; j<2; j++)
      cogXY[j] = 0.0;
    w = 0.0;
    for (i=0; i<nfitpts; i++)
      {
        if (i==iexclude)
          continue;
        for (j=0; j<2; j++)
          cogXY[j] += rho[i]*X[i][j];
        w += rho[i];
      }

    // Cannot have zero total charge in reconstruction
    if (w < 1e-3)
      return false;

    for (j=0; j<2; j++)
      cogXY[j] /= w;
    return true;
  }

integer4 p1geomfitter_class::doFit(integer4 whatFCN)
  {

    integer4 i;
    TString parname(10); // max. size of the name of any fit parameter
    integer4 nfpars;
    // start values, step sizes, lower and upper limits
    static real8 vstart, step, vlo, vup;

    // For passing lists of options to Minuit
    real8 arglist[10];
    integer4 ierflg;

    real8 stheta; // sine of zenith angle
    static integer4 n_trials = 0; // number of trial fits to get a sensible event direction


    if (gMinuit)
      delete gMinuit;
    gMinuit=0;

    switch (whatFCN)
      {
    case 0:
      // for plane fitting
      nfpars=NFPARS_PLANE;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_plane);
      break;

    case 1:
      // for modified Linsley's fitting
      nfpars=NFPARS_LINSLEY;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_linsley);
      break;

    case 2:
      // for cleaning modified Linsley's fitting
      nfpars=NFPARS_LINSLEY;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_cleaner);
      break;

    case 3:
      // for fitting into Linsley's with variable curvature
      nfpars=NFPARS_LINSLEY1;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_linsley1);
      break;

    case 4:
      // for cleaning using Linsley's with variable curvature
      nfpars=NFPARS_LINSLEY1;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_cleaner1);
      break;

    default:
      fprintf(stderr,"p1geomfitter: Option flag must be in 0-4 range\n");
      return -1;
      break;
      }

    ndof = nfitpts-nfpars; // # of d.o.f

    // Don't do any fitting if # of points is less than 3
    if (nfitpts < 3)
      return nfitpts;

    gMinuit->SetPrintLevel(-1);

    ierflg = 0;
    arglist[0] = 1;
    gMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
    for (i=0; i< nfpars; i++)
      {
        vlo=0.0;
        vup=0.0; // no limits on fit parameters, unless changed below
        switch (i)
          {
        case 0:
          parname="Theta";
          vstart=theta;
          step=0.1;
          break;
        case 1:
          parname="Phi";
          vstart=phi;
          step=0.1;
          break;
        case 2:
          parname="Core X";
          vstart=R[0];
          step=0.1;
          break;
        case 3:
          parname="Core Y";
          vstart=R[1];
          step=0.1;
          break;
        case 4:
          parname="T0";
          vstart=T0;
          step=0.1;
          break;

        case 5:
          parname="curvature";
          vstart=a;
          step=0.1;
          break;

        default:
          break;

          }

        // add fit parameter information to Minuit
        gMinuit->mnparm(i, parname, vstart, step, vlo, vup, ierflg);
      }

    gMinuit->SetMaxIterations(P1GEOM_MITR);

    // First fit with core fixed, then release the core and re-fit.
    gMinuit->FixParameter(2);
    gMinuit->FixParameter(3);
    gMinuit->Migrad();
    gMinuit->Release(2);
    gMinuit->Release(3);
    gMinuit->Migrad();
    
    // Obtain the fit parameters
    gMinuit->GetParameter(0, theta, dtheta);
    gMinuit->GetParameter(1, phi, dphi);
    phi = tacoortrans::range(phi,360.0);
    gMinuit->GetParameter(2, R[0], dR[0]);
    gMinuit->GetParameter(3, R[1], dR[1]);
    gMinuit->GetParameter(4, T0, dT0);
    if (whatFCN == 3 || whatFCN == 4)
      gMinuit->GetParameter(5, a, da);
    
    chi2 = gMinuit->fAmin;

    // Must have a shower going downwards.  If not so from the 1st trial, flip its axis and re-fit.
    stheta = sin(theta * DegToRad());
    theta = RadToDeg() * asin(stheta);
    if (stheta < 0.0)
      {
        n_trials++;
        
        theta *= -1.0;
	phi = tacoortrans::range(phi+180.0,360.0);
	
        // Caution, to avoid blowing the heap.
        // If we can't do any better
        // in 10 trials, then return what we've got.
        if (n_trials > 10)
          {
            n_trials = 0;
            return nfitpts;
          }
        return doFit(whatFCN);
      }
    else
      n_trials = 0;

    // fitter returns # of fit points
    return nfitpts;

  }

integer4 p1geomfitter_class::cleanClust(real8 deltaChi2)
  {
    real8 chi2old, chi2new;
    integer4 nDeletedPts;
    real8 dChi2;
    integer4 i, worst_i; // Keep track of the worst data point
    
    nDeletedPts = 0;

    // Get the best possible starting values with all the data points.
    // Get out of the cleaner if there are no degrees of freedom to start with.


    // First attempt plane fit
    if (doFit(0) < 1)
      {
        return nDeletedPts;
      }
    // Then, attempt modified Linsley fit
    if (doFit(1) < 1)
      {
        return nDeletedPts;
      }
    save_fpars();

    do
      {
        if (nfitpts < 1)
          {
            return nDeletedPts;
          }
        restore_fpars();
        // Fit with current data points and save the chi2 value
        iexclude = -1;
        if (!recalcCog())
          {
            return nDeletedPts;
          }
        if (doFit(2) < 1)
          {
            return nDeletedPts;
          }
        save_fpars();

        chi2old=chi2;
        // When chi2 is already smaller than deltaChi2
        if (chi2old < deltaChi2)
          {
            return nDeletedPts;
          }

        dChi2=0.0; // initialize the chi2 difference.
        worst_i = 0; // just to initialize it
        // Find the largest chi2 difference.  This will be the worst point.
        for (i=0; i<nfitpts; i++)
          {
            restore_fpars();
            // Fit w/o the i'th data pint and take the new chi2
            iexclude = i;
            if (!recalcCog())
              {
                return nDeletedPts;
              }
            if (doFit(2) < 1)
              {
                return nDeletedPts;
              }
            chi2new=chi2;
            if ((chi2old-chi2new)>dChi2)
              {
                dChi2=(chi2old-chi2new);
                worst_i = i;
              }
          }
        if (dChi2 >= deltaChi2)
          {
            remove_point(worst_i);
            ngpts = nfitpts; // To keep track of good SDs
            memcpy(goodpts, rufptnindex, ngpts*((integer4)sizeof(integer4)));
            nDeletedPts ++;
          }

      } while (dChi2>=deltaChi2);

    restore_fpars();
    // This is important.  If we don't recalculate the COG with current data points, then
    // the cog value passed to the fitter (right after this routine) will contain
    // cog computed using 1 point less (see the for-loop above). 
    iexclude = -1;
    if (!recalcCog())
      {
        return nDeletedPts;
      }
    return nDeletedPts;

  }

void p1geomfitter_class::save_fpars()
  {
    theta_old = theta;
    phi_old = phi;
    memcpy(R_old, R, (int)(2*sizeof(real8)));
    T0_old = T0;
  }

void p1geomfitter_class::restore_fpars()
  {
    theta = theta_old;
    phi = phi_old;
    memcpy(R, R_old, (int)(2*sizeof(real8)));
    T0 = T0_old;
  }

bool p1geomfitter_class::hasConverged()
  {
    Double_t amin, edm, errdef;
    Int_t nvpar, nparx, icstat;
    if (gMinuit)
      {
        gMinuit->mnstat(amin, edm, errdef, nvpar, nparx, icstat);
        if ( (SDGEN::getFitStatus((char *)gMinuit->fCstatu.Data()))
            >= GETFITSTATUS_GOOD)
          return true;

      }
    return false;
  }

