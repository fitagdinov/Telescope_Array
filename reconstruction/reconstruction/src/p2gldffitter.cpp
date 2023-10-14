#include "p2gldffitter.h"
#include "sduti.h"
#include "sdxyzclf_class.h"
#include "tacoortrans.h"
using namespace TMath;

// Fit variables
static Int_t nfsds; // number of SDs in the fit (with zero charge SDs included); most ldf and time fit SDs overlap
static Int_t nldfsds; // number of non-zero charge SDs in LDF fit
static Int_t ntfsds; // number of time fit SDs
// To keep counter lid, important in the cases when we add counters not hit
static Int_t sdxxyy[NGSDS];

// 0: Zero charge counter put in
// 1: Counter participates in time fit only
// 2: Counter participates in time and LDF fit
static Int_t pflag[NGSDS];
static Double_t X[NGSDS][3], // hit positions, counter separation units
    t[NGSDS], // hit time (relative to earliest hit, counter separation units)
    dt[NGSDS], // error on hit time (counter separation units)
    rho[NGSDS], // charge density (vem/m^2)
    drho[NGSDS]; // error on charge density (vem/m^2)

// if set to non-negative value, then i'th data point is excluded in fcn
static Int_t iexclude = -1;

static Double_t cogXY[2];

static sdxyzclf_class sdcoorclf;

// To remove i'th data point from the fitting
static void remove_point(Int_t isds)
  {
    Int_t i;
    nfsds -= 1;
    if (pflag[isds] > 1)
      ntfsds --;
    if (pflag[isds] == 2)
      nldfsds--;

    for (i=isds; i<nfsds; i++)
      {
        sdxxyy[i] = sdxxyy[i+1];
        memcpy(&X[i][0], &X[i+1][0], 3*sizeof(Double_t));
        t[i]=t[i+1];
        dt[i]=dt[i+1];
        rho[i]=rho[i+1];
        drho[i]=drho[i+1];
        pflag[i]=pflag[i+1];
      }
  }

// To recalculate the core once various cuts are applied
static bool recalcCog()
  {
    Int_t i, j;
    Double_t w;
    // Recalculate the COG core position with cuts (above) applied.
    for (j=0; j<2; j++)
      cogXY[j] = 0.0;
    w = 0.0;
    for (i=0; i<nfsds; i++)
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

// Modified Linsley Td,Ts in [nS]
// Rho is in [VEM/m^2], R is perpendicular dist. from shower axis, in [m]
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

    // Contained, >=7 counters, DCHI2 = 3.0
    //    (*td) = 0.72 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
    //    (*ts) = 0.33 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);


    // Contained, >=7 counters, DCHI2 = 4.0
    //    (*td) = 0.75 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
    //    (*ts) = 0.42 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);

    // Contained, >=4 counters, DCHI2 = 4.0
    (*td) = 0.80 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
    (*ts) = 0.70 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);

  }

// LDF function
// r     :  Perpendicular distance from shower axis, m
// theta :  Zenith angle, degree
static Double_t ldffun(Double_t r, Double_t theta)
  {
    Double_t r0; // Moliere radius
    Double_t alpha; // Constant slope parameter
    Double_t beta; // Another constant slope parameter
    Double_t eta; // Zenith angle dependent slope parameter
    Double_t rsc; // Scaling factor for r in quadratic term in power

    r0 = 91.6;
    alpha = 1.2;
    beta = 0.6;
    eta = 3.97-1.79*(1.0/Cos(DegToRad()*theta)-1.0);
    rsc = 1000.0;

    return Power(r/r0, -alpha) * Power((1.0+r/r0), -(eta-alpha)) * Power((1.0
        + r*r/rsc/rsc), -beta);
  }

// AGASA atmospheric attenuation function
// S600 = S600_0 * S600_attenuation
static Double_t S600_attenuation_AGASA(Double_t theta)
  {
    Double_t sf; // The "secant factor"
    // AGASA Attenuation parameters
    Double_t X0;
    Double_t L1;
    Double_t L2;

    X0 = 920.0;
    L1 = 500.0;
    L2 = 594;

    sf = 1.0/Cos(DegToRad()*theta) - 1;
    return Exp(-X0/L1 * sf - X0/L2 * sf*sf);
  }

// Plane function:

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
static Double_t tvsx_plane(Double_t x, Double_t y, Double_t z, Double_t *par)
  {
    Double_t degrad=DegToRad(); // to convert from degrees to radians
    Double_t d[2] =
      { (x-par[2]), (y-par[3]) }; /* vector distance from the core in xy - plane */
    // Dot product of distance from core and shower axis vector
    Double_t dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0] + sin(par[1]
        *degrad)*d[1]);

    // distance from the core in shower front plane squared
    // Double_t r2 = d[0]*d[0]+d[1]*d[1]-dotp*dotp;
    //  Double_t sum1 = par[5]*par[5]-r2;
    //  if (sum1<0.0) sum1 = 1e2;
    return par[4]+dotp - z * cos(par[0]*degrad);
  }

//par[0] - zenith angle, degrees
//par[1] - azimuthal angle, degrees
//par[2] - x-position of the core, counter separation units
//par[3] - y-position of the core, counter separation units
//par[4] - time of the core hit, relative to earliest hit time, counter separation units
//par[5] - scale factor in front of LDF
//______________________________________________________________________________
static void fcn_linsley_ldf(Int_t &npar, Double_t *gin, Double_t &f,
    Double_t *par, Int_t iflag)
  {
    Int_t i;
    Double_t ltd, lts; // Modified Linsley's time delay and time delay fluctuation
    Double_t chisq = 0;
    Double_t delta, denom;
    Double_t degrad = DegToRad();
    Double_t dotp, s;
    Double_t d[2];
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfsds; i++)
      {
        // Useful in case of cleaning
        if (i==iexclude)
          continue;
	
        d[0] = X[i][0]-par[2];
        d[1] = X[i][1]-par[3];
        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
            + sin(par[1] *degrad)*d[1]); // Dot product of distance from core and shower axis vector

        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);

        // Time fit portion of FCN
        if (pflag[i] >= 1)
          {
            ltdts(rho[i], s*1.2e3, par[0], &ltd, &lts);
            ltd *= nsecTo1200m;
            lts *= nsecTo1200m;
            denom = P2GLDF_ERR_SCALE*P2GLDF_ERR_SCALE*(lts*lts + dt[i]*dt[i]);
            delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - ltd;
            chisq += delta*delta / denom;
          }

        // LDF portion of fit
        if ( (pflag[i] == 0) || pflag[i] == 2)
          {
            delta = (rho[i] - par[5] * ldffun(s*1.2e3, par[0]));
            denom = P2GLDF_ERR_SCALE*P2GLDF_ERR_SCALE*drho[i]*drho[i];
            chisq += delta*delta/denom;
          }

      }
    // Portion that prevents the core from moving far away
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3]) *(cogXY[1]
        -par[3])) / (P2GLDF_TEMP_dr*P2GLDF_TEMP_dr);
    f = chisq;
  }

p2gldffitter_class::p2gldffitter_class()
  {
    gMinuit = 0;
    sdorigin_xy[0] = SD_ORIGIN_X_CLF;
    sdorigin_xy[1] = SD_ORIGIN_Y_CLF;
  }

p2gldffitter_class::~p2gldffitter_class()
  {
  }

bool p2gldffitter_class::loadVariables()
  {
    Int_t i;
    Double_t xy[2];
    Double_t d[2]; // Dist. from core on ground plane, vector
    Double_t dotp, cdist, sdist, smax;

    nfsds = 0;
    ntfsds = 0;
    nldfsds = 0;
    memset(sds_hit, 0, (SDMON_X_MAX*SDMON_Y_MAX*sizeof(Int_t)));
    Int_t ix, iy;
    Int_t xlo, xup;
    Int_t ylo, yup;

    // These are initial values of the fit parameters
    // Important to have them so that we know how to put in
    // zero counters
    T0 = rusdgeom_.t0[1];
    // tyro theta and phi, using upper and lower layers
    theta = rusdgeom_.theta[1];
    phi = rusdgeom_.phi[1];
    R[0] = rusdgeom_.xcore[1];
    R[1] = rusdgeom_.ycore[1];
    S = rufldf_.sc[0]; // Use LDF alone fit S for starting value

    smax = 0.0;
    for (i=0; i < rufptn_.nhits; i++)
      {
        // Use hits that are in space-time cluster only
        if (rufptn_.isgood[i] < 4)
          continue;

        // Important to know if a given SD was hit, so that we don't add the 0's for them
        ix = rufptn_.xxyy[i] / 100 - 1;
        iy = rufptn_.xxyy[i] % 100 - 1;
        sds_hit[ix][iy] = 1;

        sdxxyy[nfsds] = rufptn_.xxyy[i];

        xy[0] = rufptn_.xyzclf[i][0] - sdorigin_xy[0];
        xy[1] = rufptn_.xyzclf[i][1] - sdorigin_xy[1];
        X[nfsds][0] = xy[0];
        X[nfsds][1] = xy[1];
        X[nfsds][2] = rufptn_.xyzclf[i][2];
        t [nfsds]= (rufptn_.reltime[i][0]+rufptn_.reltime[i][1]) / 2.0;
        dt [nfsds] = 0.5 * sqrt(rufptn_.timeerr[i][0]*rufptn_.timeerr[i][0]
            + rufptn_.timeerr[i][1]*rufptn_.timeerr[i][1]);
        rho[nfsds]= ((rufptn_.pulsa[i][0]+rufptn_.pulsa[i][1])/2.0) / 3.0;
        drho[nfsds] = 0.53 * sqrt(2.0*rho[nfsds] + 0.15*0.15*rho[nfsds]
            *rho[nfsds]);

        d[0] = X[nfsds][0]-rusdgeom_.xcore[2];
        d[1] = X[nfsds][1]-rusdgeom_.ycore[2];
        cdist=sqrt(d[0]*d[0]+d[1]*d[1]);

        dotp = Sin(DegToRad()*theta)*(d[0]*Cos(DegToRad()*phi) +d[1]
            *Sin(DegToRad()*phi));
        sdist = sqrt(d[0]*d[0]+d[1]*d[1] - dotp*dotp);

        // Maximum perpendicular distance from shower axis
        if (sdist > smax)
          smax = sdist;

        // SD can participate in time and LDF fitting
        pflag[nfsds] = 2;
        ntfsds ++; // increase the number of SDs in time fitting
        nldfsds ++; // increase the number of SDs in LDF fitting
        // Central counter can only stay in time fitting
        if (cdist < 0.5)
          {
            pflag[nfsds] = 1;
            nldfsds --; // Decrease number of SDs in LDF
          }
        nfsds ++;
      }

    // Additional LDF data points (counters that were not hit)


    // This is just a range of X and Y values for counters that 
    // we need to scan and see if if they are inside the event ellipse or not.
    xlo = (Int_t)(Floor(R[0] - smax/Cos(DegToRad()*theta) ) - 1.0);
    if (xlo < 1)
      xlo = 1;
    xup = (Int_t)(Ceil(R[0] + smax/Cos(DegToRad()*theta) ) + 1.0);
    if (xup > SDMON_X_MAX)
      xup = SDMON_X_MAX;

    ylo = (Int_t)(Floor(R[1] - smax/Cos(DegToRad()*theta) ) - 1.0);
    if (ylo < 1)
      ylo = 1;
    yup = (Int_t)(Ceil(R[1] + smax/Cos(DegToRad()*theta) ) + 1.0);
    if (yup > SDMON_Y_MAX)
      yup = SDMON_Y_MAX;

    Double_t smin;
    Double_t smax1=smax;
    if (smax > 1.5)
      {
        smin = smax - 0.5;
        smax1 = smax + 0.1;
      }
    else if (smax >= 1.0 && smax <= 1.5)
      {
        smin = smax - 0.25;
        smax1 = smax + 0.1;
      }
    else
      {
        smin = smax; // Don't add counters closer than 1200m
        smax1 = smax1;
      }

    smax=smax1;

    Int_t tower_id = rusdraw_.site;
    Int_t itowid;
    Double_t xyz[3];
    if (tower_id < 0 || tower_id > 6)return false;

    for (ix = (xlo-1); ix < xup; ix++)
      {
        for (iy = (ylo-1); iy < yup; iy++)
          {
            itowid=sdcoorclf.get_towerid(rusdraw_.yymmdd, (ix+1), (iy+1));
            
            // For events read out only by one site
            if ((tower_id < 3) && (itowid != tower_id))
            continue;
            
            // For multiple site events
            if ( tower_id >= 3 )
              {

                if( (tower_id == RUSDRAW_BRLR) &&
                    (itowid != RUSDRAW_BR) &&
                    (itowid != RUSDRAW_LR) )
                continue;

                if( (tower_id == RUSDRAW_BRSK) &&
                    (itowid != RUSDRAW_BR) &&
                    (itowid != RUSDRAW_SK) )
                continue;

                if( (tower_id == RUSDRAW_LRSK) &&
                    (itowid != RUSDRAW_LR) &&
                    (itowid != RUSDRAW_SK) )
                continue;

                if( (tower_id == RUSDRAW_BRLRSK) &&
                    (itowid != RUSDRAW_BR) &&
                    (itowid != RUSDRAW_LR) &&
                    (itowid != RUSDRAW_SK) )
                continue;

              }
	    
	    // Ignore counters which don't have valid GPS coordinates
            if (!sdcoorclf.get_xyz(rusdraw_.yymmdd, (ix+1), (iy+1), &xyz[0]))
              continue;
            xyz[0] -= SD_ORIGIN_X_CLF; // Subtract SD origin
            xyz[1] -= SD_ORIGIN_Y_CLF;
	    
            if (sds_hit[ix][iy] == 1) continue;
            d[0]=xyz[0]-R[0];
            d[1]=xyz[1]-R[1];
            dotp = Sin(DegToRad()*theta)*(d[0]*Cos(DegToRad()*phi) +d[1]
                *Sin(DegToRad()*phi));
            sdist = sqrt(d[0]*d[0]+d[1]*d[1] - dotp*dotp);
            if ( sdist < smin )
            continue;
            if ( sdist> smax )
            continue;
            sdxxyy[nfsds]=100*(ix+1)+(iy+1);
            memcpy(&X[nfsds][0],&xyz[0],3*sizeof(Double_t));
            t[nfsds] = 0.0;
            dt[nfsds] = 0.0;
            rho[nfsds] = 0.0;
            drho[nfsds] = 0.53 * 3.125;
            pflag[nfsds] = 0; // counter that was put in

            nfsds ++;
          }
      }

    iexclude = -1;

    if(!recalcCog())
    return false;

    save_fitinfo();

    // Number of actuals SDs for each type of fit must be greater than 1.
    if (ntfsds < 1)
    return false;
    if (nldfsds < 1)
    return false;

    return true;
  }

//______________________________________________________________________________


int p2gldffitter_class::clean(Double_t deltaChi2)
  {
    Int_t i, nfpars;
    Double_t chi2old, chi2new;
    Int_t nDeletedPts;
    Double_t dChi2;
    Int_t worst_i; // Keep track of the worst data point

    nfpars=6;

    nDeletedPts = 0;
    doFit();
    save_fpars();

    do
      {
        if (nldfsds < 1 || ntfsds < 1)
          {
            iexclude = -1;
            return nDeletedPts;
          }
        restore_fpars();
        // Fit with current data points and save the chi2 value
        iexclude = -1;
        if (!recalcCog())
          {
            iexclude = -1;
            return nDeletedPts;
          }
        doFit();

        save_fpars();

        chi2old=chi2;
        // When chi2 is already smaller than deltaChi2
        if (chi2old < deltaChi2)
          {
            save_fitinfo();
            ndof = nfitpts-nfpars;
            iexclude = -1;
            return nDeletedPts;
          }

        dChi2=0.0; // initialize the chi2 difference.
        worst_i = 0; // just to initialize it
        // Find the largest chi2 difference.  This will be the worst point.
        for (i=0; i<nfsds; i++)
          {
            // Fit w/o the i'th data pint and take the new chi2
            restore_fpars();
            iexclude = i;
            if (!recalcCog())
              {
                iexclude = -1;
                return nDeletedPts;
              }
            doFit();
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
            nDeletedPts ++;
          }

      } while (dChi2>=deltaChi2);

    restore_fpars();

    // This is important.  If we don't recalculate the COG with current data points, then
    // the cog value passed to the fitter (right after this routine) will contain
    // cog computed using 1 point less (see the previous loop). 
    iexclude = -1;
    if (!recalcCog())
      {
        iexclude = -1;
        return nDeletedPts;
      }
    save_fitinfo();
    ndof = nfitpts-nfpars;
    iexclude = -1;
    return nDeletedPts;

  }

bool p2gldffitter_class::doFit()
  {

    Int_t i;
    TString parname(10); // max. size of the name of any fit parameter
    Int_t nfpars;
    // start values, step sizes, lower and upper limits
    static Double_t vstart, step, vlo, vup;

    // For passing lists of options to Minuit
    Double_t arglist[10];
    Int_t ierflg;

    Double_t stheta; // sine of zenith angle
    static Int_t n_trials = 0; // number of trial fits to get a sensible event direction


    if (gMinuit)
      delete gMinuit;
    gMinuit=0;

    nfpars=6;
    gMinuit = new TMinuit(nfpars);
    gMinuit->SetFCN(fcn_linsley_ldf);

    gMinuit->SetPrintLevel(-1);

    ndof = nfitpts-nfpars; // # of d.o.f
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
          parname="S";
          vstart=S;
          step=0.1;
          break;

        default:
          break;

          }

        // add fit parameter information to Minuit
        gMinuit->mnparm(i, parname, vstart, step, vlo, vup, ierflg);
      }

    // Fitting
    gMinuit->SetMaxIterations(P2GLDF_MITR);
    gMinuit->Migrad();

    calc_results();

    // Must have a shower going downwards.  If not so from the 1st trial, flip its axis and re-fit.
    stheta = sin(theta * DegToRad());
    theta = RadToDeg() * asin(stheta);
    if (stheta < 0.0)
      {
        n_trials++;
        theta *= -1.0;
	phi = tacoortrans::range(phi+180.0,360.0);
        // Caution, to avoid blowing the heap.
        if (n_trials > 10)
          {
            n_trials = 0;
            return true;
          }
        return doFit();
      }
    else
      {
        n_trials = 0;
      }

    return true;

  }

void p2gldffitter_class::save_fpars()
  {
    theta_old = theta;
    phi_old = phi;
    memcpy(R_old, R, (int)(2*sizeof(real8)));
    T0_old = T0;
    S_old = S;
  }

void p2gldffitter_class::restore_fpars()
  {
    theta = theta_old;
    phi = phi_old;
    memcpy(R, R_old, (int)(2*sizeof(real8)));
    T0 = T0_old;
    S = S_old;
  }

void p2gldffitter_class::save_fitinfo()
  {
    nfitsds = nfsds;
    nldffitsds = nldfsds;
    ntfitsds = ntfsds;
    for (Int_t isd=0; isd<nfsds; isd++)
      {
        fxxyy[isd] = sdxxyy[isd];
        fpflag[isd] = pflag[isd];
        memcpy(&fX[isd][0], &X[isd][0], 3*sizeof(Double_t));
        ft[isd] = t[isd];
        fdt[isd] = dt[isd];
        frho[isd] = rho[isd];
        fdrho[isd] = drho[isd];
      }
    // Number of fit points which goes into computing ndof, i.e. SDs w/o zero hit SDs
    // and if an SD participates in time and LDF fitting, then it countributes 2 fit points.
    nfitpts = nldfsds+ntfsds;
  }

void p2gldffitter_class::calc_results()
  {

    Int_t nfpars = 6;



    // Obtain the fit parameters
    gMinuit->GetParameter(0, theta, dtheta);
    gMinuit->GetParameter(1, phi, dphi);
    phi = tacoortrans::range(phi,360.0);
    gMinuit->GetParameter(2, R[0], dR[0]);
    gMinuit->GetParameter(3, R[1], dR[1]);
    gMinuit->GetParameter(4, T0, dT0);
    gMinuit->GetParameter(5, S, dS);
    
    // Chi2, ndof results
    chi2 = gMinuit->fAmin;
    ndof = nfitpts-nfpars;

    // Evaluate the LDF function at 600 and 800m
    s600=S*ldffun(600.0, theta);
    s600_0 = s600 / S600_attenuation_AGASA(theta);
    energy = 0.203 * s600_0;
    log10en = 18.0 + Log10(energy);
    s800=S*ldffun(800.0, theta); // Signal size at 800m from core

  }

bool p2gldffitter_class::hasConverged()
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
