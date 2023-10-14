#include "gldffitter.h"
#include "sdxyzclf_class.h"
#include "sdenergy.h"
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
    (*td) = 0.75 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
    (*ts) = 0.42 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
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
            denom = GLDF_ERR_SCALE*GLDF_ERR_SCALE*(lts*lts + dt[i]*dt[i]);
            delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - ltd;
            chisq += delta*delta / denom;
          }

        // LDF portion of fit
        if ( (pflag[i] == 0) || pflag[i] == 2)
          {
            delta = (rho[i] - par[5] * ldffun(s*1.2e3, par[0]));
            denom = GLDF_ERR_SCALE*GLDF_ERR_SCALE*drho[i]*drho[i];
            chisq += delta*delta/denom;
          }

      }
    // Portion that prevents the core from moving far away
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3]) *(cogXY[1]
        -par[3])) / (GLDF_TEMP_dr*GLDF_TEMP_dr);
    f = chisq;
  }

//____________________________________________________________________________________________
ClassImp(gldffitter)

gldffitter::gldffitter()
  {
    gMinuit = 0;
    gTrsdVsR = 0;
    gTrsdVsS = 0;
    gTvsU = 0;
    gRhoVsS = 0;
    gRhoRsdVsS = 0;
    ldfFun = 0;
    sdorigin_xy[0] = SD_ORIGIN_X_CLF;
    sdorigin_xy[1] = SD_ORIGIN_Y_CLF;

    hTrsdS = new TH2F("hTrsdS","Residuals vs S",65,0.0,6.0,80,-4.0,4.0);
    pTrsdS = new TProfile("pTrsdS","Residuals vs S",24,0.0,6.0,-4.0,4.0,"S");

    hTrsdRho = new TH2F ("hTrsdRho","Residuals vs #rho",100,0.0,100.0,80,-4.0,4.0);
    pTrsdRho = new TProfile("pTrsdRho","Residuals vs #rho",100,0.0,100.0,-4.0,4.0,"S");

    cleanRsd();
  }

gldffitter::~gldffitter()
  {
  }

bool gldffitter::loadVariables(rusdraw_class *rusdraw1, rufptn_class *rufptn1,
    rusdgeom_class *rusdgeom1, rufldf_class *rufldf1)
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
    T0 = rusdgeom1->t0[2];
    // tyro theta and phi, using upper and lower layers
    theta = rusdgeom1->theta[2];
    phi = rusdgeom1->phi[2];
    R[0] = rusdgeom1->xcore[2];
    R[1] = rusdgeom1->ycore[2];  
    S    = rufldf1->sc[0]; // Use LDF alone fit S for starting value

    smax = 0.0;
    for (i=0; i < rufptn1->nhits; i++)
      {
        // Use hits that are in space-time cluster only
        if (rufptn1->isgood[i] < 3)
          continue;
        
        // Important to know if a given SD was hit, so that we don't add the 0's for them
        ix = rufptn1->xxyy[i] / 100 - 1;
        iy = rufptn1->xxyy[i] % 100 - 1;
        sds_hit[ix][iy] = 1;

        sdxxyy[nfsds] = rufptn1->xxyy[i];

        xy[0] = rufptn1->xyzclf[i][0] - sdorigin_xy[0];
        xy[1] = rufptn1->xyzclf[i][1] - sdorigin_xy[1];
        X[nfsds][0] = xy[0];
        X[nfsds][1] = xy[1];
        X[nfsds][2] = rufptn1->xyzclf[i][2];
        t [nfsds]= (rufptn1->reltime[i][0]+rufptn1->reltime[i][1]) / 2.0;
        dt [nfsds] = 0.5 * sqrt(rufptn1->timeerr[i][0]*rufptn1->timeerr[i][0]
            + rufptn1->timeerr[i][1]*rufptn1->timeerr[i][1]);
        rho[nfsds]= ((rufptn1->pulsa[i][0]+rufptn1->pulsa[i][1])/2.0) / 3.0;
        drho[nfsds] = 0.53 * sqrt(2.0*rho[nfsds] + 0.15*0.15*rho[nfsds]*rho[nfsds]);

        d[0] = X[nfsds][0]-rusdgeom1->xcore[2];
        d[1] = X[nfsds][1]-rusdgeom1->ycore[2];
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
    xlo = (Int_t) (Floor( R[0] - smax/Cos(DegToRad()*theta) ) - 1.0);
    if (xlo < 1)
      xlo = 1;
    xup = (Int_t) (Ceil(R[0] + smax/Cos(DegToRad()*theta) ) + 1.0);
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

    Int_t tower_id = rusdraw1->site;
    Int_t itowid;
    Double_t xyz[3];
    
    if (tower_id < 0 || tower_id > 6)return false;

    for (ix = (xlo-1); ix < xup; ix++)
      {
        for (iy = (ylo-1); iy < yup; iy++)
          {
            itowid=sdcoorclf.get_towerid(rusdraw1->yymmdd, (ix+1), (iy+1));
            
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
            if (!sdcoorclf.get_xyz(rusdraw1->yymmdd, (ix+1), (iy+1), &xyz[0]))
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

void gldffitter::cleanRsd()
  {
    hTrsdS->Reset();
    pTrsdS->Reset();

    hTrsdRho->Reset();
    pTrsdRho->Reset();

  }

// Form of LDF function for plotting it in root
// x[0]   - perpendicular distance from core, [1200m] units
// par[0] - scaling constant
// par[1] - zenith angle, degree
static Double_t ldfRootFun(Double_t *x, Double_t *par)
  {
    return par[0] * ldffun(x[0]*1.2e3, par[1]);
  }

// computes various graphs and variables using new geometry fit values.
void gldffitter::compVars()
  {
    Int_t i;
    Double_t ltd, lts; // Linsley's time delay & time fluctuation
    Double_t trsd[NGSDS]; // time fit residual
    Double_t rhorsd[NGSDS]; // LDF fit residual
    Double_t ts[NGSDS]; // fluctuation on time delay
    Double_t r [NGSDS]; // dist. from core in xy (ground) plane
    Double_t s [NGSDS]; // dist. from core in shower front plane
    Double_t u [NGSDS]; // dsit. along the shower axis in ground plane
    Double_t dr[NGSDS];
    Double_t ds[NGSDS];
    Double_t ldfrho[NGSDS]; // Charge density participating in LDF fit
    Double_t ldfdrho[NGSDS]; // Error on charge density participating in LDF fit

    Double_t degrad=DegToRad(); // to convert from degrees to radians
    Double_t d[2]; // vector dist. from core in xy (ground) plane
    Double_t dotp; // dot production of distance from core and event axis vector

    Int_t ntpts; // number of all time fit points
    Int_t nldfpts; // number of all LDF fit points


    ntpts = 0;
    // Fill graph for time fit results
    for (i=0; i<nfsds; i++)
      {
        if (pflag[i] < 1)
          continue;
        d[0] = X[i][0]-R[0];
        d[1] = X[i][1]-R[1];
        /* Dot product of the distance vector from the core (in xy-plane) with the shower
         direction vector (the n-vector). */
        dotp = sin(theta*degrad) *(cos(phi*degrad)*d[0]+sin(phi*degrad) *d[1]);
        r[ntpts] = sqrt(d[0]*d[0]+d[1]*d[1]); // dist. from core in ground plane
        dr[ntpts] = 0.0;
        u[ntpts] = (cos(phi*degrad)*d[0]+sin(phi*degrad)*d[1]);
        s[ntpts] = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        ds[ntpts] = 0.0;

        ltdts(rho[i], s[i]*1.2e3, theta, &ltd, &lts);
        ts[ntpts] = GLDF_ERR_SCALE * sqrt(lts*nsecTo1200m * lts*nsecTo1200m + dt[i]*dt[i]) ;
        trsd[ntpts] = t[i]- T0 - (dotp - X[i][2]*cos(theta*degrad)) - ltd
            *nsecTo1200m;
        ntpts ++;
      }

    // Fill graphs for time fit results
    if (gTrsdVsR)
      {
        delete gTrsdVsR;
        gTrsdVsR = 0;
      }
    if (gTrsdVsS)
      {
        delete gTrsdVsS;
        gTrsdVsS = 0;
      }
    if (gTvsU)
      {
        delete gTvsU;
        gTvsU = 0;
      }

    if (ntpts < 1)
      return;
    gTrsdVsR = new TGraphErrors(ntpts,r,trsd,dr,ts);
    gTrsdVsS = new TGraphErrors(ntpts,s,trsd,ds,ts);
    gTvsU = new TGraphErrors(ntpts,u,t,dr,ts);

    // Fill graph for LDF fit results


    // ldf fuction for drawing. Define it if it hasn't been defined yet.
    if (ldfFun ==0)
      {
        ldfFun = new TF1("ldfFun1",ldfRootFun,0.0,4.0,2);
        ldfFun->GetXaxis()->SetTitle("Distance from shower axis, [1200m]");
        ldfFun->GetYaxis()->SetTitle("Charge Density, [VEM/m^{2}]");
        ldfFun->SetLineColor(2);
      }

    // Set the appropriate parameters
    ldfFun->SetParameter(0, S);
    ldfFun->SetParameter(1, theta);

    nldfpts = 0;
    for (i=0; i<nfsds; i++)
      {
        
        if (pflag[i] == 1)
          continue;
        d[0] = X[i][0]-R[0];
        d[1] = X[i][1]-R[1];
        /* Dot product of the distance vector from the core (in xy-plane) with the shower
         direction vector (the n-vector). */
        dotp = sin(theta*degrad) *(cos(phi*degrad)*d[0]+sin(phi*degrad) *d[1]);
        s[nldfpts] = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        ds[nldfpts] = 0.0;

        ldfrho[nldfpts] = rho[i];
        rhorsd[nldfpts] = ldfrho[nldfpts] - S*ldffun(s[nldfpts]*1.2e3,theta);
        ldfdrho[nldfpts] = GLDF_ERR_SCALE * drho[i];
        nldfpts ++;

      }
    if (gRhoVsS)
      {
        delete gRhoVsS;
        gRhoVsS = 0;
      }
    if (gRhoRsdVsS)
      {
        delete gRhoRsdVsS;
        gRhoRsdVsS = 0;
      }
    if (nldfpts < 1)
      return;
    gRhoVsS    = new TGraphErrors(nldfpts,s,ldfrho,ds,ldfdrho);
    gRhoRsdVsS = new TGraphErrors(nldfpts,s,rhorsd,ds,ldfdrho);
  }

//______________________________________________________________________________


int gldffitter::clean(Double_t deltaChi2, bool verbose)
  {
    Int_t xxyy;
    Int_t i, nfpars;
    Double_t chi2old, chi2new;
    Int_t nDeletedPts;
    Double_t dChi2;
    Int_t worst_i; // Keep track of the worst data point

    nfpars=6;

    nDeletedPts = 0;
    Ifit(false);
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
        Ifit(false);

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
            Ifit(false);
            chi2new=chi2;

            if ((chi2old-chi2new)>dChi2)
              {
                dChi2=(chi2old-chi2new);
                worst_i = i;
              }
          }
        if (dChi2 >= deltaChi2)
          {
            xxyy=((Int_t)(Int_t)Floor(X[worst_i][0]+0.5))*100
                + ((Int_t)(Int_t)Floor(X[worst_i][1]+0.5));
            remove_point(worst_i);
            nDeletedPts ++;
            if (verbose)
              fprintf(stdout,"Removed point: %04d\n",xxyy);
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

void gldffitter::calcRsd(Int_t isds, Double_t *trsd, Double_t *terr,
    Double_t *chrgdens, Double_t *chrgdenserr, Double_t *rcore, Double_t *score)
  {
    Double_t ltd, lts; // Modified time delay and time delay fluctuation
    Double_t rdist; // distance from core in ground plane
    Double_t sdist; // distance from core in shower front plane
    Double_t delta; // the time residual
    Double_t par[5]; // parameters needed for tvsx_plane


    Double_t degrad=DegToRad(); // to convert from degrees to radians
    Double_t dotp, d[2];

    par[0] = theta;
    par[1] = phi;
    par[2] = R[0];
    par[3] = R[1];
    par[4] = T0;

    rdist = sqrt( (X[isds][0]-par[2])*(X[isds][0]-par[2])
        + (X[isds][1] -par[3]) *(X[isds][1]-par[3]));

    d[0]=(X[isds][0]-par[2]);
    d[1]=(X[isds][1]-par[3]);
    dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0] + sin(par[1] *degrad)
        *d[1]);
    sdist = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);

    // Calculating the time delay and time delay fluctuation
    ltdts(rho[isds], sdist*1.2e3, par[0], &ltd, &lts);
    ltd *= nsecTo1200m; // time delay
    lts *= nsecTo1200m; // time delay flucluation

    // Residual
    delta = t[isds] - tvsx_plane(X[isds][0], X[isds][1], X[isds][2], par) - ltd;

    // Return the quantities
    (*trsd)        = delta; // Residual
    (*terr)        = GLDF_ERR_SCALE * sqrt(lts*lts+dt[isds]*dt[isds]); // Time delay fluctuation
    (*chrgdens)    = rho[isds]; // Charge Density
    (*chrgdenserr) = GLDF_ERR_SCALE * drho[isds]; // Charge Density
    (*rcore) = rdist; // Distance from core in ground plane
    (*score) = sdist; // Distance from core in shower front plane

  }

void gldffitter::fillRsd()
  {
    Int_t i, nfpars;
    TString parname(10);
    Double_t arglist[10];
    Int_t ierflg;

    Double_t trsd, terr, chrgdens, chrdenserr, rcore, score;

    if (gMinuit)
      delete gMinuit;
    gMinuit=0;

    nfpars=6;
    gMinuit = new TMinuit(nfpars);
    gMinuit->SetFCN(fcn_linsley_ldf);

    gMinuit->SetPrintLevel(-1);

    ierflg = 0;
    arglist[0] = 1;
    gMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
    static Double_t vstart[GLDF_NFPARS], step[GLDF_NFPARS], vlo[GLDF_NFPARS], vup[GLDF_NFPARS];
    for (i=0; i< nfpars; i++)
      {
        switch (i)
          {
        case 0:
          parname="Theta";
          vstart[i]=theta;
          step[i]=0.1;
          break;
        case 1:
          parname="Phi";
          vstart[i]=phi;
          step[i]=0.1;
          break;
        case 2:
          parname="Core X";
          vstart[i]=R[0];
          step[i]=0.1;
          break;
        case 3:
          parname="Core Y";
          vstart[i]=R[1];
          step[i]=0.1;
          break;
        case 4:
          parname="T0";
          vstart[i]=T0;
          step[i]=0.1;
          break;
        case 5:
          parname="S";
          vstart[i]=S;
          step[i]=0.1;
          break;

        default:
          break;
          }
        // No limits on fit parameters
        vlo[i]=0.0;
        vup[i]=0.0;
        // add fit parameter information to Minuit
        gMinuit->mnparm(i, parname, vstart[i], step[i], vlo[i], vup[i], ierflg);
      }

    gMinuit->SetMaxIterations(GLDF_MITR);

      {
        // Go over all the points.  For i'th point, do the fit without it and then
        // compute its residual
        for (i=0; i<nfsds; i++)
          {
            // Fit w/o the i'th data pint and take the new chi2
            iexclude = i;
            gMinuit->Migrad();
            // get the fit parameters
            gMinuit->GetParameter(0, theta, dtheta);
            gMinuit->GetParameter(1, phi, dphi);
	    phi = tacoortrans::range(phi,360.0);
            gMinuit->GetParameter(2, R[0], dR[0]);
            gMinuit->GetParameter(3, R[1], dR[1]);
            gMinuit->GetParameter(4, T0, dT0);

            // Compute the residual of the i'th point
            calcRsd(i, &trsd, &terr, &chrgdens, &chrdenserr, &rcore, &score);

            // Fill Residual vs S Scatter plot
            hTrsdS -> Fill(score, trsd);

            // Fill Residual vs S profile histogram
            pTrsdS -> Fill(score, trsd);

            // Residual vs rho 
            hTrsdRho -> Fill(chrgdens, trsd);
            pTrsdRho -> Fill(chrgdens, trsd);

          }

      }
      iexclude = -1;

  }

bool gldffitter::Ifit(bool verbose)
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

    if (verbose)
      fprintf(stdout,"Fitting into modified Linsley's with LDF\n");
    nfpars=6;
    gMinuit = new TMinuit(nfpars);
    gMinuit->SetFCN(fcn_linsley_ldf);

    if (!verbose)
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
    gMinuit->SetMaxIterations(GLDF_MITR);
    gMinuit->Migrad();

    if (verbose)
      {
        // Print results
        Double_t amin, edm, errdef;
        Int_t nvpar, nparx, icstat;
        gMinuit->mnstat(amin, edm, errdef, nvpar, nparx, icstat);
        gMinuit->mnprin(3, amin);
      }

    calc_results();

    // Must have a shower going downwards.  If not so from the 1st trial, flip its axis and re-fit.
    stheta = sin(theta * DegToRad());
    theta = RadToDeg() * asin(stheta);
    if (stheta < 0.0)
      {
        n_trials++;
        // Caution, to avoid blowing the heap.
        if (n_trials > 10)
          {
            theta *= -1.0;
	    phi = tacoortrans::range(phi+180.0,360.0);
            if (verbose)
              print_results();
            n_trials = 0;
            return true;
          }
        return Ifit(verbose);
      }
    else
      {
        n_trials = 0;
      }

    if (verbose)
      print_results();

    return true;

  }

void gldffitter::save_fpars()
  {
    theta_old = theta;
    phi_old = phi;
    memcpy(R_old, R, (int)(2*sizeof(real8)));
    T0_old = T0;
    S_old = S;
  }

void gldffitter::restore_fpars()
  {
    theta = theta_old;
    phi = phi_old;
    memcpy(R, R_old, (int)(2*sizeof(real8)));
    T0 = T0_old;
    S = S_old;
  }

void gldffitter::save_fitinfo()
  {
    nfitsds       =  nfsds;
    nldffitsds    =  nldfsds;
    ntfitsds      =  ntfsds;
    for (Int_t isd=0; isd<nfsds; isd++)
      {
        fxxyy[isd]  = sdxxyy[isd];
        fpflag[isd] = pflag[isd];
        memcpy(&fX[isd][0], &X[isd][0], 3*sizeof(Double_t));
        ft[isd]     = t[isd];
        fdt[isd]    = dt[isd];
        frho[isd]   = rho[isd];
        fdrho[isd]  = drho[isd];
      }
    // Number of fit points which goes into computing ndof, i.e. SDs w/o zero hit SDs
    // and if an SD participates in time and LDF fitting, then it countributes 2 fit points.
    nfitpts = nldfsds+ntfsds;
  }

void gldffitter::calc_results()
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

    // Evaluate the LDF function at 600 and 800m
    s600=S*ldffun(600.0, theta);
    s600_0 = s600 / S600_attenuation_AGASA(theta);
    
    s800=S*ldffun(800.0, theta); // Signal size at 800m from core

    energy = rusdenergy(s800,theta);
    log10en = 18.0 + Log10(energy);

    // Chi2, ndof results

    chi2 = gMinuit->fAmin;
    ndof = nfitpts-nfpars;
  }

void gldffitter::print_results()

  {
    fprintf(stdout, "Number of effective fit points: %d\n",nfitpts);
    fprintf(stdout, "Total number of SDs: %d\n",nfitsds);
    fprintf(stdout, "Number of actual LDF fit SDs: %d\n",nldffitsds);
    fprintf(stdout, "Number of time fit SDs: %d\n",ntfitsds);
    fprintf(stdout, "chi2  = %f\n",chi2);
    fprintf(stdout, "ndof  = %d\n",ndof);

    // Print geom. results
    fprintf(stdout, "theta = %f +/- %f\n",theta,dtheta);
    fprintf(stdout, "phi   = %f +/- %f\n",phi,dphi);
    fprintf(stdout, "xcore = %f +/- %f\n",R[0],dR[0]);
    fprintf(stdout, "ycore = %f +/- %f\n",R[1],dR[1]);
    fprintf(stdout, "T0    = %f +/- %f\n",T0,dT0);

    // Print LDF results
    fprintf(stdout, "S       = %f +/- %f\n",S,dS);
    fprintf(stdout, "S600    = %f\n",s600);
    fprintf(stdout, "S600_0  = %f\n",s600_0);
    fprintf(stdout, "S600 attenuation: %f\n",
    S600_attenuation_AGASA(theta));
    fprintf(stdout, "S800    = %f\n",s800);
    fprintf(stdout, "log10en = %.1f\n",log10en);
    fprintf(stdout, "Energy  = %.1f EeV\n",energy);

  }
