#include "p2ldffitter.h"
#include "sduti.h"
#include "sdxyzclf_class.h"
#include "ldffun.h"
using namespace TMath;

// Fit variables
static Int_t nfitpts; // number of points in the fit
static Int_t nactpts; // number of actual SDs that were hit
static Int_t pflag[NGSDS]; // Flag for the point: 0 - zero charge counter put in, 1 - actual counter
static Double_t X[NGSDS][3], // hit positions, [1200m] units
    rho[NGSDS], // charge density (vem/m^2)
    drho[NGSDS]; // error on charge density due to SD response fluctuation (vem/m^2)

static Int_t iexclude = -1; // to exclude points

static Double_t zenang, azi; // Event direction, zenith and azimuthal angles, degrees
static Double_t coreXY[2]; // Core XY position, [1200m] units

static sdxyzclf_class sdcoorclf;

static void remove_point(Int_t ipoint)
  {
    nfitpts --;
    if (pflag[ipoint]==1)
      nactpts--;
    for (Int_t i=ipoint; i<nfitpts; i++)
      {
        memcpy(&X[i][0], &X[i+1][0], 3*sizeof(Double_t));
        rho[i]=rho[i+1];
        drho[i]=drho[i+1];
        pflag[i]=pflag[i+1];
      }
  }

// FCN to minimize when fitting into AGASA LDF.
/*
 * par[0] - core x
 * par[1] - core y
 * par[2] - scaling factor
 */

//______________________________________________________________________________
static void fcn_ldf(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par,
    Int_t iflag)
  {
    Int_t i;
    Double_t d[2]; // Vector distance from core in ground plane
    Double_t r; // Distance from the shower axis
    Double_t dotp; // Dot product of shower direction and dist. from core vector.
    Double_t delta;
    Double_t denom;
    Double_t chisq = 0;
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {

        if (i==iexclude)
          continue;

        d[0] = X[i][0] - par[0];
        d[1] = X[i][1] - par[1];
        dotp = Sin(zenang*DegToRad())*(d[0]*Cos(DegToRad()*azi)+d[1]
            *Sin(DegToRad()*azi));

        // Distance from the shower axis
        r = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);

        delta = (rho[i] - par[2] * ldffun(r*1.2e3, zenang));
        denom = drho[i]*drho[i];
        chisq += delta*delta/denom;
      }
    //    
    chisq += ((coreXY[0]-par[0])*(coreXY[0]-par[0])+ (coreXY[1]-par[1])
        *(coreXY[1]-par[1])) / (P2LDF_TEMP_dr*P2LDF_TEMP_dr);
    f = chisq;
  }

//____________________________________________________________________________________________

//_______________________________________________________________________________________


p2ldffitter_class::p2ldffitter_class()
  {
    gMinuit = 0;
    sdorigin_xy[0] = SD_ORIGIN_X_CLF;
    sdorigin_xy[1] = SD_ORIGIN_Y_CLF;
  }

p2ldffitter_class::~p2ldffitter_class()
  {
  }

bool p2ldffitter_class::loadVariables()
  {
    Int_t i;
    Double_t xy[2];
    Double_t d[2]; // Dist. from core on ground plane, vector
    Double_t dotp, cdist, sdist, smax;

    iexclude = -1;

    // Event direction
    theta = rusdgeom_.theta[1];
    phi = rusdgeom_.phi[1];

    // Barycentric core calculation from tyro analysis used as core starting value
    memcpy(R, rufptn_.tyro_xymoments[2], (integer4)(2*sizeof(real8)));
    memcpy(coreXY, R, (Int_t)(2*sizeof(Double_t)));
    S = 1.0; // Starting value for LDF scale parameter

    // These are needed in FCN as global static variables
    zenang = theta;
    azi = phi;

    nfitpts = 0;
    nactpts = 0;
    memset(sds_hit, 0, (SDMON_X_MAX*SDMON_Y_MAX*sizeof(Int_t)));
    Int_t ix, iy;
    Int_t xlo, xup;
    Int_t ylo, yup;

    smax = 0.0;
    for (i=0; i < rufptn_.nhits; i++)
      {

        // Hits in space-time cluster will be used for LDF fitting
        if (rufptn_.isgood[i] < 4)
          continue;
        // Important to know if a given SD was hit, so that we don't add the 0's for them
        ix = rufptn_.xxyy[i] / 100 - 1;
        iy = rufptn_.xxyy[i] % 100 - 1;
        sds_hit[ix][iy] = 1;

        // Exclude saturated counters
        if (rufptn_.isgood[i] == 5)
          continue;

        xy[0] = rufptn_.xyzclf[i][0] - sdorigin_xy[0];
        xy[1] = rufptn_.xyzclf[i][1] - sdorigin_xy[1];
        d[0]=xy[0]-R[0];
        d[1]=xy[1]-R[1];
        cdist = sqrt(d[0]*d[0]+d[1]*d[1]);

        dotp = Sin(DegToRad()*theta)*(d[0]*Cos(DegToRad()*phi) +d[1]
            *Sin(DegToRad()*phi));
        sdist = sqrt(d[0]*d[0]+d[1]*d[1] - dotp*dotp);

        // Maximum perpendicular distance from shower axis
        if (sdist > smax)
          smax = sdist;

        if (cdist < 0.5)
          continue;

        X[nfitpts][0] = xy[0];
        X[nfitpts][1] = xy[1];
        X[nfitpts][2] = rufptn_.xyzclf[i][2];
        rho[nfitpts] = 0.5 * (rufptn_.pulsa[i][0]+rufptn_.pulsa[i][1])/ 3.0;
        drho[nfitpts] = 0.5 * sqrt(rufptn_.pulsaerr[i][0]
            *rufptn_.pulsaerr[i][0] + rufptn_.pulsaerr[i][1]
            *rufptn_.pulsaerr[i][1])/ 3.0;

        drho[nfitpts] = 0.53 * sqrt(2.0*rho[nfitpts] + 0.15*0.15*rho[nfitpts]
            *rho[nfitpts]);
        pflag[nfitpts] = 1; // counter that had a non-zero charge in it

        //        if(rho[nfitpts] < 1.0)
        //          continue;


        nfitpts++;
        nactpts++;
      }

    // Additional data points (counters that were not hit)


    // This is just a range of X and Y values for counters that 
    // we need to scan and see if if they are inside the event ellipse or not.
    xlo = (Int_t) (Floor(R[0] - smax/Cos(DegToRad()*theta) ) - 1.0);
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
            memcpy(&X[nfitpts][0],&xyz[0],3*sizeof(Double_t));
            rho[nfitpts] = 0.0;
            drho[nfitpts] = 0.53 * 3.125;
            pflag[nfitpts] = 0; // counter that was put in
            nfitpts ++;
          }
      }

    nfpts = nfitpts;
    napts = nactpts;

    memcpy(fX,X,3*NGSDS*sizeof(Double_t));
    memcpy(frho,rho,NGSDS*sizeof(Double_t));
    memcpy(fdrho,drho,NGSDS*sizeof(Double_t));

    ndof = nactpts - NFPARS;
    if(nactpts<1)
    return false;

    return true;
  }

//______________________________________________________________________________


bool p2ldffitter_class::doFit(bool fixCore)
  {

    Int_t i;
    TString parname(10); // max. size of the name of any fit parameter
    Int_t nfpars;
    // start values, step sizes, lower and upper limits
    static Double_t vstart, step, vlo, vup;

    // For passing lists of options to Minuit
    Double_t arglist[10];
    Int_t ierflg;

    if (gMinuit)
      gMinuit->Delete();
    gMinuit=0;

    nfpars=NFPARS;
    gMinuit = new TMinuit(nfpars);

    gMinuit->SetFCN(fcn_ldf);

    // No extra printing
    gMinuit->SetPrintLevel(-1);

    ndof = nactpts-nfpars; // # of d.o.f
    if (nactpts < 1)
      return false;
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
          parname="Core X";
          vstart=R[0];
          step=0.1;
          break;
        case 1:
          parname="Core Y";
          vstart=R[1];
          step=0.1;
          break;

        case 2:
          parname="Scale";
          vstart = S;
          step = 0.1;
          break;
        default:
          break;

          }

        // add fit parameter information to Minuit
        gMinuit->mnparm(i, parname, vstart, step, vlo, vup, ierflg);
      }

    if (fixCore)
      {
        gMinuit->FixParameter(0);
        gMinuit->FixParameter(1);
      }

    // Do the fitting
    gMinuit->SetMaxIterations(P2LDF_MITR);
    gMinuit->Migrad();

    // Obtain the fit parameters 
    gMinuit->GetParameter(0, R[0], dR[0]);
    gMinuit->GetParameter(1, R[1], dR[1]);
    gMinuit->GetParameter(2, S, dS);

    // Evaluate the LDF function at 600 and 800m
    S600=S*ldffun(600.0, theta);
    S600_0 = S600 / S600_attenuation_AGASA(theta);

    energy = 0.203 * S600_0;
    log10en = 18.0 + Log10(energy);

    S800=S*ldffun(800.0, theta); // Signal size at 800m from core


    chi2 = gMinuit->fAmin;
    ndof = nactpts-nfpars;

    // Copy new set of data points into array, in case
    // we were cleaning the data points
    nfpts = nfitpts;
    napts = nactpts;
    memcpy(fX, X, 3*NGSDS*sizeof(Double_t));
    memcpy(frho, rho, NGSDS*sizeof(Double_t));
    memcpy(fdrho, drho, NGSDS*sizeof(Double_t));

    return true;

  }

Int_t p2ldffitter_class::clean(Double_t deltaChi2, bool fixCore)
  {
    Int_t i;
    Double_t chi2old, chi2new;
    Int_t nDeletedPts;
    Double_t dChi2;
    Int_t worst_i; // Keep track of the worst data point


    nDeletedPts = 0;
    iexclude = -1;
    doFit(fixCore);

    do
      {
        if (nfitpts < 1)
          {
            iexclude = -1;
            return nDeletedPts;
          }
        // Fit with current data points and save the chi2 value
        iexclude = -1;
        doFit(fixCore);

        chi2old=chi2;
        // When chi2 is already smaller than deltaChi2
        if (chi2old < deltaChi2)
          {
            iexclude = -1;
            return nDeletedPts;
          }

        dChi2=0.0; // initialize the chi2 difference.
        worst_i = 0; // just to initialize it
        // Find the largest chi2 difference.  This will be the worst point.
        for (i=0; i<nfitpts; i++)
          {
            // Fit w/o the i'th data pint and take the new chi2
            iexclude = i;
            doFit(fixCore);
            chi2new=chi2;

            //       fprintf(stdout, "chi2new=%f\n",chi2new);
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

    iexclude = -1;
    return nDeletedPts;

  }
bool p2ldffitter_class::hasConverged()
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
