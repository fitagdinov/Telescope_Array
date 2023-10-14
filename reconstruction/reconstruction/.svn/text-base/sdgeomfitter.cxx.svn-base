#include "sdgeomfitter.h"
#include "sdxyzclf_class.h"
#include "tacoortrans.h"
using namespace TMath;

// Fit variables
static Int_t nfitpts; // number of points in the fit
static Double_t X[NGSDS][3], // hit positions, counter separation units
    t[NGSDS], // hit time (relative to earliest hit, counter separation units)
    dt[NGSDS], // error on hit time (counter separation units)
    rho[NGSDS], // charge density (vem/m^2)
    drho[NGSDS]; // error on charge density (vem/m^2)
    

static Int_t rufptnindex[NGSDS]; // rufptn indices of each point

// if set to non-negative value, then i'th data point is excluded in fcn_cleaner
static Int_t iexclude;

static Double_t cogXY[2];



// To remove i'th data point from the fitting
static void remove_point(Int_t ipoint)
  {
    Int_t i,j;
    nfitpts -= 1;
    for(i=ipoint;i<nfitpts;i++)
      {
        for(j=0;j<3;j++)
          X[i][j]=X[i+1][j];
        t[i]=t[i+1];
        dt[i]=dt[i+1];
        rho[i]=rho[i+1];
        drho[i]=drho[i+1];
        rufptnindex[i]=rufptnindex[i+1];
      }
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
        a=(0.6511268210e-4*(theta-.2614963683))*(theta*theta-134.7902422*theta+4558.524091);
      }
    else
      {
        a = exp( -3.2e-2 * theta + 2.0);
      }
    
    
   

    // Contained, >=7 counters, DCHI2 = 3.0
//    (*td) = 0.72 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.33 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
//     Contained, >=7 counters, DCHI2 = 4.0
//    (*td) = 0.75 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.42 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
    // Contained, >=7 counters, DCHI2 = 5.0
//    (*td) = 0.78 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.50 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
    // Contained, >=7 counters, DCHI2 = 6.0
//    (*td) = 0.78 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.55 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    // Contained, >=7 counters, DCHI2 = 7.0
//    (*td) = 0.78 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.58 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    // Contained, >=7 counters, DCHI2 = 8.0
//    (*td) = 0.8 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.6 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    // Contained, >=7 counters, DCHI2 = 9.0
//    (*td) = 0.82 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.63 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3); 
    
    
    // Contained, >=7 counters, DCHI2 = 9.0, chi2/dof < 3.0
//    (*td) = 0.82 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.63 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    // Contained, >=7 counters, DCHI2 = 10.0
//    (*td) = 0.82 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.63 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3); 
    
    // Contained, >=6 counters, DCHI2 = 3.0
//    (*td) = 0.75 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.50 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
    
    // Contained, >=6 counters, DCHI2 = 4.0
//    (*td) = 0.75 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.55 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
    // Contained, >=6 counters, DCHI2 = 5.0
//    (*td) = 0.76 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.61 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
    // Contained, >=6 counters, DCHI2 = 6.0
//    (*td) = 0.78 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.65 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
    // Contained, >=6 counters, DCHI2 = 7.0
//    (*td) = 0.80 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.68 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
    // Contained, >=6 counters, DCHI2 = 8.0
//    (*td) = 0.80 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.70 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    // Contained, >=6 counters, DCHI2 = 9.0
//    (*td) = 0.80 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.70 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
//     Contained, >=6 counters, DCHI2 = 10.0
//    (*td) = 0.80 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
//    (*ts) = 0.70 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
    
    //     Contained, >=4 counters, DCHI2 = 4.0
    (*td) = 0.80 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.5);
    (*ts) = 0.70 * a * pow( (1.0 + R / 30.0), 1.5) * pow(Rho, -0.3);
    
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


static void ltdts1(Double_t Rho, Double_t R, Double_t a, Double_t usintheta,
		   Double_t *td, Double_t *ts)
  {
    
    Double_t al;
    
    Double_t td_rp   = 1.35;
    Double_t td_rhop = -0.5;
  
    Double_t ts_rp   = 1.50;
    Double_t ts_rhop = -0.3;
    
    al = 1.56 * Power((1.-0.1 * usintheta/1200.),(td_rp-0.3));
    
    (*td) = a * Power((1.-0.1 * usintheta/1200.), (td_rp-0.3)) * Power( (1
        + R / 30.0), td_rp) * Power(Rho, td_rhop);
    (*ts) = al * Power( (1 + R / 30.0), ts_rp) * Power(Rho, ts_rhop);

  }


/* 
 AGASA Modified Linsley's Time Delay Function
 
 INPUTS:
 rho:   charge density (VEM / square meter )
 R:     distance from the core, meters
 theta: zenith angle (degree)
 
 OUTPUTS:
 td: time delay (nS) 
 ts: time delay fluctuation (nS) 
 */
static void altdts(Double_t rho, Double_t R, Double_t *td, Double_t *ts)
  {
    Double_t a = 2.6;
    Double_t f = a * Power( (1 + R / 30.0), 1.5);
    *td = f * Power(rho, -0.5);
    *ts = f * Power(rho, -0.3);
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

//______________________________________________________________________________
static void fcn_plane(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par,
    Int_t iflag)
  {
    Int_t i;
    Double_t chisq = 0;
    Double_t delta;
    Double_t denom;
    Double_t degrad = DegToRad();
    Double_t dotp, s;
    Double_t d[2];
    Double_t ltd,lts;
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
        ltdts(rho[i],s*1.2e3, par[0], &ltd, &lts);
        lts *= nsecTo1200m;
        denom = sqrt(lts*lts+dt[i]*dt[i]);
        delta = t[i] - tvsx_plane(X[i][0], X[i][1], X[i][2], par);
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3])*(cogXY[1]-par[3])) / 
    (SDGEOM_TEMP_dr*SDGEOM_TEMP_dr);
    f = chisq;
  }

//______________________________________________________________________________
static void fcn_agasa(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par,
    Int_t iflag)
  {
    Int_t i;
    // Double_t degrad=DegToRad(); // to convert from degrees to radians
    Double_t altd, alts; // AGASA Modified Linsley's time delay and time fluctuation
    Double_t ltd,lts;    // Linsley's time delay and time fluctuation
    //    Double_t cdist; // distance from core
    //calculate chisquare
    Double_t chisq = 0;
    Double_t delta;
    Double_t denom;
    Double_t d[2],cdist;
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {
        d[0]=(X[i][0]-par[2]);
        d[1]=(X[i][1]-par[3]); // vector distance from the core in xy - plane
        cdist=sqrt(d[0]*d[0]+d[1]*d[1]);
//        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
//            + sin(par[1] *degrad)*d[1]); // Dot product of distance from core and shower axis vector
//        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        //        cdist=sqrt( (X[i][0]-par[2])*(X[i][0]-par[2])+ (X[i][1]-par[3])
        //            *(X[i][1]-par[3]));
        altdts(rho[i], cdist*1.2e3, &altd, &alts);
        ltdts(rho[i],cdist*1.2e3,par[0],&ltd,&lts);
        altd/=4e3;
        alts/=4e3;
        lts/=4e3;
        // denom = dt[i]*dt[i];
        denom = lts*lts;   // Use Linsley's time fluctuation instead
        delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - altd;
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3])*(cogXY[1]-par[3])) / 
    (SDGEOM_TEMP_dr*SDGEOM_TEMP_dr);
    f = chisq;
  }

//______________________________________________________________________________
static void fcn_linsley(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par,
    Int_t iflag)
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
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3])
        *(cogXY[1]-par[3])) / (SDGEOM_TEMP_dr*SDGEOM_TEMP_dr);
    f = chisq;
  }




//______________________________________________________________________________
static void fcn_linsley1(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par,
    Int_t iflag)
  {
    
    // par[5] is the curvature parameter
    
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
    for (i=0; i<nfitpts; i++)
      {
	
        d[0] = X[i][0]-par[2];
        d[1] = X[i][1]-par[3];
        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
            + sin(par[1] *degrad)*d[1]); // Dot product of distance from core and shower axis vector
        
        dotp -= X[i][2] * cos(degrad*par[0]); // correcting the dot product for altitude
        
        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        ltdts1(rho[i], s*1.2e3, par[5], dotp*1.2e3, &ltd, &lts);

        ltd *= nsecTo1200m;
        lts *= nsecTo1200m;
        denom = lts*lts + dt[i]*dt[i];
        delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - ltd;
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3])
        *(cogXY[1]-par[3])) / (SDGEOM_TEMP_dr*SDGEOM_TEMP_dr);
    f = chisq;
  }





//______________________________________________________________________________
static void fcn_cleaner(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par,
    Int_t iflag)
  {
    Int_t i;
    Double_t ltd, lts; // Modified Linsley's time delay and time delay fluctuation
    Double_t degrad = DegToRad();
    Double_t dotp, s;
    Double_t d[2];
    
    Double_t chisq = 0;
    Double_t delta, denom;
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {
        if(i==iexclude)
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
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3])
        *(cogXY[1]-par[3])) / (SDGEOM_TEMP_dr*SDGEOM_TEMP_dr);
    f = chisq;
  }




//______________________________________________________________________________
static void fcn_sphere(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par,
    Int_t iflag)
  {
    Int_t i;
    Double_t degrad=DegToRad(); // to convert from degrees to radians
    Double_t dotp, d[2];
    Double_t s, delta, denom;
    Double_t chisq = 0;
    Double_t cdist,ltd,lts;
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {
        d[0]=(X[i][0]-par[2]);
        d[1]=(X[i][1]-par[3]); // vector distance from the core in xy - plane
        cdist=sqrt (d[0]*d[0] + d[1]*d[1]);
        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
            + sin(par[1] *degrad)*d[1]); // Dot product of distance from core and shower axis vector
        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        ltdts(rho[i],cdist*1.2e3,par[0],&ltd,&lts);
        lts *= nsecTo1200m;
        denom = lts*lts;
        delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - par[5] + sqrt(par[5]
            *par[5] - s*s);
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3])*(cogXY[1]-par[3])) / 
    (SDGEOM_TEMP_dr*SDGEOM_TEMP_dr);
    f = chisq;
  }

//______________________________________________________________________________
static void fcn_parabola(Int_t &npar, Double_t *gin, Double_t &f,
    Double_t *par, Int_t iflag)
  {
    Int_t i;
    Double_t degrad=DegToRad(); // to convert from degrees to radians
    Double_t dotp, d[2];
    Double_t s, delta, denom;
    Double_t chisq = 0;
    Double_t ltd,lts;
    (void)(npar);
    (void)(gin);
    (void)(iflag);
    for (i=0; i<nfitpts; i++)
      {
        d[0]=(X[i][0]-par[2]);
        d[1]=(X[i][1]-par[3]); // vector distance from the core in xy - plane
        dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0]
            + sin(par[1] *degrad)*d[1]); // Dot product of distance from core and shower axis vector
        s = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        altdts(rho[i],s*1.2e3,&ltd,&lts);
        lts *= nsecTo1200m;
        denom = lts*lts;
        delta = t[i]- tvsx_plane(X[i][0], X[i][1], X[i][2], par) - par[5]*s*s;
        chisq += delta*delta / denom;
      }
    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3])*(cogXY[1]-par[3])) / 
    (SDGEOM_TEMP_dr*SDGEOM_TEMP_dr);
    
    
    
//    chisq += ((cogXY[0]-par[2])*(cogXY[0]-par[2])+(cogXY[1]-par[3])*(cogXY[1]-par[3])) / 
//     (0.001 * 0.001);
    
    
    f = chisq;
  }

//____________________________________________________________________________________________
//____________________________________________________________________________________________
ClassImp(sdgeomfitter)

sdgeomfitter::sdgeomfitter()
  {
    gMinuit = 0;
    gTrsdVsR = 0;
    gTrsdVsS = 0;
    gTvsU = 0;
    gQvsS = 0;
    sdorigin_xy[0] = SD_ORIGIN_X_CLF;
    sdorigin_xy[1] = SD_ORIGIN_Y_CLF;
    
    
    
    hRsdS = new TH2F("hRsdS","Residuals vs S",65,0.0,6.0,80,-4.0,4.0);
    pRsdS = new TProfile("pRsdS","Residuals vs S",24,0.0,6.0,-4.0,4.0,"S");
    
    hRsdRho = new TH2F ("hRsdRho","Residuals vs #rho",100,0.0,100.0,80,-4.0,4.0);
    pRsdRho = new TProfile("pRsdRho","Residuals vs #rho",100,0.0,100.0,-4.0,4.0,"S");
    
    cleanRsd();
  }

sdgeomfitter::~sdgeomfitter()
  {
  }


void sdgeomfitter::cleanRsd()
  {
    hRsdS->Reset();
    pRsdS->Reset();
    
    hRsdRho->Reset();
    pRsdRho->Reset();
    
  }


// computes various graphs and variables using new geometry fit values.
void sdgeomfitter::compVars(Int_t whatFCN)
  {
    Int_t i;
    Double_t altd,alts; // AGASA time delay & time fluctuation
    Double_t ltd,lts;   // Linsley's time delay & time fluctuation
    Double_t trsd[NGSDS]; // time fit residual
    Double_t ts[NGSDS]; // fluctuation on time delay
    Double_t r [NGSDS]; // dist. from core in xy (ground) plane
    Double_t s [NGSDS]; // dist. from core in shower front plane
    Double_t u [NGSDS]; // dsit. along the shower axis in ground plane
    Double_t degrad=DegToRad(); // to convert from degrees to radians
    Double_t d[2]; // vector dist. from core in xy (ground) plane
    Double_t dotp; // dot production of distance from core and event axis vector
    
    for (i=0; i<nfitpts; i++)
      {
        d[0] = X[i][0]-R[0];
        d[1] = X[i][1]-R[1];
        /* Dot product of the distance vector from the core (in xy-plane) with the shower
         direction vector (the n-vector). */
        dotp = sin(theta*degrad)*(cos(phi*degrad)*d[0]+sin(phi*degrad)*d[1]);
        r[i] = sqrt(d[0]*d[0]+d[1]*d[1]); // dist. from core in ground plane
        u[i] = (cos(phi*degrad)*d[0]+sin(phi*degrad)*d[1]);
        s[i] = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
        trsd[i] = t[i]-dotp-T0 ;
        ltdts(rho[i],s[i]*1.2e3,theta,&ltd,&lts);
        altdts(rho[i],r[i]*1.2e3,&altd,&alts);
        ts[i] = sqrt(lts*nsecTo1200m * lts*nsecTo1200m + dt[i]*dt[i]) ; // Modified Linsley's time fluctuation
        // Subtract off the time delays to get the residuals when fitting into
        // AGASA Modified Linsley's or Modified Linsley's
        if(whatFCN==1)
            trsd[i] -= altd*nsecTo1200m;
        if(whatFCN==2)
            trsd[i] -= ltd*nsecTo1200m;
        if(whatFCN==4)
          trsd[i] -= a*s[i]*s[i];
        if (whatFCN==6)
          trsd[i] -= ltd*nsecTo1200m;
      }
    if (gTrsdVsR)
      {
        gTrsdVsR->Delete();
        gTrsdVsR = 0;
      }
    if (gTrsdVsS)
      {
        gTrsdVsS->Delete();
        gTrsdVsS = 0;
      }
    if (gTvsU)
      {
        gTvsU->Delete();
        gTvsU = 0;
      }
    if (gQvsS)
      {
        gQvsS->Delete();
        gQvsS = 0;
      }
    
    if(nfitpts < 1)
      return;
    
    gTrsdVsR = new TGraphErrors(nfitpts,r,trsd,0,ts);
    gTrsdVsS = new TGraphErrors(nfitpts,s,trsd,0,ts);
    gTvsU = new TGraphErrors(nfitpts,u,t,0,ts);
    gQvsS = new TGraphErrors(nfitpts,s,rho,0,drho);
  }




void sdgeomfitter::calcNewCore(rufptn_class *pass1, Double_t *newCore)
  {
    Int_t j,k;
    Double_t w;
    Double_t q2;
    Int_t xy[2];
    Double_t res[2];
    w = 0.0;
    q2 = 0.0;
    for(k=0; k<2; k++)
      {
        res[k] = 0.0;
        xy[k] = 0;
      }
    
    for (j=0; j<pass1->nhits; j++)
      {
        if(rufptn_.isgood[j] < 3)
          continue;
        xy[0] = pass1->xxyy[j] / 100;
        xy[1] = pass1->xxyy[j] % 100;
        q2 = sqrt((pass1->pulsa[j][0]+pass1->pulsa[j][1])/2.0);
        w += q2;
        for(k=0; k<2; k++)
          {
            res[k] += q2 * (Double_t)xy[k];
          }
        
      }
    for(k=0; k<2; k++)
        newCore[k] = res[k] / w;
  }


// To recalculate the core once various cuts are applied
static bool recalcCog()
  {
    Int_t i,j;
    Double_t w;
    // Recalculate the COG core position with cuts (above) applied.
     for (j=0; j<2; j++) cogXY[j] = 0.0;
     w = 0.0;
     for (i=0; i<nfitpts; i++)
       {
         if(i==iexclude)
           continue;
         for (j=0; j<2; j++)
           cogXY[j] += rho[i]*X[i][j];
         w += rho[i];
       }
      
      // Cannot have zero total charge in reconstruction
      if(w < 1e-3)
        return false;
      
      for(j=0;j<2;j++)
          cogXY[j] /= w;
      return true;
  }

bool sdgeomfitter::loadVariables_stclust(rufptn_class *rufptn1, rusdgeom_class *rusdgeom1)
  {
    Int_t i;
    Double_t xy[2];
    
    
    nfitpts = 0;
    for (i=0; i < rufptn1->nhits; i++)
      {
        // Use hits that are in space-time cluster only
        if(rufptn1->isgood[i] < 4)
          continue;
        
        xy[0] = rufptn1->xyzclf[i][0] - sdorigin_xy[0];
        xy[1] = rufptn1->xyzclf[i][1] - sdorigin_xy[1];
        X[nfitpts][0] = xy[0];
        X[nfitpts][1] = xy[1];
        X[nfitpts][2] = rufptn1->xyzclf[i][2];
        
        t [nfitpts]= (rufptn1->reltime[i][0]+rufptn1->reltime[i][1]) / 2.0;
        dt [nfitpts] = 0.5 * sqrt(rufptn1->timeerr[i][0]*rufptn1->timeerr[i][0]+
            rufptn1->timeerr[i][1]*rufptn1->timeerr[i][1]);
        rho[nfitpts]= ((rufptn1->pulsa[i][0]+rufptn1->pulsa[i][1])/2.0) / 3.0;
        drho[nfitpts]= (sqrt(rufptn1->pulsaerr[i][0]*rufptn1->pulsaerr[i][0]
            + rufptn1->pulsaerr[i][1]*rufptn1->pulsaerr[i][1])/2.0) / 3.0;
        
        rufptnindex[nfitpts] = i;

        
        nfitpts++;
      }
    
     
     ngpts = nfitpts;
     memcpy(goodpts,rufptnindex,ngpts*sizeof(Int_t));
               
     /* These are initial values of the fit parameters */
     T0 = rusdgeom1->t0[2];    
     // tyro theta and phi, using upper and lower layers
     theta = rusdgeom1->theta[2];
     phi = rusdgeom1->phi[2];
     R[0] = rusdgeom1->xcore[2];
     R[1] = rusdgeom1->ycore[2];
     iexclude = -1;
     if(!recalcCog())
       return false;
     
    return true;
  }

//______________________________________________________________________________


int sdgeomfitter::cleanClust(Double_t deltaChi2, bool verbose)
  {
    Int_t xxyy;
    Int_t i, nfpars;
    Double_t chi2old, chi2new;
    Int_t nDeletedPts;
    Double_t dChi2;
    Int_t worst_i; // Keep track of the worst data point
    
    nfpars=NFPARS_LINSLEY;
    
    
    nDeletedPts = 0;
    Ifit(2,false);
    save_fpars();
    
    do
      {
        if(nfitpts < 1)
          {
            return nDeletedPts;
          }
        restore_fpars();
        // Fit with current data points and save the chi2 value
        iexclude = -1;
        if(!recalcCog())
          {
            return nDeletedPts;
          }
        Ifit(5,false);
        
        save_fpars();
        
        
        chi2old=chi2;
        // When chi2 is already smaller than deltaChi2
        if (chi2old < deltaChi2)
          {
            ndof = nfitpts-nfpars; // # of d.o.f
            ngpts = nfitpts;                  // To keep track of good SDs
            memcpy(goodpts,rufptnindex,ngpts*((Int_t)sizeof(Int_t)));
            return nDeletedPts;
          }
        
        dChi2=0.0; // initialize the chi2 difference.
        worst_i = 0; // just to initialize it
        // Find the largest chi2 difference.  This will be the worst point.
        for (i=0; i<nfitpts; i++)
          {
            // Fit w/o the i'th data pint and take the new chi2
            restore_fpars();
            iexclude = i;
            if(!recalcCog())
              {
                return nDeletedPts;
              }
            Ifit(5,false);
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
            xxyy=((Int_t)Floor(X[worst_i][0]+0.5))*100
                + ((Int_t)Floor(X[worst_i][1]+0.5));
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
    if(!recalcCog())
      {
        return nDeletedPts;
      }
    ndof = nfitpts-nfpars; // # of d.o.f
    ngpts = nfitpts;                  // To keep track of good SDs
    memcpy(goodpts,rufptnindex,ngpts*((Int_t)sizeof(Int_t)));
    return nDeletedPts;

  }


void sdgeomfitter::calcRsd(Int_t ipoint, Double_t *trsd, Double_t *terr,
    Double_t *chrgdens, Double_t *rcore, Double_t *score)
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
 
    rdist = sqrt( (X[ipoint][0]-par[2])*(X[ipoint][0]-par[2]) + (X[ipoint][1]
        -par[3]) *(X[ipoint][1]-par[3]));
    
    d[0]=(X[ipoint][0]-par[2]);
    d[1]=(X[ipoint][1]-par[3]);
    dotp = sin(par[0]*degrad)*(cos(par[1]*degrad)*d[0] + sin(par[1] *degrad)
        *d[1]);
    sdist = sqrt(d[0]*d[0]+d[1]*d[1]-dotp*dotp);
    
    
    // Calculating the time delay and time delay fluctuation
    ltdts(rho[ipoint], sdist*1.2e3, par[0], &ltd, &lts);
    ltd *= nsecTo1200m; // time delay
    lts *= nsecTo1200m; // time delay flucluation
    
    // Residual
    delta = t[ipoint] - tvsx_plane(X[ipoint][0], X[ipoint][1], X[ipoint][2],
        par) - ltd;

    // Return the quantities
    (*trsd) = delta; // Residual
    (*terr) = sqrt(lts*lts+dt[ipoint]*dt[ipoint]);   // Time delay fluctuation
    (*chrgdens) = rho[ipoint]; // Charge Density
    (*rcore) = rdist;          // Distance from core in ground plane
    (*score) = sdist;          // Distance from core in shower front plane
    
  }




void sdgeomfitter::fillRsd()
  {
    Int_t i, nfpars;
    TString parname(10);
    Double_t arglist[10];
    Int_t ierflg;
    
    Double_t trsd,terr,chrgdens,rcore,score;
    
    if (gMinuit)
      delete gMinuit;
    gMinuit=0;
    
    nfpars=NFPARS_LINSLEY;
    gMinuit = new TMinuit(nfpars);
    gMinuit->SetFCN(fcn_linsley);

    gMinuit->SetPrintLevel(-1);
    
    ierflg = 0;
    arglist[0] = 1;
    gMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
    static Double_t vstart[NFPARS], step[NFPARS], vlo[NFPARS], vup[NFPARS];
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
        default:
          break;
          }
        // No limits on fit parameters
        vlo[i]=0.0;
        vup[i]=0.0;
        // add fit parameter information to Minuit
        gMinuit->mnparm(i, parname, vstart[i], step[i], vlo[i], vup[i], ierflg);
      }
    
    gMinuit->SetMaxIterations(SDGEOM_MITR);
   
      {
        // Go over all the points.  For i'th point, do the fit without it and then
        // compute its residual
        for (i=0; i<nfitpts; i++)
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
            calcRsd(i,&trsd,&terr,&chrgdens, &rcore, &score);
            
            
            // Fill Residual vs S Scatter plot
            hRsdS -> Fill (score,trsd);
            
            // Fill Residual vs S profile histogram
            pRsdS -> Fill (score,trsd);
            
            
            // Residual vs rho 
            hRsdRho -> Fill(chrgdens,trsd);
            pRsdRho -> Fill(chrgdens,trsd);
            
            
            
          }

      }

  }


bool sdgeomfitter::Ifit(Int_t whatFCN, bool verbose)
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
     static Int_t n_trials = 0;  // number of trial fits to get a sensible event direction
     
     
    if (gMinuit)
      delete gMinuit;
    gMinuit=0;

    switch (whatFCN)
      {
    case 0:
      // for plane fitting
      if (verbose)
        fprintf(stdout,"Fitting into a plane\n");
      nfpars=NFPARS_PLANE;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_plane);
      break;

    case 1:
      if (verbose)
        fprintf(stdout,"Fitting into AGASA function\n");
      nfpars=NFPARS_AGASA;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_agasa);
      break;

    case 2:
      if (verbose)
        fprintf(stdout,"Fitting into modified Linsley's\n");
      nfpars=NFPARS_LINSLEY;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_linsley);
      break;

    case 3:
      if (verbose)
        fprintf(stdout,"Fitting into a spherical shape\n");
      nfpars=NFPARS_SPHERE;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_sphere);
      break;

    case 4:
      if (verbose)
        fprintf(stdout,"Fitting into a parabolic shape\n");
      nfpars=NFPARS_PARABOLA;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_parabola);
      break;
      
    case 5:
      if (verbose)
        fprintf(stdout,"Fitting into a modified Linsley's without i=%d point\n",iexclude);
      nfpars=NFPARS_LINSLEY;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_cleaner);
      break;
      
    case 6:
      if (verbose)
        fprintf(stdout,"Fitting into a Linsley shape with curvature and development\n");
      nfpars=NFPARS_LINSLEY1;
      gMinuit = new TMinuit(nfpars);
      gMinuit->SetFCN(fcn_linsley1);
      break;
      
    default:
      fprintf(stderr,"Option flag must be 0,1,2,3,4, or 6\n");
      return false;
      break;
      }
        
    if(!verbose) 
      gMinuit->SetPrintLevel(-1);
    
    ndof = nfitpts-nfpars; // # of d.o.f
    ierflg = 0;
    arglist[0] = 1;
    gMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
    for (i=0; i< nfpars; i++)
      {
        vlo=0.0; vup=0.0; // no limits on fit parameters, unless changed below
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
          // for spherical shape
          if (whatFCN==3)
            {
              parname="Curv. Rad.";
              vstart= 100.0;
              step=1.0;
            }
          // for parabolic shape
          else if (whatFCN==4)
            {
              parname="Curvature";
              vstart= 0.01;
              step=1e-3;
            }
          // for parabolic shape
          else if (whatFCN==6)
            {
              parname="Curvature";
              vstart= 2.6;
              step=1e-3;
            }
          break;

        default:
          break;
          
          }
        
        // add fit parameter information to Minuit
        gMinuit->mnparm(i, parname, vstart, step, vlo, vup, ierflg);
      }
    
    gMinuit->SetMaxIterations(SDGEOM_MITR);
    // First fit with the fixed core, then release the core and re-fit.
    gMinuit->FixParameter(2);
    gMinuit->FixParameter(3);
    gMinuit->Migrad();
    gMinuit->Release(2);
    gMinuit->Release(3);
    gMinuit->Migrad();
    
    if (verbose)
      {
        // Print results
        Double_t amin, edm, errdef;
        Int_t nvpar, nparx, icstat;
        gMinuit->mnstat(amin, edm, errdef, nvpar, nparx, icstat);
        gMinuit->mnprin(3, amin);
      }

    // Obtain the fit parameters
   
    gMinuit->GetParameter(0, theta, dtheta);
    gMinuit->GetParameter(1, phi, dphi);
    phi = tacoortrans::range(phi,360.0);
    gMinuit->GetParameter(2, R[0], dR[0]);
    gMinuit->GetParameter(3, R[1], dR[1]);
    gMinuit->GetParameter(4, T0, dT0);
      
    

    // additional curvature parameter
    if (nfpars==6)
      gMinuit->GetParameter(5, a, da);
    
    chi2 = gMinuit->fAmin;
    
    
    
    // Must have a shower going downwards.  If not so from the 1st trial, flip its axis and re-fit.
    stheta = sin(theta * DegToRad());
    theta = RadToDeg() * asin(stheta);
    if (stheta < 0.0)
      {
        n_trials++;
        // Caution, to avoid blowing the heap.
        if(n_trials > 10)
          {
            theta *= -1.0;
	    phi = tacoortrans::range(phi+180.0,360.0);
            if (theta >= 30.0)
              {
                linsleyA = exp( -3.2e-2 * theta + 2.0);
              }
            else
              {
                linsleyA = 3.3836 - 0.01848*theta;
              }
            if (verbose)
              {
                fprintf(stdout,"Linsley a: %f\n", linsleyA);
                fprintf(stdout, "Chi2 = %f\n",chi2);
                fprintf(stdout,"ndof = %d\n",ndof);
              }
            n_trials = 0;
            return true;
          }
        return Ifit(whatFCN,verbose);
      }
    else
      {
        n_trials = 0;
      }
    
    
//    fprintf(stdout, "\n\n Core Starting Values (SQRT of CHARGE): %f %f\n",
//        cogXY[0],cogXY[1]);
    
    if (theta >= 30.0)
      {
        linsleyA = exp( -3.2e-2 * theta + 2.0);
      }
    else
      {
        linsleyA = 3.3836 - 0.01848*theta;
      }
    if (verbose)
      {
        fprintf(stdout,"Linsley a: %f\n", linsleyA);
        fprintf(stdout, "Chi2 = %f\n",chi2);
        fprintf(stdout,"ndof = %d\n",ndof);
      }
    return true;

  }



void sdgeomfitter::save_fpars()
  {
    theta_old = theta;
    phi_old = phi;
    memcpy(R_old,R,(int)(2*sizeof(real8)));          
    T0_old = T0;
  }

void sdgeomfitter::restore_fpars()
  {
    theta = theta_old;
    phi = phi_old;
    memcpy(R,R_old,(int)(2*sizeof(real8)));           
    T0 = T0_old;
  }
