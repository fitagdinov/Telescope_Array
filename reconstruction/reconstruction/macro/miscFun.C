#include "TMath.h"
#include "TRandom.h"
#include "TF1.h"
using namespace TMath;


Double_t phires(Double_t theta, Double_t phi1, Double_t phi2)
{
  Double_t dphi=phi2-phi1;
  while (dphi > 180.)
    dphi -= 360;
  while (dphi < -180.)
    dphi += 360.;
  return Sin ( DegToRad() * theta ) * dphi;
}

Double_t sec(Double_t theta)
{
  return 1. / Cos (DegToRad() * theta); 
}

Double_t delphi(Double_t theta, Double_t dphi)
{
  Double_t idphi=dphi;
  while (idphi > 180.)
    idphi -= 360;
  while (idphi < -180.)
    idphi += 360.;
  return Sin ( DegToRad() * theta ) * idphi;
}

Double_t pdErr(Double_t theta, Double_t dtheta, Double_t dphi)
{
  Double_t phierr;
  phierr=delphi(theta,dphi);
  return sqrt (dtheta*dtheta+phierr*phierr); 
}

Double_t pdRes(Double_t theta1, Double_t theta2, Double_t phi1, Double_t phi2)
{
  Double_t phr = phires(theta1,phi1,phi2);
  Double_t thr = theta2-theta1;
  return sqrt(thr*thr+phr*phr);
}

Double_t corErr(Double_t dxcore, Double_t dycore)
{
  return sqrt (dxcore*dxcore+dycore*dycore);
}

Double_t corRes(Double_t xcore1, Double_t ycore1, 
		Double_t xcore2, Double_t ycore2 )
{
  return sqrt(
	      (xcore2-xcore1)*(xcore2-xcore1)+
	      (ycore2-ycore1)*(ycore2-ycore1)
	      );
}

Double_t angRes(Double_t theta1, Double_t theta2, Double_t phi1, Double_t phi2)
{  
  return
    RadToDeg()*
    ACos( 
	 Cos(DegToRad()*theta1)*Cos(DegToRad()*theta2)
	 +
	 Sin(DegToRad()*theta1)*Sin(DegToRad()*theta2)*Cos(DegToRad()*(phi2-phi1))
	  );
}

Int_t irnd(Int_t imax)
{
  TRandom rndm;
  return (int)Floor(((Double_t)imax) * rndm.Rndm());
}


// Fit function for zenith angle distribution
// par[0] - scaling factor in front of sin(2*theta)
Double_t thetaFit(Double_t *x, Double_t *par)
{
  return par[0]*Sin(2.0*DegToRad()*x[0]);
}
TF1 *fTheta = new TF1("fTheta",thetaFit,0.0,90.0,1);

Double_t en_cic(Double_t s800, Double_t theta)
{
  Double_t ct=cos(DegToRad()*theta);
  Double_t x=ct*ct-0.77;
  return 0.48 * s800/(1.0+0.7*x-2.3*x*x);
}

// returns month number for a given year,month,day
// (in yymmdd format) since some start year and month
// in yymm format.
Int_t month_num(Int_t yymm_start, Int_t yymmdd)
{
  Int_t sm = 12*(yymm_start/100)+(yymm_start%100);
  return (12*(yymmdd/10000)+(yymmdd%10000)/100) -sm;
}

Double_t xxyy2x(Int_t xxyy)
{
  return (double)(xxyy/100);
}

Double_t xxyy2y(Int_t xxyy)
{
  return (double)(xxyy%100);
}

// Elevation vs Azimuth function for shower-detector plane normal
// Elevation (return) and Azimuth ( x[0] ) are both in degrees
// par[0],par[1],par[3] is the shower-detector plane normal unit vector
Double_t ele_vs_azi_sdp(double *x, double *par)
{
  double fval,a,fsign;
  static const double d2r = DegToRad();
  a = par[2]/(par[0]*Cos(d2r*x[0])+par[1]*Sin(d2r*x[0]));
  fval = (1.0/d2r)*ASin(1.0/Sqrt(1+a*a));
  fsign = -par[2]/(par[0]*Cos(d2r*x[0])+par[1]*Sin(d2r*x[0]));  
  return ( fsign > 0 ? fval : -fval);
}
