#include "TMath.h"

Double_t h2mo(Double_t h_cm,
	      Double_t h0,Double_t h1,Double_t h2,Double_t h3,Double_t h4,
	      Double_t a0,Double_t a1,Double_t a2,Double_t a3,Double_t a4,
	      Double_t b0,Double_t b1,Double_t b2,Double_t b3,Double_t b4,
	      Double_t c0,Double_t c1,Double_t c2,Double_t c3,Double_t c4)
{
  const Int_t nh = 5;
  Double_t h[nh] = { h0, h1, h2, h3, h4 };
  Double_t a[nh] = { a0, a1, a2, a3, a4 };
  Double_t b[nh] = { b0, b1, b2, b3, b4 };
  Double_t c[nh] = { c0, c1, c2, c3, c4 };
  Int_t i = 0;
  while (i < nh-1 && h_cm > h[i+1]) 
    i++;
  if (i < nh-1)
    return a[i]+b[i]*TMath::Exp(-h_cm/c[i]);
  return a[i]-b[i]*h_cm/c[i];
}

Double_t rho(Double_t h_cm,
	     Double_t h0,Double_t h1,Double_t h2,Double_t h3,Double_t h4,
	     Double_t a0,Double_t a1,Double_t a2,Double_t a3,Double_t a4,
	     Double_t b0,Double_t b1,Double_t b2,Double_t b3,Double_t b4,
	     Double_t c0,Double_t c1,Double_t c2,Double_t c3,Double_t c4)
{
  const Int_t nh = 5;
  Double_t h[nh] = { h0, h1, h2, h3, h4 };
  Double_t a[nh] = { a0, a1, a2, a3, a4 };
  Double_t b[nh] = { b0, b1, b2, b3, b4 };
  Double_t c[nh] = { c0, c1, c2, c3, c4 };
  Int_t i = 0;
  while (i < nh-1 && h_cm > h[i+1]) 
    i++;
  if (i < nh-1)
    return b[i]/c[i]*TMath::Exp(-h_cm/c[i]);
  return b[i]/c[i];
}

Double_t mo2h(Double_t mo_g_cm2,
	      Double_t h0,Double_t h1,Double_t h2,Double_t h3,Double_t h4,
	      Double_t a0,Double_t a1,Double_t a2,Double_t a3,Double_t a4,
	      Double_t b0,Double_t b1,Double_t b2,Double_t b3,Double_t b4,
	      Double_t c0,Double_t c1,Double_t c2,Double_t c3,Double_t c4)
{
  const Int_t nh = 5;
  Double_t h[nh] = { h0, h1, h2, h3, h4 };
  Double_t a[nh] = { a0, a1, a2, a3, a4 };
  Double_t b[nh] = { b0, b1, b2, b3, b4 };
  Double_t c[nh] = { c0, c1, c2, c3, c4 };
  Int_t i = 0;
  Double_t h_cm = 0;
  while (i < nh-1 && 
	 mo_g_cm2 < h2mo
	 (h[i+1],
	  h[0],h[1],h[2],h[3],h[4],
	  a[0],a[1],a[2],a[3],a[4],
	  b[0],b[1],b[2],b[3],b[4],
	  c[0],c[1],c[2],c[3],c[4])) i++;
  if (i < nh-1)
    h_cm = -c[i]*TMath::Log((mo_g_cm2-a[i])/b[i]);
  else
    h_cm = (a[i]-mo_g_cm2)*c[i]/b[i];
  return h_cm;
}


// Saturated water vapor pressure as function of temperature
// input: T in Kelvin
// output: pressure in Pa 
Double_t H20_Saturated_Vapor_Pressure(Double_t T)
{
  const Double_t C1  = -5.6745359e3;
  const Double_t C2  =  6.3925247e0;
  const Double_t C3  = -9.6778430e-3;
  const Double_t C4  =  6.2215701e-7;
  const Double_t C5  =  2.0747825e-9;
  const Double_t C6  = -9.4840240e-13;
  const Double_t C7  =  4.1635019e0;
  const Double_t C8  = -5.8002206e3;
  const Double_t C9  =  1.3914993e0;
  const Double_t C10 = -4.8640239e-2;
  const Double_t C11 =  4.1764768e-5;
  const Double_t C12 = -1.4452093e-8;
  const Double_t C13 =  6.5459673e0; 
  Double_t logPw = 0.0;  
  // over ice
  if(T <= 273.15)
    logPw = C1/T+C2+C3*T+C4*T*T+C5*T*T*T+C6*T*T*T*T+C7*log(T);
  // over liquid water
  else
    logPw = C8/T+C9+C10*T+C11*T*T+C12*T*T*T+C13*log(T);
  return TMath::Exp(logPw);
}

// Density of air / water mixture
// P_Pa:    Pressure of air/water vapor mixture [ K ]
// T_K:     Temperature of air/water vapor mixture [ K ]
// T_Dew_K: Dew point temperature [ K ]
// Return: air density in g/cm3 units
Double_t Air_Density_g_cm3(Double_t P_Pa, Double_t T_K, Double_t T_Dew_K)
{
  if(TMath::IsNaN(T_Dew_K))
    T_Dew_K = 273.15 - 120.0; // -120 C seems to be the lower limit on Dew Temperature
  const Double_t Rd = 287.05;  // specific gas constant for dry air [J/(kg*degK)]
  const Double_t Rv = 461.495; // specific gas constant for water vapor [J/(kg*degK)]
  Double_t P_H20   = H20_Saturated_Vapor_Pressure(T_Dew_K);
  Double_t P_D_Air = P_Pa - P_H20;
  return 1e-3 * (P_D_Air/(Rd*T_K)+P_H20/(Rv*T_K));
}



Int_t atmfun()
{
  return 0;
}
