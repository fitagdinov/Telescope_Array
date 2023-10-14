#include "sduti.h"
#include <cmath>

using namespace std;

// Saturated water vapor pressure as function of temperature
// input: T in Kelvin
// output: pressure in Pa 
double SDGEN::Get_H20_Saturated_Vapor_Pressure(double T)
{
  const double C1  = -5.6745359e3;
  const double C2  =  6.3925247e0;
  const double C3  = -9.6778430e-3;
  const double C4  =  6.2215701e-7;
  const double C5  =  2.0747825e-9;
  const double C6  = -9.4840240e-13;
  const double C7  =  4.1635019e0;
  const double C8  = -5.8002206e3;
  const double C9  =  1.3914993e0;
  const double C10 = -4.8640239e-2;
  const double C11 =  4.1764768e-5;
  const double C12 = -1.4452093e-8;
  const double C13 =  6.5459673e0; 
  double logPw = 0.0;  
  // over ice
  if(T <= 273.15)
    logPw = C1/T+C2+C3*T+C4*T*T+C5*T*T*T+C6*T*T*T*T+C7*log(T);
  // over liquid water
  else
    logPw = C8/T+C9+C10*T+C11*T*T+C12*T*T*T+C13*log(T);
  return exp(logPw);
}

// Density of air / water mixture
// P_Pa:    Pressure of air/water vapor mixture [ K ]
// T_K:     Temperature of air/water vapor mixture [ K ]
// T_Dew_K: Dew point temperature [ K ]
// Return: air density in g/cm3 units
double SDGEN::Get_Air_Density_g_cm3(double P_Pa, double T_K, double T_Dew_K)
{
  const double Rd = 287.05;  // specific gas constant for dry air [J/(kg*degK)]
  const double Rv = 461.495; // specific gas constant for water vapor [J/(kg*degK)]
  double P_H20   = Get_H20_Saturated_Vapor_Pressure(T_Dew_K);
  double P_D_Air = P_Pa - P_H20;
  return 1e-3 * (P_D_Air/(Rd*T_K)+P_H20/(Rv*T_K));
}
