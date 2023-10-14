

#ifndef _ldffun_h_
#define _ldffun_h_


#include "TMath.h"
using namespace TMath;


// LDF function
// r     :  Perpendicular distance from shower axis, m
// theta :  Zenith angle, degree
static Double_t ldffun(Double_t r, Double_t theta)
  {
    Double_t r0;    // Moliere radius
    Double_t alpha; // Constant slope parameter
    Double_t beta;  // Another constant slope parameter
    Double_t eta;   // Zenith angle dependent slope parameter
    Double_t rsc;   // Scaling factor for r in quadratic term in power
      
    r0    =  91.6;
    alpha =  1.2;
    beta  =  0.6;
    eta   =  3.97-1.79*(1.0/Cos(DegToRad()*theta)-1.0);
    rsc   = 1000.0;
    
    return  Power(r/r0, -alpha) * 
            Power((1.0+r/r0), -(eta-alpha)) * 
            Power((1.0+ r*r/rsc/rsc), -beta);
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
    
    sf = 1.0/Cos( DegToRad()*theta ) - 1;
    return Exp(-X0/L1 * sf - X0/L2 * sf*sf);
  }



#endif
