#ifndef _rusdenergy_h_
#define _rusdenergy_h_


////////////// CURRENT ENERGY ESTIMATION ROUTINE /////////////////

// This routine uses PROTON energy estimation table (use for making the spectrum)
// this energy scale needs to be lowered by 1.27 when calibrated
// against the FD energy scale
// s800 in VEM/m^2, theta in degrees, answer in EeV.
double rusdenergy(double s800, double theta);

// This routine uses IRON energy estimation table (good for CORSIKA studies only)
// this energy scale needs to be lowered by 1.17 when calibrated
// against the FD energy scale
// s800 in VEM/m^2, theta in degrees, answer in EeV
double rusdenergy_iron(double s800, double theta);

#endif
