#include "sditerator.h"

#define raddeg 57.2957795130823229
#define ln10 2.30258509299404590
#define secIn1200m 4.0027691424e-6 // Seconds in a [1200m] unit

#define DS1_TO_DS2_DATE 81111

//     FOR ANALYZING TA SD DATA IN C++
//************************************************************************
///////////////////////////////////////////////////////////////////////


static double pdErr(double theta, double dtheta, double dphi)
{
  return
    sqrt(dtheta*dtheta+sin(theta/raddeg)*sin(theta/raddeg)*dphi*dphi);
}

/////////// EXECUTED ON EACH EVENT //////////////////////
void cppanalysis(FILE *outFl)
{
  int passed_cuts;

  double gfchi2pdof;
  double ldfchi2pdof;
 
  passed_cuts = 1;


  // number of good SDs cut
  if (rufptn_.nstclust < 4)
    passed_cuts = 0;

  // Distance from the surrounding edgre should be less than 1200m
  // ( 1 counter separation unit ) 
  if ( rufldf_.bdist < 1.0 )
    passed_cuts = 0;

  // T-Shape boundary CUT
  if ( (rusdraw_.yymmdd < DS1_TO_DS2_DATE) && (rufldf_.tdist < 1.0) )
    passed_cuts = 0;
  
  // Geometry fit chi2 / dof cut
  if (rusdgeom_.ndof[1] > 0)
    {
      gfchi2pdof = rusdgeom_.chi2[1] / (double) rusdgeom_.ndof[1];
    }
  else
    {
      gfchi2pdof = rusdgeom_.chi2[1];
    }
  if (gfchi2pdof > 4.0)
    passed_cuts = 0;
  
  // LDF Fit chi2 / dof
  if (rufldf_.ndof[1] > 0)
    {
      ldfchi2pdof = rufldf_.chi2[0] / (double) rufldf_.ndof[0];
    }
  else
    {
      ldfchi2pdof = rufldf_.chi2[0];
    }
  if (ldfchi2pdof > 4.0)
    passed_cuts = 0;
  

  // Pointing direction error cut ( should be less than 5 degrees ) 
  if (pdErr(rusdgeom_.theta[1], rusdgeom_.dtheta[1], rusdgeom_.dphi[1]) > 5.0)
    passed_cuts = 0;
  
  // Cut on resolution of LDF scaling constant.  SAME as 
  // as cut on resolution of S800. 
  if (rufldf_.dsc[0] / rufldf_.sc[0] > 0.25)
    passed_cuts = 0;
  
  // Zenith angle cut:
  if (rusdgeom_.theta[1] > 45.0 )
    passed_cuts = 0;
  
  
  // Energy cut ? 


  if ( passed_cuts ) 
    {
      fprintf (outFl, "PASSED CUTS: ");
    }
  else
    {
      fprintf (outFl, "DID NOT PASS CUTS: ");
    }
  
  fprintf (
	   outFl, "%06d %06d.%06d %.2f %.2f %.2f\n",
	   rusdraw_.yymmdd,rusdraw_.hhmmss,rusdraw_.usec,
	   rusdgeom_.theta[1],rusdgeom_.phi[1],rufldf_.energy[0]
	   );
      
 
}
