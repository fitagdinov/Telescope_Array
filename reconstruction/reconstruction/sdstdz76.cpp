#include "sduti.h"
#include <cmath>

using namespace std;

// Based on U.S. Standard Atmosphere, 1976
// Ported to C/C++ from mc04(stdz76.f) and successfuly tested.
// Dmitri Ivanov, <ivanov@physics.rutgers.edu>
// Last modified: Jun 1, 2010
// INPUTS:
// h:      altitude [meter]
// OUTPUTS:
// pres:   pressure [millibar]
// xdepth: vertical depth [g/cm^2]
// tcent:  temperature in [degree Celsius]
// rho:    density [g/cm^3]
// RETURNS: 
// 1:      h < 84852m, calculated variables are non-trivial
// 0:      h > 84852m, calculated variables are at absolute zero
int SDGEN::stdz76(double h, double *pres, 
		  double *xdepth, double *tcent, double *rho)
{
  static double hb76[10]   = 
    {0.,1.1E4,2.E4,3.2E4,4.7E4,5.1E4,7.1E4,84852.0,84852.0,84852.0};
  static double ht76[10]   = 
    {1.1E4,2.E4,3.2E4,4.7E4,5.1E4,7.1E4,84852.0,84852.0,84852.0,84852.0};
  static double tb76[10]   = 
    {15.,-56.5,-56.5,-44.5,-2.5,-2.5,-58.5,-86.2,-86.2,-86.2};
  static double grad76[10] = 
    {-.0065,0.,.001,.0028,0.,-.0028,-.002,0.,0.,0.};
  static double ab76[10],pb76[11];
  static double absz       =  273.15;
  static double pzero      =  1013.250;
  static double g          =  980.665;
  static double r          =  83143200./28.9644;
  static bool check        =  false;  
  int i;
  double t;  
  if(!check)
    {
      pb76[0] = pzero;      
      for (i=0; i<8; i++)
	ab76[i] = tb76[i] + absz;
      for (i=0; i<8; i++)
	{
	  if (fabs(grad76[i])>1e-6)
	    pb76[i+1] = 
	      pb76[i]*pow((ab76[i+1]/ab76[i]),(100.0*g/(-grad76[i]*r)));
	  else
	    pb76[i+1] = 
	      pb76[i]*pow(2.7182818,((-100.*g/(r*ab76[i]))*(ht76[i]-hb76[i])));
	}
      check = true;
    }
  for (i=0; i<8; i++)
    {
      if ( h <= hb76[i+1] )
	{
	  if (fabs(grad76[i]) < 1e-6)
	    (*pres) = pb76[i]*exp((h-hb76[i])*(-g)/(0.01*r*ab76[i]));
	  else
	    (*pres) = 
	      pb76[i]*pow(((h-hb76[i])*grad76[i]/ab76[i]+1.0),
			  (-100.0*g/(grad76[i]*r)));
	  (*xdepth) = 1.e3 * (*pres) / g;
	  t = ab76[i] + grad76[i]*(h-hb76[i]);
	  (*tcent) = t - absz;
	  (*rho)   = 1.e3 * (*pres)/(r*t);
	  return 1;
	}
    }
  (*pres)   = 0.0;
  (*xdepth) = 0.0;
  (*tcent)  = absz;
  (*rho)    = 0.0;
  return 0;
}


// Get the vertical depth for a given altitude 
// (U.S. Standard Atmosphere, 1976)
// INPUTS:
// h:      altitude  [meter]
// OUTPUTS (RETURNS):
// xdepth: vertical depth  [g/cm^2]
double SDGEN::stdz76_xdepth(double h)  
{
  double pres,xdepth,tcent,rho;
  stdz76(h,&pres,&xdepth,&tcent,&rho);
  return xdepth;
}



// iverse of stdz76
// INPUTS: 
// x: vertical depth [g/cm^2]
// OUTPUTS (RETURN): 
// altitude [m] in 0-84852m range
double SDGEN::stdz76_inv(double x) 
{
  const int fNp = 102, fKstep = 0;
  const double fDelta = -1, fXmin = 0.0063564, fXmax = 1033.23;

  // at the top of the atmosphere
  if (x < fXmin)
    return 84852.0;
  
  // at the bottom of the atmosphere
  if ( x>=fXmax)
    return 0.0;
  
  const double fX[102] = 
    { 0.0063564, 0.0161676, 0.0370089, 0.0771021, 0.147382,
      0.262068, 0.438224, 0.695158, 1.06175, 1.58428,
      2.33033, 3.37975, 4.83364, 6.81784, 9.48457,
      12.9699, 17.4189, 22.9965, 29.87, 38.2039,
      48.1535, 59.8555, 73.363, 88.7196, 105.954,
      125.062, 146.01, 168.735, 193.144, 219.122,
      246.431, 274.43, 302.856, 331.517, 360.232,
      388.841, 417.198, 445.174, 472.659, 499.555,
      525.783, 551.278, 575.986, 599.866, 622.888,
      645.033, 666.288, 686.65, 706.121, 724.708,
      742.425, 759.288, 775.318, 790.536, 804.968,
      818.639, 831.578, 843.812, 855.371, 866.282,
      876.575, 886.277, 895.418, 904.025, 912.124,
      919.742, 926.904, 933.634, 939.955, 945.891,
      951.463, 956.69, 961.594, 966.193, 970.504,
      974.545, 978.332, 981.879, 985.202, 988.314,
      991.228, 993.956, 996.509, 998.899, 1001.13,
      1003.23, 1005.18, 1007.02, 1008.73, 1010.33,
      1011.83, 1013.23, 1014.54, 1015.76, 1016.91,
      1017.98, 1018.98, 1019.91, 1020.79, 1027.12,
      1032, 1033.23 
    };
  const double fY[102] = 
    { 82010, 76590, 71540, 66820, 62410,
      58290, 54450, 50860, 47500, 44370,
      41440, 38700, 36150, 33770, 31540,
      29460, 27510, 25700, 24000, 22420,
      20940, 19560, 18270, 17060, 15940,
      14890, 13900, 12990, 12130, 11330,
      10580, 9883, 9231, 8622, 8053,
      7522, 7026, 6562, 6129, 5725,
      5347, 4994, 4665, 4357, 4069,
      3801, 3550, 3316, 3097, 2893,
      2702, 2524, 2357, 2202, 2056,
      1921, 1794, 1676, 1565, 1462,
      1365, 1275, 1191, 1113, 1039,
      970.6, 906.5, 846.7, 790.9, 738.7,
      689.9, 644.4, 601.9, 562.2, 525.1,
      490.4, 458.1, 427.9, 399.6, 373.3,
      348.6, 325.6, 304.2, 284.1, 265.3,
      247.8, 231.5, 216.2, 201.9, 188.6,
      176.2, 164.6, 153.7, 143.6, 134.1,
      125.2, 117, 109.3, 102, 50,
      10, 0 
    };
  const double fB[102] = 
    { -706064, -416199, -146944, -91678, -44993.6,
      -28526.4, -17139.2, -11419.3, -7425.38, -4867.65,
      -3202.63, -2143.8, -1448.89, -1001.11, -704.452,
      -510.42, -377.121, -282.279, -216.681, -167.333,
      -132.375, -105.697, -86.6758, -71.5087, -59.4463,
      -51.0127, -43.4603, -37.384, -32.9756, -28.9501,
      -26.0778, -23.8428, -22.0564, -20.4942, -19.1596,
      -17.9932, -17.0186, -16.1571, -15.3648, -14.7004,
      -14.1251, -13.5651, -13.09, -12.7084, -12.2926,
      -11.9476, -11.6454, -11.366, -11.1049, -10.8754,
      -10.6581, -10.4935, -10.2875, -10.157, -9.99187,
      -9.83847, -9.72722, -9.62591, -9.51599, -9.43045,
      -9.35711, -9.23477, -9.10106, -9.1086, -9.07023,
      -8.94788, -8.92627, -8.85085, -8.80851, -8.77875,
      -8.73119, -8.68308, -8.64901, -8.61587, -8.60112,
      -8.55831, -8.51139, -8.52595, -8.47801, -8.4604,
      -8.46572, -8.39659, -8.38586, -8.43834, -8.3732,
      -8.34342, -8.33435, -8.33812, -8.35115, -8.27946,
      -8.26678, -8.31141, -8.30864, -8.24967, -8.30976,
      -8.25319, -8.22177, -8.30521, -8.2821, -8.20229,
      -8.15309, -8.10466 
    };
  const double fC[102] = 
    { 1.74328e+07, 1.21115e+07, 807819, 570608, 93655.8,
      49929.4, 14713.1, 7549.09, 3345.57, 1549.34,
      682.44, 326.532, 151.431, 74.2426, 37.0019,
      18.6692, 11.2924, 5.71173, 3.83183, 2.08964,
      1.42381, 0.855984, 0.552203, 0.435458, 0.264444,
      0.176921, 0.183611, 0.0837741, 0.0968293, 0.0581311,
      0.0470448, 0.0327788, 0.0300674, 0.0244372, 0.0220407,
      0.0187308, 0.0156351, 0.0151626, 0.0136627, 0.0110407,
      0.0108929, 0.011072, 0.00815811, 0.00782115, 0.010237,
      0.00534415, 0.00887482, 0.00484712, 0.00855816, 0.00379076,
      0.00847266, 0.00128835, 0.011568, -0.00299813, 0.0144432,
      -0.00322254, 0.0118203, -0.00353912, 0.0130492, -0.00521001,
      0.0123355, 0.000273925, 0.0143542, -0.0152311, 0.0199699,
      -0.00391025, 0.00692802, 0.00427908, 0.00241912, 0.00259424,
      0.00594121, 0.00326169, 0.00368648, 0.00352025, -9.89568e-05,
      0.0106911, 0.00169908, -0.0058035, 0.0202296, -0.0145688,
      0.0127439, 0.0125953, -0.00839315, -0.0135651, 0.0427658,
      -0.0285857, 0.0332373, -0.0352857, 0.0276636, 0.0171404,
      -0.00868811, -0.0231874, 0.0253053, 0.0230303, -0.0752865,
      0.128154, -0.0967297, 0.00700338, 0.0192641, -0.00665687,
      0.0167399, 1.23 
    };
  const double fD[102] = 
    { -1.8079e+08, -1.8079e+08, -1.97216e+06, -2.26216e+06, -127090,
      -66638.4, -9294.27, -3822.16, -1145.86, -387.327,
      -113.049, -40.1455, -12.9671, -4.65498, -1.75332,
      -0.552697, -0.333515, -0.0911663, -0.069683, -0.0223066,
      -0.0161747, -0.00749659, -0.00253409, -0.00330762, -0.00152681,
      0.000106462, -0.00146443, 0.000178284, -0.000496551, -0.00013532,
      -0.000169839, -3.17946e-05, -6.54812e-05, -2.78192e-05, -3.85649e-05,
      -3.63894e-05, -5.63002e-06, -1.81901e-05, -3.24953e-05, -1.87947e-06,
      2.3424e-06, -3.93112e-05, -4.70348e-06, 3.49792e-05, -7.3649e-05,
      5.537e-05, -6.59348e-05, 6.3531e-05, -8.5497e-05, 8.80867e-05,
      -0.000142013, 0.000213758, -0.000319054, 0.00040284, -0.000430735,
      0.000387531, -0.000418489, 0.000478367, -0.000557823, 0.000568202,
      -0.000414401, 0.000513447, -0.00114578, 0.00144878, -0.0010449,
      0.000504434, -0.000131201, -9.80836e-05, 9.83364e-06, 0.000200226,
      -0.000170877, 2.88736e-05, -1.20482e-05, -0.000279843, 0.000890046,
      -0.000791479, -0.000705063, 0.00261141, -0.00372734, 0.00312431,
      -1.81574e-05, -0.00274036, -0.000721333, 0.00841639, -0.0113256,
      0.010568, -0.0124136, 0.0122708, -0.00219234, -0.00573966,
      -0.00345222, 0.0123391, -0.000621584, -0.0284976, 0.0633771,
      -0.0749612, 0.0371803, 0.00464423, -0.00136498, 0.00159814,
      0.00159814, 2.3675 
    };
  int klow=0;
  if(x<=fXmin) 
    klow=0;
  else if(x>=fXmax) 
    klow=fNp-1;
  else 
    {
      if(fKstep) 
	{
	  // Equidistant knots, use histogramming
	  klow = int((x-fXmin)/fDelta);
	  if (klow < fNp-1) klow = fNp-1;
	} 
      else 
	{
	  int khig=fNp-1, khalf;
	  // Non equidistant knots, binary search
	  while(khig-klow>1)
	    if(x>fX[khalf=(klow+khig)/2]) klow=khalf;
	    else khig=khalf;
	}
    }
  // Evaluate now
  double dx=x-fX[klow];
  return (fY[klow]+dx*(fB[klow]+dx*(fC[klow]+dx*fD[klow])));
}