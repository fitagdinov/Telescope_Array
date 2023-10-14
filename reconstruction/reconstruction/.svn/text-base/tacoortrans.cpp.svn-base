//  Many transformation and time conversion routines are adopted  
//  from Lauren Scott's and Sean Stratton's development toolboxes,
//  some transformation routines where taken from Ben Stokes's analysys,
//  and some additional routines were added by Dmitri Ivanov. 

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "tacoortrans.h"

using namespace std;

const double tacoortrans::fd_origin_clf[tacoortrans_nfd][3] = 
  {
    // Position of BR FD site relative to CLF (meters)
    { 17028.099,  -12044.217, -12.095 },
    
    // Position of LR FD site relative to CLF (meters)
    { -18377.572, -9862.693,  137.922 },
    
    // Position of MD FD site relative to CLF (meters)
    { -7308.07,   19536.12,   183.828 }
  };

const double tacoortrans::fd2clf_mat[tacoortrans_nfd][3][3] = 
  {
    // BR FD site to CLF rotation matrix
    {
      { 0.999994,  -0.002173, 0.002666  },
      { 0.002178,  0.999996,  -0.001893 },
      { -0.002661, 0.001899,  0.999995  }
    },
    // LR FD site to CLF rotation matrix
    {
      { 0.999993,  0.002347,  -0.002877 },
      { -0.002351, 0.999996,  -0.001550 },
      { 0.002873,  0.001557,  0.999995  }
    },
    // MD FD site to CLF rotation matrix
    {
      { 0.999999,  0.000942, -0.001144  },
      { -0.000939, 0.999995,  0.003070  },
      { 0.001147, -0.003069,  0.999995  }
    }
  };


bool tacoortrans::chk_fdsiteid(int fdsiteid)
{
  if(fdsiteid < 0 || fdsiteid >= tacoortrans_nfd)
    {
      fprintf (stderr, "error: tacoortrans: fdsiteid = %d is not supported\n",fdsiteid);
      return false;
    }
  return true;
}



// Only rotates from FD frame to CLF frame
// fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
// xclf - 3-vector in CLF frame (INPUT)
// xfd  - 3-vector in FD frame (OUTPUT)
bool tacoortrans::rot_fdsite2clf(int fdsiteid, double* xfd, double* xclf)
{
  int i, j;
  double x[3];
  if(!chk_fdsiteid(fdsiteid))
    {
      for (i=0; i<3; i++)
	xclf[i] = 0.0;
      return false;
    }
  for (i = 0; i < 3; i++)
    x[i] = xfd[i]; // (can use same vector variable as input and output)
  for (i = 0; i < 3; i++)
    {
      xclf[i] = 0.0;
      for (j = 0; j < 3; j++)
	xclf[i] += fd2clf_mat[fdsiteid][i][j] * x[j];
    }
  return true;
}


// Only rotates from CLF to FD frame
// fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
// xclf - 3-vector in CLF frame (INPUT)
// xfd  - 3-vector in FD frame (OUTPUT)
bool tacoortrans::rot_clf2fdsite(int fdsiteid, double* xclf, double* xfd)
{
  int i, j;
  double x[3];
  if(!chk_fdsiteid(fdsiteid))
    {
      for (i=0; i<3; i++)
	xfd[i] = 0.0;
      return false;
    }
  for (i = 0; i < 3; i++)
    x[i] = xclf[i]; // (can use same vector variable as input and output)
  for (i = 0; i < 3; i++)
    {
      xfd[i] = 0.0;
      for (j = 0; j < 3; j++)
	xfd[i] += fd2clf_mat[fdsiteid][j][i] * x[j]; // m transpose for rotating from CLF to FD
    }
  return true;
}


// fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
// xfd[3]  - vector in FD frame, [meters] (INPUT)
// xclf[3] - vector in CLF frame, [meters] (OUTPUT)
bool tacoortrans::fdsite2clf(int fdsiteid, double* xfd, double* xclf)
{
  int i, j;
  double x[3];
  if(!chk_fdsiteid(fdsiteid))
    {
      for (i=0; i<3; i++)
	xclf[i] = 0.0;
      return false;
    }
  for (i = 0; i < 3; i++)
    x[i] = xfd[i]; // (can use same vector variable as input and output)
  for (i=0; i<3; i++)
    {
      xclf[i] = fd_origin_clf[fdsiteid][i];
      for (j=0; j<3; j++)
	xclf[i] +=  fd2clf_mat[fdsiteid][i][j] * x[j];
    }
  return true;
}


// fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
// xclf[3] - vector in CLF frame, [meters], (INPUT)
// xfd[3]  - vector in FD frame, [meters], (OUTPUT)
bool tacoortrans::clf2fdsite(int fdsiteid, double* xclf, double* xfd)
{
  int i, j;
  double x[3];
  if(!chk_fdsiteid(fdsiteid))
    {
      for (i=0; i<3; i++)
	xfd[i] = 0.0;
      return false;
    }
  for (i = 0; i < 3; i++)
    x[i] = xclf[i]-fd_origin_clf[fdsiteid][i]; // (can use same vector variable as input and output)
  for (i=0; i<3; i++)
    {
      xfd[i] = 0.0;
      for (j=0; j<3; j++)
	xfd[i] +=  fd2clf_mat[fdsiteid][j][i] * x[j]; // m transpose for rotating from CLF to FD
    }
  return true;
}



void tacoortrans::get_alt_azm_in_SDP(double *sdp_n,double *x, double *alt_sdp, double *azm_sdp)
{
  int i;
  double vmag,ex[3],ey[3],ez[3],y[3];
  // z-axis of the shower-detector plane coordinate system.
  // actually it is negative of the shower detector plane vector, because
  // the shower-detector plane vector was chosen so that 
  // (event axis) cross ( sdp normal) || (rp vector)
  for (i = 0; i < 3; i ++ )
    ez[i] = -sdp_n[i];
  
  vmag = sqrt (ez[0]*ez[0]+ez[1]*ez[1]);
  
  // x-axis of the shower-detector plane coordinate system,
  // a cross-product b/w FD z-axis and SDP z axis
  ex[0] = -ez[1]/vmag; ex[1] = ez[0]/vmag; ex[2] = 0.0;
  
  // y-axis of the shower-detector plane coordinate system, a cross-product
  // b/w SDP z - axis and SDP x - axis
  ey[0] = -ez[0]*ez[2]/vmag; ey[1] = -ez[1]*ez[2]/vmag; ey[2] = vmag;
  

  // Get the transformed vector in SDP frame
  y[0] = 0.0; y[1] = 0.0; y[2] = 0.0; vmag = 0.0;
  for (i=0; i<3; i++)
    {      
      vmag += x[i]*x[i];
      y[0] += x[i]*ex[i];
      y[1] += x[i]*ey[i];
      y[2] += x[i]*ez[i];
    }
  vmag = sqrt(vmag);
  for (i=0; i<3; i++)
    y[i] /= vmag;
  
  // altitude and azimuth in SDP frame
  (*alt_sdp) = tacoortrans_R2D * asin(y[2]);
  (*azm_sdp) = tacoortrans_R2D * atan2(y[1],y[0]);
  while ((*azm_sdp) < -180.0)
    (*azm_sdp) += 360.0;
  while ((*azm_sdp) >= 180.0)
    (*azm_sdp) -= 360.0;
}


void tacoortrans::rotate (double a[], double m[][3]) {
  // Multiply a vector a[3] by matrix m[3][3]

  int i, j;
  double v[3] = {0., 0., 0.};

  for (i=0; i<3; i++)
    for (j=0; j<3; j++)
      v[i] += a[j] * m[i][j];

  for (i=0; i<3; i++)
    a[i] = v[i];
}

void tacoortrans::unrotate (double a[], double m[][3]) {
  // Multiply a vector a[3] by matrix m[3][3]

  int i, j;
  double v[3] = {0., 0., 0.};

  for (i=0; i<3; i++)
    for (j=0; j<3; j++)
      v[i] += a[j] * m[j][i];

  for (i=0; i<3; i++)
    a[i] = v[i];
}

void tacoortrans::xrot (double a[], double angle) {
  // Rotate vector about x-axis
  // Positive angle rotations move z toward +y

  int i, j;
  double b[3] = {0., 0., 0.};

  double m[3][3] = {
    { 1.,          0.,         0. },
    { 0.,  cos(angle), sin(angle) },
    { 0., -sin(angle), cos(angle) }
  };
  
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      b[i] += a[j] * m[i][j];
    }
  }

  for (i=0; i<3; i++)
    a[i] = b[i];
}

void tacoortrans::yrot (double a[], double angle) {
  // Rotate vector about y-axis
  // Positive angle rotations move z toward -x

  int i, j;
  double b[3] = {0., 0., 0.};

  double m[3][3] = {
    { cos(angle), 0.,  -sin(angle) },
    { 0.,         1.,           0. },
    { sin(angle), 0.,   cos(angle) }
  };

  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      b[i] += a[j] * m[i][j];
    }
  }

  for (i=0; i<3; i++)
    a[i] = b[i];
}

void tacoortrans::zrot (double a[], double angle) {
  // Rotate vector about z-axis
  // Positive angle rotations move x toward -y

  int i, j;
  double b[3] = {0., 0., 0.};

  double m[3][3] = {
    {  cos(angle), sin(angle), 0. },
    { -sin(angle), cos(angle), 0. },
    {          0.,         0., 1. }
  };
  
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      b[i] += a[j] * m[i][j];
    }
  }

  for (i=0; i<3; i++)
    a[i] = b[i];
}


void tacoortrans::latlonalt_to_xyz (double Latitude, double Longitude, 
				    double Altitude,
				    double *X, double *Y, double *Z) {
  // Convert geodectic Latitude, Longitude, Altitude to ECEF X, Y, Z
  // Altitude is geodectic altitude
  // Latitude and Longitude in degrees
  double r0, as;
  Latitude *= tacoortrans_D2R;
  Longitude *= tacoortrans_D2R;
  r0 = tacoortrans_R_EQ/sqrt(cos(Latitude)*cos(Latitude) +
			     (1-tacoortrans_FLAT)*(1-tacoortrans_FLAT)*
			     sin(Latitude)*sin(Latitude));
  as = (1-tacoortrans_FLAT)*(1-tacoortrans_FLAT)*r0;
  *X = ( r0 + Altitude ) * cos(Latitude)*cos(Longitude);
  *Y = ( r0 + Altitude ) * cos(Latitude)*sin(Longitude);
  *Z = ( as + Altitude ) * sin(Latitude);
}

void tacoortrans::xyz_to_latlonalt (double X, double Y, double Z,
				    double *Latitude, double *Longitude, 
				    double *Altitude) {
  // Convert ECEF X, Y, Z to geodetic Latitude, Longitude, Altitude
  double ecc_prime2, R_Polar, theta;
  double p, top, bottom;
  *Longitude = atan2(Y,X);
  R_Polar = tacoortrans_R_EQ*(1-tacoortrans_FLAT);
  p = sqrt(X*X + Y*Y);
  theta = atan(Z*tacoortrans_R_EQ/(p*R_Polar));
  ecc_prime2 = (tacoortrans_R_EQ*tacoortrans_R_EQ - 
		(R_Polar*R_Polar))/(R_Polar*R_Polar);
  top = Z + ecc_prime2 * R_Polar * sin(theta)*sin(theta)*sin(theta);
  bottom = p - tacoortrans_ECC2 * tacoortrans_R_EQ * 
    cos(theta)*cos(theta)*cos(theta);
  *Latitude = atan(top/bottom);
  *Altitude = p/cos( *Latitude ) - 
    tacoortrans_R_EQ/sqrt( 1 - tacoortrans_ECC2 * 
			   sin(*Latitude) * sin(*Latitude) );
  *Latitude *= tacoortrans_R2D;
  *Longitude *= tacoortrans_R2D;
}

void tacoortrans::latlonalt_to_xyz_clf_frame(double Latitude,double Longitude,
					     double Altitude,double *xyz)
{
  // Convert GPS coordinates to xyz with respect to CLF coordinate system
  // X - EAST, Y-NORTH
  // lat,lon in degrees, alt in meters.
  // xyz components in meters.
  double XYZ[3];
  latlonalt_to_xyz (tacoortrans_CLF_Latitude,tacoortrans_CLF_Longitude,
		    tacoortrans_CLF_Altitude,
		    &XYZ[0],&XYZ[1],&XYZ[2]);
  latlonalt_to_xyz (Latitude, Longitude, Altitude, &xyz[0],&xyz[1], &xyz[2]);
  xyz[0] -= XYZ[0]; xyz[1] -= XYZ[1]; xyz[2] -= XYZ[2];
  zrot (xyz, tacoortrans_CLF_Longitude*tacoortrans_D2R); 
  yrot (xyz, (90.0-tacoortrans_CLF_Latitude)*tacoortrans_D2R); 
  zrot (xyz, 90.0*tacoortrans_D2R);
}


/* Reduces X to the range 0 to Y.  Routine courtesy of Andrew Smith */
double tacoortrans::range (double X, double Y) {
  return (X-floor(X/Y)*Y);
}


double tacoortrans::dotProduct(double a[3], double b[3]) 
{
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}


void tacoortrans::crossProduct(double a[3], double b[3], double c[3]) 
{
  c[0] = a[1]*b[2]-a[2]*b[1];
  c[1] = a[2]*b[0]-a[0]*b[2];
  c[2] = a[0]*b[1]-a[1]*b[0];
}



void tacoortrans::unitVector(double r[3], double n[3]) 
{
  double l = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];

  /* If 'r' is a null vector, return a unit vector pointing along z-axis */
  if ( l <= 0.00 ) {
    n[0] = 0.00;
    n[1] = 0.00;
    n[2] = 1.00;
  }
  else {
    l = sqrt( l );

    n[0] = r[0] / l;
    n[1] = r[1] / l;
    n[2] = r[2] / l;
  }
}

double tacoortrans::magnitude(double a[3]) 
{
  return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}



/*
 *  Gives the latitude (radians), longitude (radians), and altitude (meters) 
 *    of a coordinate (x, y, z) (meters) wrt. the center of the Earth.
 *
 *  Input:
 *    double x --- x-coordinate (Axis pointing from 180W, 0N to 0E, 0N)
 *    double y --- y-coordinate (Axis pointing from 90W, 0N to 90E, 0N)
 *    double z --- z-coordinate (Axis pointing from south pole to north pole)
 *
 *  Output:
 *    double *lat --- the latitzenazm_to_radecude cooresponding to the 
 *                    given point in radians
 *    double *lon --- the longitude in radians
 *    double *alt --- the altitude above mean sea level in meters.
 *
 *  Returns
 *    void
 */
void tacoortrans::xyz2lla(double x, double y, double z, 
			  double *lat, double *lon, double *alt) {
  double rho, sinq, cosq, num, den, norm;
  rho = tacoortrans_ECC1 * sqrt( x*x + y*y );
  norm = sqrt( rho*rho + z*z );
  sinq = z / norm;
  cosq = rho / norm;
  num = z + tacoortrans_ECC2*tacoortrans_R_EQ/tacoortrans_ECC1 * 
    sinq * sinq * sinq;
  den = rho/tacoortrans_ECC1 - tacoortrans_ECC2*tacoortrans_R_EQ * 
    cosq * cosq * cosq;
  norm = sqrt( num*num + den*den );
  *lat = atan(num/den);
  *lon = atan2(y, x);
  sinq = num / norm;
  cosq = den / norm;
  *alt = rho/(tacoortrans_ECC1*cosq) - 
    tacoortrans_R_EQ/sqrt(1.0 - tacoortrans_ECC2*sinq*sinq);
  return;
}

/*
 *  Gives the coordinate (x, y, z) (meters) relative to the center of the 
 *    Earth for the given latitude (radians), longitude (radians), and 
 *    sea-level altitude (meters).
 *
 *  Input:
 *    double lat --- a latitude (radians)
 *    double lon --- a longitude (radians)
 *    double alt --- an altitude (meters above mean sea level)
 *
 *  Output:
 *    double *x --- x-coordinate (Axis pointing from China to Africa)
 *    double *y --- y-coordinate (Axis pointing from S. America to India)
 *    double *z --- z-coordinate (Axis pointing form S Pole to N pole)
 *
 *  Returns:
 *    void
 */
void tacoortrans::lla2xyz(double lat, double lon, double alt, 
			  double *x, double *y, double *z) {
  double cosq, sinq, r0, as;
  cosq = cos(lat);
  sinq = sin(lat);
  r0 = tacoortrans_R_EQ/sqrt(cosq*cosq+tacoortrans_ECC1*
			     tacoortrans_ECC1*sinq*sinq);
  as = tacoortrans_ECC1 * tacoortrans_ECC1 * r0;
  *x = ( r0 + alt ) * cosq * cos(lon);
  *y = ( r0 + alt ) * cosq * sin(lon);
  *z = ( as + alt ) * sinq;
  return;
}


// Returns full Julian date corresponding to given UTC date and time
// (time is in seconds since midnight)
double tacoortrans::utc_to_jday(int year, int month, int day, double second)
{
  int a, b, c, e, f;
  int iyear, imonth, iday;
  iyear = year;
  imonth = month;
  iday = day;
  if (imonth <= 2)
    {
      iyear -= 1;
      imonth += 12;
    }
  a = iyear/100;
  b = a/4;
  c = 2-a+b;
  e = (int)floor(365.25 * (double)(iyear+4716));
  f = (int)floor(30.6001 * (imonth+1));
  return ((double)(c+iday+e+f)-1524.5+second/86400.0);
}


/* Returns Greenich Mean Standard Time (GMST) in radians */
double tacoortrans::jday_to_GMST (double jday) 
{
  double T0, T;
  double gmst;
  /* Getting Julian days since Jan 1 2000 at noon */
  T0 = jday - 2451545.0;
  /* Getting T as julian centuries since Jan 1 2000 at noon */
  T = T0 / 36525.0;
  /* GMST in degrees */
  gmst = 280.46061837 + 360.98564736629 * T0
    + 0.000387933 * T * T
    - (1/38710000.0) * T * T * T;
  /* Getting gmst between 0 and 360 */
  gmst = range (gmst*tacoortrans_PI/180., 2.*tacoortrans_PI);  
  return gmst;
}

// Calculate the local mean siderial time J2000
// INPUT: Julian date (in days) and longitude 
// (radians, if west then with minus sign)
// adopted from BTS routines
double tacoortrans::jday_to_LMST(double jd, double lon)
{
  double jd0, t, st, q;
  jd -= 2415020.0;
  jd0 = (int)(jd+0.5)-0.5;
  t = jd0/36525.0;
  q = (jd - jd0) * 1.0027379;
  st = 0.276919398+100.0021359*t+1.075E-6*t*t+lon/2/tacoortrans_PI+q;
  st -= (int) st;
  return st*2*tacoortrans_PI;
}


/*
 * Provide: Julian day, Latitude, Longitude, Altitude, Zenith and 
 *          Azimuth (radians)
 * Returns: HA, RA and Dec (radians)
 */
void tacoortrans::zenazm_to_radec (double jday, double lat, double lon,
				   double zen, double azm, 
				   double *ha, double *ra, double *dec) 
{  
  double lmst;
  double sinzen, coszen;
  double sinazm, cosazm;
  double sinlat, coslat;
  lmst = range (jday_to_GMST (jday) + lon, (2.*tacoortrans_PI));
  sinzen = sin(zen);
  coszen = cos(zen);
  sinazm = sin(azm);
  cosazm = cos(azm);
  sinlat = sin(lat);
  coslat = cos(lat);
  (*dec) = asin ( coszen*sinlat + sinzen*sinazm*coslat );
  (*ha) = atan2 ( -sinzen*cosazm, (coszen*coslat - sinzen*sinazm*sinlat ) );
  (*ha) = range ((*ha), (2.*tacoortrans_PI));
  (*ra) = range (lmst - (*ha), (2.*tacoortrans_PI));
}

void tacoortrans::zenazm_to_radec_ecc (double jday, double lat, double lon, 
				       double alt, double zen, double azm, 
				       double *ha, double *ra, double *dec) 
{
  double lmst;
  double v[3], modlat;
  double sinzen, coszen;
  double sinazm, cosazm;
  double sinlat, coslat;
  lmst = range (jday_to_GMST (jday) + lon, (2.*tacoortrans_PI));
  lla2xyz (lat, lon, alt, &v[0], &v[1], &v[2]);
  unitVector (v, v);
  modlat = asin (v[2]);
  sinzen = sin(zen);
  coszen = cos(zen);
  sinazm = sin(azm);
  cosazm = cos(azm);
  sinlat = sin(modlat);
  coslat = cos(modlat);
  (*dec) = asin ( coszen*sinlat + sinzen*sinazm*coslat );
  (*ha) = atan2 ( -sinzen*cosazm, (coszen*coslat - sinzen*sinazm*sinlat ) );
  (*ha) = range ((*ha), (2.*tacoortrans_PI));
  (*ra) = range (lmst - (*ha), (2.*tacoortrans_PI));
}


// return hour angle in radians, inputs are in radians
double tacoortrans::get_ha (double zen, double azm, double lat)
{
  return atan2(-cos(azm),cos(lat)/tan(zen)-sin(lat)*sin(azm));
}

// return declination in radians, inputs are in radians
double tacoortrans::get_dec (double zen, double azm, double lat)
{
  return asin(cos(zen)*sin(lat)+sin(zen)*cos(lat)*sin(azm));
}

// Convert equatorial coordinates to galactic coordinates (J2000)
// INPUT: ra,dec = right ascension, declination (radians)
// OUTPUT: l, b = galactic longitude, galatic latitude (radians)
void tacoortrans::radec_to_gal(double ra, double dec, double *l, double *b)
{
  const double gnp_ra   = tacoortrans_gnp_ra*tacoortrans_D2R;
  const double gnp_dec  = tacoortrans_gnp_dec*tacoortrans_D2R;
  const double glon_enp = tacoortrans_glon_enp*tacoortrans_D2R;
  (*b) = asin(sin(gnp_dec)*sin(dec)+cos(gnp_dec)*cos(dec)*cos(ra-gnp_ra));
  (*l) = glon_enp-atan2(cos(dec)*sin(ra-gnp_ra)/cos((*b)),
			(sin(dec)-sin((*b))*sin(gnp_dec))/cos((*b))/cos(gnp_dec));
  (*l) = range((*l),(2.*tacoortrans_PI));
}

/* Galactic longitude and latitude separately (adopted from BTS routines) */
double tacoortrans::gall(double ra, double dec)
{
  double l, tx1, tx2;
  tx1=sin(3.3660332070-ra);
  tx2=cos(3.3660332070-ra)*sin(.4734790845)-tan(dec)*cos (.4734790845);
  l=5.2871631341-atan2 (tx1, tx2);
  l=range(l,(2.*tacoortrans_PI));
  return l;
}
double tacoortrans::galb(double ra, double dec)
{
  double b, sb;
  sb = sin(dec)*sin(.4734790845)+cos(dec)*cos(.4734790845)*cos(3.3660332070-ra);
  b=asin (sb);
  return b;
}

// Convert equatorial coordinates to supergalactic coordinates (J2000)
// INPUT: ra,dec = right ascension, declination (radians)
// OUTPUT: l, b = supergalactic longitude, galatic latitude (radians)
void tacoortrans::radec_to_sgal(double ra, double dec, double *l, double *b)
{
  const double sgnp_ra   = tacoortrans_sgnp_ra*tacoortrans_D2R;
  const double sgnp_dec  = tacoortrans_sgnp_dec*tacoortrans_D2R;
  const double sglon_enp = tacoortrans_sglon_enp*tacoortrans_D2R;
  (*b) = asin(sin(sgnp_dec)*sin(dec)+cos(sgnp_dec)*cos(dec)*cos(ra-sgnp_ra));
  (*l) = sglon_enp-atan2(cos(dec)*sin(ra-sgnp_ra)/cos((*b)),
			 (sin(dec)-sin((*b))*sin(sgnp_dec))/cos((*b))/cos(sgnp_dec));
  (*l) = range((*l),(2.*tacoortrans_PI));
}

/* Supergalactic longitude and latitude separately (adopted from BTS routines) */
double tacoortrans::sgall(double ra, double dec)
{
  double sgl, tx1, tx2;
  tx1=sin(4.9524451297-ra);
  tx2=cos(4.9524451297-ra)*sin(.2741727325)-tan(dec)*cos(.2741727325);
  sgl=3.6032397315-atan2 (tx1,tx2);
  sgl=range(sgl,(2.*tacoortrans_PI));
  return sgl;
}
double tacoortrans::sgalb(double ra, double dec)
{
  double sgb,sb;
  sb=sin(dec)*sin(.2741727325)+cos(dec)*cos(.2741727325)*cos(4.9524451297-ra);
  sgb=asin(sb);
  return sgb;
}
