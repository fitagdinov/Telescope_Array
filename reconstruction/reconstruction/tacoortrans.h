#ifndef _tacoortrans_h_
#define _tacoortrans_h_

// The degree-to-radian conversion constants
#define tacoortrans_PI  3.1415926535897931159979634685441851615906 
#define tacoortrans_D2R	0.0174532925199432954743716805978692718782
#define tacoortrans_R2D	57.2957795130823228646477218717336654663086

// Earth variables
#define tacoortrans_R_EQ         6378137.0  // Radius of the Earth at the equator.
#define tacoortrans_INV_FLAT     298.257223 // Shape parameter 
#define tacoortrans_FLAT         (1./tacoortrans_INV_FLAT)
#define tacoortrans_ECC1         (1.0 - tacoortrans_FLAT)
#define tacoortrans_ECC2         (tacoortrans_FLAT*(2.0-tacoortrans_FLAT))

// Central Laser Facility GPS Coordinates
// Latitude, Longitude in degrees, Altitude in meters.
#define tacoortrans_CLF_Latitude   39.29693
#define tacoortrans_CLF_Longitude  -112.90875
#define tacoortrans_CLF_Altitude   1382.0

// Galactic coordinate system definitions (J2000)
#define tacoortrans_gnp_ra     192.85949646  // RA of the galactic north pole, degree
#define tacoortrans_gnp_dec    27.12835323   // DEC of the galactic norh pole, degree
#define tacoortrans_glon_enp   122.93200023  // Galactic longitude of the equatorial north pole, degree

// Supergalactic coordinate system definitions (J2000)
#define tacoortrans_sgnp_ra    283.75420420  // RA of the supergalactic north pole, degree
#define tacoortrans_sgnp_dec   15.70894043   // DEC of the supergalactic norh pole, degree
#define tacoortrans_sglon_enp  26.45051665   // Supergalactic longitude of the equatorial north pole, degree

namespace tacoortrans
{
  
  // Site indices for the 3 FD detectors (for the reference):
  const int tacoortrans_nfd = 3;
  const int brfd = 0;
  const int lrfd = 1;
  const int mdfd = 2;
 
  extern const double fd_origin_clf[tacoortrans_nfd][3]; // CLF XYZ coordinates of each FD detector
  extern const double fd2clf_mat[tacoortrans_nfd][3][3]; // 3x3 rotation matrix from FD to CLF frame for each FD detector
   

  // check whether the FD site ID is in a legitimate range
  bool chk_fdsiteid(int fdsiteid);

  // Only rotates from FD frame to CLF frame
  // fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
  // xclf - 3-vector in CLF frame (INPUT)
  // xfd  - 3-vector in FD frame (OUTPUT)
  bool rot_fdsite2clf(int fdsiteid, double *xfd, double *xclf);
  
  
  // fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
  // xfd[3]  - vector in FD frame, [meters] (INPUT)
  // xclf[3] - vector in CLF frame, [meters] (OUTPUT)
  bool fdsite2clf(int fdsiteid, double *xfd, double *xclf);
  
  
  // Only rotates from CLF to FD frame
  // fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
  // xclf - 3-vector in CLF frame (INPUT)
  // xfd  - 3-vector in FD frame (OUTPUT)
  bool rot_clf2fdsite(int fdsiteid, double *xclf, double *xfd);
  
  // fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
  // xclf[3] - vector in CLF frame, [meters], (INPUT)
  // xfd[3]  - vector in FD frame, [meters], (OUTPUT)
  bool clf2fdsite(int fdsiteid, double *xclf, double *xfd);
  
  // Calculate altitude and azimuth in the Shower Detector Plane for any unit vector in the FD frame
  // sdp_n      (INPUT)  FD shower-detector plane normal unit vector, FD frame
  // Shower detector plane normal must be such that (event axis vector) x ( sdp normal vector) || (rp vector)
  // x          (INPUT)  some unit vector in the FD coordinate system
  // *alt_sdp   (OUTPUT) altitude in SDP frame, Degree , 0 means vector lies completely inside the SDP
  // *azm_sdp   (OUTPUT) azimuthal angle in SDP frame, Degree, CCW about -(sdp direction), 
  // 0 means along the vector that points from the FD to the event core
  void get_alt_azm_in_SDP(double *sdp_n,double *x, double *alt_sdp, double *azm_sdp);
  
  // Multiply a vector a[3] by matrix m[3][3]
  void rotate (double a[], double m[][3]);

  // Multiply a vector a[3] by matrix m[3][3]
  void unrotate (double a[], double m[][3]);
  
  // Rotate vector about x-axis
  // Positive angle rotations move z toward +y
  void xrot (double a[], double angle);

  // Rotate vector about y-axis
  // Positive angle rotations move z toward -x
  void yrot (double a[], double angle);

  // Rotate vector about z-axis
  // Positive angle rotations move x toward -y
  void zrot (double a[], double angle);
  
  // Convert geodectic Latitude, Longitude, Altitude to ECEF X, Y, Z
  // Altitude is geodectic altitude
  // Latitude and Longitude in degrees
  void latlonalt_to_xyz (double Latitude, double Longitude, double Altitude,
			 double *X, double *Y, double *Z);

  // Convert ECEF X, Y, Z to geodetic Latitude, Longitude, Altitude
  void xyz_to_latlonalt (double X, double Y, double Z,
			 double *Latitude, double *Longitude, double *Altitude);
  
  // Convert GPS coordinates to xyz with respect to CLF coordinate system
  // X - EAST, Y-NORTH
  // lat,lon in degrees, alt in meters.
  // xyz components in meters.
  void latlonalt_to_xyz_clf_frame(double Latitude,double Longitude,
				  double Altitude,double *xyz);

  // Reduces X to the range 0 to Y.  Routine courtesy of Andrew Smith
  double range (double X, double Y);

  // Self explanatory
  double dotProduct(double a[3], double b[3]);
  void   crossProduct(double a[3], double b[3], double c[3]);
  void   unitVector(double r[3], double n[3]);
  double magnitude(double a[3]);
  
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
   *    double *lat --- the latitzenazm_to_radecude cooresponding to 
   *                    the given point in radians
   *    double *lon --- the longitude in radians
   *    double *alt --- the altitude above mean sea level in meters.
   *
   *  Returns
   *    void
   */
  void xyz2lla(double x, double y, double z, double *lat, double *lon, double *alt);
  
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
  void lla2xyz(double lat, double lon, double alt, double *x, double *y, double *z);
  
  // Returns full Julian date corresponding to given UTC date and time
  // (time is in seconds since midnight)
  double utc_to_jday(int year, int month, int day, double second);
  
  // Returns Greenich Mean Standard Time (GMST) in radians
  double jday_to_GMST (double jday);

  // Calculate the local mean siderial time J2000
  // INPUT: Julian date (in days) and longitude 
  // (radians, if west then with minus sign)
  double jday_to_LMST(double jd, double lon);


  //    EQUATORIAL COORDINATES (HA, RA, DEC) ARE CALCULATED USING THE FOLLOWING ASSUMPTIONS 
  //    ABOUT THE LOCAL SKY COORDINATES:
  //    1. SHOWER ZENITH (zen) AND AZIMUTHAL (azm) ANGLES ARE SUCH THAT IF ONE CONSTRUCTS
  //    A CARTESIAN VECTOR FROM THEM IN THE USUAL WAY 
  //    <sin(zen)*cos(azm),sin(zen)*sin(azm),cos(zen)>
  //    THEN IT POINTS TO WHERE THE EVENT CAME FROM IN THE LOCAL SKY COORDINATES.
  //    2. LOCAL SKY COORDINATES ARE: X = EAST, Y = NORTH, Z - UP 
  //    3. EVENT DATE AND TIME ARE IN UTC 
  
  // Provide: Julian day, Latitude, Longitude, Altitude, Zenith and 
  //          Azimuth (radians)
  // Returns: HA, RA and Dec (radians)
  void zenazm_to_radec (double jday, double lat, double lon,
			double zen, double azm, 
			double *ha, double *ra, double *dec); 
  
  // More accurate ra, dec calculation ( input/output in the same
  // units as for zenazm_to_radec)
  void zenazm_to_radec_ecc (double jday, double lat, double lon, double alt, 
			    double zen, double azm, 
			    double *ha, double *ra, double *dec);
  
  // return hour angle in radians, inputs are in radians
  double get_ha (double zen, double azm, double lat);
  
  // return declination in radians, inputs are in radians
  double get_dec (double zen, double azm, double lat);
  
  // Convert equatorial coordinates to galactic coordinates (J2000)
  // INPUT: ra,dec = right ascension, declination (radians)
  // OUTPUT: l, b = galactic longitude, galatic latitude (radians)
  void radec_to_gal(double ra, double dec, double *l, double *b);
  
  // J2000 galactic longitude, everything in radians
  double gall(double ra, double dec);
  
  // J2000 galatcic latitude, everything in radians
  double galb(double ra, double dec);
    
  // Convert equatorial coordinates to supergalactic coordinates (J2000)
  // INPUT: ra,dec = right ascension, declination (radians)
  // OUTPUT: l, b = supergalactic longitude, galatic latitude (radians)
  void radec_to_sgal(double ra, double dec, double *l, double *b);
  
  // J2000 supergalactic longitude, everything in radians
  double sgall(double ra, double dec);
  
  // J2000 supergalatcic latitude, everything in radians
  double sgalb(double ra, double dec);
  
};

#endif
