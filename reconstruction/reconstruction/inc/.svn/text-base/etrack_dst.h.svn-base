/*
 *     Bank for TA event track information, INDEPENDENT OF RECONSTRUCTION. This bank depends 
 *     only on physics of the cosmic ray event and not on ANY details of the detector(s) that 
 *     reconstructed it. The purpose of this bank is to be able to get out event geometry
 *     and energy without going through detector-specific DST banks which often store information in
 *     their own idiosyncratic ways that are hard to understand for non-experts.
 * 
 *     Dmitri Ivanov (ivanov@physics.rutgers.edu)
 *     Apr 23, 2011
 *     Last modified: Apr. 23, 2011
*/

#ifndef _ETRACK_
#define _ETRACK_

#define ETRACK_BANKID  9999
#define ETRACK_BANKVERSION   000

#ifdef __cplusplus
extern "C" {
#endif
integer4 etrack_common_to_bank_ ();
integer4 etrack_bank_to_dst_ (integer4 * NumUnit);
integer4 etrack_common_to_dst_ (integer4 * NumUnit);	/* combines above 2 */
integer4 etrack_bank_to_common_ (integer1 * bank);
integer4 etrack_common_to_dump_ (integer4 * opt1);
integer4 etrack_common_to_dumpf_ (FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* etrack_bank_buffer_ (integer4* etrack_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


/* 
   maximum number of additional pieces of information that are not 
   related to physics of event track and that some users may want to include.
   These pieces of information come as additional ETRACK_NUDATA float
   numbers that one may choose to store in this bank.  It is not advised to include
   any tube / surface detector - specific information in this banks. For that sort of data, 
   please use the corresponding SD or FD dst banks.
*/
#define ETRACK_NUDATA 32

typedef struct
{
  /*
    Privide all information in CLF frame: X=East, Y=North, Z=Up, Origin = CLF
    CLF origin: (LATITUDE=39.29693 degrees, LONGITUDE=-112.90875 degrees).  
    For most recent numbers, visit: 
    http://www.telescopearray.org/tawiki/index.php/CLF/FD_site_locations
  */

  real4    energy;    /* event energy in EeV units (1 EeV = 10^18 eV), 0 if not available */
  real4    xmax;      /* slant depth where maximum number of charged particles occurs, [g/cm^2], 0 if not available */
  
  /* 
     event direction in CLF frame
     <sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)> points to where event comes from. 
  */
  real4    theta;     // zenith angle [radians]
  real4    phi;       // azimuthal angle [radians], counter-clock-wise from X=East
 
  
  /* 
     shower core in CLF frame
     NOT NEEDED: a 3D point where event axis crosses Z=0 plane in some FD frame
     WANT: just a 2D point where shower axis crosses the CLF Z=0 plane
  */
  real8    t0;        /* time when the shower axis crosses CLF Z=0 plane, [uS], with respect to GPS second */
  real4    xycore[2]; /* CLF XY point where event axis croses CLF Z=0 plane [meters], with respect to CLF origin */
  
 
  
  real4    udata[ETRACK_NUDATA]; /* non-essential, user-specific pieces information */
  integer4 nudata;    /* number of user-specific pieces of information */
  integer4 yymmdd;    /* UTC date, yy=year since 2000, mm=month, dd=day */
  integer4 hhmmss;    /* UTC time, hh=hour,mm=minute,ss=second */
  
  /* optionally, one can also put in a label that tells if the event passes
     the quality cuts of the reconstruction that produced these results.  This can be anything
     starting from simple 1=YES, 0=NO to more elaborate flag that tells which cuts were passed and 
     which ones failed.  For simplicity and clarity, it is best if one uses just 1=pass, 0=fail */
  integer4 qualct;    /* flag to indicate whether events passes the quality cuts */
  
} etrack_dst_common;

extern etrack_dst_common etrack_;

#endif

