/* 
 * $Source: /hires_soft/uvm2k/geo/geohr2_dst.h,v $
 * $Log: geohr2_dst.h,v $
 * Revision 1.2  2002/03/12 18:27:54  hires
 * Add variables for mirror curvature and mirror-to-cluster separation
 *
 * Revision 1.1  1999/07/12 17:23:46  wiencke
 * Initial revision
 *
 * Stand-alone HIRES2 geometry data structure
 */

#ifndef _GEOHR2_
#define _GEOHR2_

#define HR_GEOHR2_MAXMIR 42

typedef struct  {
  
  /* 
   * mean direction vector of local vertical axis at local site origin
   * relative to absolute HIRES origin
   */
   real4 xvsite;
   real4 yvsite;
   real4 zvsite;
   
  /* position of local site origin relative to absolute HIRES origin */
   real4 xsite;
   real4 ysite;
   real4 zsite;
   
  /* position of mirror center relative to local site origin (LSO) */
   real4 xmir[HR_GEOHR2_MAXMIR];
   real4 ymir[HR_GEOHR2_MAXMIR];
   real4 zmir[HR_GEOHR2_MAXMIR];
   
  /* mean direction vectors of mirrors relative to location of mirror */
   real4 xvmir[HR_GEOHR2_MAXMIR];
   real4 yvmir[HR_GEOHR2_MAXMIR];
   real4 zvmir[HR_GEOHR2_MAXMIR];

  /* mirror azimuth E=0 N=90 */
   real4 azmir[HR_GEOHR2_MAXMIR];

  /* tilt of each cluster/mirror about the mirror axis, in degrees */
   real4 taumir[HR_GEOHR2_MAXMIR];

  /* ring number of mirror */
   integer4 ring[HR_GEOHR2_MAXMIR];
     
  /* mean direction vectors of tubes relative to location of mirror */
   real4 xvtube[HR_GEOHR2_MAXMIR][256];
   real4 yvtube[HR_GEOHR2_MAXMIR][256];
   real4 zvtube[HR_GEOHR2_MAXMIR][256];

  real4 rcurv[HR_GEOHR2_MAXMIR];
  real4 sep[HR_GEOHR2_MAXMIR];

} geohr2_dst_common ;

extern geohr2_dst_common geohr2_ ;

#endif










