/* 
 * $Source: /hires_soft/uvm2k/geo/geoh_dst.h,v $
 * $Log: geoh_dst.h,v $
 * Revision 1.2  1997/09/23 18:39:23  meyer
 * had to change a geo1_ refrence to a geoh_ refrence
 *
 * Revision 1.1  1997/09/23  17:41:24  meyer
 * Initial revision
 *
 */

#ifndef _GEOH_
#define _GEOH_

#include "univ_dst.h"

typedef struct  {
  
  /* 
   * mean direction vector of local vertical axis at local site origin
   * relative to absolute HIRES origin
   */
   real8 xvsite;
   real8 yvsite;
   real8 zvsite;
   
  /* position of local site origin relative to absolute HIRES origin */
   real8 xsite;
   real8 ysite;
   real8 zsite;
   
  /* position of mirror center relative to local site origin (LSO) */
   real8 xmir[HR_UNIV_MAXMIR];
   real8 ymir[HR_UNIV_MAXMIR];
   real8 zmir[HR_UNIV_MAXMIR];
   
  /* mean direction vectors of mirrors relative to location of mirror */
   real8 xvmir[HR_UNIV_MAXMIR];
   real8 yvmir[HR_UNIV_MAXMIR];
   real8 zvmir[HR_UNIV_MAXMIR];

  /* tilt of each cluster/mirror about the mirror axis, in degrees */
   real8 taumir[HR_UNIV_MAXMIR];

  /* mean direction vectors of tubes relative to location of mirror */
   real8 xvtube[HR_UNIV_MAXMIR][256];
   real8 yvtube[HR_UNIV_MAXMIR][256];
   real8 zvtube[HR_UNIV_MAXMIR][256];

} geoh_dst_common ;

extern geoh_dst_common geoh_ ;

#endif



