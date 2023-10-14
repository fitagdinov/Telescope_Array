//   Routines to calculate distances from various SD borders. 
//   Dmitri Ivanov, <ivanov@physics.rutgers.edu>
//   Last modified: Wed Mar 10 

#ifndef _sdparamborder_
#define _sdparamborder_






#ifdef __cplusplus

/* 
   INPUTS: x,y in [1200m] units with respect to SD origin
   xcore, ycore: core position in CLF frame with respect to SD origin.
   If you have xcore,ycore just in the CLF frame in meters, do the folowing:
   xcore = xcore_in_CLF_in_meters/1.2e3 - RUSDGEOM_ORIGIN_X_CLF;
   ycore = ycore_in_CLF_in_meters/1.2e3 - RUSDGEOM_ORIGIN_Y_CLF;
   
   OUTPUTS:

   b[2] - unit vector, points inside SD array, perpendicular to 
   closest boundary line
   bdist - distance along that vector.  It is positive if the point
   is inside the array and negative if the point is outside of the array

   
   FOR T-SHAPE BOUNDARY:
   tbr[2] - unit vect, perpendicular to closest BR T-Shape boundary,
   pointing inside the BR subarray
   
   tdistbr - distance along that vector.  Is negative if the point
   is outside of the BR subarray
   

   Simular for LR, SK subarrays
   tlr[2]
   tdistlr
   tsk[2]
   tdistks

   For all points inside the SD array (bdist is positive), AT MOST ONE
   from tdistbr,tdistlr,tdistsk  is positive, which corresponds
   to the case that the point is in one of the subarrays.  If a point
   is in neither subarrays, then all 3 distances will be negative.
   
*/
extern "C" void sdbdist(double x, double y, 
			double *b,  double *bdist, 
			double *tbr,double *tdistbr, 
			double *tlr,double *tdistlr,
			double *tsk,double *tdistsk);

/* 

   Simplification of the above routine.
   INPUTS:
   xcore, ycore: core position in CLF frame with respect to SD origin.
   If you have xcore,ycore just in the CLF frame in meters, do the folowing:
   xcore = xcore_in_CLF_in_meters/1.2e3 - RUSDGEOM_ORIGIN_X_CLF;
   ycore = ycore_in_CLF_in_meters/1.2e3 - RUSDGEOM_ORIGIN_Y_CLF;
   
   OUTPUTS:
   borderdist: closest distance to the border that goes around the entire 
   array.  If this is negative, then xcore,ycore are outside
   of the array
   
   tshapedist: closest distance to the T-shape boundary, after
   considering all 3 subarrays: BR,LR,SK. If this is negative, then 
   xcore,ycore are outside of all subarrays (somewhere b/w the subarrays)

   both borderdist, and tshapedist are in 1200m units: if you want to 
   have your answer in meters, do: 
   borderdist_in_meters = borderdist * 1.2e3;
   tshapedist_in_meters = tshapedist_in_meters * 1.2e3;
 */

extern "C" void comp_boundary_dist(double xcore, double ycore, 
				   double *borderdist, double *tshapedist);

#else


extern void sdbdist(double x, double y, 
		    double *b,  double *bdist, 
		    double *tbr,double *tdistbr, 
		    double *tlr,double *tdistlr,
		    double *tsk,double *tdistsk);


extern void comp_boundary_dist(double xcore, double ycore, 
			       double *borderdist, double *tshapedist);

#endif

#endif
