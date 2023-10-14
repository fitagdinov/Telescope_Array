#ifndef _rufptn_constants_h_
#define _rufptn_constants_h_

// if the time differences b/w some counter and the trigger time
// exceeds this value then this means that the clock count for the counter 
// has exceeded the maximum clock count and the counter time needs to be 
// corrected by a second.
// ( this effectively gives ~2x the time window size used by firmware for 
//   reading out the SDs )
#define LARGE_TIME_DIFF_uS 100.0

// ALL SD times have to be shifted by this ammount
// -260 nS is the GPS time offset correction
// -600 nS is the SD trigger time offset correction
// AFTER THIS CORRECTION IS APPLIED, there should be no SD - GPS offset
// SD timing is ready to use in the Hybrid analyses
// the time shift constant is expressed in micro seconds
#define SD_TIME_CORRECTION_uS (-0.600 - 0.260)

/* For converting time in uS to distance in units of counter
   separation distance 1200m. Numerically, this is 
   c*(10^(-6)(S/uS))/1200m, where c is the speed of light in m/s */
#define TIMDIST 0.249827048333

/* Square of the maximum distance that the counters can be apart and count
   as adjacent.  This should be an integer number.*/
#define SPACEADJ 2

/* Tolerance on time adjacency.  Given the separation distance b/w any two
   counters, the time difference should not exceed that distance in an ideal
   case, where the shower front is approximated by a plane. We will include
   some tolerance on that - the time difference is allowed to be within
   the counter separation distance plus some constant (the tolerance). */
#define TIMEADJTOL 0.0


// Counter must have charge in VEM greater than this value to participate in clusters
#define QMIN 1.4

/*
  For the following situation: have a multi-fold waveform, and signal 
  in the next FADC window does not start with NPEDSTART pedestal channels.  
  However, before that, if we had at least NPEDPREV channels of 
  pedestals in the previous fadc window, then treat the signal
  as an independent hit.
*/
#define NPEDPREV  50 

/* If for some channel the number of FADC counts exceeds 
   this number,then the channel potentially has a signal */
#define NRMSSIGNAL 5.0

/* Number of consequtive channels which should have fadc counts exceeding
   the pedestal value by NRMSSIGNAL to qualify as signal.  Counts in these
   channels are compared agains the pedestal in advance: i.e.
   if we are ont channel i, then we look at the channels i,i+1,i+2 and see
   if they have counts exceeding the pedestal by NRMSSIGNAL * ped.rms. 
   If so, start counting the signal.  If we are already at the
   end of the FADC trace, then check the remaining # of channels
   for signal. */
#define SIGNALCN 4



/* If after we've found the signal,NOSIGNALCN channels don't deviate
   from the pedestal level by more than NRMSSIGNAL * sigma, we've returned 
   back to the pedestal.  Refine the pedestal calculation until hit another
   signal, if any.  If at the end of the FADC trace, don't bother,
   count the last channels as signal.*/

#define NOSIGNALCN 4


// number of FADC channels over which 1MIP pulse areas are integrated
#define NMIPINTCH 12

// number of FADC channels over which ped pulse areas are integrated
#define NPEDINTCH 8

// Mean muon theta used in scaling 1MIP peaks.
#define MEAN_MU_THETA 35.0

// Constant time error, in counter separation units, used for tyro fitting
#define RUFPTN_TYRO_dt 0.31

// Minimum number of hits in space-time cluster for writing into a root-tree
#define MINNHITS_RT 3

// Chi2/d.o.f. cut on 1MIP fit
#define MAXMFTRCHI2 4.0

// For multiple waveform hits, it is observed that there is a correlation
// of the signals occuring after the signal chosen by pattern recognition,
// and the correlation time is about 2.5 * 1200m = 10 uS
// So, when patttern recognition picks out a signal from a multiple hit,
// we will include all other signals (in same counter) occuring
// 10uS later.
#define SAMESD_TIME_CORR 2.5


// Saturation is known to occur when we have more than
// 40 MIP in 20nS.  We express this charge in VEM, which is
// larger by a factor of sec (MEAN_MU_THETA):
#define QSAT20NSVEM  40.0/cos(MEAN_MU_THETA/57.2957795131)


// Number of iterations in Gold Deconvolution algorithm
// (Deconvolution used for finding the saturated counters)
#define NDECONV_ITER 1000


#endif
