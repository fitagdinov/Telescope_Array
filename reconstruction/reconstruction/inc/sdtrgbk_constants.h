#ifndef _sdtrgbk_const_
#define _sdtrgbk_const_

#define  NWFPSD  10      // maximum number of waveforms / SD
#define  NSDMAX  256     // maximum number of SDs / event
#define  NL1SD   256     // maximum number of level-1 trigger sds
#define  NSIGPSD 1280    // maximum number of level-1 trigger signals per SD
#define  NL1cnt  150     // number of fadc counts above pedestal for level-1 trigger
#define  SLWsize 8       // number of fadc time slices in the sliding window (NOT USED BY CORRECT LEVEL-1 SIGNAL FINDER!)
#define  L2TWND  8.0e-6  // level-2 trigger size window, [second]
#define  TRIGT_FADCTSLICE 32 // FADC time slice that corresponds to the trigger timing ( USED BY THE CORRECT LEVEL-1 SIGNAL FINDER)
#endif
