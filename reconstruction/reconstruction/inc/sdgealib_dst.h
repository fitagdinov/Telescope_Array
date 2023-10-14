/*
 *     Bank for SD Geant Library
 *     Dmitri Ivanov (ivanov@physics.rutgers.edu)
 *     Mar 17, 2009

 */

#ifndef _SDGEALIB_
#define _SDGEALIB_

#define SDGEALIB_BANKID  13111
#define SDGEALIB_BANKVERSION   000

#ifdef __cplusplus
extern "C" {
#endif
integer4 sdgealib_common_to_bank_();
integer4 sdgealib_bank_to_dst_(integer4 * NumUnit);
integer4 sdgealib_common_to_dst_(integer4 * NumUnit); /* combines above 2 */
integer4 sdgealib_bank_to_common_(integer1 * bank);
integer4 sdgealib_common_to_dump_(integer4 * opt1);
integer4 sdgealib_common_to_dumpf_(FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* sdgealib_bank_buffer_ (integer4* sdgealib_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


#define SDGEALIB_NKEMAX          63   // maximum possible number of K.E. bins ( achieved for gammas)
#define SDGEALIB_NSECTHETA       7    // sec(theta) binning
#define SDGEALIB_SECTHETAMIN     0.75
#define SDGEALIB_SECTHETAMAX     4.25

#define SDGEALIB_NLOG10ELOSS     250  // log10(eloss/MeV) binning
#define SDGEALIB_LOG10ELOSSMIN   -2.0
#define SDGEALIB_LOG10ELOSSMAX   3.0
#define SDGEALIB_NN0BINS         0x9000 // max. number of non-zero 2D energy loss histogram bins
#define SDGEALIB_MAXBVAL         32767  // maximum weight for digitization
// Type of information stored after the header
#define SDGEALIB_PINF 0  // particle information
#define SDGEALIB_HIST 1  // energy loss bins
//////////////// sdgealib header information /////////////////////
typedef struct
{
  integer4 corid; // CORSIKA ID
  integer4 itype; // information type stored
} sdgealib_head_struct;

// particle information structure
typedef struct
{
  ///////////// Interpolation rules ////////////////////////////// 

  // Energy loss probability proportionality constants for log10(KE/MeV)
  // dependence
  real8 peloss_kepc[SDGEALIB_NKEMAX][SDGEALIB_NSECTHETA];

  // Energy loss probability proportionality constants for sec(theta)
  // dependence
  real8 peloss_sepc[SDGEALIB_NKEMAX][SDGEALIB_NSECTHETA];

  // Peak channel energy loss proportionality constants for log10(KE/MeV)
  // dependence, [0] - lower, [1] - upper layers
  real8 pkeloss_kepc[SDGEALIB_NKEMAX][SDGEALIB_NSECTHETA][2];

  // Peak channel energy loss proportionality constants for sec(theta)
  // dependence, [0] - lower, [1] - upper layers
  real8 pkeloss_sepc[SDGEALIB_NKEMAX][SDGEALIB_NSECTHETA][2];

  ///////////  Binning information /////////////////////////////////
  real8 log10kemin; // minimum log10(K.E/MeV)
  real8 log10kemax; // maximum log10(K.E/MeV)
  real8 secthetamin;
  real8 secthetamax;
  integer4 nke; // number of log10(KE) bins
  integer4 nsectheta; // number of sec(theta) bins
} sdgealib_pinf_struct;

// energy loss bins (not explicitly written, nested into another structure ) ///
// ix = -1: energy loss occurs in the upper but not in the lower layer
// iy = -1: energy loss occurs in the lower but not in the upper layer
typedef struct
{
  integer2 ix; // bin x-axis index ( eloss in upper )
  integer2 iy; // bin y-axis index ( eloss in lower )
  integer2 w; // bin weight from 0 to SDGEALIB_MAXBVAL
} sdgealib_bin_struct;

// structure for energy loss (2D histogram) + probability of energy loss
typedef struct
{
  ///////////////// Energy loss histogram information ///////////////////
  real8 peloss[SDGEALIB_NSECTHETA]; // probability of non-zero energy loss
  // non-zero energy loss bins
  // number of non-zero ( > SDGEALIB_LOG10ELOSSMIN) bins
  integer4 nn0bins[SDGEALIB_NSECTHETA];
  integer4 ike; // kinetic energy index, log10(K.E./MeV)
  sdgealib_bin_struct bins[SDGEALIB_NSECTHETA][SDGEALIB_NN0BINS];
} sdgealib_hist_struct;

extern sdgealib_head_struct sdgealib_head_;
extern sdgealib_pinf_struct sdgealib_pinf_;
extern sdgealib_hist_struct sdgealib_hist_;

#endif
