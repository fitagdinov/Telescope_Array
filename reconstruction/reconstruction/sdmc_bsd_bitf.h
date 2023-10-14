#ifndef _sdmc_bsd_bitf_h_
#define _sdmc_bsd_bitf_h_

#include "dst_std_types.h"


#ifndef __cplusplus
/* This is how TA SD Monte-Carlo and reconstruction decide if the counter is live or not
INPUT: calibration buffer that's filled from tasdcalib and tasdconst DST files
OUTPUT values:
0 = working
> 0 - problems. See description in the code */ 
integer4 failed_Ben_Stokes_D_Ivanov_Live_Counter_Criteria(real4 *calbuf);

// Check if ICRR calibration fails during MC generation
integer4 is_icrr_cal_mc(integer4 bitf);

// Check if Rutgers calibration fails during MC generation
integer4 is_ru_cal_mc(integer4 bitf);

// Check if Rutgers calibration fails during event reconstruction
// (this does not prevent the detector from being simulated in the TA SD MC)
integer4 is_ru_cal_rc(integer4 bitf);

// Check if a certain bit is set (for debugging purposes)
integer4 is_cal_bit(integer4 bitf, integer4 bit_num);

// This function checked the bit flag from the bsdinfo DST bank and returns
// the following: 
// not_working_according_to_sdmc_checks: if true, then the counter is not working
// according to the checks performed by the TA SD Monte Carlo
// not_working_according_to_sd_reconstruction: if true, then the counter is not
// working according to the checks performed by the TA SD event reconstruction
void sdmc_calib_check_bitf(integer4 bsdinfo_bitf, 
			   integer4* not_working_according_to_sdmc_checks, 
			   integer4* not_working_according_to_sd_reconstruction);
#else
extern "C" integer4 failed_Ben_Stokes_D_Ivanov_Live_Counter_Criteria(real4 *calbuf);
extern "C" integer4 is_icrr_cal_mc(integer4 bitf);
extern "C" integer4 is_ru_cal_mc(integer4 bitf);
extern "C" integer4 is_ru_cal_rc(integer4 bitf);
extern "C" integer4 is_cal_bit(integer4 bitf, integer4 bit_num);
extern "C" void sdmc_calib_check_bitf(integer4 bsdinfo_bitf, 
				      integer4* not_working_according_to_sdmc_checks, 
				      integer4* not_working_according_to_sd_reconstruction);
#endif

#endif
