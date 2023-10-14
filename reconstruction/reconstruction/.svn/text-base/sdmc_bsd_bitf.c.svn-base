#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "event.h"



integer4 failed_Ben_Stokes_D_Ivanov_Live_Counter_Criteria(real4 *calbuf)
{
  integer4 ret_flag = 0;
  
  const real8 MEAN_MU_THETA = 35.0; // Mean muon theta used in scaling 1MIP peaks.
  const real8 MAXMFTRCHI2   = 4.0;  // Chi2/d.o.f. cut on 1MIP fit
  integer4 pchped[2]  = {(integer4)floor(calbuf[14]+0.5),(integer4)floor(calbuf[13])};
  real8   fadc_ped[2] = {(real8)calbuf[14]/8.0,(real8)calbuf[13]/8.0}; 
  integer4 mftndof[2] = {(integer4)floor(calbuf[20]+0.5),(integer4)floor(calbuf[19])};
  real8 mftchi2[2]    = {(real8)calbuf[22],(real8)calbuf[21]};
  real8 mip[2]        = {(real8)calbuf[8],(real8)calbuf[7]};
  integer4 pchmip[2]  = {0,0}; // needs to be calculated from mip and pchped
  real8 vem[2]        = {0,0}; // calculated from mip  
  integer4 k = 0;

  // a counter is considered as dead if any of the following conditions are satisfied
  
  if ((integer4)calbuf[2] != 0)
    ret_flag |= 1 << 0; // ICRR calibration issue, failed don't use criteria
  
  if (calbuf[3] <= 0. || calbuf[4] <= 0.)
    ret_flag |= 1 << 1; // ICRR calibration issue, Mev2pe problem
  
  if(calbuf[5] <= 0. || calbuf[6] <= 0.)
    ret_flag |= 1 << 2; // ICRR calibration issue, Mev2cnt problem

  if(calbuf[9] <= 0. || calbuf[10] <= 0.)
    ret_flag |= 1 << 3; // ICRR calibration issue, bad pedestal mean values
  
  if (calbuf[11] <= 0. || calbuf[12] <= 0.)
    ret_flag |= 1 << 4; // ICRR calibration issue, bad pedestal standard deviation
  
  if(calbuf[24] <= 0. || calbuf[25] <= 0.) 
    ret_flag |= 1 << 5; // ICRR calibration issue, saturation information not available
  
  if(calbuf[7] <= 0. || calbuf[8] <= 0.)
    ret_flag |= 1 << 6; // Rutgers calibration issue, bad mip values
  
  if(calbuf[13] <= 0. || calbuf[14] <= 0.)
    ret_flag |= 1 << 7; // Rutgers calibration issue, bad pedestal peak channel
  
  if(calbuf[17] <= 0. || calbuf[18] <= 0.)
    ret_flag |= 1 << 8; // Rutgers calibration issue, bad pedestal right half peak channel
  
  if (calbuf[19] <= 0. || calbuf[20] <= 0.)
    ret_flag |= 1 << 9; // Rutgers calibration issue, bad 1-MIP peak fit number of degrees of freedom
  
  if(calbuf[21] <= 0. || calbuf[22] <= 0.)
    ret_flag |= 1 << 10; // Rutgers calibration issue, bad 1-MIP peak fit chi2
  
  // check if the counter is considered as working according to the
  // standard SD reconstruction  
  for (k = 0; k < 2; k++)
    {
      // check the peak channel of pedestal histogram
      if ((pchped[k] < 8) || (pchped[k] > (SDMON_NMONCHAN/2 - 8)))
	ret_flag |= 1 << 11;
      // check the peak channel of 1-MIP histogram
      pchmip[k] = (integer4)rint(12.*fadc_ped[k]+mip[k]);
      if ((pchmip[k] < 12) ||(pchmip[k] > (SDMON_NMONCHAN-12)))
	ret_flag |= 1 << 12;
      // check 1-MIP histogram fit number of degrees of freedom
      if (mftndof[k]<=0)
	ret_flag |= 1 << 13;
      // check 1-MIP histogram chi2 / dof
      if (mftchi2[k]/(real8)mftndof[k] > MAXMFTRCHI2)
	ret_flag |= 1 << 14;
      // check the FADC counts per VEM
      vem[k] = mip[k] * cos(M_PI/180.0*MEAN_MU_THETA);
      if (vem[k] < 1.0)
	ret_flag |= 1 << 15;
    }
  
  return ret_flag; // final answer on all of the checks
}

// 

/* Description of the meaning of the bad counter bit flag that's produced by failed_Ben_Stokes_D_Ivanov_Live_Counter_Criteria
   icrr_cal_mc:    Mask for the ICRR calibration checks done during Monte Carlo generation
   ru_cal_mc:      Mask for the Rutgers calibration checks done during Monte Carlo generation
   ru_cal_rc:      Mask for the Rutgers calibration checks done during event reconstruction
*/

static void init_bitf_masks(integer4 *icrr_cal_mc, integer4 *ru_cal_mc, integer4 *ru_cal_rc)
{
  // Checks done during Monte Carlo generation.  If either of these fail then
  // the counter is not used in simulation of the event.
  if(icrr_cal_mc)
    {
      (*icrr_cal_mc) = 0;
      (*icrr_cal_mc) |= 1 << 0;  // ICRR calibration issue, MC throwing, failed don't use criteria
      (*icrr_cal_mc) |= 1 << 1;  // ICRR calibration issue, MC throwing, Mev2pe problem
      (*icrr_cal_mc) |= 1 << 2;  // ICRR calibration issue, MC throwing, Mev2cnt problem
      (*icrr_cal_mc) |= 1 << 3;  // ICRR calibration issue, MC throwing, bad pedestal mean values
      (*icrr_cal_mc) |= 1 << 4;  // ICRR calibration issue, MC throwing, bad pedestal standard deviation
      (*icrr_cal_mc) |= 1 << 5;  // ICRR calibration issue, MC throwing, saturation information not available
    }
  if(ru_cal_mc)
    {
      (*ru_cal_mc) = 0;
      (*ru_cal_mc) |= 1 << 6;    // Rutgers calibration issue, MC throwing, bad mip values
      (*ru_cal_mc) |= 1 << 7;    // Rutgers calibration issue, MC throwing, bad pedestal peak channel
      (*ru_cal_mc) |= 1 << 8;    // Rutgers calibration issue, MC throwing, bad pedestal right half peak channel
      (*ru_cal_mc) |= 1 << 9;    // Rutgers calibration issue, MC throwing, bad 1-MIP peak fit number of degrees of freedom
      (*ru_cal_mc) |= 1 << 10;   // Rutgers calibration issue, MC throwing, bad 1-MIP peak fit chi2
    }
  // Checks done during event reconstruction.  If either of these fail then the counter
  // is not used in the event reconstruction.
  if(ru_cal_rc)
    {
      (*ru_cal_rc) = 0;
      (*ru_cal_rc) |= 1 << 11;   // Rutgers calibration issue, Reconstruction, peak channel of pedestal histogram
      (*ru_cal_rc) |= 1 << 12;   // Rutgers calibration issue, Reconstruction, peak channel of 1-MIP histogram
      (*ru_cal_rc) |= 1 << 13;   // Rutgers calibration issue, Reconstruction, 1-MIP histogram fit number of degrees of freedom
      (*ru_cal_rc) |= 1 << 14;   // Rutgers calibration issue, Reconstruction, 1-MIP histogram chi2 / dof
      (*ru_cal_rc) |= 1 << 15;   // Rutgers calibration issue, Reconstruction, FADC counts per VEM
    }
}

// Check if ICRR calibration fails during MC generation
integer4 is_icrr_cal_mc(integer4 bitf)
{
  integer4 icrr_cal_mc = 0;
  init_bitf_masks(&icrr_cal_mc,0,0);
  return (bitf & icrr_cal_mc);
}

// Check if rutgers calibration fails during MC generation
integer4 is_ru_cal_mc(integer4 bitf)
{
  integer4 ru_cal_mc = 0;
  init_bitf_masks(0,&ru_cal_mc,0);
  return (bitf & ru_cal_mc);
}

// Check if rutgers calibration fails during event reconstruction
integer4 is_ru_cal_rc(integer4 bitf)
{
  integer4 ru_cal_rc = 0;
  init_bitf_masks(0,0,&ru_cal_rc);
  return (bitf & ru_cal_rc);
}

// Check if a certain bit is set (for debugging purposes)
integer4 is_cal_bit(integer4 bitf, integer4 bit_num)
{
  integer4 bit_mask = (1 << bit_num);
  return (bitf & bit_mask);
}

void sdmc_calib_check_bitf(integer4 bsdinfo_bitf, 
			   integer4* not_working_according_to_sdmc_checks, 
			   integer4* not_working_according_to_sd_reconstruction)
{
  
  // Check if the counter is considered as not working according to the checks done in TA SD MC
  (*not_working_according_to_sdmc_checks) = is_icrr_cal_mc(bsdinfo_bitf) | is_ru_cal_mc(bsdinfo_bitf); 
  
  // Optionally, checks if the counter would be considered as not working in TA SD event reconstruction.
  // This is not used for the trigger verification but may be useful in other programs.
  if(not_working_according_to_sd_reconstruction)
    *not_working_according_to_sd_reconstruction = is_ru_cal_rc(bsdinfo_bitf);
}
