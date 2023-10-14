// Description of the meaning of the bad counter bit flag that's produced by sdmc_calib_check program

int icrr_cal_mc = 0;  // Mask for the ICRR calibration checks done during Monte Carlo generation
int ru_cal_mc  = 0;  // Mask for the Rutgers calibration checks done during Monte Carlo generation
int ru_cal_rc  = 0;  // Mask for the Rutgers calibration checks done during event reconstruction

void sdmc_calib_check_bitf()
{
  // Checks done during Monte Carlo generation.  If either of these fail then
  // the counter is not used in simulation of the event.  
  icrr_cal_mc = 0;
  icrr_cal_mc |= 1 << 0;  // ICRR calibration issue, MC throwing, failed don't use criteria
  icrr_cal_mc |= 1 << 1;  // ICRR calibration issue, MC throwing, Mev2pe problem
  icrr_cal_mc |= 1 << 2;  // ICRR calibration issue, MC throwing, Mev2cnt problem
  icrr_cal_mc |= 1 << 3;  // ICRR calibration issue, MC throwing, bad pedestal mean values
  icrr_cal_mc |= 1 << 4;  // ICRR calibration issue, MC throwing, bad pedestal standard deviation
  icrr_cal_mc |= 1 << 5;  // ICRR calibration issue, MC throwing, saturation information not available  
  ru_cal_mc = 0;
  ru_cal_mc |= 1 << 6;    // Rutgers calibration issue, MC throwing, bad mip values
  ru_cal_mc |= 1 << 7;    // Rutgers calibration issue, MC throwing, bad pedestal peak channel
  ru_cal_mc |= 1 << 8;    // Rutgers calibration issue, MC throwing, bad pedestal right half peak channel
  ru_cal_mc |= 1 << 9;    // Rutgers calibration issue, MC throwing, bad 1-MIP peak fit number of degrees of freedom
  ru_cal_mc |= 1 << 10;   // Rutgers calibration issue, MC throwing, bad 1-MIP peak fit chi2
  
  // Checks done during event reconstruction.  If either of these fail then the counter
  // is not used in the event reconstruction.
  ru_cal_rc = 0;
  ru_cal_rc |= 1 << 11;   // Rutgers calibration issue, Reconstruction, peak channel of pedestal histogram
  ru_cal_rc |= 1 << 12;   // Rutgers calibration issue, Reconstruction, peak channel of 1-MIP histogram
  ru_cal_rc |= 1 << 13;   // Rutgers calibration issue, Reconstruction, 1-MIP histogram fit number of degrees of freedom
  ru_cal_rc |= 1 << 14;   // Rutgers calibration issue, Reconstruction, 1-MIP histogram chi2 / dof
  ru_cal_rc |= 1 << 15;   // Rutgers calibration issue, Reconstruction, FADC counts per VEM
}

// Check if ICRR calibration fails during MC generation
bool is_icrr_cal_mc(int bitf)
{
  if(!icrr_cal_mc)
    sdmc_calib_check_bitf();
  return (bitf & icrr_cal_mc);
}

// Check if rutgers calibration fails during MC generation
bool is_ru_cal_mc(int bitf)
{
  if(!ru_cal_mc)
    sdmc_calib_check_bitf();
  return (bitf & ru_cal_mc);
}

// Check if rutgers calibration fails during event reconstruction
bool is_ru_cal_rc(int bitf)
{
  if(!ru_cal_rc)
    sdmc_calib_check_bitf();
  return (bitf & ru_cal_rc);
}

bool is_cal_bit(int bitf, int bit_num)
{
  int bit_mask = (1 << bit_num);
  return (bitf & bit_mask);
}
