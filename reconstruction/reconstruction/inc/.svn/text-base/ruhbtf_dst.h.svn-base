/*
 *     Bank for hybrid time fit
 *     Dmitri Ivanov (ivanov@physics.rutgers.edu)
 *     Sep 16, 2009
 *     Last : Sep 16, 2009
 */

#ifndef _RUHBTF_
#define _RUHBTF_

#define RUHBTF_BANKID  30101
#define RUHBTF_BANKVERSION   000



#ifdef __cplusplus
extern "C" {
#endif
integer4 ruhbtf_common_to_bank_ ();
integer4 ruhbtf_bank_to_dst_ (integer4 * NumUnit);
integer4 ruhbtf_common_to_dst_ (integer4 * NumUnit);	/* combines above 2 */
integer4 ruhbtf_bank_to_common_ (integer1 * bank);
integer4 ruhbtf_common_to_dump_ (integer4 * opt1);
integer4 ruhbtf_common_to_dumpf_ (FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* ruhbtf_bank_buffer_ (integer4* ruhbtf_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


#define RUHBTF_NSD 0x100 // max. number of SDs in hybrid time fit
#define RUHBTF_NTB 0xc00 // max. number of tubes in hybrid time fit
#define RUHBTF_DST_GZ ".ruhbtf.dst.gz" /* output suffix */


typedef struct
{
  real8 tbuv_fd[RUHBTF_NTB][3]; // tube pointing direction in FD frame
  real8 sdpos_clf[RUHBTF_NSD][3]; // SD position in CLF frame, [meters]
  real8 sdpos_fd[RUHBTF_NSD][3]; // SD position in FD frame, [meters]
  real8 sd_xyzcdist[RUHBTF_NSD][3]; // SD distance from core in CLF plane, [meters]
  real8 sdsa_pos_clf[RUHBTF_NSD][3]; // corresponding shower axis point in CLF frame, [meters]
  real8 sdsa_pos_fd[RUHBTF_NSD][3]; // corresponding shower axis point in FD frame, [meters]
  real8 tbnpe[RUHBTF_NTB]; // integrated pulse above pedestal
  real8 tbtime[RUHBTF_NTB]; // weighted average pulse time
  real8 tbtime_rms[RUHBTF_NTB]; // weighted average pulse time rms
  real8 tbtime_err[RUHBTF_NTB]; // full error on tube time
  real8 tbsigma[RUHBTF_NTB]; // tube significance
  
 
  real8 tb_palt[RUHBTF_NTB]; // tube altitude in SDP, [degree]
  real8 tb_pazm[RUHBTF_NTB]; // tube azimuth in SDP, [degree]
  real8 tb_tvsa_texp[RUHBTF_NTB]; // expected tube time in t vs angle plot
  
  
  real8 sdrho[RUHBTF_NSD]; // Charge density for a given SD, [VEM/m^2]
  real8 sdtime[RUHBTF_NSD]; // SD time, [uS]
  real8 sdetime[RUHBTF_NSD]; // resolution on SD leading edge time, [uS]
  
  
  real8 sd_cdist[RUHBTF_NSD]; // SD distance from core in CLF plane, [meters]
  real8 sd_adist[RUHBTF_NSD]; // SD distance from core along the shower axis, [meters]
  real8 sd_sdist[RUHBTF_NSD]; // SD pependicular distance from shower axis, [meters]
  real8 sd_ltd[RUHBTF_NSD]; // Modified Linsley time delay, [uS]
  real8 sd_lts[RUHBTF_NSD]; // Linsley Td fluctuation, [uS]
  real8 sdsa_tm[RUHBTF_NSD]; // Shower front plane time on the shower axis, [uS]
  real8 sd_timerr[RUHBTF_NSD]; // Full SD time uncertainty, [uS]
  real8 sdsa_fddist[RUHBTF_NSD]; // corresponding shower axis point distance to FD, [meters]
  real8 sdsa_fdtime[RUHBTF_NSD]; // time when light from the corresponding point reaches FD, [uS]
  real8 sd_tvsa_texp[RUHBTF_NSD]; // expected SD time on time vs angle plot
  real8 sdsa_palt[RUHBTF_NSD]; // corresponding shower axis point altitude in SDP, [degree]
  real8 sdsa_pazm[RUHBTF_NSD]; // corresponding shower axis point azimuth in SDP, [degree]
  

  real8 axi_clf[3]; // Shower direction axis ( along shower propagation) in CLF frame
  real8 axi_fd[3]; // Shower direction axis ( along shower propagation ) in FD frame
  real8 core_fd[3]; // Shower core position in FD frame, [meters]
  real8 core_fduv[3]; // Unit vector pointing from FD to the fitted core
  real8 sdp_n[3]; // Shower-Detector plane normal vector
  real8 sd_cog_clf[3]; // SD charge center-of-gravity core position, CLF frame, [meters]
  real8 fd_cogcore[3]; // SD charge center-of-gravity core in FD frame, [meters]
  real8 r_sd[2]; // core position in SD frame, [1200m]
  


  /******** ACTUAL FIT PARAMETERS (BELOW) ********************/
  
  // Shower propagation axis
  real8 theta;  // Zenith angle, CLF frame, [Degree]
  real8 dtheta;
  
  real8 phi;    // Azimuth angle, CLF frame, [Degree] ( X=East ) 
  real8 dphi;
  
  // Shower core in CLF frame, [meters]. z-component is ALWAYS zero.
  real8 xcore;
  real8 dxcore;
  real8 ycore;
  real8 dycore;
  
  real8 t0;   // Time when the shower core hits the ground, [uS]
  real8 dt0;

  /******** ACTUAL FIT PARAMETERS (ABOVE) ********************/


  // Calculated while fitting:
  real8 cdist_fd; // Distance from fitted shower core to FD, [meters]
  real8 fd_cogcore_dist; // Distance from FD to SD charge center-of-gravity core, [meters]  
  real8 psi; // Angle in shower-detector plane, [Degree]
  real8 rp; // Distance of closest approach to FD, [meters]
  real8 t_rp; // Time when the shower front is at rp
  real8 chi2; // FCN at its minimum
  
  
  real8 tref; // Time refference, second + second fraction since midnight
  
  integer4 ifdplane[RUHBTF_NTB]; // fdplane index of the tube
  integer4 irusdgeom[RUHBTF_NSD]; // rusdgeom index of the SD
  integer4 xxyy[RUHBTF_NSD]; // SD position ID
  integer4 yymmdd; // date
  integer4 fdsiteid; // 0-BR, 1-LR
  integer4 sdsiteid; // 0-BR, 1-LR, 2-SK, 3-BRLR,4-BRSK,5-LRSK,6-BRLRSK
  integer4 ntb; // Number of tubes
  integer4 nsd; // Number of SD points
  integer4 ndof; // number of degrees of freedom in the fit
  
} ruhbtf_dst_common;

extern ruhbtf_dst_common ruhbtf_;

#endif
