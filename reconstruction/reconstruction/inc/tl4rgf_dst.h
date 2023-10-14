/*
 *     Bank for TALE / MD 4 Ring Geometry
 *     Dmitri Ivanov (dmiivanov@gmail.com)
 *     Last modified: Dec 07, 2016

 */

#ifndef _TL4RGF_
#define _TL4RGF_

#define TL4RGF_BANKID  12509
#define TL4RGF_BANKVERSION   001

#ifdef __cplusplus
extern "C" {
#endif
integer4 tl4rgf_common_to_bank_ ();
integer4 tl4rgf_bank_to_dst_ (integer4 * NumUnit);
integer4 tl4rgf_common_to_dst_ (integer4 * NumUnit);	/* combines above 2 */
integer4 tl4rgf_bank_to_common_ (integer1 * bank);
integer4 tl4rgf_common_to_dump_ (integer4 * opt1);
integer4 tl4rgf_common_to_dumpf_ (FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* tl4rgf_bank_buffer_ (integer4* tl4rgf_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



#define TL4RGF_DST_GZ ".tl4rgf.dst.gz" /* output suffix */


// Maximum number of sites
#define TL4RGF_NFDSITE 2

// Station ID's used in this bank
#define TL4RGF_MD    4
#define TL4RGF_TALE  5

// Maxinum number of mirrors per site
#define TL4RGF_NMIR 14

// Number of tubes per mirror
#define TL4RGF_NTUBE_PMIR 256

// Maximum number of tubes per site
#define TL4RGF_NTUBE (TL4RGF_NMIR * TL4RGF_NTUBE_PMIR)

typedef struct
{

  // Basic event geometry information
  integer4 yymmdd;    // UTC date
  integer4 hhmmss;    // UTC time
  real8    t0;        // Time [us] when the shower axis crosses CLF z=0 plane with respect to leading edge of GSP pulse
  real4    theta;     // Event zenith angle (where event comes from) [Degree] in CLF frame, Z=Up
  real4    phi;       // Event azimuthal angle (where event comes from) [Degree] in CLF frame, X=East
  real4    xycore[2]; // Event core position (where event axis crosses CLF z=0 plane) in CLF [m]
  real4    bdist;     // Distance of the shower core from the TALE SD array edge [m]
  
  integer1 nfdsite;                      // number of sites, 2 is maxium
  integer1 site_id[TL4RGF_NFDSITE];      // TL4RGF_MD = MD, TL4RGF_TALE = TALE
  integer1 is_frame[TL4RGF_NFDSITE];     // 1 = if rp,psi,trp fit was done in this site's frame
  //                                        0 if this site extends the fit in other site
  real4    sdp_n[TL4RGF_NFDSITE][3];     // shower detector plane normal vectors for each site
  real4    sdp_n_chi2[TL4RGF_NFDSITE];   // effective chi2 of the shower-detector plane of the site
  integer4 ngt_sdp_n[TL4RGF_NFDSITE];    // number of good tubes in the shower-detector plane fit
  real4    psi[TL4RGF_NFDSITE];          // angle in the shower detector plane for each site [Degree]
  real4    rp[TL4RGF_NFDSITE];           // distance of closest approach for each site [m]
  real4    trp[TL4RGF_NFDSITE];          // time at the distance of closest approach for each site [m]
  real4    tf_chi2[TL4RGF_NFDSITE];      // time fit chi2 contribution of each site
  integer4 ngt_tf[TL4RGF_NFDSITE];       // number of good tubes in the time fit
  real4    tracklength[TL4RGF_NFDSITE];  // tracklength [Degree] for each site
  real4    crossingtime[TL4RGF_NFDSITE]; // crossing time [us] for each site
  real4    npe[TL4RGF_NFDSITE];          // total npe for each site
  real4    npe_edge[TL4RGF_NFDSITE];     // total npe for each site from the edge tubes
  real4    d_psi;                        // fit uncertainty on psi in the primary site's frame [Degree]
  real4    d_rp;                         // fit uncertainty on rp in the primary site's frame [m]
  real4    d_trp;                        // fit uncertainty on trp in the primary site's frame [us]
  
  // Tube information
  integer4   ntube[TL4RGF_NFDSITE];                              // number of tubes in the site
  integer4   tube_raw_ind[TL4RGF_NFDSITE][TL4RGF_NTUBE];         // index in hraw1 (MD) or fpho1 (TALE) bank for the tube
  integer4   tube_fraw1_ind[TL4RGF_NFDSITE][TL4RGF_NTUBE];       // only TALE, 10000*fraw1_mirror_index+fraw1_mirror_tube_index, 0 for MD
  integer4   tube_stpln_ind[TL4RGF_NFDSITE][TL4RGF_NTUBE];       // index in stpln bank for the tube (pass1 plane fit/pattern recognition)
  integer1   tube_ig_sdp_n[TL4RGF_NFDSITE][TL4RGF_NTUBE];        // 1=tube was used in the sdp_n fit, 0 = tube was not used in the sdp_n fit
  integer1   tube_ig_tf[TL4RGF_NFDSITE][TL4RGF_NTUBE];           // 1=tube was used in the time fit, 0 = tube was not used in the time fit
  integer1   tube_sat_flag[TL4RGF_NFDSITE][TL4RGF_NTUBE];        // 0=not saturated, 
                                                                 // 1=MD saturated, 3=TALE saturated not recovered
                                                                 // 33=TALE saturated and recovered
  integer1   tube_edge[TL4RGF_NFDSITE][TL4RGF_NTUBE];            // 1=tube was on the edge, 0 tube was not on the edge
  integer4   tube_mir_and_tube_id[TL4RGF_NFDSITE][TL4RGF_NTUBE]; // mirror_id * 1000 + tube_id
  real4      tube_azm[TL4RGF_NFDSITE][TL4RGF_NTUBE];             // tube direction azimuth, CCW from X [Degree] (X=East from mirror,Y=North,Z=Up)
  real4      tube_ele[TL4RGF_NFDSITE][TL4RGF_NTUBE];             // tube direction elevation [Degree] (X=East from mirror,Y=North,Z=Up)
  real4      tube_npe[TL4RGF_NFDSITE][TL4RGF_NTUBE];             // tube NPE
  real4      tube_gfc[TL4RGF_NFDSITE][TL4RGF_NTUBE];             // tube calibration gain factor that was used, if relevant
  //                                                                (qdcb to NPE for MD, FADC to NPE for TALE, etc)
  real4      tube_sig2noise[TL4RGF_NFDSITE][TL4RGF_NTUBE];       // tube signal to noise ratio
  real4      tube_ped_or_thb[TL4RGF_NFDSITE][TL4RGF_NTUBE];      // tube pedestal [FADC/100ns] for TALE, or threshold [mV] for MD 
  real4      tube_tm[TL4RGF_NFDSITE][TL4RGF_NTUBE];              // tube time [us] with respect to t0
  real4      tube_tslew[TL4RGF_NFDSITE][TL4RGF_NTUBE];           // time slewing correction that was applied to tube_tm, [us]
  real4      tube_tmerr[TL4RGF_NFDSITE][TL4RGF_NTUBE];           // uncertainty on time, [us]
  real4      tube_tmfit[TL4RGF_NFDSITE][TL4RGF_NTUBE];           // time predicted by the fit [us]
  real4      tube_palt[TL4RGF_NFDSITE][TL4RGF_NTUBE];            // tube altitude in site's shower detector plane [Degree]
  real4      tube_pazm[TL4RGF_NFDSITE][TL4RGF_NTUBE];            // tube azimuthal angle in site's shower-detector plane [Degree]
  
  // Single SD information
  integer4   sd_xxyy;                                            // position ID if SD used in the fit, 0 = SD is not used in the fit
  integer4   sd_ind;                                             // index in tlfptn bank for the SD
  real4      sd_vem;                                             // signal size of the SD [VEM]
  real4      sd_tm;                                              // time of the single SD [us] with respect to t0
  real4      sd_tmerr;                                           // uncertainty on time of the single SD [us]
  real4      sd_tmfit;                                           // time for the single SD predicted by the fit [us]
  
  
} tl4rgf_dst_common;

extern tl4rgf_dst_common tl4rgf_;

#endif
