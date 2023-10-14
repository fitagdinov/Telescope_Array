#ifndef _TRUMPMC_DST_
#define _TRUMPMC_DST_

/*
 * TRUMP MC DST Bank.
 * DRB 20 January 2009
 */

#define TRUMPMC_BANKID      12803
#define TRUMPMC_BANKVERSION     1

// Parameters 
#define TRUMPMC_MAXSITES     3
#define TRUMPMC_MAXDEPTHS  200
#define TRUMPMC_MAXMIRRORS  14
#define TRUMPMC_MAXTUBES   256

// Short-hand parameters
#define MXS TRUMPMC_MAXSITES
#define MXD TRUMPMC_MAXDEPTHS
#define MXM TRUMPMC_MAXMIRRORS
#define MXT TRUMPMC_MAXTUBES

#ifdef __cplusplus
extern "C" {
#endif
integer4 trumpmc_common_to_bank_();
integer4 trumpmc_bank_to_dst_(integer4 *NumUnit);
integer4 trumpmc_common_to_dst_(integer4 *NumUnit);
integer4 trumpmc_bank_to_common_(integer1 *bank);
integer4 trumpmc_common_to_dump_(integer4 *long_output);
integer4 trumpmc_common_to_dumpf_(FILE* fp,integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* trumpmc_bank_buffer_ (integer4* trumpmc_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct {
  // Shower generic information
  real4    impactPoint[3];  // Vector from origin to impact point (m)
  real4    showerVector[3]; // Unit vector of shower propagation vector
  real4    energy;          // Energy of primary (eV)
  integer4 primary;         // Particle type of primary
  real4    ghParm[4];       // G-H parameters: x0, xmax, nmax, lambda (g/cm2 or part.)

  // Site specific information: geometry
  integer4 nSites;               // Number of sites
  integer4 siteid[MXS];          // site ID [BR=0, LR=1, MD=2]
  real4    siteLocation[MXS][3]; // Position of site wrt origin(m)
  real4    psi[MXS];             // Psi angle as seen from site
  real4    rp[MXS][3];           // Vector from site to Rp point (m)

  // Depth information: Flux (photons/m2/rad)
  integer4 nDepths;                 // Number of depth points this mirror
  real4    depth[MXD];              // Depth (g/cm2)
  integer4 nMirrors[MXS];           // Number of mirrors which see track
  integer4 mirror[MXS][MXM];        // Mirror number
  real4    fluoFlux[MXS][MXM][MXD]; // Fluorescence phot
  real4    aeroFlux[MXS][MXM][MXD]; // Aerosol/Mie scattered phot
  real4    raylFlux[MXS][MXM][MXD]; // Rayleigh scattered phot
  real4    dirCFlux[MXS][MXM][MXD]; // Direct Cerenkov phot
  
  // Tube information
  integer4 totalNPEMirror[MXS][MXM]; // Total number of NPE, this mirror
  integer4 nTubes[MXS][MXM];         // Number of tubes, this mirror, with NPE
  integer4 tube[MXS][MXM][MXT];      // Tube number
  real4    aveTime[MXS][MXM][MXT];   // Average time of PE in this tube
  integer4 totalNPE[MXS][MXM][MXT];  // Number of PE in this tube

  // new in version 1: core time
  integer4 julian;
  integer4 jsec;
  integer4 nano;
  
} trumpmc_dst_common;

extern trumpmc_dst_common trumpmc_;

integer4 trumpmc_struct_to_abank_(trumpmc_dst_common *trumpmc, integer1* (*pbank), integer4 id, integer4 ver);
integer4 trumpmc_abank_to_dst_(integer1 *bank, integer4 *unit);
integer4 trumpmc_struct_to_dst_(trumpmc_dst_common *trumpmc, integer1* (*pbank), integer4 *unit, integer4 id, integer4 ver);
integer4 trumpmc_abank_to_struct_(integer1 *bank, trumpmc_dst_common *trumpmc);
integer4 trumpmc_struct_to_dump_(trumpmc_dst_common *trumpmc, integer4 *opt);
integer4 trumpmc_struct_to_dumpf_(trumpmc_dst_common *trumpmc, FILE *fp, integer4 *opt);

#undef MXS
#undef MXD
#undef MXM
#undef MXT

#endif
