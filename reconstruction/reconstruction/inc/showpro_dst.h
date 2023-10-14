#ifndef _SHOWPRO_
#define _SHOWPRO_

#define MAX_LONG_SLICES 1024

#define SHOWPRO_BANKID 13999
#define SHOWPRO_BANKVERSION 000

#ifdef __cplusplus
extern "C" {
#endif
integer4 showpro_common_to_bank_ ();
integer4 showpro_bank_to_dst_ (integer4 * NumUnit);
integer4 showpro_common_to_dst_ (integer4 * NumUnit);  /* combines above 2 */
integer4 showpro_bank_to_common_ (integer1 * bank);
integer4 showpro_common_to_dump_ (integer4 * opt1);
integer4 showpro_common_to_dumpf_ (FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* showpro_bank_buffer_ (integer4* showpro_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct {
  integer4 event_num; 
  integer4 parttype;
  
  // These are used to scale x and nmax from rusdmc
  real8 eScale; // 
  real8 zen; //
  
  // the next three values make a scatter plot of the longitudinal profile
  integer4 nslices;
  real4 x[MAX_LONG_SLICES]; // slant depth [gr/cm^2]
  real4 gammas[MAX_LONG_SLICES]; // number of gammas at X
  real4 electrons[MAX_LONG_SLICES];  // number of electrons at X
  real4 positrons[MAX_LONG_SLICES]; // number of positrons at X
  real4 muonPlus[MAX_LONG_SLICES]; // number of muon plus at X
  real4 muonMinus[MAX_LONG_SLICES]; // number of muon minus at X
  real4 hadrons[MAX_LONG_SLICES]; // number of hadrons at X
  real4 charged[MAX_LONG_SLICES]; // number of charged particles at X
  real4 nuclei[MAX_LONG_SLICES]; // number of nuclei
  real4 cherenkov[MAX_LONG_SLICES]; // cherenkov photons? always zero?? 
  
  // energy deposition quantities
  integer4 nDepSlices;
  real4 xDep[MAX_LONG_SLICES]; // slant depth for energy deposition
  real4 gammaDep[MAX_LONG_SLICES]; //
  real4 emIoniz[MAX_LONG_SLICES];
  real4 emCut[MAX_LONG_SLICES];
  real4 muIoniz[MAX_LONG_SLICES];
  real4 muCut[MAX_LONG_SLICES];
  real4 hadrIoniz[MAX_LONG_SLICES];
  real4 hadrCut[MAX_LONG_SLICES];
  real4 neutrino[MAX_LONG_SLICES];
  real4 total[MAX_LONG_SLICES]; // particles lost
  
   //FIT OF THE HILLAS CURVE   N(T) = P1*((T-P2)/(P3-P2))**((P3-P2)/(P4+P5*T+P6*T**2)) * 
  //                                      EXP((P3-T)/(P4+P5*T+P6*T**2))
  real4 nmax; // P1
  real4 x0; // P2
  real4 xmax; // P3
  real4 lambda0; // P4 
  real4 lambda1; // P5
  real4 lambda2; // P6
  real4 chi2; 
  real4 aveDev;
  
} showpro_dst_common;

extern showpro_dst_common showpro_;
#endif /*SDMCLP_DST_H_*/
