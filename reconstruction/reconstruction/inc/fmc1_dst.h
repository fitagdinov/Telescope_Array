/* 
 *     fmc1 bank
 *     MRM July 27
 */

/*     bank bookkeeping */

#ifndef _FMC1_
#define _FMC1_

#define   FMC1_BANKID      12100 
#define   FMC1_BANKVERSION   002

/*     C functions */

#ifdef __cplusplus
extern "C" {
#endif
integer4 fmc1_common_to_bank_();
integer4 fmc1_bank_to_dst_(integer4 *NumUnit);
integer4 fmc1_common_to_dst_(integer4 *NumUnit);
integer4 fmc1_bank_to_common_(integer1 *bank);
integer4 fmc1_common_to_dump_(integer4 *long_output);
integer4 fmc1_common_to_dumpf_(FILE* fp,integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* fmc1_bank_buffer_ (integer4* fmc1_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


/*     Parameters */

#define fmc1_nhit_mir_max 320
#define fmc1_nmir_max 20

/*     Common Bank Variables */

typedef struct {
  real4 the;
  real4 phi;
  real4 agam;
  real4 energy;
  real4 tau0;
  real4 nmax;
  real4 xmax;
  real4 pfwd;
  real4 a;
  real4 rpuv[3];
  real4 shwn[3];
  real4 u[3];
  real4 rp_true;
  real4 psi_true;
  real4 t0_true;
  real4 R_impact;
  real4 phi_impact;
  real4 R_shwmax;       /* dist (km) to shower max */
  real4 the_shwmax;     /* viewing angle (degrees) at shower max */
  real4 R_viewmax;      /* dist (km) to max visible signal */
  real4 R_viewfirst;    /* dist (km) to first visible signal */
  real4 R_viewlast;     /* dist (km) to last visible signal */
                        /* Visible defined as npe/degree > 10.0 */
  real4 hal;            /* aerosol horizontal attenuation length */
  real4 vsh;            /* aerosol vertical scale height */
  real4 mlh;            /* aerosol mixing layer height */
  real4 atm0;           /* future atm parameter */
  real4 atm1;           /* future atm parameter */
  real4 atm2;           /* future atm parameter */
  real4 tav[fmc1_nmir_max][fmc1_nhit_mir_max];
  integer4 npe[fmc1_nmir_max][fmc1_nhit_mir_max];
  integer4 imir[fmc1_nmir_max];
  integer4 npe_mir[fmc1_nmir_max];
  integer4 ntube[fmc1_nmir_max];
  integer4 itube[fmc1_nmir_max][fmc1_nhit_mir_max];
  integer4 nmir;
  integer4 event_num;
  integer4 version;
} fmc1_dst_common;

extern fmc1_dst_common fmc1_;

#endif

