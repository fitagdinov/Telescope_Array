/* Created 2018/03/09 DI */

/* Parametrization of the atmosphere (GDAS, Radiosonde, etc) using
   the layer model that's found in CORSIKA:
   T(h) = a_i + b_i * exp(h/c_i), i=1..NHEIGHT (NHEIGHT = 5 is the curernt maximum)
   Allows easy and accurate determination of mass overburden for a given height and vice versus.
*/
   

#ifndef _ATMPAR_DST_
#define _ATMPAR_DST_

#define ATMPAR_BANKID		13113
#define ATMPAR_BANKVERSION	0

#define  ATMPAR_NHEIGHT         5 /* maximum number of layer boundary points */
#define  ATMPAR_GDAS_MODELID    0 /* ID of the atmospheric model that corresponds to GDAS atmosphere */

#ifdef __cplusplus
extern "C" {
#endif
integer4 atmpar_common_to_bank_ ();
integer4 atmpar_bank_to_dst_ (integer4 * NumUnit);
integer4 atmpar_common_to_dst_ (integer4 * NumUnit);   /* combines above 2 */
integer4 atmpar_bank_to_common_ (integer1 * bank);
integer4 atmpar_common_to_dump_ (integer4 * opt1);
integer4 atmpar_common_to_dumpf_ (FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* atmpar_bank_buffer_ (integer4* atmpar_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


/* C and C++ routines that give mass overburden and height above sea level */
#ifdef __cplusplus
extern "C" real8 atmpar_h2mo(real8 h);
extern "C" real8 atmpar_h2mo_deriv(real8 h); /* derivative of mass overburden with respect to h [g/cm^3] */
extern "C" real8 atmpar_mo2h(real8 mo);
#else
real8 atmpar_h2mo(real8 h);       /* mass overburden in [g/cm^2] at a given height above sea level in [cm] */
real8 atmpar_h2mo_deriv(real8 h); /* derivative of mass overburden with respect to h [g/cm^3] */
real8 atmpar_mo2h(real8 mo);      /* height above sea level in [cm] for a given mass overburden in [g/cm^2] */

#endif

typedef struct {
  uinteger4 dateFrom;          /* sec from 1970/1/1 */
  uinteger4 dateTo;            /* sec from 1970/1/1 */
  integer4  modelid;           /* number of models  */
  integer4  nh;                /* number of heights that distinquish layers */
  real8     h[ATMPAR_NHEIGHT]; /* layer transition heights [cm] */
  real8     a[ATMPAR_NHEIGHT]; /* parameters of the T(h) = a_i + b_i * exp(h/c_i) model determined by the fit */
  real8     b[ATMPAR_NHEIGHT];
  real8     c[ATMPAR_NHEIGHT];
  real8     chi2;              /* quality of the fit */
  integer4  ndof;              /* number of degees of freedom in the fit = number of points - number of fit parameters */
} atmpar_dst_common;

extern atmpar_dst_common atmpar_;

#endif
