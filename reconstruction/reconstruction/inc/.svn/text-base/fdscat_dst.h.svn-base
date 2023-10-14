/* 
 *     fdscat bank
 *     SRS 4.27.09
 *
 *     version 1
 *     expanded to include multiple vaod measurements per bank instance
 *     WFH 20 July 2016
 */

/*     bank bookkeeping */
#ifndef _FDSCAT_
#define _FDSCAT_

#define FDSCAT_BANKID       12410
#define FDSCAT_BANKVERSION      1 

/* maximum number of measurements stored for an individual entry, i.e., the
 * number of sites at which VAOD was measured */
#define FDSCAT_MAX_MEAS 16

/* list of accepted sites where measurements are made */
#define FDSCAT_SITE_UNDEF -1
#define FDSCAT_SITE_BR     0
#define FDSCAT_SITE_LR     1
#define FDSCAT_SITE_MD     2
#define FDSCAT_SITE_TL     3

/* list of quality codes */
#define FDSCAT_QUAL_UNDEF -4    /* catchall bad/missing/invalid measurment */
#define FDSCAT_QUAL_MISS  -3    /* missing measurement for example (e.g., detector inoperable during this period) */
#define FDSCAT_QUAL_EXCESS_VAOD_DIFF -2  /* difference between the two sites VAOD is large and should be not be used in analysis. (diff > 0.02) */
#define FDSCAT_QUAL_EXCESS_AEROSOL -1 /* sites have too large vaod (vaod > 0.1) */
#define FDSCAT_QUAL_BAD    0    /* measurement performed but quality is bad (e.g., excessive RMS) */
#define FDSCAT_QUAL_GOOD   1    /* measurement performed and is valid */

/* desciption of what the scalar summary values mean or how they were derived */
#define FDSCAT_METH_UNDEF     0  /* method not defined or specified */
#define FDSCAT_METH_AVG_ALL   1  /* weighted average of all sites */
#define FDSCAT_METH_AVG_BRLR  2  /* weighted average of only BR+LR */
#define FDSCAT_METH_BEST_RMS  3  /* used the site with the smallest RMS */
#define FDSCAT_METH_BEST_VAOD 4  /* used the site with the "best" vaod */

/*     C functions */
#ifdef __cplusplus
extern "C" {
#endif
integer4 fdscat_common_to_bank_();
integer4 fdscat_bank_to_dst_(integer4 *NumUnit);
integer4 fdscat_common_to_dst_(integer4 *NumUnit);
integer4 fdscat_bank_to_common_(integer1 *bank);
integer4 fdscat_common_to_dump_(integer4 *long_output);
integer4 fdscat_common_to_dumpf_(FILE* fp,integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* fdscat_bank_buffer_ (integer4* fdscat_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct {
  /* define a range of times for which this entry is valid. though these are
   * doubles, time here has traditionally been set as absolute time such
   * as a time_t value (see ctime (3)). ensure whatever code you use to search
   * these values is actually comparing time in the same units. */
  real8 startTime;  /* seconds since the Epoch */
  real8 endTime;    /* seconds since the Epoch */

  /* this bank was originally written to store a single measurement of aerosol
   * scattering properties for each entry. CLF analysis is now done using
   * multiple detectors for each time period. I'll extend this structure to
   * include the data from individual measurements, but to facilitate
   * backwards compatibilty with existing code, the original single measurement
   * will remain as a "summary" of any other measurements stored for each
   * time periods entry. for example if BR, LR, and MD vaod measurements
   * all exist for a single time period, the following elements of this bank
   * can represent the average of those three. code your applications
   * appropriately to use this information. */

  /* VAOD is used to describe the aerosol scattering properties of the
   * atmosphere. given an aerosol phase function (angular scattering 
   * function), a scale height, and horizontal attenuation length,
   * a transmission coefficient for fluorescence light from the aerosol
   * component of the atmosphere can be calculated. by measuring a known
   * calibrated light source, such as the CLF laser, since it's energy
   * (and therefore photon yield) and geometry to the receiver is known,
   * the amount of light scattered due to aerosols can be calculated.
   * I believe we always assume that the scale height is 1 km, so the only
   * free parameter to determine from CLF measurements is the horizontal
   * attenutation length. combining the scale height and attenuation length
   * provides VAOD.
   *
   * VAOD is calculated as schght/hzalen. in this bank express both in meters,
   * vaod is unitless.
   *
   * typical static values of these parameters used at TA are
   * schght = 1000 m, hzalen = 25000 m, vaod = 0.040
   * schght = 1000 m, hzalen = 29000 m, vaod = 0.034
   */
  real8 hzalen;  /* HoriZontal Attenuation LENgth (meters) */
  real8 vaodep;  /* Vertical Aerosol Optical DEPth  */
  real8 schght;  /* SCale HeiGHT (meters) */

  /* uniqID has traditionally been a time_t value for when the dst was created
   * or the data it stored was generated. here uniqID will be the creation
   * date of the DST */
  integer4 uniqID;  /* time_t value when the dst was created */

  /* the vaod database (independent of the DST) has it's own ID and
   * creation/modification date. include that information here as well */
  integer4 vaodID;   /* VAOD database version */
  integer4 vaodDate; /* VAOD database creation date (time_t) */

  real8 vaodep_rms;  /* vaod rms */


  /* there are times when there are measured values for BR and LR vaod, but
   * the quality code of the combined measurement will be marked as bad. see
   * JiHee's VAOD readme for details, but here are some reasons why this
   * may occur:
   *
   * 1) if VAOD for both sites is = 1 in the database it means both measurements
   * are missing
   *
   * 2) if both sites have VAOD > 0.1, then JiHee recommends to not use for
   * analysis, i.e., the measurement is performed, but the value is considered
   * bad.
   *
   * 3) if the difference between BR and LR VAOD > 0.02, then the combined
   * measurement is marked bad.
   *
   * users can code their software to utilize the individual values of vaod
   * found in the *_meas variables for these bad cases if they wish.
   */

  /* use FDSCAT_QUAL_* quality codes defined above */
  integer4 qual;      /* measurement quality */

  /* use FDSCAT_METH_* defined above */
  integer4 method;   /* how are the scalars above calculated */

  integer4 nMeas;  /* number of filled entries in the following arrays */

  /* use FDSCAT_SITE_* variables to specify which site is measured */
  real8 hzalen_meas[FDSCAT_MAX_MEAS];   /* hzalen for given site */
  real8 vaodep_meas[FDSCAT_MAX_MEAS];   /* vaodep for given site */
  real8 vaodep_rms_meas[FDSCAT_MAX_MEAS]; /* vaod rms for given site */
  real8 schght_meas[FDSCAT_MAX_MEAS];   /* scale height for given site */

  integer4 site_meas[FDSCAT_MAX_MEAS];  /* measurement done using which site */
  integer4 qual_meas[FDSCAT_MAX_MEAS];  /* quality/status code for this site */

} fdscat_dst_common;

extern fdscat_dst_common fdscat_;

#endif
