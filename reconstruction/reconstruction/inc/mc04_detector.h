/*  mc04_detector.h
 *
 * $Source: $
 * $Log: $
 *
 * detector definition and id's
 *
 */

#ifndef MC04_DETECTOR
#define MC04_DETECTOR

#define MC04_MAXEYE  8
#define MC04_MAXMIR  128
#define MC04_MAXTUBE (MC04_MAXMIR * 256)

/* mir area and ray tracing */
#define MC04_MIR_TYPE_HR  1    /* HiRes mirror */
#define MC04_MIR_TYPE_TA  2    /* Telescope Array mirror */
#define MC04_MIR_TYPE_TP  3    /* "Tower of Power" mirror */

#define MC04_MIR_ELECTRONICS_HR_SAH   3 /* HiRes S/H (REV 3) */
#define MC04_MIR_ELECTRONICS_HR_SAH4  4 /* HiRes S/H (REV 4) */
#define MC04_MIR_ELECTRONICS_TA_FADC  9 /* Telescope Array FADC */
#define MC04_MIR_ELECTRONICS_HR_FADC  5 /* HiRes (TALE) FADC ("REV 5") */

/* eye id's */

#define MC04_EYE1  1
#define MC04_EYE2  2
#define MC04_EYE3  3
#define MC04_EYE4  4  // DO NOT USE
#define MC04_EYE5  5  // DO NOT USE
#define MC04_EYE6  6
#define MC04_EYE7  7
#define MC04_EYE8  8
// NOTE:: In doing MD/TALE geometry reconstruction utafd uses the array index
// "ieye" corresponding to eyeid == 4 and 5 to store SD-Plane reconstruction
// performed by MD and TALE separately.  ... 

#define MC04_EYE_HIRES1   MC04_EYE1  /* eye id */
#define MC04_EYE_HIRES2   MC04_EYE2

#define MC04_EYE_STA_BR  MC04_EYE1  /* Black Rock */
#define MC04_EYE_STA_LR  MC04_EYE2  /* Long Ridge */
#define MC04_EYE_STA_MD  MC04_EYE3  /* Middle Drum */
#define MC04_EYE_STA_MD_R12  MC04_EYE3  /* Middle Drum */
#define MC04_EYE_STA_MD_R34  MC04_EYE3  /* Middle Drum */

// the following three sites are proposed; don't exist yet
#define MC04_EYE_STA_TLS      MC04_EYE6  /* Tale Stereo; East of Middle Drum site*/
#define MC04_EYE_STA_MD_TAx4  MC04_EYE7  /* Middle Drum site; TAx4 */
#define MC04_EYE_STA_BR_TAx4  MC04_EYE8  /* Black Rock site; TAx4 */

//______________________________________________________

#define MC04_DET_HIRES           1   /* detector id */
#define MC04_DET_HIRES1         11
#define MC04_DET_HIRES2         12

#define MC04_DET_STA            30
#define MC04_DET_STA_BR         31
#define MC04_DET_STA_LR         32
#define MC04_DET_STA_MD         33
#define MC04_DET_STA_MD_R12     34
#define MC04_DET_STA_MD_R34     35

// the following three sites are proposed; don't exist yet
#define MC04_DET_STA_TLS        36
#define MC04_DET_STA_MD_TAx4    37
#define MC04_DET_STA_BR_TAx4    38

#define MC04_DET_STA_BR_LR      312
#define MC04_DET_STA_BR_MD      313
#define MC04_DET_STA_LR_MD      323

// the following three stereo detectors are proposed; second site does exist yet
#define MC04_DET_STA_MD_TLS     336
#define MC04_DET_STA_MD_R12_TLS 346
#define MC04_DET_STA_MD_R34_TLS 356

//______________________________________________________

#define MC04_DET_HIRES_MAXEYE        2   /* number of sites */
#define MC04_DET_HIRES1_MAXEYE       1
#define MC04_DET_HIRES2_MAXEYE       1

#define MC04_DET_STA_MAXEYE          3
#define MC04_DET_STA_BR_MAXEYE       1
#define MC04_DET_STA_LR_MAXEYE       1
#define MC04_DET_STA_MD_MAXEYE       1
#define MC04_DET_STA_MD_R12_MAXEYE   1
#define MC04_DET_STA_MD_R34_MAXEYE   1

// the following three sites are proposed; don't exist yet
#define MC04_DET_STA_TLS_MAXEYE      1
#define MC04_DET_STA_MD_TAx4_MAXEYE  1
#define MC04_DET_STA_BR_TAx4_MAXEYE  1

#define MC04_DET_STA_BR_LR_MAXEYE    2
#define MC04_DET_STA_BR_MD_MAXEYE    2
#define MC04_DET_STA_LR_MD_MAXEYE    2

// the following three stereo detectors are proposed; second site does exist yet
#define MC04_DET_STA_MD_TLS_MAXEYE     2
#define MC04_DET_STA_MD_R12_TLS_MAXEYE 2
#define MC04_DET_STA_MD_R34_TLS_MAXEYE 2

//______________________________________________________

#define MC04_DET_HIRES_MAXMIR       64   /* number of mirrors */
#define MC04_DET_HIRES1_MAXMIR      22
#define MC04_DET_HIRES2_MAXMIR      42

#define MC04_DET_STA_MAXMIR         62
#define MC04_DET_STA_BR_MAXMIR      12
#define MC04_DET_STA_LR_MAXMIR      12
#define MC04_DET_STA_MD_MAXMIR      24
#define MC04_DET_STA_MD_R12_MAXMIR  14
#define MC04_DET_STA_MD_R34_MAXMIR  10

// the following three sites are proposed; don't exist yet
#define MC04_DET_STA_TLS_MAXMIR      2
#define MC04_DET_STA_MD_TAx4_MAXMIR  4
#define MC04_DET_STA_BR_TAx4_MAXMIR  8

#define MC04_DET_STA_MD_MAXMIR_SAH  14
#define MC04_DET_STA_MD_MAXMIR_FADC 10
#define MC04_DET_STA_TLS_MAXMIR_SAH4 2

#define MC04_DET_STA_BR_LR_MAXMIR   24
#define MC04_DET_STA_BR_MD_MAXMIR   36
#define MC04_DET_STA_LR_MD_MAXMIR   36

// the following three stereo detectors are proposed; second site does exist yet
#define MC04_DET_STA_MD_TLS_MAXMIR     26
#define MC04_DET_STA_MD_R12_TLS_MAXMIR 16
#define MC04_DET_STA_MD_R34_TLS_MAXMIR 12

#ifdef __cplusplus
extern "C" {
#endif

  int mc04_detector_neye_active(int detid);
  int mc04_detector_is_hires(int detid);
  int mc04_detector_is_TA(int detid);
  int mc04_detector_has_BR(int detid);
  int mc04_detector_has_LR(int detid);
  int mc04_detector_has_MD(int detid);
  int mc04_detector_has_TALE(int detid);

  int mc04_detector_has_TLS(int detid);
  int mc04_detector_has_MD_TAx4(int detid);
  int mc04_detector_has_BR_TAx4(int detid);

  int mc04_detector_is_mono(int detid);
  int mc04_detector_is_stereo(int detid);

#ifdef __cplusplus
}
#endif

#endif

