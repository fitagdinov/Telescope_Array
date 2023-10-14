/*
 *     Bank for SD GPS
 *     written by a student
 *     Time-stamp: Fri Apr 10 23:20:08 2009 JST
*/

#ifndef _TASDGPS_
#define _TASDGPS_

#define TASDGPS_BANKID  13009
#define TASDGPS_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdgps_common_to_bank_();
int tasdgps_bank_to_dst_(int *NumUnit);
int tasdgps_common_to_dst_(int *NumUnit);/*combines above 2*/
int tasdgps_bank_to_common_(char *bank);
int tasdgps_common_to_dump_(int *opt1);
int tasdgps_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdgps_bank_buffer_ (integer4* tasdgps_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdgps_nhmax 3      /* maximum number of host */
#define tasdgps_ndmax 512    /* maximum number of detector */


typedef struct {
  int lid;
  int lonmas; /* measured longitude [mas] */
  int latmas; /* measured latitude [mas] */
  int heicm;  /* measured height [cm] */
  int lonmasSet; /* input longitude [mas] */
  int latmasSet; /* input latitude [mas] */
  int heicmSet;  /* input height [cm] */
  float lonmasError;
  float latmasError;
  float heicmError;
  float delayns; /* signal cable delay */
  float ppsofs; /* PPS ofset */
  float ppsfluPH; /* PPS fluctuation in position hold mode */
  float ppsflu3D; /* PPS fluctuation in position 3D fix mode */
} SDGpsSubData;

typedef struct {
  int ndet;      /* the number of detectors       */
  int first_date;
  int first_run_id[tasdgps_nhmax];
  int last_date;
  int last_run_id[tasdgps_nhmax];
  SDGpsSubData sub[tasdgps_ndmax];
  int footer;
} tasdgps_dst_common;

extern tasdgps_dst_common tasdgps_;

#endif


