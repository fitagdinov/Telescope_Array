/*
 *     Bank for SD temperature, humidity and voltage monitor
 *     written by a student
 *     Time-stamp: Fri Apr 10 23:39:58 2009 JST
*/

#ifndef _TASDTEMP_
#define _TASDTEMP_

#define TASDTEMP_BANKID  13014
#define TASDTEMP_BANKVERSION   001

#ifdef __cplusplus
extern "C" {
#endif
int tasdtemp_common_to_bank_();
int tasdtemp_bank_to_dst_(int *NumUnit);
int tasdtemp_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdtemp_bank_to_common_(char *bank);
int tasdtemp_common_to_dump_(int *opt1) ;
int tasdtemp_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdtemp_bank_buffer_ (integer4* tasdtemp_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdtemp_ndmax 512 /*maximum number of detector per event*/


typedef struct {
  int lid;
  int livetime;		/* livetime in 10 min */
  int error;
  float scinti_temp[10];
  float elec_temp[10];
  float batt_temp[10];
  float cc_temp[10];
  float scinti_humm[10];
  float batt_vol[10];
  float batt_cur[10];
  float panel_vol[10];
  float cc_lv[10];
  float elec_gnd[10];
  float elec_1_2v[10];
  float elec_1_8v[10];
  float elec_3_3v[10];
  float elec_5v[10];
} SDTempData;


typedef struct {
  int num_det;   /* the number of detectors       */
  int date;      /* year month day */
  int time;      /* hour minute second */

  SDTempData sub[tasdtemp_ndmax];

  int footer;

} tasdtemp_dst_common;

extern tasdtemp_dst_common tasdtemp_;

#endif


