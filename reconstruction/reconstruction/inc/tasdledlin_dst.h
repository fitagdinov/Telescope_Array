/*
 *     Bank for SD LED linearity
 *     written by a student
 *     Time-stamp: Fri Apr 10 23:37:53 2009 JST
*/

#ifndef _TASDLEDLIN_
#define _TASDLEDLIN_

#define TASDLEDLIN_BANKID  13006
#define TASDLEDLIN_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdledlin_common_to_bank_();
int tasdledlin_bank_to_dst_(int *NumUnit);
int tasdledlin_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdledlin_bank_to_common_(char *bank);
int tasdledlin_common_to_dump_(int *opt1) ;
int tasdledlin_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdledlin_bank_buffer_ (integer4* tasdledlin_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdledlin_nhmax   3 /* maximum number of host */
#define tasdledlin_ndmax 512 /* maximum number of detector */
#define tasdledlin_npmax 128 /* number of time bins per channel */


typedef struct {
  int npoint;
  int date;
  int pmt_id;
  int error_flag;
  int npt[tasdledlin_npmax];
  float out[tasdledlin_npmax];
  float ain[tasdledlin_npmax];
  float bin[tasdledlin_npmax];
  float stdev_out[tasdledlin_npmax];
  float stdev_ain[tasdledlin_npmax];
  float stdev_bin[tasdledlin_npmax];
  float hv;
  float dec5p;
  char dirname[100];
} SDLEDRawData;

typedef struct {
  int position_id;
  char box_id[20];
  SDLEDRawData udat;
  SDLEDRawData ldat;
} SDLEDSubData;


typedef struct {
  int ndet;      /* the number of detectors       */
  int first_date;
  int first_run_id[tasdledlin_nhmax];
  int last_date;
  int last_run_id[tasdledlin_nhmax];
  SDLEDSubData sub[tasdledlin_ndmax];
  int footer;
} tasdledlin_dst_common;

extern tasdledlin_dst_common tasdledlin_;

#endif


