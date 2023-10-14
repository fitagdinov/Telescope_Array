/*
 *     Bank for SD ID and HV
 *     written by a student
 *     Time-stamp: Fri Apr 10 23:37:15 2009 JST
*/

#ifndef _TASDIDHV_
#define _TASDIDHV_

#define TASDIDHV_BANKID  13005
#define TASDIDHV_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdidhv_common_to_bank_();
int tasdidhv_bank_to_dst_(int *NumUnit);
int tasdidhv_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdidhv_bank_to_common_(char *bank);
int tasdidhv_common_to_dump_(int *opt1) ;
int tasdidhv_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdidhv_bank_buffer_ (integer4* tasdidhv_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdidhv_nhmax 3       /* maximum number of detector */
#define tasdidhv_ndmax 256     /* maximum number of detector */


typedef struct {
  int lid;
  int error_flag;
  unsigned int wlanidmsb;
  unsigned int wlanidlsb;
  short trig_mode0;
  short trig_mode1;
  short uthre_lvl0;
  short lthre_lvl0;
  short uthre_lvl1;
  short lthre_lvl1;
  short uhv;
  short lhv;
  char firm_version[32];
} SDIdhvSubData;


typedef struct {
  int ndet;      /* the number of detectors       */
  int site;
  int run_id;
  int year;

  SDIdhvSubData sub[tasdidhv_ndmax];

} tasdidhv_dst_common;

extern tasdidhv_dst_common tasdidhv_;

#endif


