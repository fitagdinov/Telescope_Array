/*
 *     Bank for SD LED linearity
 *     written by a student
 *     Time-stamp: Fri Apr 10 23:19:08 2009 JST
*/

#ifndef _TASDELECINFO_
#define _TASDELECINFO_

#define TASDELECINFO_BANKID  13008
#define TASDELECINFO_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdelecinfo_common_to_bank_();
int tasdelecinfo_bank_to_dst_(int *NumUnit);
int tasdelecinfo_common_to_dst_(int *NumUnit);/*combines above 2*/
int tasdelecinfo_bank_to_common_(char *bank);
int tasdelecinfo_common_to_dump_(int *opt1) ;
int tasdelecinfo_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdelecinfo_bank_buffer_ (integer4* tasdelecinfo_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdelecinfo_ndmax 700    /* maximum number of detector */


typedef struct {
  unsigned int wlanidmsb;
  unsigned int wlanidlsb;
  int ccid;
  int error_flag;
  float uoffset;
  float uslope;
  float loffset;
  float lslope;
  char elecid[8];
  char gpsid[8];
  char cpldver[8];
  char ccver[8];
} SDElecSubData;


typedef struct {
  int ndet;      /* the number of detectors       */
  SDElecSubData sub[tasdelecinfo_ndmax];
} tasdelecinfo_dst_common;

extern tasdelecinfo_dst_common tasdelecinfo_;

#endif


