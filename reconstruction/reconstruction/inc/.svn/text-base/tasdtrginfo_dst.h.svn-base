/*
 *     Bank for central PC trigger information
 *     written by a student
 *     Time-stamp: Wed Apr 29 01:41:55 2009 JST
*/

#ifndef _TASDTRGINFO_
#define _TASDTRGINFO_

#define TASDTRGINFO_BANKID  13004
#define TASDTRGINFO_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdtrginfo_common_to_bank_();
int tasdtrginfo_bank_to_dst_(int *NumUnit);
int tasdtrginfo_common_to_dst_(int *NumUnit);/* combines above 2 */
int tasdtrginfo_bank_to_common_(char *bank);
int tasdtrginfo_common_to_dump_(int *opt1) ;
int tasdtrginfo_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdtrginfo_bank_buffer_ (integer4* tasdtrginfo_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdtrginfo_nhmax    3 /*maximum number of hosts*/
#define tasdtrginfo_npmax 1000 /*maximum number of points*/


typedef struct {
  int bank;      /* 3rd col. */
  short pos;     /* 4th col. */
  short command; /* 2nd col. */
  char trgcode;  /* 5th col. */
  char daqcode;  /* 1st col. */
} SDTrginfoData;


typedef struct {
  int site;
  int run_id;
  int year;
  int npoint;   /* the number of data points */
  SDTrginfoData data[tasdtrginfo_npmax];
  int footer;
} tasdtrginfo_dst_common;


extern tasdtrginfo_dst_common tasdtrginfo_;


#endif


