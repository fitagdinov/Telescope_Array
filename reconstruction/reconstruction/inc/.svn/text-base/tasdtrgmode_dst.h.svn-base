/*
 *     Bank for central PC trigger information
 *     written by a student
 *     Time-stamp: Fri Apr 24 20:23:35 2009 JST
*/

#ifndef _TASDTRGMODE_
#define _TASDTRGMODE_

#define TASDTRGMODE_BANKID  13003
#define TASDTRGMODE_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdtrgmode_common_to_bank_();
int tasdtrgmode_bank_to_dst_(int *NumUnit);
int tasdtrgmode_common_to_dst_(int *NumUnit);/* combines above 2 */
int tasdtrgmode_bank_to_common_(char *bank);
int tasdtrgmode_common_to_dump_(int *opt1) ;
int tasdtrgmode_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdtrgmode_bank_buffer_ (integer4* tasdtrgmode_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdtrgmode_nhmax    3 /*maximum number of hosts*/
#define tasdtrgmode_npmax 3600 /*maximum number of points*/


typedef struct {
  short sec;
  short trgmode;
  int strial[tasdtrgmode_nhmax];
  int etrial[tasdtrgmode_nhmax];
} SDTrgmodePpsData;


typedef struct {
  int year;
  int run_id;
  int npoint;   /* the number of data points */
  SDTrgmodePpsData data[tasdtrgmode_npmax];
  int footer;
} tasdtrgmode_dst_common;


extern tasdtrgmode_dst_common tasdtrgmode_;


#endif


