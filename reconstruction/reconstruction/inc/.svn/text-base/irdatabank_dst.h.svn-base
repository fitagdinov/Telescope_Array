/*!
 * Bank for IR camera data 
 *
 * 2008/Dec/17 Y.Tsunesada (TokyoTech)
 * 
 *   Modified format 2009/Feb/17
 */
#ifndef ___IRDATABANK_DST_H___
#define ___IRDATABANK_DST_H___

#include "univ_dst.h"
#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_std_types.h"
#include "dst_pack_proto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// See http://www.telescopearray.org/tawiki/index.php/DSTBank_ID_List
#define IRDATABANK_BANKID 12408
#define IRDATABANK_BANKVERSION 000

/*
 * 12+2 directions:
 *    0-5: Lower elevation (12deg)
 *   6-11: Upper elevation (30deg)
 *
 *       | 6| 7| 8| 9|10|11|
 *       | 5| 4| 3| 2| 1| 0|
 * 
 *     12: Zenith
 *     13: Horizontal
 */  
/*
 * 4+1 sections of an IR picture
 *     0: Whole picture (320x236 pixels)
 *   1-4: Sections of 320x59 pixels (below)
 *
 *   <-------   320   ------>
 *   ------------------------   ^
 *   |  Section 1 (0-58)    |   |
 *   |  Section 2 (59-117)  |   | 236
 *   |  Section 3 (118-176) |   | 
 *   |  Section 4 (177-235) |   | 
 *   ------------ -----------   v
 */
/*
 * Array status[14]:
 *   Ex: status[0]:  Status of the direction 0 (lower-right most direction)
 *       status[12]: Status of the vertical direction
 *
 * Array score[14][5]:
 *   Ex:  score[0][1]: Score of the Section 1 of an IR picture of the direction 0
 *        score[3][3]: Score of the Section 3 of an IR picture of the direction 3
 *        score[8][0]: Score of a whole picture of the direction 8
 */
typedef struct _irdatabank_t {
  integer2 iSite;        // 0: BRM, 1: LR
  integer4 dateFrom;     // Unix time (seconds since 1970/01/01)
  integer4 dateTo;       // Unit time 
  integer2 iy;           // 2007, 2008, ...
  integer2 im;           // 01 - 12
  integer2 id;           // 01 - 31
  integer2 iH;           // 00 - 23
  integer2 iM;           // 00 - 59
  integer2 iS;           // 00 - 59
  integer2 status[14];   // 0: OK, direction identified, -1: Not indentified
  integer2 DA[14];       // Ambient temperature
  integer2 D50[14][5];   // D50 of 12+2 directions for sections 0-4
  integer2 score[14][5]; // Score (0/1) of 12(+2) for sections 0-4
  integer2 totalscore;   // Sum of scores of the 12 directions (0-48)
  /* Added 2010/Feb/08 */
  real4    prob[14][5];  // Cloud probability (0~1) of 12(+2) for sections 0-4
  real4    totalprob;    // Sum of probabilities of the 12 directions (0-48)
} irdatabank_dst_common;

extern irdatabank_dst_common irdatabank_;

#ifdef __cplusplus
extern "C" {
#endif
integer4 irdatabank_common_to_bank_();
integer4 irdatabank_bank_to_dst_(integer4 *NumUnit);
integer4 irdatabank_common_to_dst_(integer4 *NumUnit);
integer4 irdatabank_bank_to_common_(integer1 *bank);
integer4 irdatabank_common_to_dumpf_(FILE *fp, integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* irdatabank_bank_buffer_ (integer4* irdatabank_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


#endif
