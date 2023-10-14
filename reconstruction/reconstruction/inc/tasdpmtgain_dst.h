/*
 *     Bank for SD PMT gain curve
 *     written by a student
 *     Time-stamp: Fri Apr 10 23:39:15 2009 JST
*/

#ifndef _TASDPMTGAIN_
#define _TASDPMTGAIN_

#define TASDPMTGAIN_BANKID 13007
#define TASDPMTGAIN_BANKVERSION 002

#define tasdpmtgain_npmax 2000

typedef struct {
  integer4 serial;
  real4 v[15];
  real4 g[15];
  real4 minv;
  real4 maxv;
} SDPmtData;


typedef struct {
  int npmt;
  SDPmtData pmt[tasdpmtgain_npmax];
} tasdpmtgain_dst_common;

extern tasdpmtgain_dst_common tasdpmtgain_;

#ifdef __cplusplus
extern "C" {
#endif
integer4 tasdpmtgain_common_to_bank_();
integer4 tasdpmtgain_bank_to_dst_(integer4 *NumUnit);
integer4 tasdpmtgain_common_to_dst_(integer4 *NumUnit);
integer4 tasdpmtgain_bank_to_common_(integer1 *bank);
integer4 tasdpmtgain_common_to_dumpf_(FILE *fp, integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdpmtgain_bank_buffer_ (integer4* tasdpmtgain_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


#endif
