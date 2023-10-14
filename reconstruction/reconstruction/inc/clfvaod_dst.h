#ifndef _CLFVAOD_
#define _CLFVAOD_

#define CLFVAOD_BANKID 12427
#define CLFVAOD_BANKVERSION 001

#ifdef __cplusplus
extern "C" {
#endif
integer4 clfvaod_bank_to_common_(integer1 *bank);
integer4 clfvaod_common_to_dst_(integer4 *unit);
integer4 clfvaod_common_to_bank_();
integer4 clfvaod_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* clfvaod_bank_buffer_ (integer4* clfvaod_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _clfvaod_t {

  /** uniq ID */
  integer4 uniqID;
  
  /** available date from */
  integer4 dateFrom; //sec from 1970/1/1
  /** available date to */
  integer4 dateTo; //sec from 1970/1/1
  
  real4 laserEnergy;
  real4 vaod;         // VAOD at 5km
  real4 vaodMin;
  real4 vaodMax;

} clfvaod_dst_common;

extern clfvaod_dst_common clfvaod_;


#endif
