/*
  2010/Jan/05 K.Hayashi
 */
#ifndef ___FDCTD_CLOCK_H___
#define ___FDCTD_CLOCK_H___

//#include "dst_common.h"
#include "univ_dst.h"
#include "dst_std_types.h"
#include "dst_pack_proto.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FDCTD_CLOCK_BANKID 12440
#define FDCTD_CLOCK_BANKVERSION 000

typedef struct _fdctd_clock_t {
  integer2 stID;
  integer4 unixtime;
  real8 clock;
} fdctd_clock_dst_common;

extern fdctd_clock_dst_common fdctd_clock_;

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdctd_clock_common_to_bank_();
integer4 fdctd_clock_bank_to_dst_(integer4 *NumUnit);
integer4 fdctd_clock_common_to_dst_(integer4 *NumUnit);
integer4 fdctd_clock_bank_to_common_(integer1 *bank);
integer4 fdctd_clock_common_to_dumpf_(FILE *fp, integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* fdctd_clock_bank_buffer_ (integer4* fdctd_clock_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


#endif
