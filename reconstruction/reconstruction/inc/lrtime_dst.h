/* Created 2013/09/26 TAS */

#ifndef _LRTIME_DST_
#define _LRTIME_DST_

#include "fdtime_dst.h"

#define LRTIME_BANKID		12259
#define LRTIME_BANKVERSION	0

#ifdef __cplusplus
extern "C" {
#endif
integer4 lrtime_common_to_bank_();
integer4 lrtime_bank_to_dst_(integer4 *unit);
integer4 lrtime_common_to_dst_(integer4 *unit); // combines above 2
integer4 lrtime_bank_to_common_(integer1 *bank);
integer4 lrtime_common_to_dump_(integer4 *opt);
integer4 lrtime_common_to_dumpf_(FILE *fp, integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* lrtime_bank_buffer_ (integer4* lrtime_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef fdtime_dst_common lrtime_dst_common;

extern lrtime_dst_common lrtime_;

#endif
