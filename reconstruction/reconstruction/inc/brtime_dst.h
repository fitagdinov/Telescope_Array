/* Created 2013/09/26 TAS */

#ifndef _BRTIME_DST_
#define _BRTIME_DST_

#include "fdtime_dst.h"

#define BRTIME_BANKID		12159
#define BRTIME_BANKVERSION	0

#ifdef __cplusplus
extern "C" {
#endif
integer4 brtime_common_to_bank_();
integer4 brtime_bank_to_dst_(integer4 *unit);
integer4 brtime_common_to_dst_(integer4 *unit); // combines above 2
integer4 brtime_bank_to_common_(integer1 *bank);
integer4 brtime_common_to_dump_(integer4 *opt);
integer4 brtime_common_to_dumpf_(FILE *fp, integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* brtime_bank_buffer_ (integer4* brtime_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef fdtime_dst_common brtime_dst_common;

extern brtime_dst_common brtime_;

#endif
