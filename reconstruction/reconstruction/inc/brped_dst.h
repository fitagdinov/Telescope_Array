/* Created 2009/03/24 LMS */

#ifndef _BRPED_DST_
#define _BRPED_DST_

#include "fdped_dst.h"

#define BRPED_BANKID		12416
#define BRPED_BANKVERSION	000

#ifdef __cplusplus
extern "C" {
#endif
integer4 brped_common_to_bank_();
integer4 brped_bank_to_dst_(integer4 *NumUnit);
integer4 brped_common_to_dst_(integer4 *NumUnit);	// combines above 2
integer4 brped_bank_to_common_(integer1 *bank);
integer4 brped_common_to_dump_(integer4 *opt1) ;
integer4 brped_common_to_dumpf_(FILE* fp,integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* brped_bank_buffer_ (integer4* brped_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef fdped_dst_common brped_dst_common;
extern brped_dst_common brped_;

#endif
