/* Created 2008/11 LMS */

#ifndef _BRPROFILE_DST_
#define _BRPROFILE_DST_

#include "fdprofile_dst.h"

#define BRPROFILE_BANKID		12104
#define BRPROFILE_BANKVERSION	000

#ifdef __cplusplus
extern "C" {
#endif
integer4 brprofile_common_to_bank_();
integer4 brprofile_bank_to_dst_(integer4 *NumUnit);
integer4 brprofile_common_to_dst_(integer4 *NumUnit);	// combines above 2
integer4 brprofile_bank_to_common_(integer1 *bank);
integer4 brprofile_common_to_dump_(integer4 *opt1) ;
integer4 brprofile_common_to_dumpf_(FILE* fp,integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* brprofile_bank_buffer_ (integer4* brprofile_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef fdprofile_dst_common brprofile_dst_common;
extern brprofile_dst_common brprofile_;

#endif
