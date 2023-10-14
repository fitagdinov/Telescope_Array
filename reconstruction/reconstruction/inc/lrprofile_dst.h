/* Created 2008/11 LMS */

#ifndef _LRPROFILE_DST_
#define _LRPROFILE_DST_

#include "fdprofile_dst.h"

#define LRPROFILE_BANKID		12204
#define LRPROFILE_BANKVERSION	000

#ifdef __cplusplus
extern "C" {
#endif
integer4 lrprofile_common_to_bank_();
integer4 lrprofile_bank_to_dst_(integer4 *NumUnit);
integer4 lrprofile_common_to_dst_(integer4 *NumUnit);	// combines above 2
integer4 lrprofile_bank_to_common_(integer1 *bank);
integer4 lrprofile_common_to_dump_(integer4 *opt1) ;
integer4 lrprofile_common_to_dumpf_(FILE* fp,integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* lrprofile_bank_buffer_ (integer4* lrprofile_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef fdprofile_dst_common lrprofile_dst_common;
extern lrprofile_dst_common lrprofile_;

#endif
