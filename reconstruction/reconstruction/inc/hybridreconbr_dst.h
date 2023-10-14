/**
 * This DST bank has the information of reconstruction for hybrid
 * @author IKEDA Daisuke (ICRR) 2010-06-13
 * C/C++ version added by D. Ivanov <ivanov@physics.rutgers.edu> 
 * This is the bank for BR Hybrid ( by D. Ikeda ) bank
 */


#ifndef _HYBRIDRECONBR_DST_
#define _HYBRIDRECONBR_DST_

#include "hybridreconfd_dst.h"

#define HYBRIDRECONBR_BANKID		12031
#define HYBRIDRECONBR_BANKVERSION	004

#ifdef __cplusplus
extern "C" {
#endif
integer4 hybridreconbr_common_to_bank_();
integer4 hybridreconbr_bank_to_dst_(integer4 *NumUnit);
integer4 hybridreconbr_common_to_dst_(integer4 *NumUnit);
integer4 hybridreconbr_bank_to_common_(integer1 *bank);
integer4 hybridreconbr_common_to_dump_(integer4 *opt1) ;
integer4 hybridreconbr_common_to_dumpf_(FILE* fp,integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* hybridreconbr_bank_buffer_ (integer4* hybridreconbr_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef hybridreconfd_dst_common hybridreconbr_dst_common;
extern hybridreconbr_dst_common hybridreconbr_;

#endif
