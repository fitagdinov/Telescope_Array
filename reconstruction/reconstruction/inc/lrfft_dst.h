/*
 *  Bank for storing LR mean power spectra.
 *  SRS -- 5.20.2010
 */

#ifndef _LRFFT_
#define _LRFFT_

#include "fdfft_dst.h"

#define LRFFT_BANKID  12451
#define LRFFT_BANKVERSION   000

#ifdef __cplusplus
extern "C" {
#endif
integer4 lrfft_common_to_bank_();
integer4 lrfft_bank_to_dst_(integer4 *unit);
integer4 lrfft_common_to_dst_(integer4 *unit); /* combines above 2 */
integer4 lrfft_bank_to_common_(integer1 *bank);
integer4 lrfft_common_to_dump_(integer4 *opt) ;
integer4 lrfft_common_to_dumpf_(FILE* fp,integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* lrfft_bank_buffer_ (integer4* lrfft_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


#define lrfft_nmir_max 12             /* number of cameras per site        */
#define lrfft_nchan_mir 256           /* number of tubes per camera        */
#define lrfft_nt_chan_max 512         /* number of time bins per tube      */

typedef fdfft_dst_common lrfft_dst_common;
extern lrfft_dst_common lrfft_;

#endif
