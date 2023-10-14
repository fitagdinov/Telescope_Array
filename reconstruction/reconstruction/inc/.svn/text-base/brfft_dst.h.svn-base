/*
 * Bank for storing BR mean power spectra
 * SRS - 5.20.2010
 */

#ifndef _BRFFT_
#define _BRFFT_

#include "fdfft_dst.h"

#define BRFFT_BANKID  12450
#define BRFFT_BANKVERSION   000

#ifdef __cplusplus
extern "C" {
#endif
integer4 brfft_common_to_bank_();
integer4 brfft_bank_to_dst_(integer4 *unit);
integer4 brfft_common_to_dst_(integer4 *unit); /* combines above 2 */
integer4 brfft_bank_to_common_(integer1 *bank);
integer4 brfft_common_to_dump_(integer4 *opt) ;
integer4 brfft_common_to_dumpf_(FILE* fp,integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* brfft_bank_buffer_ (integer4* brfft_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


#define brfft_nmir_max 12             /* number of cameras per site        */
#define brfft_nchan_mir 256           /* number of tubes per camera        */
#define brfft_nt_chan_max 512         /* number of time bins per tube      */

typedef fdfft_dst_common brfft_dst_common;
extern brfft_dst_common brfft_;

#endif
