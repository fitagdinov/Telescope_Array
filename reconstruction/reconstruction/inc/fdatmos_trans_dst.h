#ifndef _FDATMOS_TRANS_
#define _FDATMOS_TRANS_

#define FDATMOS_TRANS_BANKID 12407
#define FDATMOS_TRANS_BANKVERSION 004

#define FDATMOS_TRANS_MAXITEM 1024

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdatmos_trans_bank_to_common_(integer1 *bank);
integer4 fdatmos_trans_common_to_dst_(integer4 *unit);
integer4 fdatmos_trans_common_to_bank_();
integer4 fdatmos_trans_common_to_dumpf_(FILE* fp,integer4* long_output);
/* get (packed) buffer pointer and size */
integer1* fdatmos_trans_bank_buffer_ (integer4* fdatmos_trans_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct _fdatmos_trans_t {

   /** uniq ID */
   integer4 uniqID;

   /** available date from */
   integer4 dateFrom; //sec from 1970/1/1
   /** available date to */
   integer4 dateTo; //sec from 1970/1/1

   /** number of data line */
   integer4 nItem;
   /** normalize height [km] */
   real4 normHeight;
   /** available max height [km] */
   real4 availableHeight;
   /** available lowest height [km] */
   real4 lowestHeight;
   /** bad flag -1:no data 0:normal 1:thin cloud 2:no horizontal data 3:bad data*/
   integer4 badFlag;
   /** condition of measurement for lower height 0:not using 1:low energy 2:strong energy(slant) 3:middole energy */
   integer4 lowerFlag;
   /** 0:no cloud 1: cloud */
   integer4 cloudFlag;

   /** height [km] */
   real4 height[FDATMOS_TRANS_MAXITEM];
   /** measured total alpha */
   real4 measuredAlpha[FDATMOS_TRANS_MAXITEM];
   /** error of measured total alpha +*/
   real4 measuredAlphaError_P[FDATMOS_TRANS_MAXITEM];
   /** error of measured total alpha -*/
   real4 measuredAlphaError_M[FDATMOS_TRANS_MAXITEM];
   /** alpha of Rayleigh scattering */
   real4 rayleighAlpha[FDATMOS_TRANS_MAXITEM];
   /** error of alpha of rayleigh scattering*/
   real4 rayleighAlphaError[FDATMOS_TRANS_MAXITEM];
   /** alpha of Mie scattering*/
   real4 mieAlpha[FDATMOS_TRANS_MAXITEM];
   /** error of alpha of Mie scattering +*/
   real4 mieAlphaError_P[FDATMOS_TRANS_MAXITEM];
   /** error of alpha of Mie scattering -*/
   real4 mieAlphaError_M[FDATMOS_TRANS_MAXITEM];
   /** VAOD */
   real4 vaod[FDATMOS_TRANS_MAXITEM];
   /** error of VAOD +*/
   real4 vaodError_P[FDATMOS_TRANS_MAXITEM];
   /** error of VAOD -*/
   real4 vaodError_M[FDATMOS_TRANS_MAXITEM];


} fdatmos_trans_dst_common;

extern fdatmos_trans_dst_common fdatmos_trans_;


#endif
