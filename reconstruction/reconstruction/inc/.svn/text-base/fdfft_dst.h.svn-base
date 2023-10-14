/*
 * Genericized bank for storing BR/LR power spectra.
 * SRS - 5.20.2010
 */

#ifndef _FDFFT_
#define _FDFFT_

#define FDFFT_BANKID  12459
#define FDFFT_BANKVERSION   000

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdfft_common_to_bank_();
integer4 fdfft_bank_to_dst_(integer4 *unit);
integer4 fdfft_common_to_dst_(integer4 *unit); /* combines above 2 */
integer4 fdfft_bank_to_common_(integer1 *bank);
integer4 fdfft_common_to_dump_(integer4 *opt) ;
integer4 fdfft_common_to_dumpf_(FILE* fp,integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* fdfft_bank_buffer_ (integer4* fdfft_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


#define fdfft_nmir_max 12             /* number of cameras per site        */
#define fdfft_nchan_mir 256           /* number of tubes per camera        */
#define fdfft_nt_chan_max 512         /* number of time bins per tube      */

typedef struct {

  integer4 startsec;
  integer4 stopsec;
  integer4 startnsec;
  integer4 stopnsec;

  integer4 ncamera;
  integer4 camera[fdfft_nmir_max];
  integer4 nchan[fdfft_nmir_max];
  integer4 chan[fdfft_nmir_max][fdfft_nchan_mir];

  real8 powerspec[fdfft_nmir_max][fdfft_nchan_mir][fdfft_nt_chan_max];
  real8 powerspecerr[fdfft_nmir_max][fdfft_nchan_mir][fdfft_nt_chan_max];

} fdfft_dst_common;

extern fdfft_dst_common fdfft_;
extern integer4 fdfft_blen; /* needs to be accessed by the c files of the derived banks */ 

integer4 fdfft_struct_to_abank_(fdfft_dst_common *fdfft, integer1* (*pbank), integer4 id, integer4 ver);
integer4 fdfft_abank_to_dst_(integer1 *bank, integer4 *unit);
integer4 fdfft_struct_to_dst_(fdfft_dst_common *fdfft, integer1* (*pbank), integer4 *unit, integer4 id, integer4 ver);
integer4 fdfft_abank_to_struct_(integer1 *bank, fdfft_dst_common *fdfft);
integer4 fdfft_struct_to_dump_(integer4 siteid, fdfft_dst_common *fdfft, integer4 *opt);
integer4 fdfft_struct_to_dumpf_(integer4 siteid, fdfft_dst_common *fdfft, FILE *fp, integer4 *opt);

#endif
