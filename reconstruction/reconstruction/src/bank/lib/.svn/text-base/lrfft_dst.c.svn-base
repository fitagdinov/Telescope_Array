/*
 * C functions for brfft
 * SRS - 5.20.2010
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fdfft_dst.h"  
#include "lrfft_dst.h"  

lrfft_dst_common lrfft_;  /* allocate memory to lrfft_common */
static fdfft_dst_common* lrfft = &lrfft_;

//static integer4 lrfft_blen; 
static integer4 lrfft_maxlen = sizeof(integer4) * 2 + sizeof(lrfft_dst_common);
static integer1 *lrfft_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* lrfft_bank_buffer_ (integer4* lrfft_bank_buffer_size)
{
  (*lrfft_bank_buffer_size) = fdfft_blen;
  return lrfft_bank;
}



static void lrfft_bank_init() {
  lrfft_bank = (integer1 *)calloc(lrfft_maxlen, sizeof(integer1));
  if (lrfft_bank==NULL) {
      fprintf (stderr,"lrfft_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 lrfft_common_to_bank_() {
  if (lrfft_bank == NULL) lrfft_bank_init();
  return fdfft_struct_to_abank_(lrfft, &lrfft_bank, LRFFT_BANKID, LRFFT_BANKVERSION);
}

integer4 lrfft_bank_to_dst_ (integer4 *unit) {
  return fdfft_abank_to_dst_(lrfft_bank, unit);
}

integer4 lrfft_common_to_dst_(integer4 *unit) {
  if (lrfft_bank == NULL) lrfft_bank_init();
  return fdfft_struct_to_dst_(lrfft, &lrfft_bank, unit, LRFFT_BANKID, LRFFT_BANKVERSION);
}

integer4 lrfft_bank_to_common_(integer1 *bank) {
  return fdfft_abank_to_struct_(bank, lrfft);
}

integer4 lrfft_common_to_dump_(integer4 *opt) {
  return fdfft_struct_to_dumpf_(1, lrfft, stdout, opt);
}

integer4 lrfft_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return fdfft_struct_to_dumpf_(1, lrfft, fp, opt);
}
