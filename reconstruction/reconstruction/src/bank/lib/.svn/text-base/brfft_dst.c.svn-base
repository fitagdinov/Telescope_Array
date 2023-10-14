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
#include "brfft_dst.h"  

brfft_dst_common brfft_;  /* allocate memory to brfft_common */
static fdfft_dst_common* brfft = &brfft_;

//static integer4 brfft_blen; 
static integer4 brfft_maxlen = sizeof(integer4) * 2 + sizeof(brfft_dst_common);
static integer1 *brfft_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* brfft_bank_buffer_ (integer4* brfft_bank_buffer_size)
{
  (*brfft_bank_buffer_size) = fdfft_blen;
  return brfft_bank;
}



static void brfft_bank_init() {
  brfft_bank = (integer1 *)calloc(brfft_maxlen, sizeof(integer1));
  if (brfft_bank==NULL) {
      fprintf (stderr,"brfft_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 brfft_common_to_bank_() {
  if (brfft_bank == NULL) brfft_bank_init();
  return fdfft_struct_to_abank_(brfft, &brfft_bank, BRFFT_BANKID, BRFFT_BANKVERSION);
}

integer4 brfft_bank_to_dst_ (integer4 *unit) {
  return fdfft_abank_to_dst_(brfft_bank, unit);
}

integer4 brfft_common_to_dst_(integer4 *unit) {
  if (brfft_bank == NULL) brfft_bank_init();
  return fdfft_struct_to_dst_(brfft, &brfft_bank, unit, BRFFT_BANKID, BRFFT_BANKVERSION);
}

integer4 brfft_bank_to_common_(integer1 *bank) {
  return fdfft_abank_to_struct_(bank, brfft);
}

integer4 brfft_common_to_dump_(integer4 *opt) {
  return fdfft_struct_to_dumpf_(0, brfft, stdout, opt);
}

integer4 brfft_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return fdfft_struct_to_dumpf_(0, brfft, fp, opt);
}
