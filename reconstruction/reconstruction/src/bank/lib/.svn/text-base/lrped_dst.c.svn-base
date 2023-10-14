// Created 2009/03/24 LMS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fdped_dst.h"
#include "lrped_dst.h"

lrped_dst_common lrped_;
static fdped_dst_common* lrped = &lrped_;

//static integer4 lrped_blen;
static integer4 lrped_maxlen = sizeof(integer4) * 2 + sizeof(lrped_dst_common);
static integer1 *lrped_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* lrped_bank_buffer_ (integer4* lrped_bank_buffer_size)
{
  (*lrped_bank_buffer_size) = fdped_blen;
  return lrped_bank;
}



static void lrped_bank_init() {
  lrped_bank = (integer1 *)calloc(lrped_maxlen, sizeof(integer1));
  if (lrped_bank==NULL) {
      fprintf (stderr,"lrped_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 lrped_common_to_bank_() {
  if (lrped_bank == NULL) lrped_bank_init();
  return fdped_struct_to_abank_(lrped, &lrped_bank, LRPED_BANKID, LRPED_BANKVERSION);
}

integer4 lrped_bank_to_dst_ (integer4 *unit) {
  return fdped_abank_to_dst_(lrped_bank, unit);
}

integer4 lrped_common_to_dst_(integer4 *unit) {
  if (lrped_bank == NULL) lrped_bank_init();
  return fdped_struct_to_dst_(lrped, lrped_bank, unit, LRPED_BANKID, LRPED_BANKVERSION);
}

integer4 lrped_bank_to_common_(integer1 *bank) {
  return fdped_abank_to_struct_(bank, lrped);
}

integer4 lrped_common_to_dump_(integer4 *opt) {
  return fdped_struct_to_dumpf_(lrped, stdout, opt);
}

integer4 lrped_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return fdped_struct_to_dumpf_(lrped, fp, opt);
}
