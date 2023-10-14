// Created 2013/09/26 TAS


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fdtime_dst.h"
#include "lrtime_dst.h"

lrtime_dst_common lrtime_;
static fdtime_dst_common* lrtime = &lrtime_;

//static integer4 lrtime_blen;
static integer4 lrtime_maxlen = sizeof(integer4) * 2 + sizeof(lrtime_dst_common);
static integer1 *lrtime_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* lrtime_bank_buffer_ (integer4* lrtime_bank_buffer_size)
{
  (*lrtime_bank_buffer_size) = fdtime_blen;
  return lrtime_bank;
}



static void lrtime_bank_init() {
  lrtime_bank = (integer1 *)calloc(lrtime_maxlen, sizeof(integer1));
  if (lrtime_bank==NULL) {
      fprintf (stderr,"lrtime_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 lrtime_common_to_bank_() {
  if (lrtime_bank == NULL) lrtime_bank_init();
  return fdtime_struct_to_abank_(lrtime, &lrtime_bank, LRTIME_BANKID, LRTIME_BANKVERSION);
}

integer4 lrtime_bank_to_dst_ (integer4 *unit) {
  return fdtime_abank_to_dst_(lrtime_bank, unit);
}

integer4 lrtime_common_to_dst_(integer4 *unit) {
  if (lrtime_bank == NULL) lrtime_bank_init();
  return fdtime_struct_to_dst_(lrtime, lrtime_bank, unit, LRTIME_BANKID, LRTIME_BANKVERSION);
}

integer4 lrtime_bank_to_common_(integer1 *bank) {
  return fdtime_abank_to_struct_(bank, lrtime);
}

integer4 lrtime_common_to_dump_(integer4 *opt) {
  return fdtime_struct_to_dumpf_(lrtime, stdout, opt);
}

integer4 lrtime_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return fdtime_struct_to_dumpf_(lrtime, fp, opt);
}
