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
#include "brtime_dst.h"

brtime_dst_common brtime_;
static fdtime_dst_common* brtime = &brtime_;

//static integer4 brtime_blen;
static integer4 brtime_maxlen = sizeof(integer4) * 2 + sizeof(brtime_dst_common);
static integer1 *brtime_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* brtime_bank_buffer_ (integer4* brtime_bank_buffer_size)
{
  (*brtime_bank_buffer_size) = fdtime_blen;
  return brtime_bank;
}



static void brtime_bank_init() {
  brtime_bank = (integer1 *)calloc(brtime_maxlen, sizeof(integer1));
  if (brtime_bank==NULL) {
      fprintf (stderr,"brtime_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 brtime_common_to_bank_() {
  if (brtime_bank == NULL) brtime_bank_init();
  return fdtime_struct_to_abank_(brtime, &brtime_bank, BRTIME_BANKID, BRTIME_BANKVERSION);
}

integer4 brtime_bank_to_dst_ (integer4 *unit) {
  return fdtime_abank_to_dst_(brtime_bank, unit);
}

integer4 brtime_common_to_dst_(integer4 *unit) {
  if (brtime_bank == NULL) brtime_bank_init();
  return fdtime_struct_to_dst_(brtime, brtime_bank, unit, BRTIME_BANKID, BRTIME_BANKVERSION);
}

integer4 brtime_bank_to_common_(integer1 *bank) {
  return fdtime_abank_to_struct_(bank, brtime);
}

integer4 brtime_common_to_dump_(integer4 *opt) {
  return fdtime_struct_to_dumpf_(brtime, stdout, opt);
}

integer4 brtime_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return fdtime_struct_to_dumpf_(brtime, fp, opt);
}
