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
#include "brped_dst.h"

brped_dst_common brped_;
static fdped_dst_common* brped = &brped_;

//static integer4 brped_blen;
static integer4 brped_maxlen = sizeof(integer4) * 2 + sizeof(brped_dst_common);
static integer1 *brped_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* brped_bank_buffer_ (integer4* brped_bank_buffer_size)
{
  (*brped_bank_buffer_size) = fdped_blen;
  return brped_bank;
}



static void brped_bank_init() {
  brped_bank = (integer1 *)calloc(brped_maxlen, sizeof(integer1));
  if (brped_bank==NULL) {
      fprintf (stderr,"brped_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 brped_common_to_bank_() {
  if (brped_bank == NULL) brped_bank_init();
  return fdped_struct_to_abank_(brped, &brped_bank, BRPED_BANKID, BRPED_BANKVERSION);
}

integer4 brped_bank_to_dst_ (integer4 *unit) {
  return fdped_abank_to_dst_(brped_bank, unit);
}

integer4 brped_common_to_dst_(integer4 *unit) {
  if (brped_bank == NULL) brped_bank_init();
  return fdped_struct_to_dst_(brped, brped_bank, unit, BRPED_BANKID, BRPED_BANKVERSION);
}

integer4 brped_bank_to_common_(integer1 *bank) {
  return fdped_abank_to_struct_(bank, brped);
}

integer4 brped_common_to_dump_(integer4 *opt) {
  return fdped_struct_to_dumpf_(brped, stdout, opt);
}

integer4 brped_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return fdped_struct_to_dumpf_(brped, fp, opt);
}
