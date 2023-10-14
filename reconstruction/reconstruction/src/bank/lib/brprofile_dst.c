// Created 2008/11 LMS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "fdprofile_dst.h"
#include "brprofile_dst.h"

brprofile_dst_common brprofile_;
static fdprofile_dst_common* brprofile = &brprofile_;

//static integer4 brprofile_blen;
static integer4 brprofile_maxlen = sizeof(integer4) * 2 + sizeof(brprofile_dst_common);
static integer1 *brprofile_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* brprofile_bank_buffer_ (integer4* brprofile_bank_buffer_size)
{
  (*brprofile_bank_buffer_size) = fdprofile_blen;
  return brprofile_bank;
}



static void brprofile_bank_init() {
  brprofile_bank = (integer1 *)calloc(brprofile_maxlen, sizeof(integer1));
  if (brprofile_bank==NULL) {
      fprintf (stderr,"brprofile_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 brprofile_common_to_bank_() {
  if (brprofile_bank == NULL) brprofile_bank_init();
  return fdprofile_struct_to_abank_(brprofile, &brprofile_bank, BRPROFILE_BANKID, BRPROFILE_BANKVERSION);
}

integer4 brprofile_bank_to_dst_ (integer4 *unit) {
  return fdprofile_abank_to_dst_(brprofile_bank, unit);
}

integer4 brprofile_common_to_dst_(integer4 *unit) {
  if (brprofile_bank == NULL) brprofile_bank_init();
  return fdprofile_struct_to_dst_(brprofile, brprofile_bank, unit, BRPROFILE_BANKID, BRPROFILE_BANKVERSION);
}

integer4 brprofile_bank_to_common_(integer1 *bank) {
  return fdprofile_abank_to_struct_(bank, brprofile);
}

integer4 brprofile_common_to_dump_(integer4 *opt) {
  return fdprofile_struct_to_dumpf_(brprofile, stdout, opt);
}

integer4 brprofile_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return fdprofile_struct_to_dumpf_(brprofile, fp, opt);
}
