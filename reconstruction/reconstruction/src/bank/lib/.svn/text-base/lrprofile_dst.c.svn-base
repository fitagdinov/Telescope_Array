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
#include "lrprofile_dst.h"

lrprofile_dst_common lrprofile_;
static fdprofile_dst_common* lrprofile = &lrprofile_;

//static integer4 lrprofile_blen;
static integer4 lrprofile_maxlen = sizeof(integer4) * 2 + sizeof(lrprofile_dst_common);
static integer1 *lrprofile_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* lrprofile_bank_buffer_ (integer4* lrprofile_bank_buffer_size)
{
  (*lrprofile_bank_buffer_size) = fdprofile_blen;
  return lrprofile_bank;
}



static void lrprofile_bank_init() {
  lrprofile_bank = (integer1 *)calloc(lrprofile_maxlen, sizeof(integer1));
  if (lrprofile_bank==NULL) {
      fprintf (stderr,"lrprofile_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 lrprofile_common_to_bank_() {
  if (lrprofile_bank == NULL) lrprofile_bank_init();
  return fdprofile_struct_to_abank_(lrprofile, &lrprofile_bank, LRPROFILE_BANKID, LRPROFILE_BANKVERSION);
}

integer4 lrprofile_bank_to_dst_ (integer4 *unit) {
  return fdprofile_abank_to_dst_(lrprofile_bank, unit);
}

integer4 lrprofile_common_to_dst_(integer4 *unit) {
  if (lrprofile_bank == NULL) lrprofile_bank_init();
  return fdprofile_struct_to_dst_(lrprofile, lrprofile_bank, unit, LRPROFILE_BANKID, LRPROFILE_BANKVERSION);
}

integer4 lrprofile_bank_to_common_(integer1 *bank) {
  return fdprofile_abank_to_struct_(bank, lrprofile);
}

integer4 lrprofile_common_to_dump_(integer4 *opt) {
  return fdprofile_struct_to_dumpf_(lrprofile, stdout, opt);
}

integer4 lrprofile_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return fdprofile_struct_to_dumpf_(lrprofile, fp, opt);
}
