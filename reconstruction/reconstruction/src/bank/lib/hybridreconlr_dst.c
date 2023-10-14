// Created 2010/09/17 by D. Ivanov <ivanov@physics.rutgers.edu>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "hybridreconfd_dst.h"
#include "hybridreconlr_dst.h"

hybridreconlr_dst_common hybridreconlr_;
static hybridreconfd_dst_common* hybridreconlr = &hybridreconlr_;

//static integer4 hybridreconlr_blen;
static integer4 hybridreconlr_maxlen = sizeof(integer4) * 2 + sizeof(hybridreconlr_dst_common);
static integer1 *hybridreconlr_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hybridreconlr_bank_buffer_ (integer4* hybridreconlr_bank_buffer_size)
{
  (*hybridreconlr_bank_buffer_size) = hybridreconfd_blen;
  return hybridreconlr_bank;
}



static void hybridreconlr_bank_init() {
  hybridreconlr_bank = (integer1 *)calloc(hybridreconlr_maxlen, sizeof(integer1));
  if (hybridreconlr_bank==NULL) {
    fprintf (stderr,"hybridreconlr_bank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

integer4 hybridreconlr_common_to_bank_() {
  if (hybridreconlr_bank == NULL) hybridreconlr_bank_init();
  return hybridreconfd_struct_to_abank_(hybridreconlr, &hybridreconlr_bank, HYBRIDRECONLR_BANKID, HYBRIDRECONLR_BANKVERSION);
}

integer4 hybridreconlr_bank_to_dst_ (integer4 *unit) {
  return hybridreconfd_abank_to_dst_(hybridreconlr_bank, unit);
}

integer4 hybridreconlr_common_to_dst_(integer4 *unit) {
  if (hybridreconlr_bank == NULL) hybridreconlr_bank_init();
  return hybridreconfd_struct_to_dst_(hybridreconlr, hybridreconlr_bank, unit, HYBRIDRECONLR_BANKID, HYBRIDRECONLR_BANKVERSION);
}

integer4 hybridreconlr_bank_to_common_(integer1 *bank) {
  return hybridreconfd_abank_to_struct_(bank, hybridreconlr);
}

integer4 hybridreconlr_common_to_dump_(integer4 *opt) {
  return hybridreconfd_struct_to_dumpf_(1, hybridreconlr, stdout, opt);
}

integer4 hybridreconlr_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return hybridreconfd_struct_to_dumpf_(1, hybridreconlr, fp, opt);
}
