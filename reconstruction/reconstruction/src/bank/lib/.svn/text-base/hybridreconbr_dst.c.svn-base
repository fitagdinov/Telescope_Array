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
#include "hybridreconbr_dst.h"

hybridreconbr_dst_common hybridreconbr_;
static hybridreconfd_dst_common* hybridreconbr = &hybridreconbr_;

//static integer4 hybridreconbr_blen;
static integer4 hybridreconbr_maxlen = sizeof(integer4) * 2 + sizeof(hybridreconbr_dst_common);
static integer1 *hybridreconbr_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* hybridreconbr_bank_buffer_ (integer4* hybridreconbr_bank_buffer_size)
{
  (*hybridreconbr_bank_buffer_size) = hybridreconfd_blen;
  return hybridreconbr_bank;
}



static void hybridreconbr_bank_init() {
  hybridreconbr_bank = (integer1 *)calloc(hybridreconbr_maxlen, sizeof(integer1));
  if (hybridreconbr_bank==NULL) {
    fprintf (stderr,"hybridreconbr_bank_init: fail to assign memory to bank. Abort.\n");
    exit(0);
  }
}

integer4 hybridreconbr_common_to_bank_() {
  if (hybridreconbr_bank == NULL) hybridreconbr_bank_init();
  return hybridreconfd_struct_to_abank_(hybridreconbr, &hybridreconbr_bank, HYBRIDRECONBR_BANKID, HYBRIDRECONBR_BANKVERSION);
}

integer4 hybridreconbr_bank_to_dst_ (integer4 *unit) {
  return hybridreconfd_abank_to_dst_(hybridreconbr_bank, unit);
}

integer4 hybridreconbr_common_to_dst_(integer4 *unit) {
  if (hybridreconbr_bank == NULL) hybridreconbr_bank_init();
  return hybridreconfd_struct_to_dst_(hybridreconbr, hybridreconbr_bank, unit, HYBRIDRECONBR_BANKID, HYBRIDRECONBR_BANKVERSION);
}

integer4 hybridreconbr_bank_to_common_(integer1 *bank) {
  return hybridreconfd_abank_to_struct_(bank, hybridreconbr);
}

integer4 hybridreconbr_common_to_dump_(integer4 *opt) {
  return hybridreconfd_struct_to_dumpf_(0, hybridreconbr, stdout, opt);
}

integer4 hybridreconbr_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return hybridreconfd_struct_to_dumpf_(0, hybridreconbr, fp, opt);
}
