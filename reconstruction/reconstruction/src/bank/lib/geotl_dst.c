// See geofd_dst.h for more information

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "dst_std_types.h"
#include "dst_bank_proto.h"
#include "dst_pack_proto.h"

#include "univ_dst.h"
#include "geofd_dst.h"
#include "geotl_dst.h"

geotl_dst_common geotl_;
static geofd_dst_common* geotl = &geotl_;

//static integer4 geotl_blen;
static integer4 geotl_maxlen = sizeof(integer4) * 2 + sizeof(geotl_dst_common);
static integer1 *geotl_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* geotl_bank_buffer_ (integer4* geotl_bank_buffer_size)
{
  (*geotl_bank_buffer_size) = geofd_blen;
  return geotl_bank;
}



static void geotl_bank_init() {
  geotl_bank = (integer1 *)calloc(geotl_maxlen, sizeof(integer1));
  if (geotl_bank==NULL) {
      fprintf (stderr,"geotl_bank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

integer4 geotl_common_to_bank_() {
  if (geotl_bank == NULL) geotl_bank_init();
  return geofd_struct_to_abank_(geotl, &geotl_bank, GEOTL_BANKID, GEOTL_BANKVERSION);
}

integer4 geotl_bank_to_dst_ (integer4 *unit) {
  return geofd_abank_to_dst_(geotl_bank, unit);
}

integer4 geotl_common_to_dst_(integer4 *unit) {
  if (geotl_bank == NULL) geotl_bank_init();
  return geofd_struct_to_dst_(geotl, geotl_bank, unit, GEOTL_BANKID, GEOTL_BANKVERSION);
}

integer4 geotl_bank_to_common_(integer1 *bank) {
  return geofd_abank_to_struct_(bank, geotl);
}

integer4 geotl_common_to_dump_(integer4 *opt) {
  return geofd_struct_to_dumpf_(geotl, stdout, opt);
}

integer4 geotl_common_to_dumpf_(FILE* fp, integer4 *opt) {
  return geofd_struct_to_dumpf_(geotl, fp, opt);
}
