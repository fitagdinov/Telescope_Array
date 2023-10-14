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
#include "convtime.h"

fdtime_dst_common fdtime_;

integer4 fdtime_blen = 0; /* not static because it needs to be accessed by the c files of the derived banks */
static integer4 fdtime_maxlen = sizeof(integer4) * 2 + sizeof(fdtime_dst_common);
static integer1 *fdtime_bank = NULL;

/* get (packed) buffer pointer and size */
integer1* fdtime_bank_buffer_ (integer4* fdtime_bank_buffer_size)
{
  (*fdtime_bank_buffer_size) = fdtime_blen;
  return fdtime_bank;
}



static void fdtime_abank_init(integer1* (*pbank) ) {
  *pbank = (integer1 *)calloc(fdtime_maxlen, sizeof(integer1));
  if (*pbank==NULL) {
      fprintf (stderr,"fdtime_abank_init: fail to assign memory to bank. Abort.\n");
      exit(0);
  }
}

static void fdtime_bank_init() {fdtime_abank_init(&fdtime_bank);}

integer4 fdtime_common_to_bank_() {
  if (fdtime_bank == NULL) fdtime_bank_init();
  return fdtime_struct_to_abank_(&fdtime_, &fdtime_bank, FDTIME_BANKID, FDTIME_BANKVERSION);
}
integer4 fdtime_bank_to_dst_ (integer4 *unit) {return fdtime_abank_to_dst_(fdtime_bank, unit);}
integer4 fdtime_common_to_dst_(integer4 *unit) {
  if (fdtime_bank == NULL) fdtime_bank_init();
  return fdtime_struct_to_dst_(&fdtime_, fdtime_bank, unit, FDTIME_BANKID, FDTIME_BANKVERSION);
}
integer4 fdtime_bank_to_common_(integer1 *bank) {return fdtime_abank_to_struct_(bank, &fdtime_);}
integer4 fdtime_common_to_dump_(integer4 *opt) {return fdtime_struct_to_dumpf_(&fdtime_, stdout, opt);}
integer4 fdtime_common_to_dumpf_(FILE* fp, integer4 *opt) {return fdtime_struct_to_dumpf_(&fdtime_, fp, opt);}

integer4 fdtime_struct_to_abank_(fdtime_dst_common *fdtime, integer1 *(*pbank), integer4 id, integer4 ver) {
  integer4 rcode, nobj;
  integer1 *bank;

  if (*pbank == NULL) fdtime_abank_init(pbank);

  bank = *pbank;
  rcode = dst_initbank_(&id, &ver, &fdtime_blen, &fdtime_maxlen, bank);

// Initialize fdtime_blen and pack the id and version to bank

  nobj = 1;
  rcode += dst_packi4_(&fdtime->julian,    &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_packi4_(&fdtime->jsecond,   &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_packi4_(&fdtime->nano,  &nobj, bank, &fdtime_blen, &fdtime_maxlen);
/*  rcode += dst_packi4_(&fdtime->yyyymmdd,    &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_packi4_(&fdtime->hhmmss,   &nobj, bank, &fdtime_blen, &fdtime_maxlen); */
  rcode += dst_packi2_(&fdtime->siteid,     &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_packi2_(&fdtime->part,    &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_packi4_(&fdtime->event_num,     &nobj, bank, &fdtime_blen, &fdtime_maxlen);

  rcode += dst_packi4_(&fdtime->ctdclock_rate,       &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_packi4_(&fdtime->gps1pps_tick,      &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_packi4_(&fdtime->ctdclock,  &nobj, bank, &fdtime_blen, &fdtime_maxlen);
/*  rcode += dst_packr8_(&fdtime->ns_per_cc,      &nobj, bank, &fdtime_blen, &fdtime_maxlen); */

  return rcode;
}

integer4 fdtime_abank_to_dst_(integer1 *bank, integer4 *unit) {
  return dst_write_bank_(unit, &fdtime_blen, bank);
}

integer4 fdtime_struct_to_dst_(fdtime_dst_common *fdtime, integer1 *bank, integer4 *unit, integer4 id, integer4 ver) {
  integer4 rcode;
  if ( (rcode = fdtime_struct_to_abank_(fdtime, &bank, id, ver)) ) {
      fprintf(stderr, "fdtime_struct_to_abank_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  if ( (rcode = fdtime_abank_to_dst_(bank, unit)) ) {
      fprintf(stderr, "fdtime_abank_to_dst_ ERROR : %ld\n", (long)rcode);
      exit(0);
  }
  return 0;
}

integer4 fdtime_abank_to_struct_(integer1 *bank, fdtime_dst_common *fdtime) {
  integer4 rcode = 0 ;
  integer4 nobj;
  real8 dummy;
  integer4 ver;
  
  fdtime_blen = 1 * sizeof(integer4);   /* skip id and version  */

  nobj = 1;
  rcode += dst_unpacki4_(&ver, &nobj, bank, &fdtime_blen, &fdtime_maxlen);  

  rcode += dst_unpacki4_(&fdtime->julian,    &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_unpacki4_(&fdtime->jsecond,   &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_unpacki4_(&fdtime->nano,  &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  
  mjlday2ymdsec((double)(fdtime->julian - MJLDOFF) + (double)fdtime->jsecond/SECPDAY, &fdtime->yyyymmdd, &dummy);
  
  integer4 utcsec = (fdtime->jsecond + 43200) % 86400;
  integer4 hour = utcsec / 3600;
  integer4 minute = (utcsec % 3600) / 60;
  integer4 second = utcsec % 60;
  fdtime->hhmmss = 10000 * hour + 100 * minute + second;
  
  
/*  rcode += dst_unpacki4_(&fdtime->yyyymmdd,    &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_unpacki4_(&fdtime->hhmmss,   &nobj, bank, &fdtime_blen, &fdtime_maxlen); */
  rcode += dst_unpacki2_(&fdtime->siteid,     &nobj, bank, &fdtime_blen, &fdtime_maxlen);  
  rcode += dst_unpacki2_(&fdtime->part,       &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_unpacki4_(&fdtime->event_num,      &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_unpacki4_(&fdtime->ctdclock_rate,  &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_unpacki4_(&fdtime->gps1pps_tick,     &nobj, bank, &fdtime_blen, &fdtime_maxlen);
  rcode += dst_unpacki4_(&fdtime->ctdclock,  &nobj, bank, &fdtime_blen, &fdtime_maxlen);
/*  rcode += dst_unpackr8_(&fdtime->ns_per_cc, &nobj, bank, &fdtime_blen, &fdtime_maxlen); */
  fdtime->ns_per_cc = 1e9/(double)fdtime->ctdclock_rate;

  return rcode;
}

integer4 fdtime_struct_to_dump_(fdtime_dst_common *fdtime, integer4 *long_output) {
  return fdtime_struct_to_dumpf_(fdtime, stdout, long_output);
}

integer4 fdtime_struct_to_dumpf_(fdtime_dst_common *fdtime, FILE* fp, integer4 *long_output) {
  (void)(long_output);
  if (fdtime->siteid == BR)
    fprintf (fp, "\n\nBRTIME bank (corrected waveform start time for Black Rock Mesa FD)\n");
  else if (fdtime->siteid == LR)
    fprintf (fp, "\n\nLRTIME bank (corrected waveform start time for Long Ridge FD)\n");  
  else
    fprintf (fp, "\n\nFDTIME bank (corrected waveform start time for unrecognized FD)\n");  
  
  
  fprintf (fp, "%08d %06d.%09d | Part %6d Event %6d\n", 
    fdtime->yyyymmdd, fdtime->hhmmss, fdtime->nano, fdtime->part, fdtime->event_num);

  fprintf (fp, "CTD at start of waveform:   %10u\n", fdtime->ctdclock);
  fprintf (fp, "CTD at start of GPS second: %10u (difference %c 2%c29: %d )\n", fdtime->gps1pps_tick,'%','^',((fdtime->ctdclock - fdtime->gps1pps_tick + 0x20000000) & 0x1fffff));
  fprintf (fp, "CTD clock cycles during previous GPS second: %d ( %.8f ns per cycle)\n", fdtime->ctdclock_rate, fdtime->ns_per_cc);
  
  fprintf (fp, "\n\n");

  return 0;
}


