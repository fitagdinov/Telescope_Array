/* Created 2013/09/26 TAS */

#ifndef _FDTIME_DST_
#define _FDTIME_DST_

#define FDTIME_BANKID		12059
#define FDTIME_BANKVERSION	0

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdtime_common_to_bank_();
integer4 fdtime_bank_to_dst_(integer4 *unit);
integer4 fdtime_common_to_dst_(integer4 *unit); // combines above 2
integer4 fdtime_bank_to_common_(integer1 *bank);
integer4 fdtime_common_to_dump_(integer4 *opt);
integer4 fdtime_common_to_dumpf_(FILE *fp, integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* fdtime_bank_buffer_ (integer4* fdtime_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct {
  integer4 julian;                  // Julian date of event
  integer4 jsecond;                 // second of Julian day
  integer4 nano;
  integer4 yyyymmdd;                // UTC date in ISO-8601 format
  integer4 hhmmss;
  integer2 siteid;
  integer2 part;
  integer4 event_num;

  integer4 ctdclock_rate;           // Hz, cycles between the last two GPS secs
  integer4 gps1pps_tick;            // clock at last GPS sec
  integer4 ctdclock;                // clock at start of waveform
  real8 ns_per_cc;                  // reciprocal of ctdclock_rate
  
} fdtime_dst_common;

extern fdtime_dst_common fdtime_;
extern integer4 fdtime_blen; /* needs to be accessed by the c files of the derived banks */ 

integer4 fdtime_struct_to_abank_(fdtime_dst_common *fdtime, integer1 *(*pbank), integer4 id, integer4 ver);
integer4 fdtime_abank_to_dst_(integer1 *bank, integer4 *unit);
integer4 fdtime_struct_to_dst_(fdtime_dst_common *fdtime, integer1 *bank, integer4 *unit, integer4 id, integer4 ver);
integer4 fdtime_abank_to_struct_(integer1 *bank, fdtime_dst_common *fdtime);
integer4 fdtime_struct_to_dump_(fdtime_dst_common *fdtime, integer4 *opt);
integer4 fdtime_struct_to_dumpf_(fdtime_dst_common *fdtime, FILE *fp, integer4 *opt);

#endif
