
/*
 *     Bank for the raw TALE-FD mirror snapshot data
 *     Dmitri Ivanov (dmiivanov@gmail.com)
 *     Jan 30, 2015
 *     Last modified: Feb 04, 2015
*/

#ifndef _TLMSNP_
#define _TLMSNP_

#define TLMSNP_BANKID  12507
#define TLMSNP_BANKVERSION   000

#ifdef __cplusplus
extern "C" {
#endif
integer4 tlmsnp_common_to_bank_();
integer4 tlmsnp_bank_to_dst_(integer4 *NumUnit);
integer4 tlmsnp_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 tlmsnp_bank_to_common_(integer1 *bank);
integer4 tlmsnp_common_to_dump_(integer4 *opt1) ;
integer4 tlmsnp_common_to_dumpf_(FILE* fp,integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* tlmsnp_bank_buffer_ (integer4* tlmsnp_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif

integer4 tlmsnp_time_print_(integer4 *second, integer4 *clkcnt);
integer4 tlmsnp_time_fprint(FILE* fp, integer4 *second, integer4 *clkcnt);

// for each snapshot mean and variance over 256 100nS time slices are calculated
// then the results are summed and recorded over nsnp snapshots.
// channels from 1 to 256 correspond to tubes
// channels from 257 to 320 correspond to the trigger sums information 
// 257-272 = Trigger (high-gain), cols 1-16
// 273-288 = Low-gain, cols 1-16
// 289-304 = Trigger (high-gain), rows 1-16
// 305-320 = Low-gain, rows 1-16
// the C-array index always means (channel ID - 1)
#define tlmsnp_nchan_mir 320         
typedef struct
{
  integer4 yymmdd;                                    // year, month, day
  integer4 hhmmss;                                    // hour, minute, second
  integer4 secfrac;                                   // second fraction with respect to GPS second [100nS]  
  integer4 mirid;                                     // mirror ID for which this snapshot event has been read out
  integer4 nsnp;                                      // number of snapshots that went into making the sums of the statistics
  real4    channel_mean[tlmsnp_nchan_mir];            // 256 time slice mean summed over nsnp snapshots [FADC counts]
  real4    channel_var[tlmsnp_nchan_mir];             // 256 time slice variance summed over nsnp snapshots [FADC counts]
  real4    channel_vgain[tlmsnp_nchan_mir];           // vertical ADC gain for the channel
  real4    channel_hgain[tlmsnp_nchan_mir];           // horizontal ADC gain for the channel
} tlmsnp_dst_common;

extern tlmsnp_dst_common tlmsnp_;

#endif
