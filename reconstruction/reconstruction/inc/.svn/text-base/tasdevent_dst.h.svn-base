/*
 *     Bank for raw SD fadc 1 values
 *     written by a student
 *     This work is based on JB and SRS work at least.
 *     Time-stamp: Fri Apr 10 23:19:45 2009 JST
*/

#ifndef _TASDEVENT_
#define _TASDEVENT_

#define TASDEVENT_BANKID  13001
#define TASDEVENT_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdevent_common_to_bank_();
int tasdevent_bank_to_dst_(int *NumUnit);
int tasdevent_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdevent_bank_to_common_(char *bank);
int tasdevent_common_to_dump_(int *opt1) ;
int tasdevent_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdevent_bank_buffer_ (integer4* tasdevent_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdevent_ndmax 1024 /*maximum number of detector per event      */
#define tasdevent_nhmax   3 /*maximum number of host      */
#define tasdevent_nfadc 128 /*number of time bins per channel  */


typedef struct {
  int clock;		/* clock count at the trigger timing	*/
  int max_clock;	/* maximum clock count between 1PPS	*/

  short lid;		/* logical ID				*/
  short usum;		/* summation value of the upper layer	*/
  short lsum;		/* summation value of the lower layer	*/
  short uavr;		/* average of the FADC of the upper layer*/
  short lavr;		/* average of the FADC of the lower layer*/
  short wf_id;		/* waveform id in the trigger		*/
  short num_trgwf;	/* number of triggered waveforms */
  short bank;		/* ID of the triggered waveform		*/
  short num_retry;	/* the number of the retry		*/
  short trig_code;	/* level-1 trigger code			*/
  short wf_error;	/* broken waveform data			*/

  short uwf[tasdevent_nfadc];	/* waveform of the upper layer */
  short lwf[tasdevent_nfadc];	/* waveform of the lower layer */

} SDEventSubData;


typedef struct {
  int event_code;	/* 1=data, 0=Monte Carlo	*/
  int run_id;		/* run id			*/
  int site;		/* site id			*/
  int trig_id;		/* trigger ID			*/

  int trig_code;	/* level-2 trigger code,0 is internal,
			   others are external		*/
  int code;		/* internal trigger code	*/
  int num_trgwf;	/* number of triggered waveform	*/
  int num_wf;		/* number of aquired waveforms	*/

  int bank;		/* bank id			*/
  int date;		/* triggered date		*/
  int time;		/* triggered time		*/
  int date_org;		/* original triggered date	*/
  int time_org;		/* original triggered time	*/
  int usec;		/* triggered usec		*/
  int gps_error;
  int pos;		/* triggered position		*/
  int pattern[16];	/* trigger pattern		*/

  SDEventSubData sub[tasdevent_ndmax];

} tasdevent_dst_common;

extern tasdevent_dst_common tasdevent_;

#endif


