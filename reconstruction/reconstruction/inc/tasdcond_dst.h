/*
 *     Bank for SD condition
 *     written by a student
 *     Time-stamp: Sun May 03 15:35:32 2009 JST
*/

#ifndef _TASDCOND_
#define _TASDCOND_

#define TASDCOND_BANKID  13015
#define TASDCOND_BANKVERSION   003

#ifdef __cplusplus
extern "C" {
#endif
int tasdcond_common_to_bank_();
int tasdcond_bank_to_dst_(int *NumUnit);
int tasdcond_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdcond_bank_to_common_(char *bank);
int tasdcond_common_to_dump_(int *opt1) ;
int tasdcond_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdcond_bank_buffer_ (integer4* tasdcond_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdcond_nhmax   3 /*maximum number of host*/
#define tasdcond_ndmax 512 /*maximum number of detector*/
#define tasdcond_ntmax 100 /*maximum number of trigger in 10 min.*/


typedef struct {
  int numTrg;
  int trgBank[tasdcond_ntmax];  /* Trigger bank ID */
  int trgSec[tasdcond_ntmax];   /* Trigger sending time in 10 min.
				   [0-599] */
  short trgPos[tasdcond_ntmax]; /* Triggered position */
  int daqMode[tasdcond_ntmax];    /* daq code from central PC */
  char miss[600];      /* condition of DAQ,
			  0 means OK, 1 means error.
			  LSB : stop DAQ,  bit-1 : DAQ timeout */
  char gpsError[600];  /* condition of GPS timestamp.
			  0 : no probrem
			  1 : stop or skip 1 sec, but recovered.
			  2 : stop more than 1 sec, but recovered.
			  3 : skip more than 1 sec, but recovered.
			  4 : can be recovered from hybrid event.
			  5 : broken data or bug. */
  short run_id[600];
} SDCondHostData;


typedef struct {
  int slowCond[10]; /* condition of sensors and trigger rate.
		       0 means OK, 1 means error.
		       LSB    : level-0 trigger rate
		       bit-1  : level-1 trigger rate
		       bit-2  : temperature sensor on scinti.
		       bit-3  : temperature sensor on elec.
		       bit-4  : temperature sensor on battery
		       bit-5  : temperature sensor on charge cont.
		       bit-6  : humidity sensor on scinti.
		       bit-7  : battery voltage
		       bit-8  : solar panel voltage
		       bit-9  : LV value of charge cont.
		       bit-10 : current from solar panel
		       bit-11 : ground level
		       bit-12 : 1.2V
		       bit-13 : 1.8V
		       bit-14 : 3.3V
		       bit-15 : 5.0V
		       bit-16 : clock count vs pedestal
		    */
  float clockFreq; /* clock frequency [Hs] */
  float clockChirp;/* time deviation of clock frequency [Hs/s] */
  float clockError;  /* fluctuation of clock [ns] */
  char miss[600];  //for current 1sec. from L
  short site;	     /* site id */
  short lid;	     /* position id */
  char gpsHealth;  //for past 10minutes from pps557
  char gpsRunMode; //for past 10minutes from pps557
} SDCondSubData;


typedef struct {
  int num_det;   /* the number of detectors       */
  int date;      /* year month day */
  int time;      /* hour minute second */
  char trgMode[600];  /* Trigger mode */
  SDCondHostData host[tasdcond_nhmax];
  SDCondSubData sub[tasdcond_ndmax];
  int footer;
} tasdcond_dst_common;


extern tasdcond_dst_common tasdcond_;


#endif


