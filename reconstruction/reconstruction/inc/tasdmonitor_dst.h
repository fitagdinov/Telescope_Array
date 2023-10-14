/*
 *     Bank for raw SD monitor values
 *     written by a student
 *     This work is based on JB and SRS work at least.
 *     Time-stamp: Fri Apr 10 23:38:21 2009 JST
*/

#ifndef _TASDMONITOR_
#define _TASDMONITOR_

#define TASDMONITOR_BANKID  13002
#define TASDMONITOR_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdmonitor_common_to_bank_();
int tasdmonitor_bank_to_dst_(int *NumUnit);
int tasdmonitor_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdmonitor_bank_to_common_(char *bank);
int tasdmonitor_common_to_dump_(int *opt1);
int tasdmonitor_common_to_dumpf_(FILE* fp,int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdmonitor_bank_buffer_ (integer4* tasdmonitor_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


int tasdmonitor_dst_to_common_(FILE* fp, int bank);

#define tasdmonitor_nhmax 3	  /* number of Host */
#define tasdmonitor_ndmax 512	 /* number of SD, actual value is
				    503 and 9 is for extention */


typedef struct {
  int run_id; /* run ID */
  int trial;	  /* number of trial */
  int date;	   /* year month day */
  int time;	   /* hour minute second */
  int date_org;	      /* year month day from data file*/
  int time_org;	      /* hour minute second from data file */
  int bank[10];       /* DAQ bank ID */
  int eventInfoCode;  /* trigger code sent to SDs*/
  short num_tbl;      /* number of trigger table */
  short num_trigger;  /* number of trigger */
  short num_bank;     /* number of DAQ bank */
  short cur_time;     /* gps comm. check */
  short num_sat;      /* number of sattelite */
  short num_retry;    /* number of retry of WLAN comm. */
  short num_error;    /* number of error of WLAN comm. */
  short num_debug;    /* number of #t (for debug) . */
  short vol[4];       /* voltage of power supply */
  short cc[4];	/* voltage from charge controller */
  short gps_error;
  short error_flag;   /* error flag(bit-code, code(0) i.e. LSB is
			 trigger over run, code(1) is trigger
			 number overflow, code(2) is broken data
			 file) */
} SDMonitorHostPPSData;


typedef struct {
  int run_id; /* run ID */
  short runtime;      /* run time in 10 min */
  short deadtime;     /* dead time in 10min */
  short num_det;      /* number of detector in the site */
  short error_flag;   /* error flag(bit-code, code(0) i.e. LSB is
			 trigger over run, code(1) is trigger
			 number overflow) */
  short num_retry;    /* number of retry of WLAN comm. in 10 min */
  short num_error;    /* number of error of WLAN comm. in 10 min */
  SDMonitorHostPPSData pps[600];
} SDMonitorHostData;


typedef struct {
  int max_clock;      /* maximum clock count between 1PPS */
  int wlan_health;    /* status of then WLAN comm. */
  short num_retry;    /* number of retry of WLAN comm. */
  short cur_time;     /* 10 minutes counter, 0-599 */
  short num_wf;       /* number of triggered waveform */
  short num_tbl;      /* number of trigger table */
} SDMonitorSubPPSData;


typedef struct {
  int mip1[0x200];   /*    0- 511 (0x0000-0x01ff) : MIP histgram */
  int mip2[0x200];   /*  512-1023 (0x0200-0x03ff) : MIP histgram */
  int ped1[0x100];   /* 1024-1279 (0x0400-0x04ff) : pedestal     */
  int ped2[0x100];   /* 1280-1535 (0x0500-0x05ff) : pedestal     */
  int phl1[0x80];    /* 1536-1663 (0x0600-0x067f) :
			pulse height linearity */
  int phl2[0x80];    /* 1664-1791 (0x0680-0x06ff) :
			pulse height linearity */
  int pcl1[0x80];    /* 1792-1919 (0x0700-0x077f) :
			pulse charge linearity */
  int pcl2[0x80];    /* 1920-2047 (0x0780-0x07ff) :
			pulse charge linearity */
  int cc_adc[10][8]; /* 2048-2127 :
			ADC values of charge controller	       */
  int sd_adc[10][8]; /* 2128-2207 : ADC values of main board   */
  int rate[10][2];   /* 2208-2227 : 1st level trigger rate     */
  int date;	     /* 2228			     */
  int time;	     /* 2229			     */
  int gps_flag;      /* 2230			     */
  int cur_rate2;     /* 2231 : 1st level trigger rate,
			just for debugging	*/
  int num_packet;    /* 2232 : number of recieved packet,
			just for debugging     */
  int num_sat;       /* 2233 : number of satellite	       */
  int gps_lat;       /* 2234 : gps latitude, not installed     */
  int gps_lon;       /* 2235 : gps longitude, not installed    */
  int gps_hei;       /* 2236 : gps height, not installed       */
  int dummy[10];     /* 2237-2240 : for extention	       */
} SDMonitorSub10mData;


typedef struct {
  short site;	/* site id, 0 is BR, 1 is LR, 2 is SK */
  short lid;	 /* logical id (positin id) */

  SDMonitorSubPPSData pps[600];
  SDMonitorSub10mData mon;

  /* for status check*/
  int num_error;     /* number of WLAN error in 10 min */
  int num_retry;     /* number of retry in 10 min */
  int livetime;      /* livetime in 10 min */

} SDMonitorSubData;


/* fixed data size, 1.3 GB/day  */
typedef struct {
  int date;				   /* year month day */
  int time;				   /* hour minute second */
  int num_det;				/* number of detector */
  short lid[tasdmonitor_ndmax];	       /* logical id list */
  SDMonitorHostData host[tasdmonitor_nhmax]; /*HOST monitor data*/
  SDMonitorSubData  sub[tasdmonitor_ndmax];  /*SD monitor data */
  int footer;				     /* for data check */
} tasdmonitor_dst_common;

extern tasdmonitor_dst_common tasdmonitor_;

#endif


