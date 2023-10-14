/*
 *     Bank for calibrated event
 *     written by a student
 *     Time-stamp: Sun May 03 15:45:25 2009 JST
*/

#ifndef _TASDCALIB_
#define _TASDCALIB_

#define TASDCALIB_BANKID  13022
#define TASDCALIB_BANKVERSION   003

#ifdef __cplusplus
extern "C" {
#endif
int tasdcalib_common_to_bank_();
int tasdcalib_bank_to_dst_(int *NumUnit);
int tasdcalib_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdcalib_bank_to_common_(char *bank);
int tasdcalib_common_to_dump_(int *opt1) ;
int tasdcalib_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdcalib_bank_buffer_ (integer4* tasdcalib_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdcalib_nhmax   3 /* maximum number of host */
#define tasdcalib_nwmax   3 /* maximum number of weather station */
#define tasdcalib_ndmax 512 /* maximum number of detector/event */
#define tasdcalib_ntmax 100 /* maximum number of trigger in 10 min. */


typedef struct {
  int site;	     /* site id */
  int numTrg;
  int trgBank[tasdcalib_ntmax]; /* Trigger bank ID */
  int trgSec[tasdcalib_ntmax];   /* Trigger sending time in 10 min.
				   [0-599] */
  short trgPos[tasdcalib_ntmax]; /* Triggered position */
  int daqMode[tasdcalib_ntmax];    /* daq code from central PC */
  char miss[600];     /* DAQ error or GPS timestamp error
		         0 means OK, 1 means error.
			 LSB   : DAQ stop
			 bit-1 : DAQ timeout
			 bit-2 : timestamp miss 1 sec
			 bit-3 : timestamp miss more than 1 sec
			 bit-4 : timestamp miss more than 10 min.
			 bit-5 : critical error
		      */
  short run_id[600];
} SDCalibHostData;


typedef struct {
  int site;	     /* site id */
  int lid;	     /* position id */
  int livetime;	     /* livetime in 10 min */

  int warning; /* condition of sensors and trigger rate.
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

  char dontUse;     /* bad detector flag.
		       0 means OK, 1 means error.
		       LSB   : gps
		       bit-1 : clock
		       bit-2 : upper pedestal
		       bit-3 : lower pedestal
		       bit-4 : upper mip info
		       bit-5 : lower mip info
		       bit-6 : trigger rate
		       bit-7 : temperature
		    */
  char dataQuality; /* condtion of data
		       0 means exist, 1 means interpolated
		       LSB   : gps
		       bit-1 : clock
		       bit-2 : upper pedestal
		       bit-3 : lower pedestal
		       bit-4 : upper mip info
		       bit-5 : lower mip info
		       bit-6 : trigger rate
		       bit-7 : temperature
		    */

  char gpsRunMode;   /* 1 is 3D fix, 2 is position hold*/

  char miss[75];     /* comm. error bit field */

  float clockFreq; /* clock frequency [Hs] */
  float clockChirp;/* time deviation of clock frequency [Hs/s] */
  float clockError;  /* fluctuation of clock [ns] */

  float upedAvr;     /* average of pedestal (upper layer) */
  float lpedAvr;     /* average of pedestal (lower layer) */
  float upedStdev;   /* standard deviation of pedestal
			(upper layer) */
  float lpedStdev;   /* standard deviation of pedestal
			(lower layer) */
  float upedChisq;   /* Chi square value (upper layer) */
  float lpedChisq;   /* Chi square value (lower layer) */

  float umipNonuni;       /* Non-uniformity (upper layer) */
  float lmipNonuni;       /* Non-uniformity (lower layer) */
  float umipMev2cnt; /* Mev to count conversion factor
			(upper layer) */
  float lmipMev2cnt; /* Mev to count conversion factor
			(lower layer) */
  float umipMev2pe;  /* Mev to photo-electron conversion factor
			(upper layer) */
  float lmipMev2pe;  /* Mev to photo-electron conversion factor
			(lower layer) */
  float umipChisq;   /* Chi square value (upper layer) */
  float lmipChisq;   /* Chi square value (lower layer) */

  float lvl0Rate;	  /* level-0 trigger rate */
  float lvl1Rate;	  /* level-1 trigger rate */

  float scinti_temp;


  // [0] - lower layer, [1] - upper layer
  int pchmip[2];     /* peak channel of 1MIP histograms */
  int pchped[2];     /* peak channel of pedestal histograms */
  int lhpchmip[2];   /* left half-peak channel for 1mip histogram */
  int lhpchped[2];   /* left half-peak channel of pedestal histogram */
  int rhpchmip[2];   /* right half-peak channel for 1mip histogram */
  int rhpchped[2];   /* right half-peak channel of pedestal histograms */
  
  /* Results from fitting 1MIP histograms */
  int mftndof[2]; /* number of degrees of freedom in 1MIP fit */
  float mip[2];        /* 1MIP value (ped. subtracted) */
  float mftchi2[2];    /* chi2 of the 1MIP fit */
  
  /* 
     1MIP Fit function: 
     [3]*(1+[2]*(x-[0]))*exp(-(x-[0])*(x-[0])/2/[1]/[1])/sqrt(2*PI)/[1]
     4 fit parameters:
     [0]=Gauss Mean
     [1]=Gauss Sigma
     [2]=Linear Coefficient
     [3]=Overall Scalling Factor
  */
  float mftp[2][4];    /* 1MIP fit parameters */
  float mftpe[2][4];   /* Errors on 1MIP fit parameters */


} SDCalibSubData;


typedef struct {
  int site; /* 0 is BRFD, 1 is LRFD, 4 will be CLF */
  float averageWindSpeed;	/* [m/s] */
  float maximumWindSpeed;	/* [m/s] */
  float windDirection;		/* 0 is north, 90 is east [deg] */
  float atmosphericPressure;	/* [hPa] */
  float temperature;		/* [C]   */
  float humidity;		/* [%RH] */
  float rainfall;		/* [mm/hour] */
  float numberOfHails;		/* [hits/cm^2/hour]*/
} SDCalibWeatherData;


typedef struct {

  int num_host;     /* the number of hosts */
  int num_det;      /* the number of detectors */
  int num_weather;   /* the number of weather stations */
  int date;      /* year month day */
  int time;      /* hour minute second */
  char trgMode[600]; /* Trigger Mode */

  SDCalibHostData    host[tasdcalib_nhmax];
  SDCalibSubData     sub[tasdcalib_ndmax];
  SDCalibWeatherData weather[tasdcalib_nwmax];

  int footer;

} tasdcalib_dst_common;

extern tasdcalib_dst_common tasdcalib_;

#endif


