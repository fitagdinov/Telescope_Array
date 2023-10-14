/*
 *     Bank for calibrated event
 *     written by a student
 *     Time-stamp: Wed May 06 20:20:42 2009 JST
*/

#ifndef _TASDCALIBEV_
#define _TASDCALIBEV_

#define TASDCALIBEV_BANKID  13021
#define TASDCALIBEV_BANKVERSION   003

#ifdef __cplusplus
extern "C" {
#endif
int tasdcalibev_common_to_bank_();
int tasdcalibev_bank_to_dst_(int *NumUnit);
int tasdcalibev_common_to_dst_(int *NumUnit);/* combines above 2 */
int tasdcalibev_bank_to_common_(char *bank);
int tasdcalibev_common_to_dump_(int *opt1) ;
int tasdcalibev_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdcalibev_bank_buffer_ (integer4* tasdcalibev_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdcalibev_nhmax 3 /* maximum number of hosts */
#define tasdcalibev_nwmax 3 /* maximum number of weather stations*/
#define tasdcalibev_ndmax 512 /* maximum number of detector/event*/
#define tasdcalibev_nfadc 128 /* number of FADC bins */


typedef struct {
  short site;   /* site id */
  short lid;    /* position id */

  int clock;	/* clock count at the trigger timing	*/
  int maxClock; /* maximum clock count between 1PPS	*/
  char wfId;	/* waveform id in the trigger		*/
  char numTrgwf;/* number of triggered waveforms */
  char trgCode; /* level-1 trigger code			*/
  char wfError; /* broken waveform data	or saturate
		   LSB   : upper PMT saturate (with LED data)
		   bit-1 : lower PMT saturate (with LED data)
		   bit-2 : upper PMT saturate (with MIP data)
		   bit-3 : lower PMT saturate (with MIP data)
		   bit-4 : upper FADC saturate
		   bit-5 : lower FADC saturate
		   bit-6 : daq error (FPGA to SDRAM) */
  short uwf[tasdcalibev_nfadc];	/* waveform of the upper layer */
  short lwf[tasdcalibev_nfadc];	/* waveform of the lower layer */


  float clockError; /* fluctuation of maximum clock count [ns] */
  float upedAvr;    /* average of pedestal (upper) */
  float lpedAvr;    /* average of pedestal (lower) */
  float upedStdev;  /* standard deviation of pedestal (upper) */
  float lpedStdev;  /* standard deviation of pedestal (lower) */

  float umipNonuni;       /* Non-uniformity (upper layer) */
  float lmipNonuni;       /* Non-uniformity (lower layer) */
  float umipMev2cnt;/* Mev to count conversion factor (upper) */
  float lmipMev2cnt;/* Mev to count conversion factor (lower) */
  float umipMev2pe;  /* Mev to photo-electron conversion factor
			(upper layer) */
  float lmipMev2pe;  /* Mev to photo-electron conversion factor
			(lower layer) */
  float lvl0Rate;    /* level-0 trigger rate */
  float lvl1Rate;    /* level-1 trigger rate */
  float scintiTemp;


  int warning;      /* condition of sensors and trigger rate.
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
		       bit-16 : clock count vs pedestal */
  char dontUse;     /* bad detector flag.
		       0 means OK, 1 means error.
		       LSB   : gps
		       bit-1 : clock
		       bit-2 : upper pedestal
		       bit-3 : lower pedestal
		       bit-4 : upper mip info
		       bit-5 : lower mip info
		       bit-6 : trigger rate
		       bit-7 : communication */
  char dataQuality; /* condtion of data
		       0 means exist, 1 means interpolated
		       LSB   : gps
		       bit-1 : clock
		       bit-2 : upper pedestal
		       bit-3 : lower pedestal
		       bit-4 : upper mip info
		       bit-5 : lower mip info
		       bit-6 : trigger rate
		       bit-7 : temperature */


  char trgMode0;     /* level-0 trigger mode */
  char trgMode1;     /* level-1 trigger mode */
  char gpsRunMode;   /* 1 is 3D fix, 2 is position hold*/
  short uthreLvl0;   /* threshold of level-0 trigger (upper) */
  short lthreLvl0;   /* threshold of level-0 trigger (lower) */
  short uthreLvl1;   /* threshold of level-1 trigger (upper) */
  short lthreLvl1;   /* threshold of level-1 trigger (lower) */

  float posX;	/* relative position [m], positive is east */
  float posY;	/* relative position [m], positive is north */
  float posZ;	/* relative position [m], positive is up */
  float delayns;/* signal cable delay */
  float ppsofs;	/* PPS ofset [ns] */
  float ppsflu;	/* PPS fluctuation [ns] */
  int lonmas;	/* longitude [mas] */
  int latmas;	/* latitude [mas] */
  int heicm;	/* height [cm] */

  short udec5pled;  /* maximun lineality range of upper layer
		     [FADC count] */
  short ldec5pled;  /* maximun lineality range of lower layer
		     [FADC count] */
  short udec5pmip;  /* maximun lineality range of upper layer
		     [FADC count] */
  short ldec5pmip;  /* maximun lineality range of lower layer
		     [FADC count] */


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


} SDCalibevData;


typedef struct {
  int site; /* 0 is BRFD, 1 is LRFD, 4 will be CLF */
  float atmosphericPressure;	/* [hPa] */
  float temperature;		/* [C]   */
  float humidity;		/* [%RH] */
  float rainfall;		/* [mm/hour] */
  float numberOfHails;		/* [hits/cm^2/hour]*/
} SDCalibevWeatherData;



typedef struct {
  char interactionModel[24];
  char primaryParticleType[8];
  float primaryEnergy;
  float primaryCosZenith;
  float primaryAzimuth;
  float primaryFirstIntDepth;
  float primaryArrivalTimeFromPps;
  float primaryCorePosX;
  float primaryCorePosY;
  float primaryCorePosZ;
  float thinRatio;
  float maxWeight;
  int trgCode;
  int userInfo;
  float detailUserInfo[10];
} SDCalibevSimInfo;



typedef struct {
  int eventCode;/* 1=data, 0=Monte Carlo */

  int date;     /* triggered date */
  int time;     /* triggered time */
  int usec;     /* triggered usec */
  int trgBank; /* Trigger bank ID */
  int trgPos;   /* triggered position (detector ID) */
  int trgMode;  /* LSB:BR, bit-1:LR, bit-2:SK,
		 * 0 means branch crossing trigger is not working */
  int daqMode;  /* LSB:BR, bit-1:LR, bit-2:SK, 
		 * 0 means branch crossing DAQ is not working */
  int numWf;	/* number of aquired waveforms	*/
  int numTrgwf;	/* number of triggered waveform	*/
  int numWeather;   /* the number of weather stations */
  int numAlive;
  int numDead;

  int runId[tasdcalibev_nhmax];/* run id */
  int daqMiss[tasdcalibev_nhmax];
  /* DAQ error or GPS timestamp error
     0 means OK, 1 means error.
     LSB   : DAQ stop
     bit-1 : DAQ timeout
     bit-2 : timestamp miss 1 sec
     bit-3 : timestamp miss more than 1 sec
     bit-4 : timestamp miss more than 10 min.
     bit-5 : critical error */

  SDCalibevData sub[tasdcalibev_ndmax];
  SDCalibevWeatherData weather[tasdcalibev_nwmax];
  SDCalibevSimInfo sim;


  /* the alive detectors information */
  short aliveDetLid[tasdcalibev_ndmax];
  short aliveDetSite[tasdcalibev_ndmax];
  float aliveDetPosX[tasdcalibev_ndmax];
  float aliveDetPosY[tasdcalibev_ndmax];
  float aliveDetPosZ[tasdcalibev_ndmax];

  /* the dead detectors information */
  short deadDetLid[tasdcalibev_ndmax];
  short deadDetSite[tasdcalibev_ndmax];
  float deadDetPosX[tasdcalibev_ndmax];
  float deadDetPosY[tasdcalibev_ndmax];
  float deadDetPosZ[tasdcalibev_ndmax];

  int footer;

} tasdcalibev_dst_common;

extern tasdcalibev_dst_common tasdcalibev_;

#endif


