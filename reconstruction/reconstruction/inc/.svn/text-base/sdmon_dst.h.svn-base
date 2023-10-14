/*
 *     Bank for SD raw monitoring data
 *     Dmitri Ivanov (dmiivanov@gmail.com)
 *     Jun 17, 2008


 *     Last modified: Dmitri Ivanov, Apr. 30, 2019
 
*/

#ifndef _SDMON_
#define _SDMON_

#define SDMON_BANKID  13102
#define SDMON_BANKVERSION   001

#ifdef __cplusplus
extern "C" {
#endif
integer4 sdmon_common_to_bank_ ();
integer4 sdmon_bank_to_dst_ (integer4 * NumUnit);
integer4 sdmon_common_to_dst_ (integer4 * NumUnit);	/* combines above 2 */
integer4 sdmon_bank_to_common_ (integer1 * bank);
integer4 sdmon_common_to_dump_ (integer4 * opt1);
integer4 sdmon_common_to_dumpf_ (FILE * fp, integer4 * opt2);
/* get (packed) buffer pointer and size */
integer1* sdmon_bank_buffer_ (integer4* sdmon_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



/* site IDs */
#define SDMON_BR 0
#define SDMON_LR 1
#define SDMON_SK 2
#define SDMON_BRLR 3
#define SDMON_BRSK 4
#define SDMON_LRSK 5
#define SDMON_BRLRSK 6




// Maximum x,y values for SDs.
#define SDMON_X_MAX 24
#define SDMON_Y_MAX 28

#define SDMON_MAXSDS 512		/* max number of sd detectors */
#define SDMON_NMONCHAN 512		/* # of channels for MIP histograms */


#define SDMON_DST_GZ ".sdmon.dst.gz" /* suffix of the output dst file */


// Clean values of sd monitoring information
#define SDMON_CL_VAL -1


typedef struct
{
  integer4 event_num;		/* monitoring cycle number */
  integer4 site;                /* BR=0,LR=1,SK=2,BRLR=3,BRSK=4,LRSK=5,BRLRSK=6 */
  integer4 run_id[3];           /* run number for [0]-BR,[1]-LR,[2]-SK, -1 if irrelevant */
  integer4 errcode;             /* 2,3 are reserved for first and last mon. cycles. All others should have 0. */
  integer4 yymmddb;		/* year, month, day at the beginning of the cycle */
  integer4 hhmmssb;		/* time at the beginning of the monitoring cycle */
  integer4 yymmdde;		/* year, month, day at the end of the cycle */
  integer4 hhmmsse;		/* time at the end of the monitoring cycle */
  
  integer4 nsds[3];             /* # of SDs present in the setting file: [0]-BR,[1]-LR,[2]-SK, -1 if irrelevant */
  integer4 lind;		/* largest detector index in the monitoring cycle.*/
  /* These arrays contain the monitoring cycle information (for each mon. cycle) */
  integer4 xxyy[SDMON_MAXSDS];	/* detector location */
  /* 2nd index: [0] - lower, [1] - upper layers */
  integer4 hmip[SDMON_MAXSDS][2][SDMON_NMONCHAN];	/* 1mip histograms (0-lower, 1-upper) */
  integer4 hped[SDMON_MAXSDS][2][SDMON_NMONCHAN / 2];	/* pedestal histogram */
  integer4 hpht[SDMON_MAXSDS][2][SDMON_NMONCHAN / 4];	/* pulse height histograms */
  integer4 hpcg[SDMON_MAXSDS][2][SDMON_NMONCHAN / 4];	/* pulse charge histograms */
  integer4 pchmip[SDMON_MAXSDS][2];	/* peak channel of 1MIP histograms */
  integer4 pchped[SDMON_MAXSDS][2];   /* peak channel of pedestal histograms */
  integer4 lhpchmip[SDMON_MAXSDS][2];	/* left half-peak channel for 1mip histogram */
  integer4 lhpchped[SDMON_MAXSDS][2]; /* left half-peak channel of pedestal histogram */
  integer4 rhpchmip[SDMON_MAXSDS][2];	/* right half-peak channel for 1mip histogram */
  integer4 rhpchped[SDMON_MAXSDS][2]; /* right half-peak channel of pedestal histograms */





  /* These are the detector status variables ** */

  integer4 tgtblnum[SDMON_MAXSDS][600];   // Number of trigger tables for every second
  integer4 mclkcnt [SDMON_MAXSDS][600];   // Maximum clock count monitoring, for every second

  


  /* CC */
  integer4 ccadcbvt[SDMON_MAXSDS][10];	      /* CC ADC value Batt Voltage, for 10 minutes */
  integer4 blankvl1[SDMON_MAXSDS][10];        /* 1st blank value in b/w, in case later it will have something */
  integer4 ccadcbct[SDMON_MAXSDS][10];	      /* CC ADC Value Batt Current */
  integer4 blankvl2[SDMON_MAXSDS][10];        /* 2nd blank value in b/w, in case later it will have something */
  integer4 ccadcrvt[SDMON_MAXSDS][10];	      /* CC ADC Value Ref Voltage */
  integer4 ccadcbtm[SDMON_MAXSDS][10];	      /* CC ADC Value Batt Temp */
  integer4 ccadcsvt[SDMON_MAXSDS][10];        /* CC ADC Value SolarVoltage */
  integer4 ccadctmp[SDMON_MAXSDS][10];        /* CC ADC Value CC Temp */ 
  


  /* Mainboard */
  integer4 mbadcgnd[SDMON_MAXSDS][10];        /* Main board ADC value "GND" */
  integer4 mbadcsdt[SDMON_MAXSDS][10];        /* Main board ADC value SDTemp */
  integer4 mbadc5vt[SDMON_MAXSDS][10];        /* Main board ADC value 5.0V */ 
  integer4 mbadcsdh[SDMON_MAXSDS][10];        /* Main board ADC value SDHum */
  integer4 mbadc33v[SDMON_MAXSDS][10];        /* Main board ADC value 3.3V */
  integer4 mbadcbdt[SDMON_MAXSDS][10];        /* Main board ADC value BDTemp */
  integer4 mbadc18v[SDMON_MAXSDS][10];        /* Main boad ADC value 1.8V */ 
  integer4 mbadc12v[SDMON_MAXSDS][10];        /* Main boad ADC value 1.2V */ 


  /* Trigger Rate Monitor */
  integer4 crminlv2[SDMON_MAXSDS][10];        /* 1min count rate Lv2(>3mip) */
  integer4 crminlv1[SDMON_MAXSDS][10];        /* 1min count rate Lv1(>0.3mip) */


  /* GPS Monitor */
  integer4 gpsyymmdd[SDMON_MAXSDS];           /* Date(YMD) */ 
  integer4 gpshhmmss[SDMON_MAXSDS];           /* Time(HMS) */ 
  integer4 gpsflag  [SDMON_MAXSDS];           /* GPSFLAG */
  integer4 curtgrate[SDMON_MAXSDS];           /* CURRENT TRIGGER Rate */

  integer4 num_sat[SDMON_MAXSDS];             /* number of satellites seen by the SD */

  /* Results from fitting 1MIP histograms */
  /* 2nd index: [0] - lower, [1] - upper layers */
  integer4 mftndof[SDMON_MAXSDS][2];  /* number of degrees of freedom in 1MIP fit */
  real8 mip[SDMON_MAXSDS][2];         /* 1MIP value (ped. subtracted) */
  real8 mftchi2[SDMON_MAXSDS][2];     /* chi2 of the 1MIP fit */
  
  /* 
     1MIP Fit function: 
     [3]*(1+[2]*(x-[0]))*exp(-(x-[0])*(x-[0])/2/[1]/[1])/sqrt(2*PI)/[1]
     4 fit parameters:
     [0]=Gauss Mean
     [1]=Gauss Sigma
     [2]=Linear Coefficient
     [3]=Overall Scalling Factor
  */
  real8 mftp[SDMON_MAXSDS][2][4];    /* 1MIP fit parameters */
  real8 mftpe[SDMON_MAXSDS][2][4];   /* Errors on 1MIP fit parameters */
  

} sdmon_dst_common;

extern sdmon_dst_common sdmon_;

#endif
