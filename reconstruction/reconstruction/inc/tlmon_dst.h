/*
 *     Bank for TALE + OTHER SD raw monitoring data
 *     Dmitri Ivanov (dmiivanov@gmail.com)
 *     Jun 23, 2013


 *     Last modified: DI, Nov 22, 2019
 
*/

#ifndef _TLMON_
#define _TLMON_
#include "talex00_dst.h"


#define TLMON_BANKID  13205
#define TLMON_BANKVERSION   001

#ifdef __cplusplus
extern "C" {
#endif
integer4 tlmon_common_to_bank_ ();
integer4 tlmon_bank_to_dst_ (integer4 * NumUnit);
integer4 tlmon_common_to_dst_ (integer4 * NumUnit);	/* combines above 2 */
integer4 tlmon_bank_to_common_ (integer1 * bank);
integer4 tlmon_common_to_dump_ (integer4 * opt1);
integer4 tlmon_common_to_dumpf_ (FILE * fp, integer4 * opt2);
integer1* tlmon_bank_buffer_ (integer4* tlmon_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


#define TLMON_MD 0                      /* site ID */
#define TLMON_MAXSDS   1280		/* max number of sd detectors */
#define TLMON_NMONCHAN 512		/* # of channels for MIP histograms */
#define TLMON_NCT      TALEX00_NCT      /* number of communication towers */

#define TLMON_DST_GZ ".tlmon.dst.gz" /* suffix of the output dst file */


// Clean values of sd monitoring information
#define TLMON_CL_VAL -1


typedef struct
{
  integer4 event_num;		/* monitoring cycle number */
  integer4 site;                /* site bitflag index (bit0=BR,1=LR,2=SK,[3-8]=[BF,DM,KM,SC,SN,SR],bit9=MD */
  integer4 run_id;              /* run number of the tower, if only one tower is read out; -1 if irrelevant */
  integer4 run_num[TLMON_NCT];  /* run numbers of all towers, -1 if irrelevant */
  integer4 errcode;             /* 2,3 are reserved for first and last mon. cycles. All others should have 0. */
  integer4 yymmddb;		/* year, month, day at the beginning of the cycle */
  integer4 hhmmssb;		/* time at the beginning of the monitoring cycle */
  integer4 yymmdde;		/* year, month, day at the end of the cycle */
  integer4 hhmmsse;		/* time at the end of the monitoring cycle */
  integer4 lind;		/* largest detector index in the monitoring cycle.*/
  /* These arrays contain the monitoring cycle information (for each mon. cycle) */
  integer4 xxyy[TLMON_MAXSDS];	/* detector location */
  /* 2nd index: [0] - lower, [1] - upper layers */
  integer4 hmip[TLMON_MAXSDS][2][TLMON_NMONCHAN];	/* 1mip histograms (0-lower, 1-upper) */
  integer4 hped[TLMON_MAXSDS][2][TLMON_NMONCHAN / 2];	/* pedestal histogram */
  integer4 hpht[TLMON_MAXSDS][2][TLMON_NMONCHAN / 4];	/* pulse height histograms */
  integer4 hpcg[TLMON_MAXSDS][2][TLMON_NMONCHAN / 4];	/* pulse charge histograms */
  integer4 pchmip[TLMON_MAXSDS][2];	/* peak channel of 1MIP histograms */
  integer4 pchped[TLMON_MAXSDS][2];   /* peak channel of pedestal histograms */
  integer4 lhpchmip[TLMON_MAXSDS][2];	/* left half-peak channel for 1mip histogram */
  integer4 lhpchped[TLMON_MAXSDS][2]; /* left half-peak channel of pedestal histogram */
  integer4 rhpchmip[TLMON_MAXSDS][2];	/* right half-peak channel for 1mip histogram */
  integer4 rhpchped[TLMON_MAXSDS][2]; /* right half-peak channel of pedestal histograms */


  /* These are the detector status variables ** */

  integer4 tgtblnum[TLMON_MAXSDS][600];   // Number of trigger tables for every second
  integer4 mclkcnt [TLMON_MAXSDS][600];   // Maximum clock count monitoring, for every second

  
  /* CC */
  integer4 ccadcbvt[TLMON_MAXSDS][10];	      /* CC ADC value Batt Voltage, for 10 minutes */
  integer4 blankvl1[TLMON_MAXSDS][10];        /* 1st blank value in b/w, in case later it will have something */
  integer4 ccadcbct[TLMON_MAXSDS][10];	      /* CC ADC Value Batt Current */
  integer4 blankvl2[TLMON_MAXSDS][10];        /* 2nd blank value in b/w, in case later it will have something */
  integer4 ccadcrvt[TLMON_MAXSDS][10];	      /* CC ADC Value Ref Voltage */
  integer4 ccadcbtm[TLMON_MAXSDS][10];	      /* CC ADC Value Batt Temp */
  integer4 ccadcsvt[TLMON_MAXSDS][10];        /* CC ADC Value SolarVoltage */
  integer4 ccadctmp[TLMON_MAXSDS][10];        /* CC ADC Value CC Temp */ 
  


  /* Mainboard */
  integer4 mbadcgnd[TLMON_MAXSDS][10];        /* Main board ADC value "GND" */
  integer4 mbadcsdt[TLMON_MAXSDS][10];        /* Main board ADC value SDTemp */
  integer4 mbadc5vt[TLMON_MAXSDS][10];        /* Main board ADC value 5.0V */ 
  integer4 mbadcsdh[TLMON_MAXSDS][10];        /* Main board ADC value SDHum */
  integer4 mbadc33v[TLMON_MAXSDS][10];        /* Main board ADC value 3.3V */
  integer4 mbadcbdt[TLMON_MAXSDS][10];        /* Main board ADC value BDTemp */
  integer4 mbadc18v[TLMON_MAXSDS][10];        /* Main boad ADC value 1.8V */ 
  integer4 mbadc12v[TLMON_MAXSDS][10];        /* Main boad ADC value 1.2V */ 


  /* Trigger Rate Monitor */
  integer4 crminlv2[TLMON_MAXSDS][10];        /* 1min count rate Lv2(>3mip) */
  integer4 crminlv1[TLMON_MAXSDS][10];        /* 1min count rate Lv1(>0.3mip) */


  /* GPS Monitor */
  integer4 gpsyymmdd[TLMON_MAXSDS];           /* Date(YMD) */ 
  integer4 gpshhmmss[TLMON_MAXSDS];           /* Time(HMS) */ 
  integer4 gpsflag  [TLMON_MAXSDS];           /* GPSFLAG */
  integer4 curtgrate[TLMON_MAXSDS];           /* CURRENT TRIGGER Rate */
  
  integer4 num_sat[TLMON_MAXSDS];             /* number of satellites seen by the SD */

  /* Results from fitting 1MIP histograms */
  /* 2nd index: [0] - lower, [1] - upper layers */
  integer4 mftndof[TLMON_MAXSDS][2];  /* number of degrees of freedom in 1MIP fit */
  real8 mip[TLMON_MAXSDS][2];         /* 1MIP value (ped. subtracted) */
  real8 mftchi2[TLMON_MAXSDS][2];     /* chi2 of the 1MIP fit */
  
  /* 
     1MIP Fit function: 
     [3]*(1+[2]*(x-[0]))*exp(-(x-[0])*(x-[0])/2/[1]/[1])/sqrt(2*PI)/[1]
     4 fit parameters:
     [0]=Gauss Mean
     [1]=Gauss Sigma
     [2]=Linear Coefficient
     [3]=Overall Scalling Factor
  */
  real8 mftp[TLMON_MAXSDS][2][4];    /* 1MIP fit parameters */
  real8 mftpe[TLMON_MAXSDS][2][4];   /* Errors on 1MIP fit parameters */
  
  real8 lat_lon_alt[TLMON_MAXSDS][3]; /* GPS coordinates: latitude, longitude, altitude
					 [0] - latitude in degrees,  positive = North
					 [1] - longitude in degrees, positive = East
					 [2] - altitude is in meters */
  
  real8 xyz_cor_clf[TLMON_MAXSDS][3];     /* XYZ coordinates in CLF frame:
					     origin=CLF, X=East,Y=North,Z=Up, [meters] */
  

} tlmon_dst_common;

extern tlmon_dst_common tlmon_;

#endif
