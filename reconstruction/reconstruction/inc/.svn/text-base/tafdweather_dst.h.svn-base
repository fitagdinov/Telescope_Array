/*
 *     Bank for FD weather monitor
 *     written by a student
 *     Time-stamp: Fri Apr 10 23:18:24 2009 JST
*/

#ifndef _TAFDWEATHER_
#define _TAFDWEATHER_

#define TAFDWEATHER_BANKID  13016
#define TAFDWEATHER_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tafdweather_common_to_bank_();
int tafdweather_bank_to_dst_(int *NumUnit);
int tafdweather_common_to_dst_(int *NumUnit);/*combines above 2*/
int tafdweather_bank_to_common_(char *bank);
int tafdweather_common_to_dump_(int *opt1);
int tafdweather_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tafdweather_bank_buffer_ (integer4* tafdweather_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tafdweather_nhmax   3  /* maximum number of hosts */
#define tafdweather_npmax 144  /* maximum number of data points */


typedef struct {
  int timeFrom;			/* time, hhmmss */
  int timeTo;			/* time, hhmmss */
  float averageWindSpeed;	/* [m/s] */
  float maximumWindSpeed;	/* [m/s] */
  float windDirection;		/* 0 is north, 90 is east [deg] */
  float atmosphericPressure;	/* [hPa] */
  float temperature;		/* [C] */
  float humidity;		/* [%RH] */
  float totalRainfall;		/* [mm] */
  float rainfall;		/* [mm/hour] */
  float numberOfHails;		/* [hits/cm^2/hour]*/
} FDWeather10mData;


typedef struct {
  int site;	/* 0 is BRM, 1 is LR, 4 will be CLF */
  int num_data;	/* number of data points */
  FDWeather10mData data[tafdweather_npmax];
} FDWeatherData;


typedef struct {
  int nsite;	/* number of sites */
  int date;	/* date, yymmdd */
  FDWeatherData st[tafdweather_nhmax];
  int footer;	/* for debugging */
} tafdweather_dst_common;


extern tafdweather_dst_common tafdweather_;


#endif


