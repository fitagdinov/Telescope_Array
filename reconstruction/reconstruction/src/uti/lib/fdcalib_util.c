#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "fdcalib_util.h"

/* convert date (yyyy MM dd HH mm ss) to sec (from 1970/1/1) */
time_t convertDate2Sec(int year,int month,int day,int hour,int min,int sec){
   struct tm stamp; 
   stamp.tm_year = year-1900;
   stamp.tm_mon = month-1;
   stamp.tm_mday=day;
   stamp.tm_hour=hour;
   stamp.tm_min=min;
   stamp.tm_sec=sec;
   return timegm(&stamp); 
}

/* convert dateline (yyyy/MM/dd HH:mm:ss) to sec (from 1970/1/1) */
time_t convertDateLine2Sec(char* dateLine){
   int year;
   int month;
   int day;
   int hour;
   int min;
   int sec;
   sscanf(dateLine,"%04d/%02d/%02d %02d:%02d:%02d",&year,&month,&day,&hour,&min,&sec);
   return convertDate2Sec(year,month,day,hour,min,sec);
}

/* convert sec (from 1970/1/1) to date (yyyy MM dd HH mm ss) */
void convertSec2Date(time_t time,int* year,int* month,int* day,int* hour,int* min,int* sec){
   struct tm stamp = *(gmtime(&time));
   *year = stamp.tm_year + 1900;
   *month = stamp.tm_mon + 1;
   *day = stamp.tm_mday;
   *hour = stamp.tm_hour;
   *min = stamp.tm_min;
   *sec = stamp.tm_sec;
}

/* convert sec (from 1970/1/1) to dateline(yyyy/MM/dd HH:mm:ss) */
void convertSec2DateLine(time_t time,char* dateLine){
   int year;
   int month;
   int day;
   int hour;
   int min;
   int sec;
   convertSec2Date(time,&year,&month,&day,&hour,&min,&sec);
   sprintf(dateLine,"%04d/%02d/%02d %02d:%02d:%02d",year,month,day,hour,min,sec); 
}

