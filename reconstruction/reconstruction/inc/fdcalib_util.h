#ifndef _FDCALIB_UTIL_
#define _FDCALIB_UTIL_
#include <time.h>
#ifdef __cplusplus
extern "C" time_t convertDate2Sec(int year,int month,int day,int hour,int min,int sec); 
extern "C" time_t convertDateLine2Sec(char* dateLine);
extern "C" void convertSec2Date(time_t time,int* year,int* month,int* day,int* hour,int* min,int* sec); 
extern "C" void convertSec2DateLine(time_t time,char* dateLine);
#else
time_t convertDate2Sec(int year,int month,int day,int hour,int min,int sec); 
time_t convertDateLine2Sec(char* dateLine);
void convertSec2Date(time_t time,int* year,int* month,int* day,int* hour,int* min,int* sec); 
void convertSec2DateLine(time_t time,char* dateLine);
#endif


#endif
