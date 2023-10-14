#ifndef _sdmc_tadate_h_
#define _sdmc_tadate_h_
#ifndef __cplusplus
float Date2TADay( float D, int M, int Y ); 
void TADay2Date( float J, int *yymmdd, int *hhmmss);
#else
extern "C" float Date2TADay( float D, int M, int Y ); 
extern "C" void TADay2Date( float J, int *yymmdd, int *hhmmss);
#endif
#endif
