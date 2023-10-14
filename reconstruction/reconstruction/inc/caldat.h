#ifndef _CALDAT_
#define _CALDAT_

#ifndef __cplusplus
void caldat(double julian, int *mm, int *id, int *iyyy);
#else
extern "C" void caldat(double julian, int *mm, int *id, int *iyyy);
#endif

#endif
