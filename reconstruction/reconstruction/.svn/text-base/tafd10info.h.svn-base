#ifndef _tafd10info_
#define _tafd10info_

// ADDED: 20101025 Dmitri Ivanov <ivanov@physics.rutgers.edu>
// Obtain tube pointing directions, time for various FDs
namespace tafd10info
{ 
  // BR,LR tubes are labled by mirror number (from 0 to 11)
  // and by tube inside number (from 0 to 255)
  // v[] is the tube pointing direction (size 3)
  bool get_br_tube_pd(int mir_num, int tube_num, double *v);
  bool get_lr_tube_pd(int mir_num, int tube_num, double *v);
  
  // MD tubes are labeled by mirror number   (from 1 to 14)
  // and by tube inside the mirror number (from 1 to 256)
  // v[] is the tube pointing direction (size 3)
  bool get_md_tube_pd(int mir_num, int tube_num, double *v);

  // BR/LR FD time (julian, jsecond are those appearing in fdplane banks)
  void get_brlr_time(int julian, int jsecond, int *yymmdd, int *hhmmss);
  
  // MD time (jday, jsec are those appearing in hraw1, mcraw banks)
  void get_md_time(int jday, int jsec, int *yymmdd, int *hhmmss);

  // Obtain hraw1 bank from mcraw
  void mcraw2hraw1();
  
};

#endif
