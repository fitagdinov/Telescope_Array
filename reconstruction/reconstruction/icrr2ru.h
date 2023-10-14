#ifndef _ICRR2RU_H_
#define _ICRR2RU_H_

//  TO TRANSFER THE CONTENTS OF tasdcalibev DST bank to 
//  rusdraw DST bank format.
//  Created: Aug 21, 2009
//  Last modified: Aug 27, 2020
//  Dmitri Ivanov <dmiivanov@gmail.com>


#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "event.h"

class icrr2ru
{

 public:
  icrr2ru();
  virtual ~icrr2ru();
  
  // to be able to reset the dst event number, if necessary
  void reset_event_num();
  
  // converts from tasdcalibev to rusdraw format
  bool Convert();

 private:
  
  int event_num;                 // event number in the DST file
  tasdcalibev_dst_common *icrr;  // ICRR raw SD data (tasdcalibev) format
  rusdraw_dst_common *ru;        // Rutgers raw SD data (rusdraw) format
  rusdmc_dst_common *rumc;       // Rutgers MC thrown (rusdmc) format

 protected:

  // convert the thrown MC information
  bool addThrownMCinfo();
};

#endif
