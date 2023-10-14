#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "tafd10info.h"

// Obtain hraw1 bank from mcraw
void tafd10info::mcraw2hraw1()
{
  int imir, itube;
  hraw1_.jday   = mcraw_.jday;
  hraw1_.jsec   = mcraw_.jsec;
  hraw1_.msec   = mcraw_.msec;
  hraw1_.status = 1; // no status variable stored in mcraw
  hraw1_.nmir   = mcraw_.nmir;
  hraw1_.ntube  = mcraw_.ntube;
  for (imir = 0; imir < hraw1_.nmir; imir++)
    {
      hraw1_.mir[imir] = mcraw_.mirid[imir];
      hraw1_.mir_rev[imir] = mcraw_.mir_rev[imir];
      hraw1_.mirevtno[imir] = mcraw_.mirevtno[imir];
      hraw1_.mirntube[imir] = mcraw_.mir_ntube[imir];
      hraw1_.miraccuracy_ns[imir] = mcraw_.mirtime_ns[imir];
      hraw1_.mirtime_ns[imir] = mcraw_.mirtime_ns[imir];
    }  
  for (itube = 0; itube < hraw1_.ntube; itube++)
    {
      hraw1_.tubemir[itube] =  mcraw_.tube_mir[itube];
      hraw1_.tube[itube]    =  mcraw_.tubeid[itube];
      hraw1_.qdca[itube]    =  mcraw_.qdca[itube];
      hraw1_.qdcb[itube]    =  mcraw_.qdcb[itube];
      hraw1_.tdc[itube]     =  mcraw_.tdc[itube];
      hraw1_.tha[itube]     =  mcraw_.tha[itube];
      hraw1_.thb[itube]     =  mcraw_.thb[itube];
      hraw1_.prxf[itube]    =  mcraw_.prxf[itube];
      hraw1_.thcal1[itube]  =  mcraw_.thcal1[itube];
    }
}
