/* Created 2009/03/24 LMS */

#ifndef _FDPED_DST_
#define _FDPED_DST_

#include "geofd_dst.h"

#define FDPED_BANKID		12496
#define FDPED_BANKVERSION	000

#define FDPED_NMIN  500
#define FDPED_NCAM  (12)
#define FDPED_NPMT  (GEOFD_MIRTUBE)

#ifdef __cplusplus
extern "C" {
#endif
integer4 fdped_common_to_bank_();
integer4 fdped_bank_to_dst_(integer4 *unit);
integer4 fdped_common_to_dst_(integer4 *unit); // combines above 2
integer4 fdped_bank_to_common_(integer1 *bank);
integer4 fdped_common_to_dump_(integer4 *opt);
integer4 fdped_common_to_dumpf_(FILE *fp, integer4 *opt);
/* get (packed) buffer pointer and size */
integer1* fdped_bank_buffer_ (integer4* fdped_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct {
  integer4 julian_start;   // part start day
  integer4 jsecond_start;  // part start second (from start of julian_start)
  integer4 jsecfrac_start; // part start nanosecond (from start of jsecond_start)

  integer4 julian_end;     // part end day
  integer4 jsecond_end;    // part end second (from start of julian_end)
  integer4 jsecfrac_end;   // part end nanosecond (from start of jsecond_end)

  integer4 siteid;                                      // site ID (BR = 0, LR = 1)
  integer4 part;                                        // part number
  integer4 num_minutes;                                 // number of minutes in part

  integer4 pedestal[FDPED_NCAM][FDPED_NPMT][FDPED_NMIN]; // pedestal values
  integer4 pedrms[FDPED_NCAM][FDPED_NPMT][FDPED_NMIN];   // pedestal rms values
  integer4 liveflag[FDPED_NCAM][FDPED_NPMT][FDPED_NMIN]; // 1 if tube has a pedestal value, 0 if not
} fdped_dst_common;

extern fdped_dst_common fdped_;
extern integer4 fdped_blen; /* needs to be accessed by the c files of the derived banks */ 

integer4 fdped_struct_to_abank_(fdped_dst_common *fdped, integer1 *(*pbank), integer4 id, integer4 ver);
integer4 fdped_abank_to_dst_(integer1 *bank, integer4 *unit);
integer4 fdped_struct_to_dst_(fdped_dst_common *fdped, integer1 *bank, integer4 *unit, integer4 id, integer4 ver);
integer4 fdped_abank_to_struct_(integer1 *bank, fdped_dst_common *fdped);
integer4 fdped_struct_to_dump_(fdped_dst_common *fdped, integer4 *opt);
integer4 fdped_struct_to_dumpf_(fdped_dst_common *fdped, FILE *fp, integer4 *opt);

#endif
