/*  mcsdd_dst.h
 *
 * $Source: $
 * $Log: $
 *
 * mc surface detector data
 * version 0 is a simple "cheat" implementation!
 * only want shower core location, assumed to be exactly known if
 * the shower impact point is within the array boundries.
 * array boundries are defined based on detector id given in mc04_detector.h
 *
 */

#ifndef _MCSDD_
#define _MCSDD_

#define  MCSDD_BANKID 15044
#define  MCSDD_BANKVERSION 0 

#include "univ_dst.h"
#include "mc04_detector.h"
//#include "mcsdd_detector.h"


/***********************************************/

#ifdef __cplusplus
extern "C" {
#endif
integer4  mcsdd_common_to_bank_(void);
integer4  mcsdd_bank_to_dst_(integer4 *NumUnit);
integer4  mcsdd_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4  mcsdd_bank_to_common_(integer1 *bank);
integer4  mcsdd_common_to_dump_(integer4 *long_output);
integer4  mcsdd_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* mcsdd_bank_buffer_ (integer4* mcsdd_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


/***********************************************/
/* Define common block structures    */

typedef struct {

  real8    Rcore[3];    /* shower core (from origin),z=0 (m) */
  integer4 have_core;   /* = 1 if true */   

}  mcsdd_dst_common ;

extern  mcsdd_dst_common  mcsdd_ ; 

#endif

