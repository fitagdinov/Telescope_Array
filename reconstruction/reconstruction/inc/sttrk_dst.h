/*
 * sttrk_dst.h
 *
 */

#ifndef _STTRK_
#define _STTRK_

#define STTRK_BANKID 15047
#define STTRK_BANKVERSION 0 

#define STTRK_MAXFIT 32

#define STTRK_TIMINFO_USED    1
#define STTRK_TIMINFO_UNUSED  0

#define STTRK_FIT_NOT_REQUESTED         1
#define STTRK_NOT_IMPLEMENTED           2
#define STTRK_REQUIRED_BANKS_MISSING    3
#define STTRK_MISSING_TRAJECTORY_INFO   4
#define STTRK_UPWARD_GOING_TRACK       10
#define STTRK_TOO_FEW_GOOD_TUBES       11
#define STTRK_FITTER_FAILURE           12
#define STTRK_INSANE_TRAJECTORY        13

#define STTRK_STAT_ERROR_FAILURE        1
#define STTRK_RIGHT_ERROR_FAILURE       2
#define STTRK_LEFT_ERROR_FAILURE        4

#define STTRK_FIT_TYPE_STEREO        1
#define STTRK_FIT_TYPE_STEREO_TIMING 2
#define STTRK_FIT_TYPE_STEREO_HYBRID 3
#define STTRK_FIT_TYPE_MONO          4
#define STTRK_FIT_TYPE_MONO_HYBRID   5
#define STTRK_FIT_TYPE_GLOBAL        6
#define STTRK_FIT_TYPE_GLOBAL_HYBRID 7
#define STTRK_FIT_TYPE_PCF           8

#ifdef __cplusplus
extern "C" {
#endif
integer4 sttrk_common_to_bank_(void);
integer4 sttrk_bank_to_dst_(integer4 *NumUnit);
integer4 sttrk_common_to_dst_(integer4 *NumUnit);
integer4 sttrk_bank_to_common_(integer1 *bank);
integer4 sttrk_common_to_dump_(integer4 *long_output);
integer4 sttrk_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* sttrk_bank_buffer_ (integer4* sttrk_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



#include "mc04_detector.h"

/*
  Hires:
  fit 1 hr1 mono or stereo ( timing )
  fit 2 hr2 mono or stereo ( no timing )

  Tale:

  fit  1 stereo (eye1,eye2) ( timing or no )
  fit  2 stereo (eye1,eye3) ( timing or no )
  fit  3 stereo (eye1,eye4) ( timing or no )
  fit  4 stereo (eye2,eye3) ( timing or no )
  fit  5 stereo (eye2,eye3) ( timing or no )
  fit  6 stereo (eye3,eye4) ( timing or no )

  fit  7 br hybrid
  fit  8 lr_ta hybrid
  fit  9 md hybrid
  fit 10 lr_hr hybrid

  fit 11 stereo hybrid (eye1,eye2) ( timing or no )
  fit 12 stereo hybrid (eye1,eye3) ( timing or no )
  fit 13 stereo hybrid (eye1,eye4) ( timing or no )
  fit 14 stereo hybrid (eye2,eye3) ( timing or no )
  fit 15 stereo hybrid (eye2,eye3) ( timing or no )
  fit 16 stereo hybrid (eye3,eye4) ( timing or no )

  fit 17 br mono
  fit 18 lr_ta mono
  fit 19 md mono
  fit 20 lr_hr mono

  fit 21 global 3 or more eyes
  fit 22 global hybrid 3 or more eyes

 */

typedef struct  {

  real8 uthat [STTRK_MAXFIT][3];
  real8 rpvec [STTRK_MAXFIT][3];  // with respect to Origin
  real8 rcvec [STTRK_MAXFIT][3];  // shower core position

  real8 chi2  [STTRK_MAXFIT][MC04_MAXEYE];
  real8 rp    [STTRK_MAXFIT][MC04_MAXEYE];
  real8 psi   [STTRK_MAXFIT][MC04_MAXEYE];
  real8 theta [STTRK_MAXFIT][MC04_MAXEYE];
  real8 phi   [STTRK_MAXFIT][MC04_MAXEYE];

  /* tube/mir info. from main fit */

  real8 time   [STTRK_MAXFIT][MC04_MAXTUBE]; /* tube time */
  real8 timefit[STTRK_MAXFIT][MC04_MAXTUBE]; /* time from best fit */
  real8 thetb  [STTRK_MAXFIT][MC04_MAXTUBE]; /* viewing angle */
  real8 sgmt   [STTRK_MAXFIT][MC04_MAXTUBE]; /* sigma time */ 
  real8 asx    [STTRK_MAXFIT][MC04_MAXTUBE];
  real8 asy    [STTRK_MAXFIT][MC04_MAXTUBE];
  real8 asz    [STTRK_MAXFIT][MC04_MAXTUBE];

  integer4 tube_eye[STTRK_MAXFIT][MC04_MAXTUBE]; 
  integer4 tube_mir[STTRK_MAXFIT][MC04_MAXTUBE]; 
  integer4 tubeid  [STTRK_MAXFIT][MC04_MAXTUBE];  /* tube number */
  integer4 ig      [STTRK_MAXFIT][MC04_MAXTUBE];  /* tube flag */

  integer4 ntube   [STTRK_MAXFIT];

  integer4 fittype [STTRK_MAXFIT];
  integer4 eyelist [STTRK_MAXFIT][MC04_MAXEYE];

  integer4 failmode[STTRK_MAXFIT];  
  integer4 timinfo [STTRK_MAXFIT];

} sttrk_dst_common ;

extern sttrk_dst_common sttrk_ ; 

#endif
