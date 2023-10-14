/*
 * ontime2_dst.h
 *  
 * 
 * $Source: /hires_soft/cvsroot/bank/ontime2_dst.h,v $
 * $Log: ontime2_dst.h,v $
 * Revision 1.1  1999/07/21 22:05:31  stokes
 * Initial revision
 *
 *
 * Contains the output for ontime2, a monthly mirror by mirror statistics 
 * program
 * 
 *
 */

#ifndef _ONTIME2_
#define _ONTIME2_

#define ONTIME2_BANKID 15022
#define ONTIME2_BANKVERSION 0

#define ONTIME2_SUCCESS 1
#define ONTIME2_MAX_WEAT 500
#define ONTIME2_MAX_AB 2000
#define ONTIME2_MAX_TXT_LEN   512




#ifdef __cplusplus
extern "C" {
#endif
integer4 ontime2_common_to_bank_ (void);
integer4 ontime2_bank_to_dst_ (integer4 * NumUnit);
integer4 ontime2_common_to_dst_ (integer4 * NumUnit);	/* combines above 2 */
integer4 ontime2_bank_to_common_ (integer1 * bank);
integer4 ontime2_common_to_dump_ (integer4 * long_output);
integer4 ontime2_common_to_dumpf_ (FILE * fp, integer4 * long_output);
/* get (packed) buffer pointer and size */
integer1* ontime2_bank_buffer_ (integer4* ontime2_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



typedef struct
  {

    integer1 weat[ONTIME2_MAX_WEAT][ONTIME2_MAX_TXT_LEN];
    /* Operator code string */
    integer1 stat[ONTIME2_MAX_TXT_LEN];		/*Name of the first part in summary */
    integer4 nweat, nab;
    /* Number of operator comments and abnormal trigger rates */
    integer4 yweat[ONTIME2_MAX_WEAT], moweat[ONTIME2_MAX_WEAT], dweat[ONTIME2_MAX_WEAT],
      hweat[ONTIME2_MAX_WEAT], mweat[ONTIME2_MAX_WEAT], sweat[ONTIME2_MAX_WEAT];	/*time of operator comment in years,... */

    integer4 hdur, mdur, sdur;	/*duration of permit in hours,... */

    integer4 hduro[HR_MAX_MIR], mduro[HR_MAX_MIR], sduro[HR_MAX_MIR];
    /* duration of permit for each individual mirror */
    integer4 ntrig[HR_MAX_MIR];	/*absolute number of triggers on a mirror */
    real8 ntrim[HR_MAX_MIR];
    /*truncated mean number of triggers/min on a mirror */

    real8 ntrimab[ONTIME2_MAX_AB];	/* abnormal trigger rates */

    integer4 mirab[ONTIME2_MAX_AB];	/* mirror# of abnormal trigger rates */
    integer4 ytrimab[ONTIME2_MAX_AB], motrimab[ONTIME2_MAX_AB], dtrimab[ONTIME2_MAX_AB];	/*time of abnormal trigger rate in years,... */


  }
ontime2_dst_common;

extern ontime2_dst_common ontime2_;
#endif
