/*
 * hsum_dst.c
 *
 * $Source: /hires_soft/cvsroot/bank/hsum_dst.h,v $
 * $Log: hsum_dst.h,v $
 * Revision 1.1  1999/07/06 21:32:05  stokes
 * Initial revision
 *
 * Revision 1.0  1999/03/04  bts
 *
 * Summary for individual pkt1 parts
 *
 */

#ifndef _HSUM_
#define _HSUM_

#define HSUM_BANKID 15021
#define HSUM_BANKVERSION 0

#define HSUM_SUCCESS 1
#define HSUM_MAXPERMIT 20
#define HSUM_MAXPERMIT_INDV 100
#define HSUM_MAX_TXT_LEN   512
#define HR_MAX_MIR 25

#ifdef __cplusplus
extern "C" {
#endif
integer4 hsum_common_to_bank_ ();
integer4 hsum_bank_to_dst_ (integer4 * NumUnit);
integer4 hsum_common_to_dst_ (integer4 * NumUnit);	/* combines above 2 */
integer4 hsum_bank_to_common_ (integer1 * bank);
integer4 hsum_common_to_dump_ (integer4 * long_output);
integer4 hsum_common_to_dumpf_ (FILE * fp, integer4 * long_output);
/* get (packed) buffer pointer and size */
integer1* hsum_bank_buffer_ (integer4* hsum_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif



typedef struct
  {
    integer1 weat[HSUM_MAXPERMIT][HSUM_MAX_TXT_LEN];	/* Weather code string */
    integer1 stat[HSUM_MAX_TXT_LEN];	/*part file name */
    real8 jdsta;		/* Start time in Julian daze */
    real8 jdsto;		/* Stop time in Julian daze */
    real8 jdper[HSUM_MAXPERMIT];	/* Global permit time in Julian daze */
    real8 jdpero[HSUM_MAXPERMIT_INDV];	/* Individual permit time in Julian daze */

    real8 jdinh[HSUM_MAXPERMIT];	/* Global inhibit time in Julian daze */
    real8 jdinho[HSUM_MAXPERMIT_INDV];	/* Individual inhibit time in Julian daze */

    real8 jddur[HSUM_MAXPERMIT];	/*duration of global permit in julian time */
    real8 jdwea[HSUM_MAXPERMIT];	/* time of weather code in julian time */
    integer4 hsta, msta, ssta;	/* Start time in UT */
    integer4 hsto, msto, ssto;	/* Stop time in UT */
    integer4 nperm, npermo, ninho;	/* number of permits */
    integer4 hper[HSUM_MAXPERMIT], mper[HSUM_MAXPERMIT], sper[HSUM_MAXPERMIT],
      msper[HSUM_MAXPERMIT];	/* Permit time in UT */
    integer4 hinh[HSUM_MAXPERMIT], minh[HSUM_MAXPERMIT], sinh[HSUM_MAXPERMIT],
      msinh[HSUM_MAXPERMIT];	/* Inhibit time in UT */
    integer4 hpero[HSUM_MAXPERMIT_INDV], mpero[HSUM_MAXPERMIT_INDV], spero[HSUM_MAXPERMIT_INDV],
      mspero[HSUM_MAXPERMIT_INDV], mirpero[HSUM_MAXPERMIT_INDV];
    /* Mirror number and permit time in UT for an individual mirror */
    integer4 hinho[HSUM_MAXPERMIT_INDV], minho[HSUM_MAXPERMIT_INDV], sinho[HSUM_MAXPERMIT_INDV],
      msinho[HSUM_MAXPERMIT_INDV], mirinho[HSUM_MAXPERMIT_INDV];
    /* Mirror number and inhibit time in UT for an individual mirror */
    integer4 hdur[HSUM_MAXPERMIT], mdur[HSUM_MAXPERMIT], sdur[HSUM_MAXPERMIT];
    /*duration of permit in hours,... */
    integer4 ntrig, nweat;	/* number of triggers an weather codesy */
    integer4 ntri[HR_MAX_MIR];	/*number of triggers on a specific mirror */
    real8 ntrim[HR_MAX_MIR];	/*number of triggers/min on a mirror */
    integer4 hoper[HR_MAX_MIR], moper[HR_MAX_MIR], soper[HR_MAX_MIR];
    integer4 hwea[HSUM_MAXPERMIT], mwea[HSUM_MAXPERMIT], swea[HSUM_MAXPERMIT];
    integer1 staflag, permflag;	/*Flags for problems in the part */
    /* time of weather code in hours... */
  }
hsum_dst_common;
extern hsum_dst_common hsum_;
#endif
