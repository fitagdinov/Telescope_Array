C     -*- fortran -*-  (force emacs to treat .inc file as fortran source)
C
C  hsum_dst.inc
C
C  $Source: /hires_soft/cvsroot/bank/hsum_dst.inc,v $
C  $Log: hsum_dst.inc,v $
C  Revision 1.1  1999/07/06 21:35:17  stokes
C  Initial revision
C
C
C  Automatically generated from TDV's PERL script: dst_bnk_cnvrtr
C  on Tue Jul  6 15:35:01 MDT 1999



C  Bank Bookkeeping

      integer*4  HSUM_BANKID
      parameter (HSUM_BANKID = 15021)

      integer*4  HSUM_BANKVERSION
      parameter (HSUM_BANKVERSION = 0)

      integer*4  HSUM_SUCCESS
      parameter (HSUM_SUCCESS = 1)

      integer*4  HSUM_MAXPERMIT
      parameter (HSUM_MAXPERMIT = 20)

      integer*4  HSUM_MAXPERMIT_INDV
      parameter (HSUM_MAXPERMIT_INDV = 100)

      integer*4  HSUM_MAX_TXT_LEN
      parameter (HSUM_MAX_TXT_LEN = 512)

      integer*4  HR_MAX_MIR
      parameter (HR_MAX_MIR = 25)


C  Function prototypes

      integer*4  hsum_common_to_bank
      external   hsum_common_to_bank
      integer*4  hsum_bank_to_dst
      external   hsum_bank_to_dst

C  /* combines above 2 */
      integer*4  hsum_common_to_dst
      external   hsum_common_to_dst
      integer*4  hsum_bank_to_common
      external   hsum_bank_to_common
      integer*4  hsum_common_to_dump
      external   hsum_common_to_dump
      integer*4  hsum_common_to_dumpf
      external   hsum_common_to_dumpf

C  Common Blocks

      common /hsum/WARNING: can't handle:       msper[HSUM_MAXPERMIT];	/* Permit time in UT */
WARNING: can't handle:       msinh[HSUM_MAXPERMIT];	/* Inhibit time in UT */
WARNING: can't handle:       mspero[HSUM_MAXPERMIT_INDV], mirpero[HSUM_MAXPERMIT_INDV];
WARNING: can't handle:       msinho[HSUM_MAXPERMIT_INDV], mirinho[HSUM_MAXPERMIT_INDV];
WARNING: can't handle:   }
WARNING: can't handle: hsum_dst_common;

     $  hsum_weat, hsum_stat, hsum_jdsta, hsum_jdsto, 
     $  hsum_jdper, hsum_jdpero, hsum_jdinh, hsum_jdinho, 
     $  hsum_jddur, hsum_jdwea, hsum_hsta, hsum_msta, 
     $  hsum_ssta, hsum_hsto, hsum_msto, hsum_ssto, 
     $  hsum_nperm, hsum_npermo, hsum_ninho, hsum_hper, 
     $  hsum_mper, hsum_sper, hsum_hinh, hsum_minh, 
     $  hsum_sinh, hsum_hpero, hsum_mpero, hsum_spero, 
     $  hsum_hinho, hsum_minho, hsum_sinho, hsum_hdur, 
     $  hsum_mdur, hsum_sdur, hsum_ntrig, hsum_nweat, 
     $  hsum_ntri, hsum_ntrim, hsum_hoper, hsum_moper, 
     $  hsum_soper, hsum_hwea, hsum_mwea, hsum_swea, 
     $  hsum_staflag, hsum_permflag


C     /* Weather code string */
      integer*1 hsum_weat(HSUM_MAX_TXT_LEN,HSUM_MAXPERMIT)

C     /*part file name */
      integer*1 hsum_stat(HSUM_MAX_TXT_LEN)

C     /* Start time in Julian daze */
      real*8 hsum_jdsta

C     /* Stop time in Julian daze */
      real*8 hsum_jdsto

C     /* Global permit time in Julian daze */
      real*8 hsum_jdper(HSUM_MAXPERMIT)

C     /* Individual permit time in Julian daze */
      real*8 hsum_jdpero(HSUM_MAXPERMIT_INDV)

C     /* Global inhibit time in Julian daze */
      real*8 hsum_jdinh(HSUM_MAXPERMIT)

C     /* Individual inhibit time in Julian daze */
      real*8 hsum_jdinho(HSUM_MAXPERMIT_INDV)

C     /*duration of global permit in julian time */
      real*8 hsum_jddur(HSUM_MAXPERMIT)

C     /* time of weather code in julian time */
      real*8 hsum_jdwea(HSUM_MAXPERMIT)

C     /* Start time in UT */
      integer*4 hsum_hsta
      integer*4 hsum_msta
      integer*4 hsum_ssta

C     /* Stop time in UT */
      integer*4 hsum_hsto
      integer*4 hsum_msto
      integer*4 hsum_ssto

C     /* number of permits */
      integer*4 hsum_nperm
      integer*4 hsum_npermo
      integer*4 hsum_ninho
      integer*4 hsum_hper(HSUM_MAXPERMIT)
      integer*4 hsum_mper(HSUM_MAXPERMIT)
      integer*4 hsum_sper(HSUM_MAXPERMIT)

C     /* Permit time in UT */
      integer*4 hsum_hinh(HSUM_MAXPERMIT)
      integer*4 hsum_minh(HSUM_MAXPERMIT)
      integer*4 hsum_sinh(HSUM_MAXPERMIT)

C     /* Inhibit time in UT */
      integer*4 hsum_hpero(HSUM_MAXPERMIT_INDV)
      integer*4 hsum_mpero(HSUM_MAXPERMIT_INDV)
      integer*4 hsum_spero(HSUM_MAXPERMIT_INDV)

C     /* Mirror number and permit time in UT for an individual mirror */
      integer*4 hsum_hinho(HSUM_MAXPERMIT_INDV)
      integer*4 hsum_minho(HSUM_MAXPERMIT_INDV)
      integer*4 hsum_sinho(HSUM_MAXPERMIT_INDV)

C     /* Mirror number and inhibit time in UT for an individual mirror */
      integer*4 hsum_hdur(HSUM_MAXPERMIT)
      integer*4 hsum_mdur(HSUM_MAXPERMIT)
      integer*4 hsum_sdur(HSUM_MAXPERMIT)

C     /*duration of permit in hours,... */

C     /* number of triggers an weather codesy */
      integer*4 hsum_ntrig
      integer*4 hsum_nweat

C     /*number of triggers on a specific mirror */
      integer*4 hsum_ntri(HR_MAX_MIR)

C     /*number of triggers/min on a mirror */
      real*8 hsum_ntrim(HR_MAX_MIR)
      integer*4 hsum_hoper(HR_MAX_MIR)
      integer*4 hsum_moper(HR_MAX_MIR)
      integer*4 hsum_soper(HR_MAX_MIR)
      integer*4 hsum_hwea(HSUM_MAXPERMIT)
      integer*4 hsum_mwea(HSUM_MAXPERMIT)
      integer*4 hsum_swea(HSUM_MAXPERMIT)

C     /*Flags for problems in the part */
      integer*1 hsum_staflag
      integer*1 hsum_permflag

C     /* time of weather code in hours... */
