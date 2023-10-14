C univ_dst.inc
C universal global variables
C FORTRAN version of univ_dst.h
C $Source: /home/bergman/hires/prog/mcru/inc/RCS/univ_dst.inc,v $
C created by LW at 13:12 on 3/16/94
C
C
      integer*4 TA_UNIV_MAXMIR, TA_UNIV_MAXTUBE
      PARAMETER (TA_UNIV_MAXMIR=14)
      PARAMETER (TA_UNIV_MAXTUBE=256*TA_UNIV_MAXMIR)
      integer*4 TA_UNIV_MIRTUBE
      PARAMETER (TA_UNIV_MIRTUBE=256)
c      integer*4 TA_UNIV_PRO2MIR
c      PARAMETER (TA_UNIV_PRO2MIR=4)

      integer*4 FDPROF_MAXTSLICE
      PARAMETER (FDPROF_MAXTSLICE=512)

      integer*4 FDTUBEPROF_MAXTUBE
      PARAMETER (FDTUBEPROF_MAXTUBE=2000)

      integer*4 STTUBEPROF_MAXTUBE
      PARAMETER (STTUBEPROF_MAXTUBE=2000)

C
C....mode constants for dst_open_unit_()
      integer*4 MODE_READ_DST, MODE_WRITE_DST
      integer*4 MODE_APPEND_DST
      PARAMETER (MODE_READ_DST=1)
      PARAMETER (MODE_WRITE_DST=2)
      PARAMETER (MODE_APPEND_DST=3)
C
C....diagnostic level constants for dst_read_bank_()
      integer*4 DIAG_NONE_DST, DIAG_WARN_DST
      integer*4 DIAG_FULL_DST
      PARAMETER (DIAG_NONE_DST=0)
C         no checking
      PARAMETER (DIAG_WARN_DST=1)
C         check, issue warning, take no action
      PARAMETER (DIAG_FULL_DST=2)
C         check, issue warning, take appropriate action
C....default bank size
      integer*4 DFLT_DST_BANK_SIZE
      PARAMETER (DFLT_DST_BANK_SIZE=100000) 

C constants
      double precision PI,D2R,R2D,CSPEED
      PARAMETER (PI=3.1415926535897931159979634685441851615906)
      PARAMETER (D2R=0.0174532925199432954743716805978692718782)
      PARAMETER (R2D=57.2957795130823228646477218717336654663086)
      PARAMETER (CSPEED=0.299792458d0)  ! meters / ns

C Central Laser Facility GPS
      double precision CLF_LATITUDE,CLF_LONGITUDE,CLF_ALTITUDE
      PARAMETER (CLF_LATITUDE=0.68586060387114d0)  ! (  39.296917936112 degrees)
      PARAMETER (CLF_LONGITUDE=-1.9706291367663d0)  ! (-112.90873252222 degrees)
      PARAMETER (CLF_ALTITUDE=1370.046d0)  ! meters

!       double precision GEOBR_LATITUDE,GEOBR_LONGITUDE,GEOBR_ALTITUDE
!       PARAMETER (GEOBR_LATITUDE=0.68396486d0)     ! (  39.18830 degrees)
!       PARAMETER (GEOBR_LONGITUDE=-1.96719027d0)   ! (-112.71170 degrees)
!       PARAMETER (GEOBR_ALTITUDE=1404.d0)          ! meters
! 
!       double precision GEOLR_LATITUDE,GEOLR_LONGITUDE,GEOLR_ALTITUDE
!       PARAMETER (GEOLR_LATITUDE=0.68430695d0)     ! (  39.20790 degrees)
!       PARAMETER (GEOLR_LONGITUDE=-1.97434385d0)   ! (-113.12157 degrees)
!       PARAMETER (GEOLR_ALTITUDE=1554.d0)          ! meters
