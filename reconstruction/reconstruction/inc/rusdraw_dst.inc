c     Bank for SD raw data
c     Dmitri Ivanov (dmiivanov@gmail.com)
c     Jun 17, 2008


c     Last modified: Nov. 6, 2018
      
      integer RUSDRAW_BANKID
      parameter(RUSDRAW_BANKID=13101)
      integer RUSDRAW_BANKVERSION
      parameter (RUSDRAW_BANKVERSION=1)

c     to determine the tower
      integer RUSDRAW_BR
      parameter(RUSDRAW_BR=0)
      integer RUSDRAW_LR
      parameter(RUSDRAW_LR=1)
      integer RUSDRAW_SK
      parameter (RUSDRAW_SK=2)
      integer RUSDRAW_BRLR
      parameter(RUSDRAW_BRLR=3)
      integer RUSDRAW_BRSK
      parameter(RUSDRAW_BRSK=4)
      integer RUSDRAW_LRSK
      parameter (RUSDRAW_LRSK=5)
      integer RUSDRAW_BRLRSK
      parameter (RUSDRAW_BRLRSK=6)
      
      integer RUSDRAWMWF 
      parameter (RUSDRAWMWF=1024) ! maximum number of waveforms in the event

      integer RUSDRAW_NCHAN_SD  !  128 fadc channels
      parameter (RUSDRAW_NCHAN_SD=128)

      integer*4 rusdraw_common_to_bank
      external  rusdraw_common_to_bank
      integer*4 rusdraw_bank_to_dst
      external  rusdraw_bank_to_dst
      integer*4 rusdraw_common_to_dst
      external  rusdraw_common_to_dst
      integer*4 rusdraw_bank_to_common
      external  rusdraw_bank_to_common
      integer*4 rusdraw_common_to_dump
      external  rusdraw_common_to_dump
      integer*4 rusdraw_common_to_dumpf
      external  rusdraw_common_to_dumpf

      common /rusdraw/
     &     rusdraw_event_num,rusdraw_event_code,rusdraw_site,
     &     rusdraw_trig_id,rusdraw_run_id,rusdraw_errcode,
     &     rusdraw_yymmdd,rusdraw_hhmmss,rusdraw_usec,rusdraw_monyymmdd,
     &     rusdraw_monhhmmss,rusdraw_nofwf,rusdraw_nretry,rusdraw_wf_id,
     &     rusdraw_trig_code,rusdraw_xxyy,rusdraw_clkcnt,
     &     rusdraw_mclkcnt,rusdraw_fadcti,rusdraw_fadcav,rusdraw_fadc,
     &     rusdraw_pchmip,rusdraw_pchped,rusdraw_lhpchmip,
     &     rusdraw_lhpchped,rusdraw_rhpchmip,rusdraw_rhpchped,
     &     rusdraw_mftndof,rusdraw_mip,rusdraw_mftchi2,rusdraw_mftp,
     &     rusdraw_mftpe

      integer*4 rusdraw_event_num ! event number
      integer*4 rusdraw_event_code ! 1=data, 0=MC
      integer*4 rusdraw_site    ! BR=0,LR=1,SK=2,BRSK=4,LRSK=5,BRLRSK=6
      integer*4 rusdraw_run_id(3) ! run number, (1) - for BR, (2) - for LR, (3) - for SK. -1 if irrelevant
      integer*4 rusdraw_trig_id(3) ! event trigger id for each tower, -1 if irrelevant
      integer*4 rusdraw_errcode ! should be zero if there were no readout problems
      integer*4 rusdraw_yymmdd  ! event year,month,day
      integer*4 rusdraw_hhmmss  ! event hour,minute,second
      integer*4 rusdraw_usec    ! event micro second
      integer*4 rusdraw_monyymmdd ! yymmdd at the beginning of the mon. cycle used in this event 
      integer*4 rusdraw_monhhmmss ! hhmmss at the beginning of the mon. cycle used in this event
      integer*4 rusdraw_nofwf   ! number of waveforms
c     This is for each waveform (maximum number of waveforms is RUSDRAWMWF)
      integer*4 rusdraw_nretry(RUSDRAWMWF) ! number of retries to get the waveform
      integer*4 rusdraw_wf_id(RUSDRAWMWF) ! waveform id in the trigger
      integer*4 rusdraw_trig_code(RUSDRAWMWF) ! level1 trigger code
      integer*4 rusdraw_xxyy(RUSDRAWMWF) ! SDs that were hit
      integer*4 rusdraw_clkcnt(RUSDRAWMWF) ! Clock counts at the beginning of the waveforms
      integer*4 rusdraw_mclkcnt(RUSDRAWMWF) ! max. clock counts

c     1st index: 1=lower, 2=upper SD layers
      integer*4 rusdraw_fadcti(2,RUSDRAWMWF) ! fadc trace integral
      integer*4 rusdraw_fadcav(2,RUSDRAWMWF) ! fadc average average

c     2nd index: 1=lower, 2=upper SD layers
      integer*4 rusdraw_fadc(128,2,RUSDRAWMWF) ! fadc trace for lower (1) and upper (2)
      

c     Useful calibration information:
c     1st index: 1=lower, 2=upper layers
      integer*4 rusdraw_pchmip(2,RUSDRAWMWF) ! peak channel of 1mip histograms (+1 to get the array index)
      integer*4 rusdraw_pchped(2,RUSDRAWMWF) ! peak channel of pedestal histograms (+1 to get the array index)
      integer*4 rusdraw_lhpchmip(2,RUSDRAWMWF) ! left half-peak channels of 1mip histograms  (+1 to get the array index)
      integer*4 rusdraw_lhpchped(2,RUSDRAWMWF) ! left half-peak channels of ped histograms  (+1 to get the array index)
      integer*4 rusdraw_rhpchmip(2,RUSDRAWMWF) ! right half-peak channels of 1mip histograms (+1 to get the array index)
      integer*4 rusdraw_rhpchped(2,RUSDRAWMWF) ! right half-peak channels of 1ped histograms (+1 to get the array index)
      
      
c  Results from fitting 1MIP histograms
      integer*4 rusdraw_mftndof(2,RUSDRAWMWF) ! numbers of degrees of freedom in 1MIP fits
      real*8 rusdraw_mip(2,RUSDRAWMWF) ! 1MIP value (ped. subtracted) 
      real*8 rusdraw_mftchi2(2,RUSDRAWMWF) ! chi2's of 1MIP fits
      
c     2nd index: 1=lower, 2=upper
c     1MIP Fit function:
c     mftp(4,*,*)*(1+mftp(3,*,*)*(x-mftp(1,*,*)))*exp(-(x-mftp(1,*,*))**2/2/mftp(2,*,*)**2)/sqrt(2*PI)/mftp(2,*,*)
c     Fit parameters:
c     (1)=Gauss Mean
c     (2)=Gauss Sigma
c     (3)=Linear Coefficient
c     (4)=Overall Scalling Factor
      real*8 rusdraw_mftp(4,2,RUSDRAWMWF) ! 1MIP fit parameters 
      real*8 rusdraw_mftpe(4,2,RUSDRAWMWF)! Errors on 1MIP fit parameters 
