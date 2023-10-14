c     Bank for SD raw monitoring data
c     Dmitri Ivanov (dmiivanov@gmail.com)
c     Jun 17, 2008

c     Last modified: Dmitri Ivanov, Apr. 30, 2019

      integer SDMON_BANKID
      parameter (SDMON_BANKID=13102)
      integer SDMON_BANKVERSION
      parameter (SDMON_BANKVERSION=001)



c     to determine the tower
      integer SDMON_BR
      parameter(SDMON_BR=0)
      integer SDMON_LR
      parameter(SDMON_LR=1)
      integer SDMON_SK
      parameter (SDMON_SK=2)
      integer SDMON_BRLR
      parameter(SDMON_BRLR=3)
      integer SDMON_BRSK
      parameter(SDMON_BRSK=4)
      integer SDMON_LRSK
      parameter (SDMON_LRSK=5)
      integer SDMON_BRLRSK
      parameter (SDMON_BRLRSK=6)

      
      
      integer SDMON_X_MAX          ! maximum value of x coordinate
      parameter (SDMON_X_MAX=24)
      
      integer SDMON_Y_MAX
      parameter (SDMON_Y_MAX=28)   ! maximum value of y coordinate
      
      
      integer SDMON_MAXSDS 
      parameter (SDMON_MAXSDS=512)    ! maximum number of SDs
      
      integer SDMON_NMONCHAN    ! number of 1MIP histogram channels
      parameter (SDMON_NMONCHAN=512)

      integer SDMON_CL_VAL      ! clean values of monitoring information
      parameter (SDMON_CL_VAL=-1)

      
      integer*4 sdmon_common_to_bank
      external  sdmon_common_to_bank
      integer*4 sdmon_bank_to_dst
      external  sdmon_bank_to_dst
      integer*4 sdmon_common_to_dst
      external  sdmon_common_to_dst
      integer*4 sdmon_bank_to_common
      external  sdmon_bank_to_common
      integer*4 sdmon_common_to_dump
      external  sdmon_common_to_dump
      integer*4 sdmon_common_to_dumpf
      external  sdmon_common_to_dumpf

      common /sdmon/
     &     sdmon_event_num,sdmon_site,sdmon_run_id,sdmon_errcode,
     &     sdmon_yymmddb,sdmon_hhmmssb,sdmon_yymmdde,sdmon_hhmmsse,
     &     sdmon_nsds,sdmon_lind,sdmon_xxyy,sdmon_hmip,
     &     sdmon_hped,sdmon_hpht,sdmon_hpcg,sdmon_pchmip,sdmon_pchped,
     &     sdmon_lhpchmip,sdmon_lhpchped,sdmon_rhpchmip,sdmon_rhpchped,
     &     sdmon_tgtblnum,sdmon_mclkcnt,sdmon_ccadcbvt,sdmon_blankvl1,
     &     sdmon_ccadcbct,sdmon_blankvl2,sdmon_ccadcrvt,sdmon_ccadcbtm,
     &     sdmon_ccadcsvt,sdmon_ccadctmp,sdmon_mbadcgnd,sdmon_mbadcsdt,
     &     sdmon_mbadc5vt,sdmon_mbadcsdh,sdmon_mbadc33v,sdmon_mbadcbdt,
     &     sdmon_mbadc18v,sdmon_mbadc12v,sdmon_crminlv2,sdmon_crminlv1,
     &     sdmon_gpsyymmdd,sdmon_gpshhmmss,sdmon_gpsflag,sdmon_mip,
     &     sdmon_curtgrate,sdmon_num_sat,sdmon_mftndof,sdmon_mftchi2,
     &     sdmon_mftp,sdmon_mftpe
      
      
      integer*4 sdmon_event_num ! monitoring cycle number
      integer*4 sdmon_site      ! BR=0,LR=1,SK=2,BRSK=4,LRSK=5,BRLRSK=6
      integer*4 sdmon_run_id(3) ! run number, (1) - for BR, (2) - for LR, (3) - for SK. -1 if irrelevant
      integer*4 sdmon_errcode   ! 2,3 are reserved for 1st and last mon. cycles. All others should have zero.
      integer*4 sdmon_yymmddb   ! year,month,day at the beginning of the cycle
      integer*4 sdmon_hhmmssb   ! hour,minute,second at the beginning of the cycle
      integer*4 sdmon_yymmdde   ! year,month,day at the end of the cycle
      integer*4 sdmon_hhmmsse   ! hour,minute,second at the end of the cycle
      
      integer*4 sdmon_nsds(3)   ! # of SDs present in the setting file: (1)-BR, (2)-LR, (3)-SK, -1 if irrelevant  
    
      integer*4 sdmon_lind      ! largest detector index in the monitoring cycle. (+1 in fortran to get the largest index <= SDMON_MAXSDS)
    
      integer*4 sdmon_xxyy(SDMON_MAXSDS) ! SD location
      integer*4 sdmon_hmip(512,2,SDMON_MAXSDS) ! 1MIP hist for lower(1) and upper(2), 512 channels, ped. not subtracted
      integer*4 sdmon_hped(256,2,SDMON_MAXSDS) ! ped hist for lower and upper, 256 channels
      integer*4 sdmon_hpht(128,2,SDMON_MAXSDS) ! pulse height histograms, 128 channels
      integer*4 sdmon_hpcg(128,2,SDMON_MAXSDS) ! pulse charge histograms, 128 channels
      integer*4 sdmon_pchmip(2,SDMON_MAXSDS)   ! peak channel of 1mip histograms (+1 to get the array index)
      integer*4 sdmon_pchped(2,SDMON_MAXSDS)   ! peak channel of pedestal histograms (+1 to get the array index)
      integer*4 sdmon_lhpchmip(2,SDMON_MAXSDS) ! left half-peak channels of 1mip histograms  (+1 to get the array index)
      integer*4 sdmon_lhpchped(2,SDMON_MAXSDS) ! left half-peak channels of ped histograms  (+1 to get the array index)
      integer*4 sdmon_rhpchmip(2,SDMON_MAXSDS) ! right half-peak channels of 1mip histograms (+1 to get the array index)
      integer*4 sdmon_rhpchped(2,SDMON_MAXSDS) ! right half-peak channels of 1ped histograms (+1 to get the array index)
  
      
      integer*4 sdmon_tgtblnum(600,SDMON_MAXSDS) ! Number of trigger tables for every second
      integer*4 sdmon_mclkcnt (600,SDMON_MAXSDS) ! Maximum clock count monitoring, for every second
      
c     CC
      integer*4 sdmon_ccadcbvt(10,SDMON_MAXSDS) ! CC ADC value Batt Voltage, for 10 minutes 
      integer*4 sdmon_blankvl1(10,SDMON_MAXSDS) ! 1st blank value in b/w, in case later it will have something 
      integer*4 sdmon_ccadcbct(10,SDMON_MAXSDS) ! CC ADC Value Batt Current 
      integer*4 sdmon_blankvl2(10,SDMON_MAXSDS) ! 2nd blank value in b/w, in case later it will have something 
      integer*4 sdmon_ccadcrvt(10,SDMON_MAXSDS) ! CC ADC Value Ref Voltage 
      integer*4 sdmon_ccadcbtm(10,SDMON_MAXSDS) ! CC ADC Value Batt Temp 
      integer*4 sdmon_ccadcsvt(10,SDMON_MAXSDS) ! CC ADC Value SolarVoltage 
      integer*4 sdmon_ccadctmp(10,SDMON_MAXSDS) ! CC ADC Value CC Temp  
      
      
      
c     Mainboard
      integer*4 sdmon_mbadcgnd(10,SDMON_MAXSDS) ! Main board ADC value "GND" 
      integer*4 sdmon_mbadcsdt(10,SDMON_MAXSDS) ! Main board ADC value SDTemp 
      integer*4 sdmon_mbadc5vt(10,SDMON_MAXSDS) ! Main board ADC value 5.0V  
      integer*4 sdmon_mbadcsdh(10,SDMON_MAXSDS) ! Main board ADC value SDHum 
      integer*4 sdmon_mbadc33v(10,SDMON_MAXSDS) ! Main board ADC value 3.3V 
      integer*4 sdmon_mbadcbdt(10,SDMON_MAXSDS) ! Main board ADC value BDTemp 
      integer*4 sdmon_mbadc18v(10,SDMON_MAXSDS) ! Main boad ADC value 1.8V  
      integer*4 sdmon_mbadc12v(10,SDMON_MAXSDS) ! Main boad ADC value 1.2V  
      
c     Rate Monitor
      integer*4 sdmon_crminlv2(10,SDMON_MAXSDS) ! 1min count rate Lv2(>3mip) 
      integer*4 sdmon_crminlv1(10,SDMON_MAXSDS) ! 1min count rate Lv1(>0.3mip) 
      
      
c     GPS Monitor
      integer*4 sdmon_gpsyymmdd(SDMON_MAXSDS) ! Date(YMD)  
      integer*4 sdmon_gpshhmmss(SDMON_MAXSDS) ! Time(HMS)  
      integer*4 sdmon_gpsflag(SDMON_MAXSDS)   ! GPSFLAG 
      integer*4 sdmon_curtgrate(SDMON_MAXSDS) ! CURRENT TRIGGER Rate
      integer*4 sdmon_num_sat(SDMON_MAXSDS)   ! number of satellites seen by the SD
      
c  Results from fitting 1MIP histograms
c     1st index: 1=lower, 2=upper layers
      integer*4 sdmon_mftndof(2,SDMON_MAXSDS) ! numbers of degrees of freedom in 1MIP fits 
      real*8 sdmon_mip(2,SDMON_MAXSDS) ! 1MIP value (ped. subtracted) 
      real*8 sdmon_mftchi2(2,SDMON_MAXSDS) ! chi2's of 1MIP fits
      
c     2nd index: 1=lower, 2=upper layers
c     1MIP Fit function: 
c     mftp(4,*,*)*(1+mftp(3,*,*)*(x-mftp(1,*,*)))*exp(-(x-mftp(1,*,*))**2/2/mftp(2,*,*)**2)/sqrt(2*PI)/mftp(2,*,*)
c     Fit parameters:
c     (1)=Gauss Mean
c     (2)=Gauss Sigma
c     (3)=Linear Coefficient
c     (4)=Overall Scalling Factor
      real*8 sdmon_mftp (4,2,SDMON_MAXSDS) ! 1MIP fit parameters 
      real*8 sdmon_mftpe(4,2,SDMON_MAXSDS) ! Errors on 1MIP fit parameters 
