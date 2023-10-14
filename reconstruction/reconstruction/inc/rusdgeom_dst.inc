c     Bank for SD pass1 geometry fit information
c     Dmitri Ivanov (dmiivanov@gmail.com)
c     Nov 4, 2008

c     Last Modified: Nov 6, 2018


      integer RUSDGEOM_BANKID
      parameter (RUSDGEOM_BANKID=13104)
      integer RUSDGEOM_BANKVERSION
      parameter (RUSDGEOM_BANKVERSION=001)
      

      integer RUSDGEOM_MSDS        ! max number of SDS in the trigger
      parameter(RUSDGEOM_MSDS=512)

      integer RUSDGEOM_MSIG        ! max number of signals in one SD
      parameter(RUSDGEOM_MSIG=100)

c     SD origin with respect to CLF origin in CLF frame, in [1200m] units
      real*8 RUSDGEOM_ORIGIN_X_CLF
      parameter (RUSDGEOM_ORIGIN_X_CLF=-12.2435)
      real*8 RUSDGEOM_ORIGIN_Y_CLF
      parameter (RUSDGEOM_ORIGIN_Y_CLF=-16.4406)

c     For converting time in uS to distance in units of counter
c     separation distance 1200m. Numerically, this is 
c     c*(10^(-6)(S/uS))/1200m, where c is the speed of light in m/s 
      real*8 RUSDGEOM_TIMDIST
      parameter(RUSDGEOM_TIMDIST=0.249827048333)

      integer*4 rusdgeom_common_to_bank
      external  rusdgeom_common_to_bank
      integer*4 rusdgeom_bank_to_dst
      external  rusdgeom_bank_to_dst
      integer*4 rusdgeom_common_to_dst
      external  rusdgeom_common_to_dst
      integer*4 rusdgeom_bank_to_common
      external  rusdgeom_bank_to_common
      integer*4 rusdgeom_common_to_dump
      external  rusdgeom_common_to_dump
      integer*4 rusdgeom_common_to_dumpf
      external  rusdgeom_common_to_dumpf



      common /rusdgeom/ 
     &     rusdgeom_sdsigq,rusdgeom_sdsigt,rusdgeom_sdsigte,
     &     rusdgeom_xyzclf,rusdgeom_pulsa,rusdgeom_sdtime,
     &     rusdgeom_sdterr,rusdgeom_xcore,rusdgeom_dxcore,
     &     rusdgeom_ycore,rusdgeom_dycore,rusdgeom_t0,rusdgeom_dt0,
     &     rusdgeom_theta,rusdgeom_dtheta,rusdgeom_phi,rusdgeom_dphi,
     &     rusdgeom_chi2,rusdgeom_a,rusdgeom_da,rusdgeom_tearliest,
     &     rusdgeom_igsig,rusdgeom_irufptn,rusdgeom_igsd,rusdgeom_xxyy,
     &     rusdgeom_nsig,rusdgeom_sdirufptn,rusdgeom_ndof,rusdgeom_nsds


c     Relative time of each signal in each counter in [1200m] units. To convert this into time after midnight, do
c     'time = tearliest + sdsigt(j,i)/RUSDGEOM_TIMDIST * (1e-6)' 
      real*8 rusdgeom_sdsigt(RUSDGEOM_MSIG,RUSDGEOM_MSDS)

c     Time resolution of each signal in each counter in [1200m] units. To convert this into time after midnight, do
c     'time = tearliest + sdsigt(j,i)/RUSDGEOM_TIMDIST * (1e-6)' 
      real*8 rusdgeom_sdsigte(RUSDGEOM_MSIG,RUSDGEOM_MSDS)
      
      real*8 rusdgeom_sdsigq(RUSDGEOM_MSIG,RUSDGEOM_MSDS) ! charge in VEM of the given signal in the given counter  
      
c     clf-frame xyz coordinates of each sd, [1200m] units, with respect to CLF origin 
      real*8 rusdgeom_xyzclf(3,RUSDGEOM_MSDS)
      
c     These two variables normally use a signal that's a part of event. If the sd is not a part of the event 
c     (see igsd - variable),then these variables will be calculated using the first signal seen by this sd 
      real*8 rusdgeom_pulsa (RUSDGEOM_MSDS) ! charge of the i'th counter  

c     To convert this to time after midnight in seconds, do 'time = tearliest + sdtime(i)/RUSDGEOM_TIMDIST * (1e-6)'
      real*8 rusdgeom_sdtime(RUSDGEOM_MSDS) ! relative time of the i'th counter, [1200m] units  
      
      real*8 rusdgeom_sdterr(RUSDGEOM_MSDS) ! time resolution of the i'th counter, [1200m] units
      
c     Results of geometry fits (double precision data)
c     (1) - for plane fit
c     (2) - for Modified Linsley's fit
c     (3) - for the final result
      
c     To find the core position in CLF frame in meters with respect to CLF origin, do
c     coreX = (xcore+RUSDGEOM_ORIGIN_X_CLF)*1200.0, coreY = (ycore+RUSDGEOM_ORIGIN_Y_CLF)*1200.0
c     CLF XY plane is used as 'SD ground plane', that's why zcore is absent.
      real*8 rusdgeom_xcore(3)  ! core X and Y, in 1200m units, with respect to CLF, SD origin subtracted off
      real*8 rusdgeom_dxcore(3) ! uncertainty on xcore
      real*8 rusdgeom_ycore(3)
      real*8 rusdgeom_dycore(3)
      real*8 rusdgeom_t0(3)     ! relative time when the core hits the CLF plane, [1200m] units 
      real*8 rusdgeom_dt0(3)
      real*8 rusdgeom_theta(3)  ! event zenith angle, degrees 
      real*8 rusdgeom_dtheta(3)
      real*8 rusdgeom_phi(3)    ! event azimuthal angle, degrees 
      real*8 rusdgeom_dphi(3)
      real*8 rusdgeom_chi2(3)   ! chi2 of the fit 
      real*8 rusdgeom_a         ! curvature parameter
      real*8 rusdgeom_da
      
c     Earliest signal time in the trigger in seconds after midnight. All other quoted times are relative to this time, 
c     and are converted to [1200m] units for convenience.
      real*8 rusdgeom_tearliest ! Earliest signal time in the trigger in seconds after midnight
      
c     igsig(signal_index,sd_index):
c     0 - given SD signal corresponds to counter that was not working properly
c     1 - given SD is not a part of any clusters
c     2 - given SD is a part of space-cluster
c     3 - given SD signal passed a rough time pattern recognition
c     4 - given SD signal is a part of the event
c     5 - given SD signal saturates the counter
      integer*4 rusdgeom_igsig(RUSDGEOM_MSIG,RUSDGEOM_MSDS)
      
      
c     irufptn(signal_index,sd_index): points to the signal in rufptn bank, which is a signal-based list of variables
      integer*4 rusdgeom_irufptn(RUSDGEOM_MSIG,RUSDGEOM_MSDS)



c     igsd(sd_index):
c     0 - sd was not working properly (bad 1MIP fit,etc)
c     1 - sd was working but is none of its signals is a part of event
c     2 - sd is a part of event
c     3 - sd is saturated
      integer*4 rusdgeom_igsd(RUSDGEOM_MSDS)
      
      integer*4 rusdgeom_xxyy(RUSDGEOM_MSDS) ! sd position IDs
      
      integer*4 rusdgeom_nsig(RUSDGEOM_MSDS) ! number of independent signals (hits) in each SD 
      
c     For each counter that's a part of the event there is only one signal chosen.
c     sdirufptn[sd_index] contains rufptn index (rufptn is a signal-based list of variables) of the chosen signal.  
c     If this sd is not a part of the event (see igsd variable), then we quote here the rufptn index of the first 
c     singnal seen by the sd.
      integer*4 rusdgeom_sdirufptn(RUSDGEOM_MSDS)
      
c     # of d.o.f. for geom. fitting, (1) - plane fitting, (2) - Modified Linsley fit, (3) - Final result
c     Calculated as  (# of counters in the fit) - (# of fit parameters)
      integer*4 rusdgeom_ndof(3) 
      
      integer*4 rusdgeom_nsds   ! number of sds in the trigger   
      
      
      
      
      
      
