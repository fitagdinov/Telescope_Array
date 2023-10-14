c
c     Bank for variables calculated from rusdmc bank.  
c     Usefull for applying cuts on thrown MC variables
c     (e.g. when studying trigger/reconstruction efficiency) and resolution studies.
c     
c     Dmitri Ivanov (ivanov@physics.rutgers.edu)
c     Apr 11, 2009
      
c     Last modified: Nov 29, 2009
      
c

      integer RUSDMC1_BANKID  
      parameter(RUSDMC1_BANKID=13106)
      integer RUSDMC1_BANKVERSION
      parameter (RUSDMC1_BANKVERSION=1)

      real*8 RUSDMC1_SD_ORIGIN_X_CLF
      parameter (RUSDMC1_SD_ORIGIN_X_CLF=-12.2435)
      real*8 RUSDMC1_SD_ORIGIN_Y_CLF
      parameter (RUSDMC1_SD_ORIGIN_Y_CLF=-16.4406)
      
      integer*4 rusdmc1_common_to_bank
      external  rusdmc1_common_to_bank
      integer*4 rusdmc1_bank_to_dst
      external  rusdmc1_bank_to_dst
      integer*4 rusdmc1_common_to_dst
      external  rusdmc1_common_to_dst
      integer*4 rusdmc1_bank_to_common
      external  rusdmc1_bank_to_common
      integer*4 rusdmc1_common_to_dump
      external  rusdmc1_common_to_dump
      integer*4 rusdmc1_common_to_dumpf
      external  rusdmc1_common_to_dumpf
      
      common /rusdmc1/
     &     rusdmc1_xcore,rusdmc1_ycore,rusdmc1_t0,rusdmc1_bdist,
     &     rusdmc1_tdistbr,rusdmc1_tdistlr,rusdmc1_tdistsk,rusdmc1_tdist
      
      real*8 rusdmc1_xcore      ! Thrown MC core position, CLF frame, [1200m] units, SD origin subtracted off
      real*8 rusdmc1_ycore
      real*8 rusdmc1_t0         ! Core hit time, [1200m], with respect to the earliest SD time in the event readout
      real*8 rusdmc1_bdist      ! Distance of the core from the edge of the array.If negative, then the core is outside.  
      real*8 rusdmc1_tdistbr    ! Distance of the core position from BR T-Shaped boundary, negative if not in BR
      real*8 rusdmc1_tdistlr    ! same for LR
      real*8 rusdmc1_tdistsk    ! same for SK
      real*8 rusdmc1_tdist      ! Closest distance to any T-shaped boundary (BR,LR,SK)
      
      
