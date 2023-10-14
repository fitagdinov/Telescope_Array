c     Bank for SDMC inputs
c     Benjamin Stokes (stokes@physics.rutgers.edu)
c     Jan 22, 2009

c     Last modified: Jan 22, 2009

      integer RUSDMC_BANKID  
      parameter(RUSDMC_BANKID=13105)
      integer RUSDMC_BANKVERSION
      parameter (RUSDMC_BANKVERSION=0)

      integer*4 rusdmc_common_to_bank
      external  rusdmc_common_to_bank
      integer*4 rusdmc_bank_to_dst
      external  rusdmc_bank_to_dst
      integer*4 rusdmc_common_to_dst
      external  rusdmc_common_to_dst
      integer*4 rusdmc_bank_to_common
      external  rusdmc_bank_to_common
      integer*4 rusdmc_common_to_dump
      external  rusdmc_common_to_dump
      integer*4 rusdmc_common_to_dumpf
      external  rusdmc_common_to_dumpf

      common /rusdmc/
     &     rusdmc_event_num, rusdmc_parttype, rusdmc_corecounter,
     &	   rusdmc_energy, rusdmc_height, rusdmc_theta, rusdmc_phi,
     &	   rusdmc_corexy   

      integer*4 rusdmc_event_num ! event number
      integer*4 rusdmc_parttype ! Corsika particle code [proton=14, iron=5626, for others, consult Corsika manual]
      integer*4 rusdmc_corecounter ! counter closest to core
      integer*4 rusdmc_tc ! clock count corresponding to t0
      real*4 rusdmc_energy ! total energy of primary particle [hypot(p,m)]
      real*4 rusdmc_height ! height of first interation [cm] */
      real*4 rusdmc_theta ! zenith angle [rad] 
      real*4 rusdmc_phi ! azimuthal angle (N of E) [rad]
      real*4 rusdmc_corexyz(3) ! separation vector between position of corecounter and actual core position
