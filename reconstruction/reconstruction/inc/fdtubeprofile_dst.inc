c Created 2010/01 LMS
C     added X0 and Lambda fits (SS 2011/09)

      integer FDTUBEPROFILE_BANKID
      parameter (FDTUBEPROFILE_BANKID=12096)
      integer FDTUBEPROFILE_BANKVERSION
      parameter (FDTUBEPROFILE_BANKVERSION=3)

      integer*4 fdtubeprofile_common_to_bank
      external  fdtubeprofile_common_to_bank
      integer*4 fdtubeprofile_bank_to_dst
      external  fdtubeprofile_bank_to_dst
      integer*4 fdtubeprofile_common_to_dst
      external  fdtubeprofile_common_to_dst
      integer*4 fdtubeprofile_bank_to_common
      external  fdtubeprofile_bank_to_common
      integer*4 fdtubeprofile_common_to_dump
      external  fdtubeprofile_common_to_dump
      integer*4 fdtubeprofile_common_to_dumpf
      external  fdtubeprofile_common_to_dumpf

      common /fdtubeprofile/
     &     fdtubeprofile_ntube,fdtubeprofile_ngtube,
     &     fdtubeprofile_rp,fdtubeprofile_psi,fdtubeprofile_t0,
     &     fdtubeprofile_Xmax,fdtubeprofile_eXmax,
     &     fdtubeprofile_Nmax,fdtubeprofile_eNmax,
     &     fdtubeprofile_Energy,fdtubeprofile_eEnergy,
     &     fdtubeprofile_chi2,
     &     fdtubeprofile_X0, fdtubeprofile_eX0, 
     &     fdtubeprofile_Lambda, fdtubeprofile_eLambda, 
     &     fdtubeprofile_x, fdtubeprofile_npe,
     &     fdtubeprofile_enpe,fdtubeprofile_eacptfrac,
     &     fdtubeprofile_acpt,fdtubeprofile_eacpt,
     &     fdtubeprofile_flux,fdtubeprofile_eflux,
     &     fdtubeprofile_simnpe,fdtubeprofile_nfl,
     &     fdtubeprofile_ncvdir,fdtubeprofile_ncvmie,
     &     fdtubeprofile_ncvray,fdtubeprofile_simflux,
     &     fdtubeprofile_ne,fdtubeprofile_ene,
     &     fdtubeprofile_tres,fdtubeprofile_tchi2,
     &     fdtubeprofile_camera,fdtubeprofile_tube,
     &     fdtubeprofile_tube_qual,fdtubeprofile_status,
     &     fdtubeprofile_siteid,fdtubeprofile_mc,
     &     fdtubeprofile_simtime,fdtubeprofile_simtrms,
     &     fdtubeprofile_simtres,fdtubeprofile_timechi2

      integer*4 fdtubeprofile_ntube                           ! total number of tubes
      integer*4 fdtubeprofile_ngtube(3)                       ! number of good tubes          

      real*8 fdtubeprofile_rp(3)                              ! Impact parameter (meters)
      real*8 fdtubeprofile_psi(3)                             ! Shower-detector plane angle (radians)
      real*8 fdtubeprofile_t0(3)                              ! Detection time at Rp, less Rp travel time (ns)

      real*8 fdtubeprofile_Xmax(3)                            ! Shower maximum (g/cm2)
      real*8 fdtubeprofile_eXmax(3)                           ! uncertainty on xmax
      real*8 fdtubeprofile_Nmax(3)                            ! Number of charged particles at shower maximum
      real*8 fdtubeprofile_eNmax(3)                           ! uncertainty on nmax
      real*8 fdtubeprofile_Energy(3)                          ! Initial cosmic-ray energy
      real*8 fdtubeprofile_eEnergy(3)                         ! uncertainty on energy
      real*8 fdtubeprofile_X0(3)                              ! effective depth of first interaction
      real*8 fdtubeprofile_eX0(3)                             ! uncertainty of X0
      real*8 fdtubeprofile_Lambda(3)                          ! profile width parameter
      real*8 fdtubeprofile_eLambda(3)                         ! uncertainty in Lambda
      real*8 fdtubeprofile_chi2(3)                            ! Total chi2 of fit

      real*8 fdtubeprofile_x(FDTUBEPROF_MAXTUBE,3)            ! slant depth at middle of time slice (g/cm2)

      real*8 fdtubeprofile_npe(FDTUBEPROF_MAXTUBE,3)          ! number of photoelectrons in tube
      real*8 fdtubeprofile_enpe(FDTUBEPROF_MAXTUBE,3)         ! uncertainty on NPE, including uncertainty from acceptance
      real*8 fdtubeprofile_eacptfrac(FDTUBEPROF_MAXTUBE,3)    ! fraction of uncertainty due to acceptance.  

      real*8 fdtubeprofile_acpt(FDTUBEPROF_MAXTUBE,3)         ! PMT acceptance
      real*8 fdtubeprofile_eacpt(FDTUBEPROF_MAXTUBE,3)        ! binomial uncertainty on acceptance 

      real*8 fdtubeprofile_flux(FDTUBEPROF_MAXTUBE,3)         ! flux at the mirror [detectable npe / (m2 * radian)]
      real*8 fdtubeprofile_eflux(FDTUBEPROF_MAXTUBE,3)        ! uncertainty on flux

      real*8 fdtubeprofile_simnpe(FDTUBEPROF_MAXTUBE,3)       ! simulated photo-electrons in tube

      real*8 fdtubeprofile_nfl(FDTUBEPROF_MAXTUBE,3)          ! Flux of simulated fluorescence photons
      real*8 fdtubeprofile_ncvdir(FDTUBEPROF_MAXTUBE,3)       ! Flux of simulated direct cerenkov photons
      real*8 fdtubeprofile_ncvmie(FDTUBEPROF_MAXTUBE,3)       ! Flux of simulated Mie scattered cerenkov photons
      real*8 fdtubeprofile_ncvray(FDTUBEPROF_MAXTUBE,3)       ! Flux of simulated Rayleigh scattered cerenkov photons
      real*8 fdtubeprofile_simflux(FDTUBEPROF_MAXTUBE,3)      ! Total flux of simluated photons

      real*8 fdtubeprofile_ne(FDTUBEPROF_MAXTUBE,3)           ! Number of charged particles
      real*8 fdtubeprofile_ene(FDTUBEPROF_MAXTUBE,3)          ! uncertainty on ne

      real*8 fdtubeprofile_tres(FDTUBEPROF_MAXTUBE,3)         ! Time-slice fit residual
      real*8 fdtubeprofile_tchi2(FDTUBEPROF_MAXTUBE,3)        ! Time-slice fit chi2 contribution

      integer*4 fdtubeprofile_camera(FDTUBEPROF_MAXTUBE)      ! Camera number for this tube
      integer*4 fdtubeprofile_tube(FDTUBEPROF_MAXTUBE)        ! Tube ID 
      integer*4 fdtubeprofile_tube_qual(FDTUBEPROF_MAXTUBE,3) ! tube quality (good = 1, bad = 0)
      integer*4 fdtubeprofile_status(3)                       ! status[0] is for fdplane_.psi
                                                              ! status[1] is for fdplane_.psi - fdplane_.epsi
                                                              ! status[2] is for fdplane_.psi + fdplane_.epsi
                                                              ! (-2 if bank is not filled
                                                              !  -1 if bad geometry fit
                                                              !   0 if bad profile fit   
                                                              !   1 if good profile fit)

      integer*4 fdtubeprofile_siteid                          ! site ID (BR = 0, LR = 1)
      integer*4 fdtubeprofile_mc                              ! [0 = don't use trumpmc bank info, 1 = use trumpmc bank]
      real*8 fdtubeprofile_simtime(FDTUBEPROF_MAXTUBE,3)
      real*8 fdtubeprofile_simtrms(FDTUBEPROF_MAXTUBE,3)
      real*8 fdtubeprofile_simtres(FDTUBEPROF_MAXTUBE,3)
      real*8 fdtubeprofile_timechi2(FDTUBEPROF_MAXTUBE,3)
