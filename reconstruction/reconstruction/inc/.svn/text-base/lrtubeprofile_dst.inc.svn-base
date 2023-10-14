c Created 2010/01 LMS
C     added X0 and Lambda fits (SS 2011/09)

      integer LRTUBEPROFILE_BANKID
      parameter (LRTUBEPROFILE_BANKID=12206)
      integer LRTUBEPROFILE_BANKVERSION
      parameter (LRTUBEPROFILE_BANKVERSION=3)

      integer*4 lrtubeprofile_common_to_bank
      external  lrtubeprofile_common_to_bank
      integer*4 lrtubeprofile_bank_to_dst
      external  lrtubeprofile_bank_to_dst
      integer*4 lrtubeprofile_common_to_dst
      external  lrtubeprofile_common_to_dst
      integer*4 lrtubeprofile_bank_to_common
      external  lrtubeprofile_bank_to_common
      integer*4 lrtubeprofile_common_to_dump
      external  lrtubeprofile_common_to_dump
      integer*4 lrtubeprofile_common_to_dumpf
      external  lrtubeprofile_common_to_dumpf

      common /lrtubeprofile/
     &     lrtubeprofile_ntube,lrtubeprofile_ngtube,
     &     lrtubeprofile_rp,lrtubeprofile_psi,lrtubeprofile_t0,
     &     lrtubeprofile_Xmax,lrtubeprofile_eXmax,
     &     lrtubeprofile_Nmax,lrtubeprofile_eNmax,
     &     lrtubeprofile_Energy,lrtubeprofile_eEnergy,
     &     lrtubeprofile_chi2,
     &     lrtubeprofile_X0, lrtubeprofile_eX0, 
     &     lrtubeprofile_Lambda, lrtubeprofile_eLambda, 
     &     lrtubeprofile_x, lrtubeprofile_npe,
     &     lrtubeprofile_enpe,lrtubeprofile_eacptfrac,
     &     lrtubeprofile_acpt,lrtubeprofile_eacpt,
     &     lrtubeprofile_flux,lrtubeprofile_eflux,
     &     lrtubeprofile_simnpe,lrtubeprofile_nfl,
     &     lrtubeprofile_ncvdir,lrtubeprofile_ncvmie,
     &     lrtubeprofile_ncvray,lrtubeprofile_simflux,
     &     lrtubeprofile_ne,lrtubeprofile_ene,
     &     lrtubeprofile_tres,lrtubeprofile_tchi2,
     &     lrtubeprofile_camera,lrtubeprofile_tube,
     &     lrtubeprofile_tube_qual,lrtubeprofile_status,
     &     lrtubeprofile_siteid,lrtubeprofile_mc
     &     lrtubeprofile_simtime,lrtubeprofile_simtrms,
     &     lrtubeprofile_simtres, lrtubeprofile_timechi2

      integer*4 lrtubeprofile_ntube                           ! total number of tubes
      integer*4 lrtubeprofile_ngtube(3)                       ! number of good tubes          

      real*8 lrtubeprofile_rp(3)                              ! Impact parameter (meters)
      real*8 lrtubeprofile_psi(3)                             ! Shower-detector plane angle (radians)
      real*8 lrtubeprofile_t0(3)                              ! Detection time at Rp, less Rp travel time (ns)

      real*8 lrtubeprofile_Xmax(3)                            ! Shower maximum (g/cm2)
      real*8 lrtubeprofile_eXmax(3)                           ! uncertainty on xmax
      real*8 lrtubeprofile_Nmax(3)                            ! Number of charged particles at shower maximum
      real*8 lrtubeprofile_eNmax(3)                           ! uncertainty on nmax
      real*8 lrtubeprofile_Energy(3)                          ! Initial cosmic-ray energy
      real*8 lrtubeprofile_eEnergy(3)                         ! uncertainty on energy
      real*8 lrtubeprofile_X0(3)                              ! effective depth of first interaction
      real*8 lrtubeprofile_eX0(3)                             ! uncertainty of X0
      real*8 lrtubeprofile_Lambda(3)                          ! profile width parameter
      real*8 lrtubeprofile_eLambda(3)                         ! uncertainty in Lambda
      real*8 lrtubeprofile_chi2(3)                            ! Total chi2 of fit

      real*8 lrtubeprofile_x(FDTUBEPROF_MAXTUBE,3)            ! slant depth at middle of time slice (g/cm2)

      real*8 lrtubeprofile_npe(FDTUBEPROF_MAXTUBE,3)          ! number of photoelectrons in tube
      real*8 lrtubeprofile_enpe(FDTUBEPROF_MAXTUBE,3)         ! uncertainty on NPE, including uncertainty from acceptance
      real*8 lrtubeprofile_eacptfrac(FDTUBEPROF_MAXTUBE,3)    ! fraction of uncertainty due to acceptance.  

      real*8 lrtubeprofile_acpt(FDTUBEPROF_MAXTUBE,3)         ! PMT acceptance
      real*8 lrtubeprofile_eacpt(FDTUBEPROF_MAXTUBE,3)        ! binomial uncertainty on acceptance 

      real*8 lrtubeprofile_flux(FDTUBEPROF_MAXTUBE,3)         ! flux at the mirror [detectable npe / (m2 * radian)]
      real*8 lrtubeprofile_eflux(FDTUBEPROF_MAXTUBE,3)        ! uncertainty on flux

      real*8 lrtubeprofile_simnpe(FDTUBEPROF_MAXTUBE,3)       ! simulated photo-electrons in tube

      real*8 lrtubeprofile_nfl(FDTUBEPROF_MAXTUBE,3)          ! Flux of simulated fluorescence photons
      real*8 lrtubeprofile_ncvdir(FDTUBEPROF_MAXTUBE,3)       ! Flux of simulated direct cerenkov photons
      real*8 lrtubeprofile_ncvmie(FDTUBEPROF_MAXTUBE,3)       ! Flux of simulated Mie scattered cerenkov photons
      real*8 lrtubeprofile_ncvray(FDTUBEPROF_MAXTUBE,3)       ! Flux of simulated Rayleigh scattered cerenkov photons
      real*8 lrtubeprofile_simflux(FDTUBEPROF_MAXTUBE,3)      ! Total flux of simluated photons

      real*8 lrtubeprofile_ne(FDTUBEPROF_MAXTUBE,3)           ! Number of charged particles
      real*8 lrtubeprofile_ene(FDTUBEPROF_MAXTUBE,3)          ! uncertainty on ne

      real*8 lrtubeprofile_tres(FDTUBEPROF_MAXTUBE,3)         ! Time-slice fit residual
      real*8 lrtubeprofile_tchi2(FDTUBEPROF_MAXTUBE,3)        ! Time-slice fit chi2 contribution

      integer*4 lrtubeprofile_camera(FDTUBEPROF_MAXTUBE)      ! Camera number for this tube
      integer*4 lrtubeprofile_tube(FDTUBEPROF_MAXTUBE)        ! Tube ID 
      integer*4 lrtubeprofile_tube_qual(FDTUBEPROF_MAXTUBE,3) ! tube quality (good = 1, bad = 0)
      integer*4 lrtubeprofile_status(3)                       ! status[0] is for fdplane_.psi
                                                              ! status[1] is for fdplane_.psi - fdplane_.epsi
                                                              ! status[2] is for fdplane_.psi + fdplane_.epsi
                                                              ! (-2 if bank is not filled
                                                              !  -1 if bad geometry fit
                                                              !   0 if bad profile fit   
                                                              !   1 if good profile fit)

      integer*4 lrtubeprofile_siteid                          ! site ID (BR = 0, LR = 1)
      integer*4 lrtubeprofile_mc                              ! [0 = don't use trumpmc bank info, 1 = use trumpmc bank]
      real*8 lrtubeprofile_simtime(FDTUBEPROF_MAXTUBE,3)
      real*8 lrtubeprofile_simtrms(FDTUBEPROF_MAXTUBE,3)
      real*8 lrtubeprofile_simtres(FDTUBEPROF_MAXTUBE,3)
      real*8 lrtubeprofile_timechi2(FDTUBEPROF_MAXTUBE,3)
      