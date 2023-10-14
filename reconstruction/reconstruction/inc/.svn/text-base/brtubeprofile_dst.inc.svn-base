c Created 2010/01 LMS
C     added X0 and Lambda fits (SS 2011/09)

      integer BRTUBEPROFILE_BANKID
      parameter (BRTUBEPROFILE_BANKID=12106)
      integer BRTUBEPROFILE_BANKVERSION
      parameter (BRTUBEPROFILE_BANKVERSION=3)

      integer*4 brtubeprofile_common_to_bank
      external  brtubeprofile_common_to_bank
      integer*4 brtubeprofile_bank_to_dst
      external  brtubeprofile_bank_to_dst
      integer*4 brtubeprofile_common_to_dst
      external  brtubeprofile_common_to_dst
      integer*4 brtubeprofile_bank_to_common
      external  brtubeprofile_bank_to_common
      integer*4 brtubeprofile_common_to_dump
      external  brtubeprofile_common_to_dump
      integer*4 brtubeprofile_common_to_dumpf
      external  brtubeprofile_common_to_dumpf

      common /brtubeprofile/
     &     brtubeprofile_ntube,brtubeprofile_ngtube,
     &     brtubeprofile_rp,brtubeprofile_psi,brtubeprofile_t0,
     &     brtubeprofile_Xmax,brtubeprofile_eXmax,
     &     brtubeprofile_Nmax,brtubeprofile_eNmax,
     &     brtubeprofile_Energy,brtubeprofile_eEnergy,
     &     brtubeprofile_chi2,
     &     brtubeprofile_X0, brtubeprofile_eX0, 
     &     brtubeprofile_Lambda, brtubeprofile_eLambda, 
     &     brtubeprofile_x, brtubeprofile_npe,
     &     brtubeprofile_enpe,brtubeprofile_eacptfrac,
     &     brtubeprofile_acpt,brtubeprofile_eacpt,
     &     brtubeprofile_flux,brtubeprofile_eflux,
     &     brtubeprofile_simnpe,brtubeprofile_nfl,
     &     brtubeprofile_ncvdir,brtubeprofile_ncvmie,
     &     brtubeprofile_ncvray,brtubeprofile_simflux,
     &     brtubeprofile_ne,brtubeprofile_ene,
     &     brtubeprofile_tres,brtubeprofile_tchi2,
     &     brtubeprofile_camera,brtubeprofile_tube,
     &     brtubeprofile_tube_qual,brtubeprofile_status,
     &     brtubeprofile_siteid,brtubeprofile_mc
     &     brtubeprofile_simtime,brtubeprofile_simtrms,
     &     brtubeprofile_simtres, brtubeprofile_timechi2

      integer*4 brtubeprofile_ntube                           ! total number of tubes
      integer*4 brtubeprofile_ngtube(3)                       ! number of good tubes          

      real*8 brtubeprofile_rp(3)                              ! Impact parameter (meters)
      real*8 brtubeprofile_psi(3)                             ! Shower-detector plane angle (radians)
      real*8 brtubeprofile_t0(3)                              ! Detection time at Rp, less Rp travel time (ns)

      real*8 brtubeprofile_Xmax(3)                            ! Shower maximum (g/cm2)
      real*8 brtubeprofile_eXmax(3)                           ! uncertainty on xmax
      real*8 brtubeprofile_Nmax(3)                            ! Number of charged particles at shower maximum
      real*8 brtubeprofile_eNmax(3)                           ! uncertainty on nmax
      real*8 brtubeprofile_Energy(3)                          ! Initial cosmic-ray energy
      real*8 brtubeprofile_eEnergy(3)                         ! uncertainty on energy
      real*8 brtubeprofile_X0(3)                              ! effective depth of first interaction
      real*8 brtubeprofile_eX0(3)                             ! uncertainty of X0
      real*8 brtubeprofile_Lambda(3)                          ! profile width parameter
      real*8 brtubeprofile_eLambda(3)                         ! uncertainty in Lambda
      real*8 brtubeprofile_chi2(3)                            ! Total chi2 of fit

      real*8 brtubeprofile_x(FDTUBEPROF_MAXTUBE,3)            ! slant depth at middle of time slice (g/cm2)

      real*8 brtubeprofile_npe(FDTUBEPROF_MAXTUBE,3)          ! number of photoelectrons in tube
      real*8 brtubeprofile_enpe(FDTUBEPROF_MAXTUBE,3)         ! uncertainty on NPE, including uncertainty from acceptance
      real*8 brtubeprofile_eacptfrac(FDTUBEPROF_MAXTUBE,3)    ! fraction of uncertainty due to acceptance.  

      real*8 brtubeprofile_acpt(FDTUBEPROF_MAXTUBE,3)         ! PMT acceptance
      real*8 brtubeprofile_eacpt(FDTUBEPROF_MAXTUBE,3)        ! binomial uncertainty on acceptance 

      real*8 brtubeprofile_flux(FDTUBEPROF_MAXTUBE,3)         ! flux at the mirror [detectable npe / (m2 * radian)]
      real*8 brtubeprofile_eflux(FDTUBEPROF_MAXTUBE,3)        ! uncertainty on flux

      real*8 brtubeprofile_simnpe(FDTUBEPROF_MAXTUBE,3)       ! simulated photo-electrons in tube

      real*8 brtubeprofile_nfl(FDTUBEPROF_MAXTUBE,3)          ! Flux of simulated fluorescence photons
      real*8 brtubeprofile_ncvdir(FDTUBEPROF_MAXTUBE,3)       ! Flux of simulated direct cerenkov photons
      real*8 brtubeprofile_ncvmie(FDTUBEPROF_MAXTUBE,3)       ! Flux of simulated Mie scattered cerenkov photons
      real*8 brtubeprofile_ncvray(FDTUBEPROF_MAXTUBE,3)       ! Flux of simulated Rayleigh scattered cerenkov photons
      real*8 brtubeprofile_simflux(FDTUBEPROF_MAXTUBE,3)      ! Total flux of simluated photons

      real*8 brtubeprofile_ne(FDTUBEPROF_MAXTUBE,3)           ! Number of charged particles
      real*8 brtubeprofile_ene(FDTUBEPROF_MAXTUBE,3)          ! uncertainty on ne

      real*8 brtubeprofile_tres(FDTUBEPROF_MAXTUBE,3)         ! Time-slice fit residual
      real*8 brtubeprofile_tchi2(FDTUBEPROF_MAXTUBE,3)        ! Time-slice fit chi2 contribution

      integer*4 brtubeprofile_camera(FDTUBEPROF_MAXTUBE)      ! Camera number for this tube
      integer*4 brtubeprofile_tube(FDTUBEPROF_MAXTUBE)        ! Tube ID 
      integer*4 brtubeprofile_tube_qual(FDTUBEPROF_MAXTUBE,3) ! tube quality (good = 1, bad = 0)
      integer*4 brtubeprofile_status(3)                       ! status[0] is for fdplane_.psi
                                                              ! status[1] is for fdplane_.psi - fdplane_.epsi
                                                              ! status[2] is for fdplane_.psi + fdplane_.epsi
                                                              ! (-2 if bank is not filled
                                                              !  -1 if bad geometry fit
                                                              !   0 if bad profile fit   
                                                              !   1 if good profile fit)

      integer*4 brtubeprofile_siteid                          ! site ID (BR = 0, LR = 1)
      integer*4 brtubeprofile_mc                              ! [0 = don't use trumpmc bank info, 1 = use trumpmc bank]
      real*8 brtubeprofile_simtime(FDTUBEPROF_MAXTUBE,3)
      real*8 brtubeprofile_simtrms(FDTUBEPROF_MAXTUBE,3)
      real*8 brtubeprofile_simtres(FDTUBEPROF_MAXTUBE,3)
      real*8 brtubeprofile_timechi2(FDTUBEPROF_MAXTUBE,3)