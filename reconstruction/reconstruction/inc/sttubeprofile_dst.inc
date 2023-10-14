c Created 2010/04 LMS

      integer STTUBEPROFILE_BANKID
      parameter (STTUBEPROFILE_BANKID=20002)
      integer STTUBEPROFILE_BANKVERSION
      parameter (STTUBEPROFILE_BANKVERSION=001)

      integer*4 sttubeprofile_common_to_bank
      external  sttubeprofile_common_to_bank
      integer*4 sttubeprofile_bank_to_dst
      external  sttubeprofile_bank_to_dst
      integer*4 sttubeprofile_common_to_dst
      external  sttubeprofile_common_to_dst
      integer*4 sttubeprofile_bank_to_common
      external  sttubeprofile_bank_to_common
      integer*4 sttubeprofile_common_to_dump
      external  sttubeprofile_common_to_dump
      integer*4 sttubeprofile_common_to_dumpf
      external  sttubeprofile_common_to_dumpf

      common /sttubeprofile/
     &     sttubeprofile_ntube,sttubeprofile_ngtube,
     &     sttubeprofile_status,
     &     sttubeprofile_rp,sttubeprofile_psi,sttubeprofile_t0,
     &     sttubeprofile_Xmax,sttubeprofile_eXmax,
     &     sttubeprofile_Nmax,sttubeprofile_eNmax,
     &     sttubeprofile_Energy,sttubeprofile_eEnergy,
     &     sttubeprofile_chi2,sttubeprofile_x,
     &     sttubeprofile_npe,sttubeprofile_enpe,sttubeprofile_eacptfrac,
     &     sttubeprofile_acpt,sttubeprofile_eacpt,
     &     sttubeprofile_flux,sttubeprofile_eflux,
     &     sttubeprofile_simnpe,sttubeprofile_nfl,
     &     sttubeprofile_ncvdir,sttubeprofile_ncvmie,
     &     sttubeprofile_ncvray,sttubeprofile_simflux,
     &     sttubeprofile_ne,sttubeprofile_ene,
     &     sttubeprofile_tres,sttubeprofile_tchi2,
     &     sttubeprofile_camera,sttubeprofile_tube,
     &     sttubeprofile_tube_qual,sttubeprofile_mc,
     &     sttubeprofile_X0, sttubeprofile_eX0,
     &     sttubeprofile_Lambda, sttubeprofile_eLambda,
     &     sttubeprofile_siteid

      integer*4 sttubeprofile_ntube(2)                        ! total number of tubes
      integer*4 sttubeprofile_ngtube                          ! number of good tubes          
      integer*4 sttubeprofile_status                          !	(-2 if bank is not filled
       	       	       	       	       	       	       	      !	 -1 if bad geometry fit
       	       	       	       	       	       	       	      !	  0 if bad profile fit
                                                              !   1 if good profile fit)

      real*8 sttubeprofile_rp(2)                              ! Impact parameter (meters)
      real*8 sttubeprofile_psi(2)                             ! Shower-detector plane angle (radians)
      real*8 sttubeprofile_t0(2)                              ! Detection time at Rp, less Rp travel time (ns)

      real*8 sttubeprofile_Xmax                               ! Shower maximum (g/cm2)
      real*8 sttubeprofile_eXmax                              ! uncertainty on xmax
      real*8 sttubeprofile_Nmax                               ! Number of charged particles at shower maximum
      real*8 sttubeprofile_eNmax                              ! uncertainty on nmax
      real*8 sttubeprofile_Energy                             ! Initial cosmic-ray energy
      real*8 sttubeprofile_eEnergy                            ! uncertainty on energy
      real*8 sttubeprofile_chi2                               ! Total chi2 of fit

      real*8 sttubeprofile_x(STTUBEPROF_MAXTUBE,2)            ! slant depth at middle of time slice (g/cm2)

      real*8 sttubeprofile_npe(STTUBEPROF_MAXTUBE,2)          ! number of photoelectrons in tube
      real*8 sttubeprofile_enpe(STTUBEPROF_MAXTUBE,2)         ! uncertainty on NPE, including uncertainty from acceptance
      real*8 sttubeprofile_eacptfrac(STTUBEPROF_MAXTUBE,2)    ! fraction of uncertainty due to acceptance.  

      real*8 sttubeprofile_acpt(STTUBEPROF_MAXTUBE,2)         ! PMT acceptance
      real*8 sttubeprofile_eacpt(STTUBEPROF_MAXTUBE,2)        ! binomial uncertainty on acceptance 

      real*8 sttubeprofile_flux(STTUBEPROF_MAXTUBE,2)         ! flux at the mirror [detectable npe / (m2 * radian)]
      real*8 sttubeprofile_eflux(STTUBEPROF_MAXTUBE,2)        ! uncertainty on flux

      real*8 sttubeprofile_simnpe(STTUBEPROF_MAXTUBE,2)       ! simulated photo-electrons in tube

      real*8 sttubeprofile_nfl(STTUBEPROF_MAXTUBE,2)          ! Flux of simulated fluorescence photons
      real*8 sttubeprofile_ncvdir(STTUBEPROF_MAXTUBE,2)       ! Flux of simulated direct cerenkov photons
      real*8 sttubeprofile_ncvmie(STTUBEPROF_MAXTUBE,2)       ! Flux of simulated Mie scattered cerenkov photons
      real*8 sttubeprofile_ncvray(STTUBEPROF_MAXTUBE,2)       ! Flux of simulated Rayleigh scattered cerenkov photons
      real*8 sttubeprofile_simflux(STTUBEPROF_MAXTUBE,2)      ! Total flux of simluated photons

      real*8 sttubeprofile_ne(STTUBEPROF_MAXTUBE,2)           ! Number of charged particles
      real*8 sttubeprofile_ene(STTUBEPROF_MAXTUBE,2)          ! uncertainty on ne

      real*8 sttubeprofile_tres(STTUBEPROF_MAXTUBE,2)         ! Time-slice fit residual
      real*8 sttubeprofile_tchi2(STTUBEPROF_MAXTUBE,2)        ! Time-slice fit chi2 contribution

      integer*4 sttubeprofile_camera(STTUBEPROF_MAXTUBE,2)    ! Camera number for this tube
      integer*4 sttubeprofile_tube(STTUBEPROF_MAXTUBE,2)      ! Tube ID 
      integer*4 sttubeprofile_tube_qual(STTUBEPROF_MAXTUBE,2) ! tube quality (good = 1, bad = 0)

      integer*4 sttubeprofile_mc                              ! [0 = don't use trumpmc bank info, 1 = use trumpmc bank]
      
      real*8 sttubeprofile_X0
      real*8 sttubeprofile_eX0
      real*8 sttubeprofile_Lambda
      real*8 sttubeprofile_eLambda
      
      integer*4 sttubeprofile_siteid(2)
