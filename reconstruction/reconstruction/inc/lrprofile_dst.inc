c Created 2008/12 LMS

      integer LRPROFILE_BANKID
      parameter (LRPROFILE_BANKID=12204)
      integer LRPROFILE_BANKVERSION
      parameter (LRPROFILE_BANKVERSION=000)

      integer*4 lrprofile_common_to_bank
      external  lrprofile_common_to_bank
      integer*4 lrprofile_bank_to_dst
      external  lrprofile_bank_to_dst
      integer*4 lrprofile_common_to_dst
      external  lrprofile_common_to_dst
      integer*4 lrprofile_bank_to_common
      external  lrprofile_bank_to_common
      integer*4 lrprofile_common_to_dump
      external  lrprofile_common_to_dump
      integer*4 lrprofile_common_to_dumpf
      external  lrprofile_common_to_dumpf

      common /lrprofile/
     &     lrprofile_siteid,lrprofile_ntslice,
     &     lrprofile_ngtslice,lrprofile_status,
     &     lrprofile_timebin,
     &     lrprofile_rp,lrprofile_psi,lrprofile_t0,
     &     lrprofile_Xmax,lrprofile_eXmax,
     &     lrprofile_Nmax,lrprofile_eNmax,
     &     lrprofile_Energy,lrprofile_eEnergy,lrprofile_chi2,
     &     lrprofile_npe,lrprofile_enpe,
     &     lrprofile_x,
     &     lrprofile_dtheta,lrprofile_darea,
     &     lrprofile_acpt,lrprofile_eacpt,
     &     lrprofile_flux,lrprofile_eflux,
     &     lrprofile_nfl,lrprofile_ncvdir,lrprofile_ncvmie,
     &     lrprofile_ncvray,lrprofile_simflux,
     &     lrprofile_tres,lrprofile_tchi2,
     &     lrprofile_ne,lrprofile_ene,
     &     lrprofile_mc

      integer*4 lrprofile_siteid                              ! site ID (BR = 0, LR = 1)
      integer*4 lrprofile_ntslice                             ! number of time slices (FADC bins)
      integer*4 lrprofile_ngtslice(3)                         ! number of good time slices (acceptance)
      integer*4 lrprofile_status(3)                           ! status[0] is for fdplane_.psi
                                                              ! status[1] is for fdplane_.psi - fdplane_.epsi
                                                              ! status[2] is for fdplane_.psi + fdplane_.epsi
                                                              ! (-1 if bank is not filled,
                                                              !   0 if bad profile fit
                                                              !   1 if good profile fit)

      integer*4 lrprofile_timebin(FDPROF_MAXTSLICE)           ! FADC bin time slice

      real*8 lrprofile_rp(3)                                  ! Impact parameter (meters)
      real*8 lrprofile_psi(3)                                 ! Shower-detector plane angle (radians)
      real*8 lrprofile_t0(3)                                  ! Detection time at Rp, less Rp travel time (ns)

      real*8 lrprofile_Xmax(3)                                ! Shower maximum (g/cm2)
      real*8 lrprofile_eXmax(3)                               ! uncertainty on xmax
      real*8 lrprofile_Nmax(3)                                ! Number of charged particles at shower maximum
      real*8 lrprofile_eNmax(3)                               ! uncertainty on nmax
      real*8 lrprofile_Energy(3)                              ! Initial cosmic-ray energy
      real*8 lrprofile_eEnergy(3)                             ! uncertainty on energy
      real*8 lrprofile_chi2(3)                                ! Total chi2 of fit

      real*8 lrprofile_npe(FDPROF_MAXTSLICE)                  ! number of photoelectrons by time slice
      real*8 lrprofile_enpe(FDPROF_MAXTSLICE)                 ! uncertainty on npe

      real*8 lrprofile_x(FDPROF_MAXTSLICE,3)                  ! slant depth at middle of time slice (g/cm2)

      real*8 lrprofile_dtheta(FDPROF_MAXTSLICE,3)             ! angular size of bin (radians)
      real*8 lrprofile_darea(FDPROF_MAXTSLICE,3)              ! cosine-corrected active area of mirror (sq. meter)

      real*8 lrprofile_acpt(FDPROF_MAXTSLICE,3)               ! PMT acceptance by time slice
      real*8 lrprofile_eacpt(FDPROF_MAXTSLICE,3)              ! binomial uncertainty on acceptance

      real*8 lrprofile_flux(FDPROF_MAXTSLICE,3)               ! flux at the mirror [photons / (m2 * radian)]
      real*8 lrprofile_eflux(FDPROF_MAXTSLICE,3)              ! uncertainty on flux

      real*8 lrprofile_nfl(FDPROF_MAXTSLICE,3)                ! Flux of simulated fluorescence photons
      real*8 lrprofile_ncvdir(FDPROF_MAXTSLICE,3)             ! Flux of simulated direct cerenkov photons
      real*8 lrprofile_ncvmie(FDPROF_MAXTSLICE,3)             ! Flux of simulated Mie scattered cerenkov photons
      real*8 lrprofile_ncvray(FDPROF_MAXTSLICE,3)             ! Flux of simulated Rayleigh scattered cerenkov photons
      real*8 lrprofile_simflux(FDPROF_MAXTSLICE,3)            ! Total flux of simluated photons

      real*8 lrprofile_tres(FDPROF_MAXTSLICE,3)               ! Time-slice fit residual
      real*8 lrprofile_tchi2(FDPROF_MAXTSLICE,3)              ! Time-slice fit chi2 contribution

      real*8 lrprofile_ne(FDPROF_MAXTSLICE,3)                 ! Number of charged particles
      real*8 lrprofile_ene(FDPROF_MAXTSLICE,3)                ! uncertainty on ne
      integer*4 lrprofile_mc
