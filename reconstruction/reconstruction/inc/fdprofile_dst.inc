c Created 2008/12 LMS

      integer FDPROFILE_BANKID
      parameter (FDPROFILE_BANKID=12094)
      integer FDPROFILE_BANKVERSION
      parameter (FDPROFILE_BANKVERSION=000)

      integer*4 fdprofile_common_to_bank
      external  fdprofile_common_to_bank
      integer*4 fdprofile_bank_to_dst
      external  fdprofile_bank_to_dst
      integer*4 fdprofile_common_to_dst
      external  fdprofile_common_to_dst
      integer*4 fdprofile_bank_to_common
      external  fdprofile_bank_to_common
      integer*4 fdprofile_common_to_dump
      external  fdprofile_common_to_dump
      integer*4 fdprofile_common_to_dumpf
      external  fdprofile_common_to_dumpf

      common /fdprofile/
     &     fdprofile_siteid,fdprofile_ntslice,
     &     fdprofile_ngtslice,fdprofile_status,
     &     fdprofile_timebin,
     &     fdprofile_rp,fdprofile_psi,fdprofile_t0,
     &     fdprofile_Xmax,fdprofile_eXmax,
     &     fdprofile_Nmax,fdprofile_eNmax,
     &     fdprofile_Energy,fdprofile_eEnergy,fdprofile_chi2,
     &     fdprofile_npe,fdprofile_enpe,
     &     fdprofile_x,
     &     fdprofile_dtheta,fdprofile_darea,
     &     fdprofile_acpt,fdprofile_eacpt,
     &     fdprofile_flux,fdprofile_eflux,
     &     fdprofile_nfl,fdprofile_ncvdir,fdprofile_ncvmie,
     &     fdprofile_ncvray,fdprofile_simflux,
     &     fdprofile_tres,fdprofile_tchi2,
     &     fdprofile_ne,fdprofile_ene,
     &     fdprofile_mc

      integer*4 fdprofile_siteid                              ! site ID (BR = 0, LR = 1)
      integer*4 fdprofile_ntslice                             ! number of time slices (FADC bins)
      integer*4 fdprofile_ngtslice(3)                         ! number of good time slices (acceptance)
      integer*4 fdprofile_status(3)                           ! status(1) is for fdplane_psi
                                                              ! status(2) is for fdplane_psi - fdplane_epsi
                                                              ! status(3) is for fdplane_psi + fdplane_epsi
                                                              ! (-1 if bank is not filled,
                                                              !   0 if bad profile fit
                                                              !   1 if good profile fit)

      integer*4 fdprofile_timebin(FDPROF_MAXTSLICE)           ! FADC bin time slice

      real*8 fdprofile_rp(3)                                  ! Impact parameter (meters)
      real*8 fdprofile_psi(3)                                 ! Shower-detector plane angle (radians)
      real*8 fdprofile_t0(3)                                  ! Detection time at Rp, less Rp travel time (ns)

      real*8 fdprofile_Xmax(3)                                ! Shower maximum (g/cm2)
      real*8 fdprofile_eXmax(3)                               ! uncertainty on xmax
      real*8 fdprofile_Nmax(3)                                ! Number of charged particles at shower maximum
      real*8 fdprofile_eNmax(3)                               ! uncertainty on nmax
      real*8 fdprofile_Energy(3)                              ! Initial cosmic-ray energy
      real*8 fdprofile_eEnergy(3)                             ! uncertainty on energy
      real*8 fdprofile_chi2(3)                                ! Total chi2 of fit

      real*8 fdprofile_npe(FDPROF_MAXTSLICE)                  ! number of photoelectrons by time slice
      real*8 fdprofile_enpe(FDPROF_MAXTSLICE)                 ! uncertainty on npe

      real*8 fdprofile_x(FDPROF_MAXTSLICE,3)                  ! slant depth at middle of time slice (g/cm2)

      real*8 fdprofile_dtheta(FDPROF_MAXTSLICE,3)             ! angular size of bin (radians)
      real*8 fdprofile_darea(FDPROF_MAXTSLICE,3)              ! cosine-corrected active area of mirror (sq. meter)

      real*8 fdprofile_acpt(FDPROF_MAXTSLICE,3)               ! PMT acceptance by time slice
      real*8 fdprofile_eacpt(FDPROF_MAXTSLICE,3)              ! binomial uncertainty on acceptance

      real*8 fdprofile_flux(FDPROF_MAXTSLICE,3)               ! flux at the mirror [photons / (m2 * radian)]
      real*8 fdprofile_eflux(FDPROF_MAXTSLICE,3)              ! uncertainty on flux

      real*8 fdprofile_nfl(FDPROF_MAXTSLICE,3)                ! Flux of simulated fluorescence photons
      real*8 fdprofile_ncvdir(FDPROF_MAXTSLICE,3)             ! Flux of simulated direct cerenkov photons
      real*8 fdprofile_ncvmie(FDPROF_MAXTSLICE,3)             ! Flux of simulated Mie scattered cerenkov photons
      real*8 fdprofile_ncvray(FDPROF_MAXTSLICE,3)             ! Flux of simulated Rayleigh scattered cerenkov photons
      real*8 fdprofile_simflux(FDPROF_MAXTSLICE,3)            ! Total flux of simluated photons

      real*8 fdprofile_tres(FDPROF_MAXTSLICE,3)               ! Time-slice fit residual
      real*8 fdprofile_tchi2(FDPROF_MAXTSLICE,3)              ! Time-slice fit chi2 contribution

      real*8 fdprofile_ne(FDPROF_MAXTSLICE,3)                 ! Number of charged particles
      real*8 fdprofile_ene(FDPROF_MAXTSLICE,3)                ! uncertainty on ne
      integer*4 fdprofile_mc
