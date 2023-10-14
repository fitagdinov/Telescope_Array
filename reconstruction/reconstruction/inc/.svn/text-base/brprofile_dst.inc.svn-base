c Created 2008/12 LMS

      integer BRPROFILE_BANKID
      parameter (BRPROFILE_BANKID=12104)
      integer BRPROFILE_BANKVERSION
      parameter (BRPROFILE_BANKVERSION=000)

      integer*4 brprofile_common_to_bank
      external  brprofile_common_to_bank
      integer*4 brprofile_bank_to_dst
      external  brprofile_bank_to_dst
      integer*4 brprofile_common_to_dst
      external  brprofile_common_to_dst
      integer*4 brprofile_bank_to_common
      external  brprofile_bank_to_common
      integer*4 brprofile_common_to_dump
      external  brprofile_common_to_dump
      integer*4 brprofile_common_to_dumpf
      external  brprofile_common_to_dumpf

      common /brprofile/
     &     brprofile_siteid,brprofile_ntslice,
     &     brprofile_ngtslice,brprofile_status,
     &     brprofile_timebin,
     &     brprofile_rp,brprofile_psi,brprofile_t0,
     &     brprofile_Xmax,brprofile_eXmax,
     &     brprofile_Nmax,brprofile_eNmax,
     &     brprofile_Energy,brprofile_eEnergy,brprofile_chi2,
     &     brprofile_npe,brprofile_enpe,
     &     brprofile_x,
     &     brprofile_dtheta,brprofile_darea,
     &     brprofile_acpt,brprofile_eacpt,
     &     brprofile_flux,brprofile_eflux,
     &     brprofile_nfl,brprofile_ncvdir,brprofile_ncvmie,
     &     brprofile_ncvray,brprofile_simflux,
     &     brprofile_tres,brprofile_tchi2,
     &     brprofile_ne,brprofile_ene,
     &     brprofile_mc

      integer*4 brprofile_siteid                              ! site ID (BR = 0, LR = 1)
      integer*4 brprofile_ntslice                             ! number of time slices (FADC bins)
      integer*4 brprofile_ngtslice(3)                         ! number of good time slices (acceptance)
      integer*4 brprofile_status(3)                           ! status[0] is for fdplane_.psi
                                                              ! status[1] is for fdplane_.psi - fdplane_.epsi
                                                              ! status[2] is for fdplane_.psi + fdplane_.epsi
                                                              ! (-1 if bank is not filled,
                                                              !   0 if bad profile fit
                                                              !   1 if good profile fit)

      integer*4 brprofile_timebin(FDPROF_MAXTSLICE)           ! FADC bin time slice

      real*8 brprofile_rp(3)                                  ! Impact parameter (meters)
      real*8 brprofile_psi(3)                                 ! Shower-detector plane angle (radians)
      real*8 brprofile_t0(3)                                  ! Detection time at Rp, less Rp travel time (ns)

      real*8 brprofile_Xmax(3)                                ! Shower maximum (g/cm2)
      real*8 brprofile_eXmax(3)                               ! uncertainty on xmax
      real*8 brprofile_Nmax(3)                                ! Number of charged particles at shower maximum
      real*8 brprofile_eNmax(3)                               ! uncertainty on nmax
      real*8 brprofile_Energy(3)                              ! Initial cosmic-ray energy
      real*8 brprofile_eEnergy(3)                             ! uncertainty on energy
      real*8 brprofile_chi2(3)                                ! Total chi2 of fit

      real*8 brprofile_npe(FDPROF_MAXTSLICE)                  ! number of photoelectrons by time slice
      real*8 brprofile_enpe(FDPROF_MAXTSLICE)                 ! uncertainty on npe

      real*8 brprofile_x(FDPROF_MAXTSLICE,3)                  ! slant depth at middle of time slice (g/cm2)

      real*8 brprofile_dtheta(FDPROF_MAXTSLICE,3)             ! angular size of bin (radians)
      real*8 brprofile_darea(FDPROF_MAXTSLICE,3)              ! cosine-corrected active area of mirror (sq. meter)

      real*8 brprofile_acpt(FDPROF_MAXTSLICE,3)               ! PMT acceptance by time slice
      real*8 brprofile_eacpt(FDPROF_MAXTSLICE,3)              ! binomial uncertainty on acceptance

      real*8 brprofile_flux(FDPROF_MAXTSLICE,3)               ! flux at the mirror [photons / (m2 * radian)]
      real*8 brprofile_eflux(FDPROF_MAXTSLICE,3)              ! uncertainty on flux

      real*8 brprofile_nfl(FDPROF_MAXTSLICE,3)                ! Flux of simulated fluorescence photons
      real*8 brprofile_ncvdir(FDPROF_MAXTSLICE,3)             ! Flux of simulated direct cerenkov photons
      real*8 brprofile_ncvmie(FDPROF_MAXTSLICE,3)             ! Flux of simulated Mie scattered cerenkov photons
      real*8 brprofile_ncvray(FDPROF_MAXTSLICE,3)             ! Flux of simulated Rayleigh scattered cerenkov photons
      real*8 brprofile_simflux(FDPROF_MAXTSLICE,3)            ! Total flux of simluated photons

      real*8 brprofile_tres(FDPROF_MAXTSLICE,3)               ! Time-slice fit residual
      real*8 brprofile_tchi2(FDPROF_MAXTSLICE,3)              ! Time-slice fit chi2 contribution

      real*8 brprofile_ne(FDPROF_MAXTSLICE,3)                 ! Number of charged particles
      real*8 brprofile_ene(FDPROF_MAXTSLICE,3)                ! uncertainty on ne
      integer*4 brprofile_mc
