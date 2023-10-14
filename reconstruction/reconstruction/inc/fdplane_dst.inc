c Created 2008/09/24 LMS DRB

      integer FDPLANE_BANKID
      parameter (FDPLANE_BANKID=12093)
      integer FDPLANE_BANKVERSION
      parameter (FDPLANE_BANKVERSION=002)

      integer*4 fdplane_common_to_bank
      external  fdplane_common_to_bank
      integer*4 fdplane_bank_to_dst
      external  fdplane_bank_to_dst
      integer*4 fdplane_common_to_dst
      external  fdplane_common_to_dst
      integer*4 fdplane_bank_to_common
      external  fdplane_bank_to_common
      integer*4 fdplane_common_to_dump
      external  fdplane_common_to_dump
      integer*4 fdplane_common_to_dumpf
      external  fdplane_common_to_dumpf

      common /fdplane/
     &     fdplane_part,fdplane_event_num,fdplane_julian,
     &     fdplane_jsecond,fdplane_jsecfrac,fdplane_second,
     &     fdplane_secfrac,fdplane_ntube,
     &     fdplane_uniqID,fdplane_fmode,
     &     fdplane_npe,fdplane_adc,fdplane_ped,
     &     fdplane_time,fdplane_time_rms,fdplane_sigma,
     &     fdplane_sdp_n,fdplane_sdp_en,fdplane_sdp_n_cov,
     &     fdplane_sdp_the,fdplane_sdp_phi,fdplane_sdp_chi2,
     &     fdplane_alt,fdplane_azm,fdplane_plane_alt,fdplane_plane_azm,
     &     fdplane_linefit_slope,fdplane_linefit_eslope,
     &     fdplane_linefit_int,fdplane_linefit_eint,
     &     fdplane_linefit_chi2,fdplane_linefit_cov,
     &     fdplane_linefit_res,fdplane_linefit_tchi2,
     &     fdplane_ptanfit_rp,fdplane_ptanfit_erp,
     &     fdplane_ptanfit_t0,fdplane_ptanfit_et0,
     &     fdplane_ptanfit_chi2,fdplane_ptanfit_cov,
     &     fdplane_ptanfit_res,fdplane_ptanfit_tchi2,
     &     fdplane_rp,fdplane_erp,fdplane_psi,fdplane_epsi,
     &     fdplane_t0,fdplane_et0,fdplane_tanfit_chi2,
     &     fdplane_tanfit_cov,fdplane_tanfit_res,fdplane_tanfit_tchi2,
     &     fdplane_azm_extent,fdplane_time_extent,
     &     fdplane_shower_zen,fdplane_shower_azm,fdplane_shower_axis,
     &     fdplane_rpuv,fdplane_core,
     &     fdplane_camera,fdplane_tube,fdplane_it0,fdplane_it1,
     &     fdplane_knex_qual,fdplane_tube_qual,
     &     fdplane_ngtube,fdplane_seed,fdplane_type,
     &     fdplane_status,fdplane_siteid

      integer*4 fdplane_part                       ! part number
      integer*4 fdplane_event_num                  ! event number
      integer*4 fdplane_julian                     ! run start day
      integer*4 fdplane_jsecond                    ! run start second (from start of julian)
      integer*4 fdplane_jsecfrac                   ! run start nanosecond (from start of jsecond)
      integer*4 fdplane_second                     ! event start second (from run start)
      integer*4 fdplane_secfrac                    ! event start nanosecond (from start of second
      integer*4 fdplane_ntube                      ! number of tubes in event
      integer*4 fdplane_uniqID
      integer*4 fdplane_fmode
      
      real*8 fdplane_npe(GEOFD_MAXTUBE)          ! integrated pulse above pedestal in NPE
      real*8 fdplane_adc(GEOFD_MAXTUBE);         ! integrated pulse above pedestal in FADC counts
      real*8 fdplane_ped(GEOFD_MAXTUBE);         ! pedestal value under the pulse in FADC counts

      real*8 fdplane_time(GEOFD_MAXTUBE)         ! weighted average pulse time
      real*8 fdplane_time_rms(GEOFD_MAXTUBE)     ! weighted average pulse time rms
      real*8 fdplane_sigma(GEOFD_MAXTUBE)     ! tube significance

      real*8 fdplane_sdp_n(3)                       ! shower-detector plane normal (SDPN)
      real*8 fdplane_sdp_en(3)                      ! uncertainty on SDPN fit
      real*8 fdplane_sdp_n_cov(3,3)                 ! covariance matrix of SDPN fit
      real*8 fdplane_sdp_the                        ! shower-detector plane theta angle
      real*8 fdplane_sdp_phi                        ! shower-detector plane phi angle
      real*8 fdplane_sdp_chi2                       ! SDPN fit chi2

      real*8 fdplane_alt(GEOFD_MAXTUBE)           ! altitude of tube
      real*8 fdplane_azm(GEOFD_MAXTUBE)           ! azimuth of tube
      real*8 fdplane_plane_alt(GEOFD_MAXTUBE)     ! altitude of tube rotated into SDP coordinate system
      real*8 fdplane_plane_azm(GEOFD_MAXTUBE)     ! azimuth of tube rotated into SDP coordinate system

      real*8 fdplane_linefit_slope                  ! linear fit to time vs. angle slope (ns / degree)
      real*8 fdplane_linefit_eslope                 ! linear fit to time vs. angle slope uncertainty (ns / degree)
      real*8 fdplane_linefit_int                    ! linear fit to time vs. angle intercept (ns)
      real*8 fdplane_linefit_eint                   ! linear fit to time vs. angle intercept uncertainty (ns)
      real*8 fdplane_linefit_chi2                   ! linear fit chi2
      real*8 fdplane_linefit_cov(2,2)               ! linear fit covariance
      real*8 fdplane_linefit_res(GEOFD_MAXTUBE)   ! linear fit tube residual (ns)
      real*8 fdplane_linefit_tchi2(GEOFD_MAXTUBE) ! linear fit tube chi2 contribution

      real*8 fdplane_ptanfit_rp                     ! pseudo-tangent fit rp (meters)
      real*8 fdplane_ptanfit_erp                    ! pseudo-tangent fit rp uncertainty (meters)
      real*8 fdplane_ptanfit_t0                     ! pseudo-tangent fit t0 (ns)
      real*8 fdplane_ptanfit_et0                    ! pseudo-tangent fit t0 uncertainty (ns)
      real*8 fdplane_ptanfit_chi2                   ! pseudo-tangent fit chi2
      real*8 fdplane_ptanfit_cov(2,2)               ! pseudo-tangent fit covariance
      real*8 fdplane_ptanfit_res(GEOFD_MAXTUBE)   ! pseudo-tangent fit tube residual (ns)
      real*8 fdplane_ptanfit_tchi2(GEOFD_MAXTUBE) ! pseudo-tangent fit tube chi2

      real*8 fdplane_rp                             ! tangent-fit rp (meters)
      real*8 fdplane_erp                            ! tangent-fit rp uncertainty (meters)
      real*8 fdplane_psi                            ! tangent-fit psi (radians)
      real*8 fdplane_epsi                           ! tangent-fit psi uncertainty (radians)
      real*8 fdplane_t0                             ! tangent-fit t0 (ns)
      real*8 fdplane_et0                            ! tangent-fit t0 uncertainty (ns)
      real*8 fdplane_tanfit_chi2                    ! tangent-fit chi2
      real*8 fdplane_tanfit_cov(3,3)                ! pseudo-tangent fit covariance
      real*8 fdplane_tanfit_res(GEOFD_MAXTUBE)    ! pseudo-tangent fit tube residual (ns)
      real*8 fdplane_tanfit_tchi2(GEOFD_MAXTUBE)  ! pseudo-tangent fit tube chi2 contribution

      real*8 fdplane_azm_extent                     ! azimuthal extent of good tubes rotated into SDP coordinate system
      real*8 fdplane_time_extent                    ! time extent of good tubes (ns)

      real*8 fdplane_shower_zen                     ! Shower zenith angle (radians)   
      real*8 fdplane_shower_azm                     ! Shower azimuthal angle (pointing back to source, radians, E=0, N=PI/2)
      real*8 fdplane_shower_axis(3)                 ! Shower axis vector (along direction of shower propagation)
      real*8 fdplane_rpuv(3)                        ! Rp unit vector
      real*8 fdplane_core(3)                        ! Shower core location (meters)

      integer*4 fdplane_camera(GEOFD_MAXTUBE)     ! camera number
      integer*4 fdplane_tube(GEOFD_MAXTUBE)       ! tube number
      integer*4 fdplane_it0(GEOFD_MAXTUBE)        ! FADC index of start of pulse
      integer*4 fdplane_it1(GEOFD_MAXTUBE)        ! FADC index of end of pulse
      integer*4 fdplane_knex_qual(GEOFD_MAXTUBE)  ! 1 = good connectivity, 0 = bad connectivity
      integer*4 fdplane_tube_qual(GEOFD_MAXTUBE)  ! total tube quality
                                                    ! good = 1
                                                    ! bad  = decimal (-[bad_knex][bad_sdpn][bad_tvsa])

      integer*4 fdplane_ngtube                      ! number of good tubes in event
      integer*4 fdplane_seed                        ! original knex seed
      integer*4 fdplane_type                        ! type of event (down=2, up=3, intime=4, noise=5)
      integer*4 fdplane_status                      ! decimal time fit status ([good linear][good pseudotan][good tangent])
      integer*4 fdplane_siteid                      ! site ID (BR = 0, LR = 1)
