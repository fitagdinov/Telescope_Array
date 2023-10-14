c Created 2008/09/24 LMS DRB

      integer LRPLANE_BANKID
      parameter (LRPLANE_BANKID=12203)
      integer LRPLANE_BANKVERSION
      parameter (LRPLANE_BANKVERSION=002)

      integer*4 lrplane_common_to_bank
      external  lrplane_common_to_bank
      integer*4 lrplane_bank_to_dst
      external  lrplane_bank_to_dst
      integer*4 lrplane_common_to_dst
      external  lrplane_common_to_dst
      integer*4 lrplane_bank_to_common
      external  lrplane_bank_to_common
      integer*4 lrplane_common_to_dump
      external  lrplane_common_to_dump
      integer*4 lrplane_common_to_dumpf
      external  lrplane_common_to_dumpf

      common /lrplane/
     &     lrplane_part,lrplane_event_num,lrplane_julian,
     &     lrplane_jsecond,lrplane_jsecfrac,lrplane_second,
     &     lrplane_secfrac,lrplane_ntube,
     &     lrplane_npe,lrplane_adc,lrplane_ped,
     &     lrplane_time,lrplane_time_rms,lrplane_sigma,
     &     lrplane_sdp_n,lrplane_sdp_en,lrplane_sdp_n_cov,
     &     lrplane_sdp_the,lrplane_sdp_phi,lrplane_sdp_chi2,
     &     lrplane_alt,lrplane_azm,lrplane_plane_alt,lrplane_plane_azm,
     &     lrplane_linefit_slope,lrplane_linefit_eslope,
     &     lrplane_linefit_int,lrplane_linefit_eint,
     &     lrplane_linefit_chi2,lrplane_linefit_cov,
     &     lrplane_linefit_res,lrplane_linefit_tchi2,
     &     lrplane_ptanfit_rp,lrplane_ptanfit_erp,
     &     lrplane_ptanfit_t0,lrplane_ptanfit_et0,
     &     lrplane_ptanfit_chi2,lrplane_ptanfit_cov,
     &     lrplane_ptanfit_res,lrplane_ptanfit_tchi2,
     &     lrplane_rp,lrplane_erp,lrplane_psi,lrplane_epsi,
     &     lrplane_t0,lrplane_et0,lrplane_tanfit_chi2,
     &     lrplane_tanfit_cov,lrplane_tanfit_res,lrplane_tanfit_tchi2,
     &     lrplane_azm_extent,lrplane_time_extent,
     &     lrplane_shower_zen,lrplane_shower_azm,lrplane_shower_axis,
     &     lrplane_rpuv,lrplane_core,
     &     lrplane_camera,lrplane_tube,lrplane_it0,lrplane_it1,
     &     lrplane_knex_qual,lrplane_tube_qual,
     &     lrplane_ngtube,lrplane_seed,lrplane_type,
     &     lrplane_status,lrplane_siteid

      integer*4 lrplane_part                        ! part number
      integer*4 lrplane_event_num                   ! event number
      integer*4 lrplane_julian                      ! run start day
      integer*4 lrplane_jsecond                     ! run start second (from start of julian)
      integer*4 lrplane_jsecfrac                    ! run start nanosecond (from start of jsecond)
      integer*4 lrplane_second                      ! event start second (from run start)
      integer*4 lrplane_secfrac                     ! event start nanosecond (from start of second
      integer*4 lrplane_ntube                       ! number of tubes in event
      integer*4 fdplane_uniqID
      integer*4 fdplane_fmode
      real*8 lrplane_npe(TA_UNIV_MAXTUBE)           ! integrated pulse above pedestal in NPE
      real*8 lrplane_adc(TA_UNIV_MAXTUBE);          ! integrated pulse above pedestal in FADC counts
      real*8 lrplane_ped(TA_UNIV_MAXTUBE);          ! pedestal value under the pulse in FADC counts
      real*8 lrplane_time(TA_UNIV_MAXTUBE)          ! weighted average pulse time
      real*8 lrplane_time_rms(TA_UNIV_MAXTUBE)      ! weighted average pulse time rms
      real*8 lrplane_sigma(TA_UNIV_MAXTUBE)         ! tube significance

      real*8 lrplane_sdp_n(3)                       ! shower-detector plane normal (SDPN)
      real*8 lrplane_sdp_en(3)                      ! uncertainty on SDPN fit
      real*8 lrplane_sdp_n_cov(3,3)                 ! covariance matrix of SDPN fit
      real*8 lrplane_sdp_the                        ! shower-detector plane theta angle
      real*8 lrplane_sdp_phi                        ! shower-detector plane phi angle
      real*8 lrplane_sdp_chi2                       ! SDPN fit chi2

      real*8 lrplane_alt(TA_UNIV_MAXTUBE)           ! altitude of tube
      real*8 lrplane_azm(TA_UNIV_MAXTUBE)           ! azimuth of tube
      real*8 lrplane_plane_alt(TA_UNIV_MAXTUBE)     ! altitude of tube rotated into SDP coordinate system
      real*8 lrplane_plane_azm(TA_UNIV_MAXTUBE)     ! azimuth of tube rotated into SDP coordinate system

      real*8 lrplane_linefit_slope                  ! linear fit to time vs. angle slope (ns / degree)
      real*8 lrplane_linefit_eslope                 ! linear fit to time vs. angle slope uncertainty (ns / degree)
      real*8 lrplane_linefit_int                    ! linear fit to time vs. angle intercept (ns)
      real*8 lrplane_linefit_eint                   ! linear fit to time vs. angle intercept uncertainty (ns)
      real*8 lrplane_linefit_chi2                   ! linear fit chi2
      real*8 lrplane_linefit_cov(2,2)               ! linear fit covariance
      real*8 lrplane_linefit_res(TA_UNIV_MAXTUBE)   ! linear fit tube residual (ns)
      real*8 lrplane_linefit_tchi2(TA_UNIV_MAXTUBE) ! linear fit tube chi2 contribution

      real*8 lrplane_ptanfit_rp                     ! pseudo-tangent fit rp (meters)
      real*8 lrplane_ptanfit_erp                    ! pseudo-tangent fit rp uncertainty (meters)
      real*8 lrplane_ptanfit_t0                     ! pseudo-tangent fit t0 (ns)
      real*8 lrplane_ptanfit_et0                    ! pseudo-tangent fit t0 uncertainty (ns)
      real*8 lrplane_ptanfit_chi2                   ! pseudo-tangent fit chi2
      real*8 lrplane_ptanfit_cov(2,2)               ! pseudo-tangent fit covariance
      real*8 lrplane_ptanfit_res(TA_UNIV_MAXTUBE)   ! pseudo-tangent fit tube residual (ns)
      real*8 lrplane_ptanfit_tchi2(TA_UNIV_MAXTUBE) ! pseudo-tangent fit tube chi2

      real*8 lrplane_rp                             ! tangent-fit rp (meters)
      real*8 lrplane_erp                            ! tangent-fit rp uncertainty (meters)
      real*8 lrplane_psi                            ! tangent-fit psi (radians)
      real*8 lrplane_epsi                           ! tangent-fit psi uncertainty (radians)
      real*8 lrplane_t0                             ! tangent-fit t0 (ns)
      real*8 lrplane_et0                            ! tangent-fit t0 uncertainty (ns)
      real*8 lrplane_tanfit_chi2                    ! tangent-fit chi2
      real*8 lrplane_tanfit_cov(3,3)                ! pseudo-tangent fit covariance
      real*8 lrplane_tanfit_res(TA_UNIV_MAXTUBE)    ! pseudo-tangent fit tube residual (ns)
      real*8 lrplane_tanfit_tchi2(TA_UNIV_MAXTUBE)  ! pseudo-tangent fit tube chi2 contribution

      real*8 lrplane_azm_extent                     ! azimuthal extent of good tubes rotated into SDP coordinate system
      real*8 lrplane_time_extent                    ! time extent of good tubes (ns)

      real*8 lrplane_shower_zen                     ! Shower zenith angle (radians)   
      real*8 lrplane_shower_azm                     ! Shower azimuthal angle (pointing back to source, radians, E=0, N=PI/2)
      real*8 lrplane_shower_axis(3)                 ! Shower axis vector (along direction of shower propagation)
      real*8 lrplane_rpuv(3)                        ! Rp unit vector
      real*8 lrplane_core(3)                        ! Shower core location (meters)

      integer*4 lrplane_camera(TA_UNIV_MAXTUBE)     ! camera number
      integer*4 lrplane_tube(TA_UNIV_MAXTUBE)       ! tube number
      integer*4 lrplane_it0(TA_UNIV_MAXTUBE)        ! FADC index of start of pulse
      integer*4 lrplane_it1(TA_UNIV_MAXTUBE)        ! FADC index of end of pulse
      integer*4 lrplane_knex_qual(TA_UNIV_MAXTUBE)  ! 1 = good connectivity, 0 = bad connectivity
      integer*4 lrplane_tube_qual(TA_UNIV_MAXTUBE)  ! total tube quality
                                                    ! good = 1
                                                    ! bad  = decimal (-[bad_knex][bad_sdpn][bad_tvsa])

      integer*4 lrplane_ngtube                      ! number of good tubes in event
      integer*4 lrplane_seed                        ! original knex seed
      integer*4 lrplane_type                        ! type of event (down=2, up=3, intime=4, noise=5)
      integer*4 lrplane_status                      ! decimal time fit status ([good linear][good pseudotan][good tangent])
      integer*4 lrplane_siteid                      ! site ID (BR = 0, LR = 1)
