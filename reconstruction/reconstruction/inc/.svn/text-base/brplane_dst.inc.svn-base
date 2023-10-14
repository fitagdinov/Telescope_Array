c Created 2008 LMS DRB

      integer BRPLANE_BANKID
      parameter (BRPLANE_BANKID=12103)
      integer BRPLANE_BANKVERSION
      parameter (BRPLANE_BANKVERSION=002)

      integer*4 brplane_common_to_bank
      external  brplane_common_to_bank
      integer*4 brplane_bank_to_dst
      external  brplane_bank_to_dst
      integer*4 brplane_common_to_dst
      external  brplane_common_to_dst
      integer*4 brplane_bank_to_common
      external  brplane_bank_to_common
      integer*4 brplane_common_to_dump
      external  brplane_common_to_dump
      integer*4 brplane_common_to_dumpf
      external  brplane_common_to_dumpf

      common /brplane/
     &     brplane_part,brplane_event_num,brplane_julian,
     &     brplane_jsecond,brplane_jsecfrac,brplane_second,
     &     brplane_secfrac,brplane_ntube,
     &     brplane_uniqID,brplane_fmode,
     &     brplane_npe,brplane_adc,brplane_ped,
     &     brplane_time,brplane_time_rms,brplane_sigma,
     &     brplane_sdp_n,brplane_sdp_en,brplane_sdp_n_cov,
     &     brplane_sdp_the,brplane_sdp_phi,brplane_sdp_chi2,
     &     brplane_alt,brplane_azm,brplane_plane_alt,brplane_plane_azm,
     &     brplane_linefit_slope,brplane_linefit_eslope,
     &     brplane_linefit_int,brplane_linefit_eint,
     &     brplane_linefit_chi2,brplane_linefit_cov,
     &     brplane_linefit_res,brplane_linefit_tchi2,
     &     brplane_ptanfit_rp,brplane_ptanfit_erp,
     &     brplane_ptanfit_t0,brplane_ptanfit_et0,
     &     brplane_ptanfit_chi2,brplane_ptanfit_cov,
     &     brplane_ptanfit_res,brplane_ptanfit_tchi2,
     &     brplane_rp,brplane_erp,brplane_psi,brplane_epsi,
     &     brplane_t0,brplane_et0,brplane_tanfit_chi2,
     &     brplane_tanfit_cov,brplane_tanfit_res,brplane_tanfit_tchi2,
     &     brplane_azm_extent,brplane_time_extent,
     &     brplane_shower_zen,brplane_shower_azm,brplane_shower_axis,
     &     brplane_rpuv,brplane_core,
     &     brplane_camera,brplane_tube,brplane_it0,brplane_it1,
     &     brplane_knex_qual,brplane_tube_qual,
     &     brplane_ngtube,brplane_seed,brplane_type,
     &     brplane_status,brplane_siteid

      integer*4 brplane_part                        ! part number
      integer*4 brplane_event_num                   ! event number
      integer*4 brplane_julian                      ! run start day
      integer*4 brplane_jsecond                     ! run start second (from start of julian)
      integer*4 brplane_jsecfrac                    ! run start nanosecond (from start of jsecond)
      integer*4 brplane_second                      ! event start second (from run start)
      integer*4 brplane_secfrac                     ! event start nanosecond (from start of second
      integer*4 brplane_ntube                       ! number of tubes in event
      integer*4 brplane_uniqID
      integer*4 brplane_fmode
      real*8 brplane_npe(TA_UNIV_MAXTUBE)           ! integrated pulse above pedestal in NPE
      real*8 brplane_adc(TA_UNIV_MAXTUBE);          ! integrated pulse above pedestal in FADC counts
      real*8 brplane_ped(TA_UNIV_MAXTUBE);          ! pedestal value under the pulse in FADC counts

      real*8 brplane_time(TA_UNIV_MAXTUBE)          ! weighted average pulse time
      real*8 brplane_time_rms(TA_UNIV_MAXTUBE)      ! weighted average pulse time rms
      real*8 brplane_sigma(TA_UNIV_MAXTUBE)         ! tube significance

      real*8 brplane_sdp_n(3)                       ! shower-detector plane normal (SDPN)
      real*8 brplane_sdp_en(3)                      ! uncertainty on SDPN fit
      real*8 brplane_sdp_n_cov(3,3)                 ! covariance matrix of SDPN fit
      real*8 brplane_sdp_the                        ! shower-detector plane theta angle
      real*8 brplane_sdp_phi                        ! shower-detector plane phi angle
      real*8 brplane_sdp_chi2                       ! SDPN fit chi2

      real*8 brplane_alt(TA_UNIV_MAXTUBE)           ! altitude of tube
      real*8 brplane_azm(TA_UNIV_MAXTUBE)           ! azimuth of tube
      real*8 brplane_plane_alt(TA_UNIV_MAXTUBE)     ! altitude of tube rotated into SDP coordinate system
      real*8 brplane_plane_azm(TA_UNIV_MAXTUBE)     ! azimuth of tube rotated into SDP coordinate system

      real*8 brplane_linefit_slope                  ! linear fit to time vs. angle slope (ns / degree)
      real*8 brplane_linefit_eslope                 ! linear fit to time vs. angle slope uncertainty (ns / degree)
      real*8 brplane_linefit_int                    ! linear fit to time vs. angle intercept (ns)
      real*8 brplane_linefit_eint                   ! linear fit to time vs. angle intercept uncertainty (ns)
      real*8 brplane_linefit_chi2                   ! linear fit chi2
      real*8 brplane_linefit_cov(2,2)               ! linear fit covariance
      real*8 brplane_linefit_res(TA_UNIV_MAXTUBE)   ! linear fit tube residual (ns)
      real*8 brplane_linefit_tchi2(TA_UNIV_MAXTUBE) ! linear fit tube chi2 contribution

      real*8 brplane_ptanfit_rp                     ! pseudo-tangent fit rp (meters)
      real*8 brplane_ptanfit_erp                    ! pseudo-tangent fit rp uncertainty (meters)
      real*8 brplane_ptanfit_t0                     ! pseudo-tangent fit t0 (ns)
      real*8 brplane_ptanfit_et0                    ! pseudo-tangent fit t0 uncertainty (ns)
      real*8 brplane_ptanfit_chi2                   ! pseudo-tangent fit chi2
      real*8 brplane_ptanfit_cov(2,2)               ! pseudo-tangent fit covariance
      real*8 brplane_ptanfit_res(TA_UNIV_MAXTUBE)   ! pseudo-tangent fit tube residual (ns)
      real*8 brplane_ptanfit_tchi2(TA_UNIV_MAXTUBE) ! pseudo-tangent fit tube chi2

      real*8 brplane_rp                             ! tangent-fit rp (meters)
      real*8 brplane_erp                            ! tangent-fit rp uncertainty (meters)
      real*8 brplane_psi                            ! tangent-fit psi (radians)
      real*8 brplane_epsi                           ! tangent-fit psi uncertainty (radians)
      real*8 brplane_t0                             ! tangent-fit t0 (ns)
      real*8 brplane_et0                            ! tangent-fit t0 uncertainty (ns)
      real*8 brplane_tanfit_chi2                    ! tangent-fit chi2
      real*8 brplane_tanfit_cov(3,3)                ! pseudo-tangent fit covariance
      real*8 brplane_tanfit_res(TA_UNIV_MAXTUBE)    ! pseudo-tangent fit tube residual (ns)
      real*8 brplane_tanfit_tchi2(TA_UNIV_MAXTUBE)  ! pseudo-tangent fit tube chi2 contribution

      real*8 brplane_azm_extent                     ! azimuthal extent of good tubes rotated into SDP coordinate system
      real*8 brplane_time_extent                    ! time extent of good tubes (ns)

      real*8 brplane_shower_zen                     ! Shower zenith angle (radians)   
      real*8 brplane_shower_azm                     ! Shower azimuthal angle (pointing back to source, radians, E=0, N=PI/2)
      real*8 brplane_shower_axis(3)                 ! Shower axis vector (along direction of shower propagation)
      real*8 brplane_rpuv(3)                        ! Rp unit vector
      real*8 brplane_core(3)                        ! Shower core location (meters)

      integer*4 brplane_camera(TA_UNIV_MAXTUBE)     ! camera number
      integer*4 brplane_tube(TA_UNIV_MAXTUBE)       ! tube number
      integer*4 brplane_it0(TA_UNIV_MAXTUBE)        ! FADC index of start of pulse
      integer*4 brplane_it1(TA_UNIV_MAXTUBE)        ! FADC index of end of pulse
      integer*4 brplane_knex_qual(TA_UNIV_MAXTUBE)  ! 1 = good connectivity, 0 = bad connectivity
      integer*4 brplane_tube_qual(TA_UNIV_MAXTUBE)  ! total tube quality
                                                    ! good = 1
                                                    ! bad  = decimal (-[bad_knex][bad_sdpn][bad_tvsa])

      integer*4 brplane_ngtube                      ! number of good tubes in event
      integer*4 brplane_seed                        ! original knex seed
      integer*4 brplane_type                        ! type of event (down=2, up=3, intime=4, noise=5)
      integer*4 brplane_status                      ! decimal time fit status ([good linear][good pseudotan][good tangent])
      integer*4 brplane_siteid                      ! site ID (BR = 0, LR = 1)
