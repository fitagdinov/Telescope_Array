c Locations and pointing directions for FADC cameras and tubes
c All azimuthal angles are east = 0, north = 90
c X-axis to the east, Y-axis to the north
c geobr_dst.inc LMS - 2008/10

      integer*4 GEOBR_BANKID,GEOBR_BANKVERSION
      parameter(GEOBR_BANKID=12101)
      parameter(GEOBR_BANKVERSION=003)

      integer*4  GEOBR_MAXMIR
      parameter (GEOBR_MAXMIR=52)
      integer*4  GEOBR_MIRTUBE
      parameter (GEOBR_MIRTUBE=256)

      integer*4 GEOBR_MAXTUBE
      parameter (GEOBR_MAXTUBE=GEOBR_MAXMIR * GEOBR_MIRTUBE)

      integer*4  GEOBR_ROW        ! Number of rows of PMTs
      parameter (GEOBR_ROW=16)
      integer*4  GEOBR_COL        ! Number of columns of PMTs
      parameter (GEOBR_COL=16)
      integer*4  GEOBR_SEGMENT    ! Number of mirror segments
      parameter (GEOBR_SEGMENT=38)

      real*8     GEOBR_PMT_GAP    ! Gap between flat sides of PMTs (meters)
      parameter (GEOBR_PMT_GAP=0.002d0)

      common /geobr/
     &     geobr_uniqID,
     &     geobr_latitude,geobr_longitude,geobr_altitude,
     &     geobr_vclf,geobr_vsite,geobr_local_vsite,
     &     geobr_site2earth,geobr_site2clf,
     &     geobr_local_vmir,geobr_local_vcam,geobr_vmir,
     &     geobr_mir_lat,geobr_mir_lon,geobr_mir_alt,
     &     geobr_mir_the,geobr_mir_phi,geobr_rcurve,geobr_sep,
     &     geobr_site2cam,
     &     geobr_xtube,geobr_ytube,
     &     geobr_vtube,
     &     geobr_vseg,
     &     geobr_diameter,
     &     geobr_cam_width,geobr_cam_height,geobr_cam_depth,
     &     geobr_pmt_flat2flat,geobr_pmt_point2point,
     &     geobr_seg_flat2flat,geobr_seg_point2point,
     &     geobr_ring,geobr_siteid,
     &     geobr_nmir,geobr_nseg,geobr_ring3,
     &     geobr_diameters,geobr_local_vmir3,geobr_local_vcam3,
     &     geobr_vmir3,geobr_mir_lat3,geobr_mir_lon3,
     &     geobr_mir_alt3,geobr_mir_the3,geobr_mir_phi3,
     &     geobr_rcurve3,geobr_sep3,geobr_site2cam3,
     &     geobr_vtube3,geobr_camtype,geobr_vseg3,
     &     geobr_seg_center,geobr_rotation,
     &     geobr_seg_rcurve,geobr_seg_spot,geobr_seg_orient,
     &     geobr_seg_rcurvex,geobr_seg_rcurvey,
     &     geobr_seg_spotx,geobr_seg_spoty

      integer*4 geobr_uniqID

      real*8 geobr_latitude                       ! Site latitude in radians
      real*8 geobr_longitude                      ! Site longitude in radians
      real*8 geobr_altitude                       ! Site altitude in meters above sea level

      real*8 geobr_vclf(3)    ! vector to vertical axis at CLF relative to center of earth (meters)
      real*8 geobr_vsite(3)   ! vector to vertical axis at site relative to center of earth (meters)
      real*8 geobr_local_vsite(3) ! position of local site origin relative to CLF (east, north, relative altitude in meters)

      real*8 geobr_site2earth(3,3)  ! Rotation matrix to rotate from site coordinates to earth coordinates
      real*8 geobr_site2clf(3,3)    ! Rotation matrix to rotate from site coordinates to CLF coordinates

      real*8 geobr_local_vmir(3,12)   ! vector to mirror from site origin
      real*8 geobr_local_vcam(3,12)  ! vector to camera from site origin
      real*8 geobr_vmir(3,12)         ! Mirror pointing directions relative to local site origin

      real*8 geobr_mir_lat(12) ! Mirror latitude (radians)
      real*8 geobr_mir_lon(12) ! Mirror longitude (radians)
      real*8 geobr_mir_alt(12) ! Mirror altitude above sea level (meters)

      real*8 geobr_mir_the(12) ! Zenith angle of mirror pointing direction relative to local site origin (radians)
      real*8 geobr_mir_phi(12) ! Azimuthal angle of mirror pointing directions relative to local site origin (radians)
      real*8 geobr_rcurve(12)  ! effective mirror radius of curvature (meters)
      real*8 geobr_sep(12)     ! effective mirror-camera separation (meters)

      real*8 geobr_site2cam(3,3,12) ! Rotation matrix to rotate vmir to (0, 0, 1)

c x and y coordinates of tubes in camera box
c (origin at center, when facing camera box, +x is to the right, +y is up)
      real*8 geobr_xtube(GEOBR_MIRTUBE)
      real*8 geobr_ytube(GEOBR_MIRTUBE)

c mean direction vectors of tubes relative to local site origin
      real*8 geobr_vtube(3,GEOBR_MIRTUBE,12)

c unit vectors to mirror segments center relative to center of curvature of mirror
      real*8 geobr_vseg(3,18)

      real*8 geobr_diameter   ! largest-distance diameter of mirror (meters)

      real*8 geobr_cam_width   ! Width of camera box (meters)
      real*8 geobr_cam_height  ! Height of camera box (meters)
      real*8 geobr_cam_depth   ! Depth of camera box (meters)

      real*8 geobr_pmt_flat2flat   ! flat-to-flat distance on PMT (meters)
      real*8 geobr_pmt_point2point ! point-to-point distance on PMT (meters)

c The following segment distances are along the chord of the sphere (two dimensional distance)
      real*8 geobr_seg_flat2flat    ! mirror segment 2D flat-to-flat distance (meters)
      real*8 geobr_seg_point2point  ! mirror segment 2D point-to-point distance (meters)

      integer*4 geobr_ring(18)  ! mirror ring number
      integer*4 geobr_siteid   ! site id (BR = 0, LR = 1, MD = 2, TL = 3)

      integer*4 geobr_nmir
      integer*4 geobr_nseg(GEOBR_MAXMIR)
      integer*4 geobr_ring3(GEOBR_MAXMIR)
      
      real*8 geobr_diameters(GEOBR_MAXMIR)
      real*8 geobr_local_vmir3(3,GEOBR_MAXMIR)
      real*8 geobr_local_vcam3(3,GEOBR_MAXMIR)
      real*8 geobr_vmir3(3,GEOBR_MAXMIR)
      
      real*8 geobr_mir_lat3(GEOBR_MAXMIR)
      real*8 geobr_mir_lon3(GEOBR_MAXMIR)
      real*8 geobr_mir_alt3(GEOBR_MAXMIR)
      
      real*8 geobr_mir_the3(GEOBR_MAXMIR)
      real*8 geobr_mir_phi3(GEOBR_MAXMIR)
      real*8 geobr_rcurve3(GEOBR_MAXMIR)
      real*8 geobr_sep3(GEOBR_MAXMIR)
      real*8 geobr_site2cam3(3,3,GEOBR_MAXMIR)
      real*8 geobr_vtube3(3,GEOBR_MIRTUBE,GEOBR_MAXMIR)
      
      integer*4 geobr_camtype(GEOBR_MAXMIR)
      
      real*8 geobr_vseg3(3,GEOBR_SEGMENT,GEOBR_MAXMIR)
      real*8 geobr_seg_center(3,GEOBR_SEGMENT,GEOBR_MAXMIR)
      real*8 geobr_rotation(GEOBR_MAXMIR)
      
      real*8 geobr_seg_rcurve(GEOBR_SEGMENT,GEOBR_MAXMIR)
      real*8 geobr_seg_spot(GEOBR_SEGMENT,GEOBR_MAXMIR)
      real*8 geobr_seg_orient(GEOBR_SEGMENT,GEOBR_MAXMIR)
      real*8 geobr_seg_rcurvex(GEOBR_SEGMENT,GEOBR_MAXMIR)
      real*8 geobr_seg_rcurvey(GEOBR_SEGMENT,GEOBR_MAXMIR)
      real*8 geobr_seg_spotx(GEOBR_SEGMENT,GEOBR_MAXMIR)
      real*8 geobr_seg_spoty(GEOBR_SEGMENT,GEOBR_MAXMIR)