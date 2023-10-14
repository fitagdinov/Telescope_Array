c Locations and pointing directions for FADC cameras and tubes
c All azimuthal angles are east = 0, north = 90
c X-axis to the east, Y-axis to the north
c geolr_dst.inc LMS - 2008/10

      integer*4 GEOLR_BANKID,GEOLR_BANKVERSION
      parameter(GEOLR_BANKID=12202)
      parameter(GEOLR_BANKVERSION=003)

      integer*4  GEOLR_MAXMIR
      parameter (GEOLR_MAXMIR=52)
      integer*4  GEOLR_MIRTUBE
      parameter (GEOLR_MIRTUBE=256)

      integer*4 GEOLR_MAXTUBE
      parameter (GEOLR_MAXTUBE=GEOLR_MAXMIR * GEOLR_MIRTUBE)

      integer*4  GEOLR_ROW        ! Number of rows of PMTs
      parameter (GEOLR_ROW=16)
      integer*4  GEOLR_COL        ! Number of columns of PMTs
      parameter (GEOLR_COL=16)
      integer*4  GEOLR_SEGMENT    ! Number of mirror segments
      parameter (GEOLR_SEGMENT=38)

      real*8     GEOLR_PMT_GAP    ! Gap between flat sides of PMTs (meters)
      parameter (GEOLR_PMT_GAP=0.002d0)

      common /geolr/
     &     geolr_uniqID,
     &     geolr_latitude,geolr_longitude,geolr_altitude,
     &     geolr_vclf,geolr_vsite,geolr_local_vsite,
     &     geolr_site2earth,geolr_site2clf,
     &     geolr_local_vmir,geolr_local_vcam,geolr_vmir,
     &     geolr_mir_lat,geolr_mir_lon,geolr_mir_alt,
     &     geolr_mir_the,geolr_mir_phi,geolr_rcurve,geolr_sep,
     &     geolr_site2cam,
     &     geolr_xtube,geolr_ytube,
     &     geolr_vtube,
     &     geolr_vseg,
     &     geolr_diameter,
     &     geolr_cam_width,geolr_cam_height,geolr_cam_depth,
     &     geolr_pmt_flat2flat,geolr_pmt_point2point,
     &     geolr_seg_flat2flat,geolr_seg_point2point,
     &     geolr_ring,geolr_siteid,
     &     geolr_nmir,geolr_nseg,geolr_ring3,
     &     geolr_diameters,geolr_local_vmir3,geolr_local_vcam3,
     &     geolr_vmir3,geolr_mir_lat3,geolr_mir_lon3,
     &     geolr_mir_alt3,geolr_mir_the3,geolr_mir_phi3,
     &     geolr_rcurve3,geolr_sep3,geolr_site2cam3,
     &     geolr_vtube3,geolr_camtype,geolr_vseg3,
     &     geolr_seg_center,geolr_rotation,
     &     geolr_seg_rcurve,geolr_seg_spot,geolr_seg_orient,
     &     geolr_seg_rcurvex,geolr_seg_rcurvey,
     &     geolr_seg_spotx,geolr_seg_spoty

      integer*4 geolr_uniqID

      real*8 geolr_latitude                       ! Site latitude in radians
      real*8 geolr_longitude                      ! Site longitude in radians
      real*8 geolr_altitude                       ! Site altitude in meters above sea level

      real*8 geolr_vclf(3)    ! vector to vertical axis at CLF relative to center of earth (meters)
      real*8 geolr_vsite(3)   ! vector to vertical axis at site relative to center of earth (meters)
      real*8 geolr_local_vsite(3) ! position of local site origin relative to CLF (east, north, relative altitude in meters)

      real*8 geolr_site2earth(3,3)  ! Rotation matrix to rotate from site coordinates to earth coordinates
      real*8 geolr_site2clf(3,3)    ! Rotation matrix to rotate from site coordinates to CLF coordinates

      real*8 geolr_local_vmir(3,12)   ! vector to mirror from site origin
      real*8 geolr_local_vcam(3,12)  ! vector to camera from site origin
      real*8 geolr_vmir(3,12)         ! Mirror pointing directions relative to local site origin

      real*8 geolr_mir_lat(12) ! Mirror latitude (radians)
      real*8 geolr_mir_lon(12) ! Mirror longitude (radians)
      real*8 geolr_mir_alt(12) ! Mirror altitude above sea level (meters)

      real*8 geolr_mir_the(12) ! Zenith angle of mirror pointing direction relative to local site origin (radians)
      real*8 geolr_mir_phi(12) ! Azimuthal angle of mirror pointing directions relative to local site origin (radians)
      real*8 geolr_rcurve(12)  ! effective mirror radius of curvature (meters)
      real*8 geolr_sep(12)     ! effective mirror-camera separation (meters)

      real*8 geolr_site2cam(3,3,12) ! Rotation matrix to rotate vmir to (0, 0, 1)

c x and y coordinates of tubes in camera box
c (origin at center, when facing camera box, +x is to the right, +y is up)
      real*8 geolr_xtube(GEOLR_MIRTUBE)
      real*8 geolr_ytube(GEOLR_MIRTUBE)

c mean direction vectors of tubes relative to local site origin
      real*8 geolr_vtube(3,GEOLR_MIRTUBE,12)

c unit vectors to mirror segments center relative to center of curvature of mirror
      real*8 geolr_vseg(3,18)

      real*8 geolr_diameter   ! largest-distance diameter of mirror (meters)

      real*8 geolr_cam_width   ! Width of camera box (meters)
      real*8 geolr_cam_height  ! Height of camera box (meters)
      real*8 geolr_cam_depth   ! Depth of camera box (meters)

      real*8 geolr_pmt_flat2flat   ! flat-to-flat distance on PMT (meters)
      real*8 geolr_pmt_point2point ! point-to-point distance on PMT (meters)

c The following segment distances are along the chord of the sphere (two dimensional distance)
      real*8 geolr_seg_flat2flat    ! mirror segment 2D flat-to-flat distance (meters)
      real*8 geolr_seg_point2point  ! mirror segment 2D point-to-point distance (meters)

      integer*4 geolr_ring(18)  ! mirror ring number
      integer*4 geolr_siteid   ! site id (BR = 0, LR = 1, MD = 2, TL = 3)

      integer*4 geolr_nmir
      integer*4 geolr_nseg(GEOLR_MAXMIR)
      integer*4 geolr_ring3(GEOLR_MAXMIR)
      
      real*8 geolr_diameters(GEOLR_MAXMIR)
      real*8 geolr_local_vmir3(3,GEOLR_MAXMIR)
      real*8 geolr_local_vcam3(3,GEOLR_MAXMIR)
      real*8 geolr_vmir3(3,GEOLR_MAXMIR)
      
      real*8 geolr_mir_lat3(GEOLR_MAXMIR)
      real*8 geolr_mir_lon3(GEOLR_MAXMIR)
      real*8 geolr_mir_alt3(GEOLR_MAXMIR)
      
      real*8 geolr_mir_the3(GEOLR_MAXMIR)
      real*8 geolr_mir_phi3(GEOLR_MAXMIR)
      real*8 geolr_rcurve3(GEOLR_MAXMIR)
      real*8 geolr_sep3(GEOLR_MAXMIR)
      real*8 geolr_site2cam3(3,3,GEOLR_MAXMIR)
      real*8 geolr_vtube3(3,GEOLR_MIRTUBE,GEOLR_MAXMIR)
      
      integer*4 geolr_camtype(GEOLR_MAXMIR)
      
      real*8 geolr_vseg3(3,GEOLR_SEGMENT,GEOLR_MAXMIR)
      real*8 geolr_seg_center(3,GEOLR_SEGMENT,GEOLR_MAXMIR)
      real*8 geolr_rotation(GEOLR_MAXMIR)
      
      real*8 geolr_seg_rcurve(GEOLR_SEGMENT,GEOLR_MAXMIR)
      real*8 geolr_seg_spot(GEOLR_SEGMENT,GEOLR_MAXMIR)
      real*8 geolr_seg_orient(GEOLR_SEGMENT,GEOLR_MAXMIR)
      real*8 geolr_seg_rcurvex(GEOLR_SEGMENT,GEOLR_MAXMIR)
      real*8 geolr_seg_rcurvey(GEOLR_SEGMENT,GEOLR_MAXMIR)
      real*8 geolr_seg_spotx(GEOLR_SEGMENT,GEOLR_MAXMIR)
      real*8 geolr_seg_spoty(GEOLR_SEGMENT,GEOLR_MAXMIR)