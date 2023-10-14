c Locations and pointing directions for FADC cameras and tubes
c All azimuthal angles are east = 0, north = 90
c X-axis to the east, Y-axis to the north
c geofd_dst.inc LMS - 2008/10

      integer*4 GEOFD_BANKID,GEOFD_BANKVERSION
      parameter(GEOFD_BANKID=12091)
      parameter(GEOFD_BANKVERSION=003)

      integer*4  GEOFD_MAXMIR
      parameter (GEOFD_MAXMIR=52)
      integer*4  GEOFD_MIRTUBE
      parameter (GEOFD_MIRTUBE=256)

      integer*4 GEOFD_MAXTUBE
      parameter (GEOFD_MAXTUBE=GEOFD_MAXMIR * GEOFD_MIRTUBE)

      integer*4  GEOFD_ROW        ! Number of rows of PMTs
      parameter (GEOFD_ROW=16)
      integer*4  GEOFD_COL        ! Number of columns of PMTs
      parameter (GEOFD_COL=16)
      integer*4  GEOFD_SEGMENT    ! Number of mirror segments
      parameter (GEOFD_SEGMENT=38)

      real*8     GEOFD_PMT_GAP    ! Gap between flat sides of PMTs (meters)
      parameter (GEOFD_PMT_GAP=0.002d0)

      common /geofd/
     &     geofd_uniqID,
     &     geofd_latitude,geofd_longitude,geofd_altitude,
     &     geofd_vclf,geofd_vsite,geofd_local_vsite,
     &     geofd_site2earth,geofd_site2clf,
     &     geofd_local_vmir,geofd_local_vcam,geofd_vmir,
     &     geofd_mir_lat,geofd_mir_lon,geofd_mir_alt,
     &     geofd_mir_the,geofd_mir_phi,geofd_rcurve,geofd_sep,
     &     geofd_site2cam,
     &     geofd_xtube,geofd_ytube,
     &     geofd_vtube,
     &     geofd_vseg,
     &     geofd_diameter,
     &     geofd_cam_width,geofd_cam_height,geofd_cam_depth,
     &     geofd_pmt_flat2flat,geofd_pmt_point2point,
     &     geofd_seg_flat2flat,geofd_seg_point2point,
     &     geofd_ring,geofd_siteid,
     &     geofd_nmir,geofd_nseg,geofd_ring3,
     &     geofd_diameters,geofd_local_vmir3,geofd_local_vcam3,
     &     geofd_vmir3,geofd_mir_lat3,geofd_mir_lon3,
     &     geofd_mir_alt3,geofd_mir_the3,geofd_mir_phi3,
     &     geofd_rcurve3,geofd_sep3,geofd_site2cam3,
     &     geofd_vtube3,geofd_camtype,geofd_vseg3,
     &     geofd_seg_center,geofd_rotation,
     &     geofd_seg_rcurve,geofd_seg_spot,geofd_seg_orient,
     &     geofd_seg_rcurvex,geofd_seg_rcurvey,
     &     geofd_seg_spotx,geofd_seg_spoty

      integer*4 geofd_uniqID

      real*8 geofd_latitude                       ! Site latitude in radians
      real*8 geofd_longitude                      ! Site longitude in radians
      real*8 geofd_altitude                       ! Site altitude in meters above sea level

      real*8 geofd_vclf(3)    ! vector to vertical axis at CLF relative to center of earth (meters)
      real*8 geofd_vsite(3)   ! vector to vertical axis at site relative to center of earth (meters)
      real*8 geofd_local_vsite(3) ! position of local site origin relative to CLF (east, north, relative altitude in meters)

      real*8 geofd_site2earth(3,3)  ! Rotation matrix to rotate from site coordinates to earth coordinates
      real*8 geofd_site2clf(3,3)    ! Rotation matrix to rotate from site coordinates to CLF coordinates

      real*8 geofd_local_vmir(3,12)   ! vector to mirror from site origin
      real*8 geofd_local_vcam(3,12)  ! vector to camera from site origin
      real*8 geofd_vmir(3,12)         ! Mirror pointing directions relative to local site origin

      real*8 geofd_mir_lat(12) ! Mirror latitude (radians)
      real*8 geofd_mir_lon(12) ! Mirror longitude (radians)
      real*8 geofd_mir_alt(12) ! Mirror altitude above sea level (meters)

      real*8 geofd_mir_the(12) ! Zenith angle of mirror pointing direction relative to local site origin (radians)
      real*8 geofd_mir_phi(12) ! Azimuthal angle of mirror pointing directions relative to local site origin (radians)
      real*8 geofd_rcurve(12)  ! effective mirror radius of curvature (meters)
      real*8 geofd_sep(12)     ! effective mirror-camera separation (meters)

      real*8 geofd_site2cam(3,3,12) ! Rotation matrix to rotate vmir to (0, 0, 1)

c x and y coordinates of tubes in camera box
c (origin at center, when facing camera box, +x is to the right, +y is up)
      real*8 geofd_xtube(GEOFD_MIRTUBE)
      real*8 geofd_ytube(GEOFD_MIRTUBE)

c mean direction vectors of tubes relative to local site origin
      real*8 geofd_vtube(3,GEOFD_MIRTUBE,12)

c unit vectors to mirror segments center relative to center of curvature of mirror
      real*8 geofd_vseg(3,18)

      real*8 geofd_diameter   ! largest-distance diameter of mirror (meters)

      real*8 geofd_cam_width   ! Width of camera box (meters)
      real*8 geofd_cam_height  ! Height of camera box (meters)
      real*8 geofd_cam_depth   ! Depth of camera box (meters)

      real*8 geofd_pmt_flat2flat   ! flat-to-flat distance on PMT (meters)
      real*8 geofd_pmt_point2point ! point-to-point distance on PMT (meters)

c The following segment distances are along the chord of the sphere (two dimensional distance)
      real*8 geofd_seg_flat2flat    ! mirror segment 2D flat-to-flat distance (meters)
      real*8 geofd_seg_point2point  ! mirror segment 2D point-to-point distance (meters)

      integer*4 geofd_ring(18)  ! mirror ring number
      integer*4 geofd_siteid   ! site id (BR = 0, LR = 1, MD = 2, TL = 3)

      integer*4 geofd_nmir
      integer*4 geofd_nseg(GEOFD_MAXMIR)
      integer*4 geofd_ring3(GEOFD_MAXMIR)
      
      real*8 geofd_diameters(GEOFD_MAXMIR)
      real*8 geofd_local_vmir3(3,GEOFD_MAXMIR)
      real*8 geofd_local_vcam3(3,GEOFD_MAXMIR)
      real*8 geofd_vmir3(3,GEOFD_MAXMIR)
      
      real*8 geofd_mir_lat3(GEOFD_MAXMIR)
      real*8 geofd_mir_lon3(GEOFD_MAXMIR)
      real*8 geofd_mir_alt3(GEOFD_MAXMIR)
      
      real*8 geofd_mir_the3(GEOFD_MAXMIR)
      real*8 geofd_mir_phi3(GEOFD_MAXMIR)
      real*8 geofd_rcurve3(GEOFD_MAXMIR)
      real*8 geofd_sep3(GEOFD_MAXMIR)
      real*8 geofd_site2cam3(3,3,GEOFD_MAXMIR)
      real*8 geofd_vtube3(3,GEOFD_MIRTUBE,GEOFD_MAXMIR)
      
      integer*4 geofd_camtype(GEOFD_MAXMIR)
      
      real*8 geofd_vseg3(3,GEOFD_SEGMENT,GEOFD_MAXMIR)
      real*8 geofd_seg_center(3,GEOFD_SEGMENT,GEOFD_MAXMIR)
      real*8 geofd_rotation(GEOFD_MAXMIR)
      
      real*8 geofd_seg_rcurve(GEOFD_SEGMENT,GEOFD_MAXMIR)
      real*8 geofd_seg_spot(GEOFD_SEGMENT,GEOFD_MAXMIR)
      real*8 geofd_seg_orient(GEOFD_SEGMENT,GEOFD_MAXMIR)
      real*8 geofd_seg_rcurvex(GEOFD_SEGMENT,GEOFD_MAXMIR)
      real*8 geofd_seg_rcurvey(GEOFD_SEGMENT,GEOFD_MAXMIR)
      real*8 geofd_seg_spotx(GEOFD_SEGMENT,GEOFD_MAXMIR)
      real*8 geofd_seg_spoty(GEOFD_SEGMENT,GEOFD_MAXMIR)