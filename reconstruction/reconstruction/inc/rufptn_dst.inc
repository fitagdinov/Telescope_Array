c     Bank for SD pass1 data
c     Dmitri Ivanov (dmiivanov@gmail.com)
c     Jul 5, 2008

c     Last Modified: Nov 6, 2018


      integer RUFPTN_BANKID
      parameter (RUFPTN_BANKID=13103)
      integer RUFPTN_BANKVERSION
      parameter (RUFPTN_BANKVERSION=001)
      

      integer RUFPTNMH          ! max number of hits
      parameter(RUFPTNMH=1024)

c     SD origin with respect to CLF origin in CLF frame, in [1200m] units
      real*8 RUFPTN_ORIGIN_X_CLF
      parameter (RUFPTN_ORIGIN_X_CLF=-12.2435)
      real*8 RUFPTN_ORIGIN_Y_CLF
      parameter (RUFPTN_ORIGIN_Y_CLF=-16.4406)

c     For converting time in uS to distance in units of counter
c     separation distance 1200m. Numerically, this is 
c     c*(10^(-6)(S/uS))/1200m, where c is the speed of light in m/s */
      real*8 RUFPTN_TIMDIST
      parameter(RUFPTN_TIMDIST=0.249827048333)

      integer*4 rufptn_common_to_bank
      external  rufptn_common_to_bank
      integer*4 rufptn_bank_to_dst
      external  rufptn_bank_to_dst
      integer*4 rufptn_common_to_dst
      external  rufptn_common_to_dst
      integer*4 rufptn_bank_to_common
      external  rufptn_bank_to_common
      integer*4 rufptn_common_to_dump
      external  rufptn_common_to_dump
      integer*4 rufptn_common_to_dumpf
      external  rufptn_common_to_dumpf



      common /rufptn/ 
     &     rufptn_nhits,rufptn_nsclust,rufptn_nstclust,rufptn_nborder,
     &     rufptn_isgood,rufptn_wfindex,rufptn_xxyy,rufptn_nfold,
     &     rufptn_sstart,rufptn_sstop,rufptn_lderiv,rufptn_zderiv,
     &     rufptn_xyzclf,rufptn_qtot,rufptn_tearliest,rufptn_reltime,
     &     rufptn_timeerr,rufptn_fadcpa,rufptn_fadcpaerr,rufptn_pulsa,
     &     rufptn_pulsaerr,rufptn_ped,rufptn_pederr,rufptn_vem,
     &     rufptn_vemerr,rufptn_tyro_cdist,rufptn_tyro_xymoments,
     &     rufptn_tyro_xypmoments,rufptn_tyro_u,rufptn_tyro_v,
     &     rufptn_tyro_tfitpars,rufptn_tyro_chi2,rufptn_tyro_ndof,
     &     rufptn_tyro_theta,rufptn_tyro_phi
      
      
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     fadc trace and monitoring data parsing, integer data 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      
      integer*4 rufptn_nhits    ! number of independent signals (hits) in the trigger
      
      integer*4 rufptn_nsclust  ! number of hits in the largest space cluster
      
c     Only one hit from each SD can be a part of space-time cluster
      integer*4 rufptn_nstclust ! number of SDs in space-time cluster
      
      integer*4 rufptn_nborder  ! number of counters that are a part of the event and lie on the border
      
c     isgood - variable:
c     isgood(i) = 0 : the counter to which i'th hit corresonds was not working properly
c     isgood(i) = 1 : i'th hit is not a part of any clusters
c     isgood(i) = 2 : i'th hit is a part of space cluster
c     isgood(i) = 3:  i'th hit passed rough time pattern recognition
c     isgood(i) = 4:  i'th hit is a part of the event
c     isgood(i) = 5:  i'th hit corresponds to a saturated counter 
      integer*4 rufptn_isgood(RUFPTNMH)

      integer*4 rufptn_wfindex(RUFPTNMH) ! rusdraw - index of the 1st waveform in each hit
      integer*4 rufptn_xxyy(RUFPTNMH) ! position of the hit
      integer*4 rufptn_nfold(RUFPTNMH) ! foldedness of the hit (over how many 128 fadc widnows this signal extends)

c     1st index: 1 for lower, 2 for upper layers
      integer*4 rufptn_sstart(2,RUFPTNMH) ! channel where the signal starts. 
      integer*4 rufptn_sstop(2,RUFPTNMH) ! channel where the signal stops 
      
c     Channel after which fadc makes biggest jump since the signal started. 
c     Signal point of inflection channel. Use the right edgre of this channel.
      integer*4 rufptn_lderiv(2,RUFPTNMH) 
      
c     First channel since the point of inflection, after which derivative first becomes negative.
c     Take the right edge of the bin when computing rms on time using fadc derivative.
      integer*4 rufptn_zderiv(2,RUFPTNMH)
      

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     fadc trace and monitoring, double precision data
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


c     1st index:
c     (1,*) = X position of i'th counter in CLF frame
c     (2,*) = Y position of i'th counter in CLF frame
c     (3,*) = Z position of i'th counter in CLF frame
      real*8 rufptn_xyzclf(3,RUFPTNMH) ! SD coordinates with respect to CLF frame, [1200m] units

c     (1) - lower layers, (2) - upper layers
      real*8 rufptn_qtot(2)     ! Total charge in the event (sum over counters in space-time cluster) 
      
      
      
c     Time of the earliest waveform in the trigger in seconds since midnight.
c     To find time in seconds since midnight of any hit, do
c     tearliest + (reltime of a given hit) * 1200m / (c*t).
      real*8 rufptn_tearliest(2) ! (1) - for lower, (2) - for upper layers
      

c     1st index: 1 - for lower, 2 - for upper layers
      
      real*8 rufptn_reltime(2,RUFPTNMH) ! hit time, relative to to EARLIEST hit, in units of counter sep. dist 
      real*8 rufptn_timeerr(2,RUFPTNMH) ! error on time, in counter separation units 
      real*8 rufptn_fadcpa(2,RUFPTNMH) ! pulse area, in fadc counts, peds subtracted 
      real*8 rufptn_fadcpaerr(2,RUFPTNMH) ! errror on (pulse area - peds) in fadc counts  
      real*8 rufptn_pulsa(2,RUFPTNMH) ! pulse area in VEM (pedestals subtracted) 
      real*8 rufptn_pulsaerr(2,RUFPTNMH) ! error on pulse area in VEM (pedestals subtracted) 
      real*8 rufptn_ped(2,RUFPTNMH) ! pedestals taken from monitoring  
      real*8 rufptn_pederr(2,RUFPTNMH) ! pedestal errors computed from the monitoring information (FWHM/2.33)  
      real*8 rufptn_vem(2,RUFPTNMH) ! FADC counts/VEM, from monitoring 
      real*8 rufptn_vemerr(2,RUFPTNMH) ! errors on the FADC Counts/VEM (FWHM/2.33), using monitoring 
      

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Tyro geometry reconstruction (double precision data) 
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c     2nd index interpreted as follows:
c     (*,1): reconstruction using lower counters
c     (*,2): reconstruction using upper counters
c     (*,3): reconstruction using upper and lower counters (averaging over upper/lower)
 
      real*8 rufptn_tyro_cdist(RUFPTNMH,3) ! distances from the core for all counters that were hit

c     (1-2,*): <x>,<y> in CLF frame with SD origin subtracted off, [1200m] units.
c     (3,*): <x**2> about the core
c     (4,*): <xy> about the core
c     (5,*): <y**2> about the core
      real*8 rufptn_tyro_xymoments(5,3)
      

      real*8 rufptn_tyro_xypmoments(2,3) ! principal moments (eigenvalues) 
      real*8 rufptn_tyro_u(2,3) ! long axis, corresponding to larger eigenvalue 
      real*8 rufptn_tyro_v(2,3) ! short axis, corresponding to smaller eigenvalue 
      
c     Time fit to a straight line of a t (rel. time) vs u plot, for points
c     in the st-cluster such that t<u (demand physicsally plausible timing)
c     (*,1)-constant offset, (*,2)-slope
      real*8 rufptn_tyro_tfitpars(2,3)

c     here indext labels the reconstruction: 1-using lower, 2-using upper, 3-averaging b/w upper and lower
      real*8 rufptn_tyro_chi2(3) ! chi2 value for T vs U fit
      real*8 rufptn_tyro_ndof(3) ! # of d.o.f. for T vs U fit
      real*8 rufptn_tyro_theta(3) ! event zenith angle 
      real*8 rufptn_tyro_phi(3) ! event azimuthal angle 
      
