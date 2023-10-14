
*
*     TRUMP MC DST Bank.
*     DRB 20 January 2009
*     SS 2 July 2009  --- converted to fortran .inc
*

      integer TRUMPMC_BANKID
      parameter(TRUMPMC_BANKID=12803)
      integer TRUMPMC_BANKVERSION
      parameter(TRUMPMC_BANKVERSION=1)

C     Parameters 
      integer TRUMPMC_MAXSITES
      integer TRUMPMC_MAXDEPTHS
      integer TRUMPMC_MAXMIRRORS
      integer TRUMPMC_MAXTUBES
      parameter(TRUMPMC_MAXSITES=3)
      parameter(TRUMPMC_MAXDEPTHS=200)
      parameter(TRUMPMC_MAXMIRRORS=14)
      parameter(TRUMPMC_MAXTUBES=256)

C Short-hand parameters
      integer MXS
      integer MXD
      integer MXM
      integer MXT
      parameter(MXS=TRUMPMC_MAXSITES)
      parameter(MXD=TRUMPMC_MAXDEPTHS)
      parameter(MXM=TRUMPMC_MAXMIRRORS)
      parameter(MXT=TRUMPMC_MAXTUBES)

      external trumpmc_common_to_bank
      external trumpmc_bank_to_dst
      external trumpmc_common_to_dst
      external trumpmc_bank_to_common
      external trumpmc_common_to_dump
      external trumpmc_common_to_dumpf
      integer*4 trumpmc_common_to_bank
      integer*4 trumpmc_bank_to_dst
      integer*4 trumpmc_common_to_dst
      integer*4 trumpmc_bank_to_common
      integer*4 trumpmc_common_to_dump
      integer*4 trumpmc_common_to_dumpf

      common /trumpmc/
     &     trumpmc_impact_point,trumpmc_shower_vector,trumpmc_energy,
     &     trumpmc_primary,trumpmc_gh_parm,trumpmc_n_sites,
     &     trumpmc_siteid,trumpmc_site_location,trumpmc_psi,trumpmc_rp,
     &     trumpmc_n_depths,trumpmc_depth,trumpmc_n_mirrors,
     &     trumpmc_mirror,trumpmc_fluo_flux,trumpmc_aero_flux,
     &     trumpmc_rayl_flux,trumpmc_dirc_flux,trumpmc_total_npe_mirror,
     &     trumpmc_n_tubes,trumpmc_tube,trumpmc_ave_time,
     &     trumpmc_total_npe,trumpmc_julian,trumpmc_jsec,trumpmc_nano

C     Shower generic information
      real*4 trumpmc_impact_point(3) ! Vector from origin to impact point (m)
      real*4 trumpmc_shower_vector(3) ! Unit vector of shower propagation vector
      real*4 trumpmc_energy     ! Energy of primary (eV)
      integer*4 trumpmc_primary ! Particle type of primary
      real*4 trumpmc_gh_parm(4) ! G-H parameters: x0, xmax, nmax, lambda (g/cm2 or part.)

C     Site specific information: geometry
      integer*4 trumpmc_n_sites ! Number of sites
      integer*4 trumpmc_siteid(MXS) ! site ID (BR=0, LR=1, MD=2)
      real*4 trumpmc_site_location(3,MXS) ! Position of site wrt origin(m)
      real*4 trumpmc_psi(MXS)   ! Psi angle as seen from site
      real*4 trumpmc_rp(3,MXS) ! Vector from site to Rp point (m)

C     Depth information: Flux (photons/m2/rad)
      integer*4 trumpmc_n_depths ! Number of depth points this mirror
      real*4 trumpmc_depth(MXD) ! Depth (g/cm2)
      integer*4 trumpmc_n_mirrors(MXS) ! Number of mirrors which see track
      integer*4 trumpmc_mirror(MXM,MXS) ! Mirror number
      real*4 trumpmc_fluo_flux(MXD,MXM,MXS) ! Fluorescence phot
      real*4 trumpmc_aero_flux(MXD,MXM,MXS) ! Aerosol/Mie scattered phot
      real*4 trumpmc_rayl_flux(MXD,MXM,MXS) ! Rayleigh scattered phot
      real*4 trumpmc_dirc_flux(MXD,MXM,MXS) ! Direct Cerenkov phot
  
C     Tube information
      integer*4 trumpmc_total_npe_mirror(MXM,MXS) ! Total number of NPE, this mirror
      integer*4 trumpmc_n_tubes(MXM,MXS) ! Number of tubes, this mirror, with NPE
      integer*4 trumpmc_tube(MXT,MXM,MXS) ! Tube number
      real*4 trumpmc_ave_time(MXT,MXM,MXS) ! Average time of PE in this tube
      integer*4 trumpmc_total_npe(MXT,MXM,MXS) ! Number of PE in this tube

C     New in version 1: core time
      integer*4 trumpmc_julian
      integer*4 trumpmc_jsec
      integer*4 trumpmc_nano
      
      external trumpmc_struct_to_abank
      external trumpmc_abank_to_dst
      external trumpmc_struct_to_dst
      external trumpmc_abank_to_struct
      external trumpmc_struct_to_dump
      external trumpmc_struct_to_dumpf
      integer*4 trumpmc_struct_to_abank
      integer*4 trumpmc_abank_to_dst
      integer*4 trumpmc_struct_to_dst
      integer*4 trumpmc_abank_to_struct
      integer*4 trumpmc_struct_to_dump
      integer*4 trumpmc_struct_to_dumpf

