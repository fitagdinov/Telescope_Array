c     Bank for SD pass2 data
c     Dmitri Ivanov <dmiivanov@gmail.com>
c     Mar 14, 2009
      
c     Last Modified: May 16, 2019
      
      integer RUFLDF_BANKID
      parameter (RUFLDF_BANKID=13107)
      integer RUFLDF_BANKVERSION
      parameter (RUFLDF_BANKVERSION=001)
      
      
      integer*4 rufldf_common_to_bank
      external  rufldf_common_to_bank
      integer*4 rufldf_bank_to_dst
      external  rufldf_bank_to_dst
      integer*4 rufldf_common_to_dst
      external  rufldf_common_to_dst
      integer*4 rufldf_bank_to_common
      external  rufldf_bank_to_common
      integer*4 rufldf_common_to_dump
      external  rufldf_common_to_dump
      integer*4 rufldf_common_to_dumpf
      external  rufldf_common_to_dumpf



      common /rufldf/
     &     rufldf_xcore,rufldf_dxcore,rufldf_ycore,rufldf_dycore,
     &	   rufldf_sc,rufldf_dsc,rufldf_s600,rufldf_s600_0,rufldf_s800,
     &	   rufldf_s800_0,rufldf_aenergy,rufldf_energy,rufldf_atmcor,
     &     rufldf_chi2,rufldf_theta,rufldf_dtheta,rufldf_phi,
     &     rufldf_dphi,rufldf_t0, rufldf_dt0,rufldf_bdist,
     &     rufldf_tdistbr,rufldf_tdistlr,rufldf_tdistsk,rufldf_tdist,
     &     rufldf_ndof
      
      
      
c     (1): LDF alone fit, (2): combined LDF and geometry fitting
      real*8 rufldf_xcore(2)
      real*8 rufldf_dxcore(2)
      real*8 rufldf_ycore(2)
      real*8 rufldf_dycore(2)
      real*8 rufldf_sc(2)       ! Scalinig constant in front of LDF
      real*8 rufldf_dsc(2)
      
      real*8 rufldf_s600(2)
      real*8 rufldf_s600_0(2)
      real*8 rufldf_s800(2)
      real*8 rufldf_s800_0(2)
      real*8 rufldf_aenergy(2)  ! AGASA formula energy, EeV
      real*8 rufldf_energy(2)   ! Rutgers formula energy, EeV
      real*8 rufldf_atmcor(2)   ! Energy atmopsheric correction factor that was applied
      real*8 rufldf_chi2(2)
      
c     These variables are for combined geom + LDF fit      
      real*8 rufldf_theta
      real*8 rufldf_dtheta
      real*8 rufldf_phi
      real*8 rufldf_dphi
      real*8 rufldf_t0
      real*8 rufldf_dt0
      
      
c     These variables are required for efficiency / exposure
      
      
      
c     Distance of the shower core from a closes SD array edge boundary.
c     If it is negative, then the core is outside of the array
      real*8 rufldf_bdist
      
c     Distance of the shower core from the closest T-shape bounday for BR,LR,SK
c     At most only one such distance is non-negagtive, as the shower core can
c     hit only one of the subarrays.  If all distances are negative, this means that
c     the shower core either hits outside of the array or outside of BR,LR,SK subarrays
      
      real*8 rufldf_tdistbr
      real*8 rufldf_tdistlr
      real*8 rufldf_tdistsk
      real*8 rufldf_tdist       ! Distance to a T-shape boundary
      
      
c     (1): LDF alone fit, (2): LDF and geometry combined fit
      integer*4 rufldf_ndof(2)
      
