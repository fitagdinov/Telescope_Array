C     Created 2000/04/22 DRB

C     Bank Bookkeeping

      integer*4 FSCN1_BANKID,FSCN1_BANKVERSION
      parameter(FSCN1_BANKID = 12013)
      parameter(FSCN1_BANKVERSION = 000)

C     Functions

      integer*4 fscn1_common_to_bank
      external  fscn1_common_to_bank
      integer*4 fscn1_bank_to_dst
      external  fscn1_bank_to_dst
      integer*4 fscn1_common_to_dst
      external  fscn1_common_to_dst
      integer*4 fscn1_bank_to_common
      external  fscn1_bank_to_common
      integer*4 fscn1_common_to_dump
      external  fscn1_common_to_dump
      integer*4 fscn1_common_to_dumpf
      external  fscn1_common_to_dumpf

      common /fscn1/
     1     fscn1_ntube,
     2     fscn1_mir,fscn1_tube,fscn1_ig,fscn1_ped,fscn1_pedrms,
     3     fscn1_pamp,fscn1_pmaxt,fscn1_pnpe,fscn1_pt0,
     4     fscn1_pnt,fscn1_ptav,fscn1_pfilt

      integer*4 fscn1_ntube
      integer*4 fscn1_mir(HR_UNIV_MAXTUBE)    ! mir number  
      integer*4 fscn1_tube(HR_UNIV_MAXTUBE)   ! tube number 
      integer*4 fscn1_ig(HR_UNIV_MAXTUBE)     ! tube flag
                                    ! ig=1: good tube  
                                    ! ig=0: rejected by scan or has no signal 
      real*4    fscn1_ped(HR_UNIV_MAXTUBE)    ! pedestal 
      real*4    fscn1_pedrms(HR_UNIV_MAXTUBE) ! RMS of pedestal 
      integer*4 fscn1_pamp(HR_UNIV_MAXTUBE)   ! Max amp of filtered pulse 
      integer*4 fscn1_pmaxt(HR_UNIV_MAXTUBE)  ! Time index of max 
      real*4    fscn1_pnpe(HR_UNIV_MAXTUBE)   ! Integrated pulse above ped 
      integer*4 fscn1_pt0(HR_UNIV_MAXTUBE)    ! Index of first slice in pulse 
      integer*4 fscn1_pnt(HR_UNIV_MAXTUBE)    ! Number of slices in pulse 
      real*4    fscn1_ptav(HR_UNIV_MAXTUBE)   ! Weighted average time 
      real*4    fscn1_pfilt(HR_UNIV_MAXTUBE)  ! Filter time scale in slices 



