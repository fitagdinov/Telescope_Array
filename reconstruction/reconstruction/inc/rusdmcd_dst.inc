      

      integer RUSDMCD_BANKID  
      parameter(RUSDMCD_BANKID=13108)
      integer RUSDMCD_BANKVERSION
      parameter (RUSDMCD_BANKVERSION=0)

c     Max. number of sds / event
      integer RUSDMCD_MSDS
      parameter (RUSDMCD_MSDS=256)
     
      integer*4 rusdmcd_common_to_bank
      external  rusdmcd_common_to_bank
      integer*4 rusdmcd_bank_to_dst
      external  rusdmcd_bank_to_dst
      integer*4 rusdmcd_common_to_dst
      external  rusdmcd_common_to_dst
      integer*4 rusdmcd_bank_to_common
      external  rusdmcd_bank_to_common
      integer*4 rusdmcd_common_to_dump
      external  rusdmcd_common_to_dump
      integer*4 rusdmcd_common_to_dumpf
      external  rusdmcd_common_to_dumpf

      common /rusdmcd/
     &     rusdmcd_nsds,rusdmcd_xxyy,rusdmcd_igsd,rusdmcd_edep


      integer*4 rusdmcd_nsds ! number of sds in event
      integer*4 rusdmcd_xxyy(RUSDMCD_MSDS) ! sd position ID
      integer*4 rusdmcd_igsd(RUSDMCD_MSDS) ! is part of the shower? 1- yes, 0-no
      real*8    rusdmcd_edep(2,RUSDMCD_MSDS) ! energy deposits, MeV, (1,*) - lower, (2,*) - upper
      
