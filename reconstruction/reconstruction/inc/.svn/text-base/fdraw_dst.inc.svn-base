c Genericized BRM/LR raw data
c LMS 2008/10

      integer*4 FDRAW_BANKID,FDRAW_BANKVERSION
      parameter(FDRAW_BANKID=12092)
      parameter(FDRAW_BANKVERSION=000)

      integer*4 fdraw_common_to_bank
      external  fdraw_common_to_bank
      integer*4 fdraw_bank_to_dst
      external  fdraw_bank_to_dst
      integer*4 fdraw_common_to_dst
      external  fdraw_common_to_dst
      integer*4 fdraw_bank_to_common
      external  fdraw_bank_to_common
      integer*4 fdraw_common_to_dump
      external  fdraw_common_to_dump
      integer*4 fdraw_common_to_dumpf
      external  fdraw_common_to_dumpf

      integer*4 fdraw_nmir_max,fdraw_nchan_mir,fdraw_nt_chan_max
      parameter(fdraw_nmir_max=12)
      parameter(fdraw_nchan_mir=256)
      parameter(fdraw_nt_chan_max=512)

      common /fdraw/
     &   fdraw_event_code,fdraw_part,fdraw_num_mir,fdraw_event_num,
     &   fdraw_julian,fdraw_jsecond,fdraw_gps1pps_tick,
     &   fdraw_ctdclock,
     &   fdraw_ctd_version,fdraw_tf_version,fdraw_sdf_version,
     &   fdraw_trig_code(fdraw_nmir_max),
     &   fdraw_second(fdraw_nmir_max),
     &   fdraw_microsec(fdraw_nmir_max),
     &   fdraw_clkcnt(fdraw_nmir_max),
     &   fdraw_mir_num(fdraw_nmir_max),
     &   fdraw_num_chan(fdraw_nmir_max),
     &   fdraw_tf_mode(fdraw_nmir_max),
     &   fdraw_tf_mode2(fdraw_nmir_max),
     &   fdraw_hit_pt(fdraw_nchan_mir+1,fdraw_nmir_max),
     &   fdraw_channel(fdraw_nchan_mir,fdraw_nmir_max),
     &   fdraw_sdf_peak(fdraw_nchan_mir,fdraw_nmir_max),
     &   fdraw_sdf_tmphit(fdraw_nchan_mir,fdraw_nmir_max),
     &   fdraw_sdf_mode(fdraw_nchan_mir,fdraw_nmir_max),
     &   fdraw_sdf_ctrl(fdraw_nchan_mir,fdraw_nmir_max),
     &   fdraw_sdf_thre(fdraw_nchan_mir,fdraw_nmir_max),
     &   fdraw_mean(4,fdraw_nchan_mir,fdraw_nmir_max),
     &   fdraw_disp(4,fdraw_nchan_mir,fdraw_nmir_max),
     &   fdraw_m_fadc(fdraw_nt_chan_max,
     &                fdraw_nchan_mir,fdraw_nmir_max)

      integer*2 fdraw_event_code     ! 1=normal, 0=monte carlo
      integer*2 fdraw_part           ! = run_id % 100
      integer*4 fdraw_num_mir        ! number of participating cameras
      integer*4 fdraw_event_num      ! trigger id number
  
c CTD trigger time   
      integer*4 fdraw_julian         ! julian day
      integer*4 fdraw_jsecond        ! second into julian day
      integer*4 fdraw_gps1pps_tick   ! last 1pps tick from gps
      integer*4 fdraw_ctdclock       ! ctd 40MHz clock tick

c Hardware version info
      integer*4 fdraw_ctd_version
      integer*4 fdraw_tf_version
      integer*4 fdraw_sdf_version

c selected TF data
      integer*4 fdraw_trig_code      ! tf trigger code:                  
                                     !   0 = not a primary trigger       
                                     !   1 = primary trigger             
                                     !   2 = joint trigger               
                                     !   3, 4 = very large signals       
      integer*4 fdraw_second         ! camera store time rel. to 0:00 UT 
      integer*4 fdraw_microsec       ! microsec of store time          
      integer*4 fdraw_clkcnt         ! camera 40 MHz clock tick          
 
      integer*2 fdraw_mir_num        ! mirror id number (0-11)           
      integer*2 fdraw_num_chan       ! number of channels with FADC data 

      integer*4 fdraw_tf_mode
      integer*4 fdraw_tf_mode2

c array of triggered tubes by camera (idx 0-255 are tubes, 256 is empty) 
      integer*2 fdraw_hit_pt

c selected SDF data
c channel ID number 
      integer*2 fdraw_channel

c peak timing of input pulse 
      integer*2 fdraw_sdf_peak
c internal value for trigg 
      integer*2 fdraw_sdf_tmphit
c run mode 
      integer*2 fdraw_sdf_mode
c communication mode 
      integer*2 fdraw_sdf_ctrl
c for threshold and trigger mode 
      integer*2 fdraw_sdf_thre 

c average of bkgnd. 0ms, 6ms, 53ms, and 79ms ago. 
      integer*2 fdraw_mean 

c rms of bkgnd. 0ms, 6ms, 53ms, and 79ms ago. 
      integer*2 fdraw_disp 

c raw 14-bit fadc data 
      integer*2 fdraw_m_fadc

