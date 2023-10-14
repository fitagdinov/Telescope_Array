c Created 2008/03/24 LMS

      integer*4 LRRAW_BANKID,LRRAW_BANKVERSION
      parameter(LRRAW_BANKID=12201)
      parameter(LRRAW_BANKVERSION=001)

      integer*4 lrraw_common_to_bank
      external  lrraw_common_to_bank
      integer*4 lrraw_bank_to_dst
      external  lrraw_bank_to_dst
      integer*4 lrraw_common_to_dst
      external  lrraw_common_to_dst
      integer*4 lrraw_bank_to_common
      external  lrraw_bank_to_common
      integer*4 lrraw_common_to_dump
      external  lrraw_common_to_dump
      integer*4 lrraw_common_to_dumpf
      external  lrraw_common_to_dumpf

      integer*4 lrraw_nmir_max,lrraw_nchan_mir,lrraw_nt_chan_max
      parameter(lrraw_nmir_max=12)
      parameter(lrraw_nchan_mir=256)
      parameter(lrraw_nt_chan_max=512)

      common /lrraw/
     &   lrraw_event_code,lrraw_part,lrraw_num_mir,lrraw_event_num,
     &   lrraw_julian,lrraw_jsecond,lrraw_gps1pps_tick,
     &   lrraw_ctdclock,
     &   lrraw_ctd_version,lrraw_tf_version,lrraw_sdf_version,
     &   lrraw_trig_code(lrraw_nmir_max),
     &   lrraw_second(lrraw_nmir_max),
     &   lrraw_microsec(lrraw_nmir_max),
     &   lrraw_clkcnt(lrraw_nmir_max),
     &   lrraw_mir_num(lrraw_nmir_max),
     &   lrraw_num_chan(lrraw_nmir_max),
     &   lrraw_tf_mode(lrraw_nmir_max),
     &   lrraw_tf_mode2(lrraw_nmir_max),
     &   lrraw_hit_pt(lrraw_nchan_mir+1,lrraw_nmir_max),
     &   lrraw_channel(lrraw_nchan_mir,lrraw_nmir_max),
     &   lrraw_sdf_peak(lrraw_nchan_mir,lrraw_nmir_max),
     &   lrraw_sdf_tmphit(lrraw_nchan_mir,lrraw_nmir_max),
     &   lrraw_sdf_mode(lrraw_nchan_mir,lrraw_nmir_max),
     &   lrraw_sdf_ctrl(lrraw_nchan_mir,lrraw_nmir_max),
     &   lrraw_sdf_thre(lrraw_nchan_mir,lrraw_nmir_max),
     &   lrraw_mean(4,lrraw_nchan_mir,lrraw_nmir_max),
     &   lrraw_disp(4,lrraw_nchan_mir,lrraw_nmir_max),
     &   lrraw_m_fadc(lrraw_nt_chan_max,
     &                lrraw_nchan_mir,lrraw_nmir_max)

      integer*2 lrraw_event_code     ! 1=normal, 0=monte carlo
      integer*2 lrraw_part           ! = run_id % 100
      integer*4 lrraw_num_mir        ! number of participating cameras
      integer*4 lrraw_event_num      ! trigger id number
  
c CTD trigger time   
      integer*4 lrraw_julian         ! julian day
      integer*4 lrraw_jsecond        ! second into julian day
      integer*4 lrraw_gps1pps_tick   ! last 1pps tick from gps
      integer*4 lrraw_ctdclock       ! ctd 40MHz clock tick

c Hardware version info
      integer*4 lrraw_ctd_version
      integer*4 lrraw_tf_version
      integer*4 lrraw_sdf_version

c selected TF data
      integer*4 lrraw_trig_code      ! tf trigger code:                  
                                     !   0 = not a primary trigger       
                                     !   1 = primary trigger             
                                     !   2 = joint trigger               
                                     !   3, 4 = very large signals       
      integer*4 lrraw_second         ! camera store time rel. to 0:00 UT 
      integer*4 lrraw_microsec       ! microsec of store time          
      integer*4 lrraw_clkcnt         ! camera 40 MHz clock tick          
 
      integer*2 lrraw_mir_num        ! mirror id number (0-11)           
      integer*2 lrraw_num_chan       ! number of channels with FADC data 

      integer*4 lrraw_tf_mode
      integer*4 lrraw_tf_mode2

c array of triggered tubes by camera (idx 0-255 are tubes, 256 is empty) 
      integer*2 lrraw_hit_pt

c selected SDF data
c channel ID number 
      integer*2 lrraw_channel

c peak timing of input pulse 
      integer*2 lrraw_sdf_peak
c internal value for trigg 
      integer*2 lrraw_sdf_tmphit
c run mode 
      integer*2 lrraw_sdf_mode
c communication mode 
      integer*2 lrraw_sdf_ctrl
c for threshold and trigger mode 
      integer*2 lrraw_sdf_thre 

c average of bkgnd. 0ms, 6ms, 53ms, and 79ms ago. 
      integer*2 lrraw_mean 

c rms of bkgnd. 0ms, 6ms, 53ms, and 79ms ago. 
      integer*2 lrraw_disp 

c raw 14-bit fadc data 
      integer*2 lrraw_m_fadc

