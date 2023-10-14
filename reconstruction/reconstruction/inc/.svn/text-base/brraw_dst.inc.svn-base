c Created 2008/03/24 LMS

      integer*4 BRRAW_BANKID,BRRAW_BANKVERSION
      parameter(BRRAW_BANKID=12102)
      parameter(BRRAW_BANKVERSION=000)

      integer*4 brraw_common_to_bank
      external  brraw_common_to_bank
      integer*4 brraw_bank_to_dst
      external  brraw_bank_to_dst
      integer*4 brraw_common_to_dst
      external  brraw_common_to_dst
      integer*4 brraw_bank_to_common
      external  brraw_bank_to_common
      integer*4 brraw_common_to_dump
      external  brraw_common_to_dump
      integer*4 brraw_common_to_dumpf
      external  brraw_common_to_dumpf

      integer*4 brraw_nmir_max,brraw_nchan_mir,brraw_nt_chan_max
      parameter(brraw_nmir_max=12)
      parameter(brraw_nchan_mir=256)
      parameter(brraw_nt_chan_max=512)

      common /brraw/
     &   brraw_event_code,brraw_part,brraw_num_mir,brraw_event_num,
     &   brraw_julian,brraw_jsecond,brraw_gps1pps_tick,
     &   brraw_ctdclock,
     &   brraw_ctd_version,brraw_tf_version,brraw_sdf_version,
     &   brraw_trig_code(brraw_nmir_max),
     &   brraw_second(brraw_nmir_max),
     &   brraw_microsec(brraw_nmir_max),
     &   brraw_clkcnt(brraw_nmir_max),
     &   brraw_mir_num(brraw_nmir_max),
     &   brraw_num_chan(brraw_nmir_max),
     &   brraw_tf_mode(brraw_nmir_max),
     &   brraw_tf_mode2(brraw_nmir_max),
     &   brraw_hit_pt(brraw_nchan_mir+1,brraw_nmir_max),
     &   brraw_channel(brraw_nchan_mir,brraw_nmir_max),
     &   brraw_sdf_peak(brraw_nchan_mir,brraw_nmir_max),
     &   brraw_sdf_tmphit(brraw_nchan_mir,brraw_nmir_max),
     &   brraw_sdf_mode(brraw_nchan_mir,brraw_nmir_max),
     &   brraw_sdf_ctrl(brraw_nchan_mir,brraw_nmir_max),
     &   brraw_sdf_thre(brraw_nchan_mir,brraw_nmir_max),
     &   brraw_mean(4,brraw_nchan_mir,brraw_nmir_max),
     &   brraw_disp(4,brraw_nchan_mir,brraw_nmir_max),
     &   brraw_m_fadc(brraw_nt_chan_max,
     &                brraw_nchan_mir,brraw_nmir_max)

      integer*2 brraw_event_code     ! 1=normal, 0=monte carlo
      integer*2 brraw_part           ! = run_id % 100
      integer*4 brraw_num_mir        ! number of participating cameras
      integer*4 brraw_event_num      ! trigger id number
  
c CTD trigger time   
      integer*4 brraw_julian         ! julian day
      integer*4 brraw_jsecond        ! second into julian day
      integer*4 brraw_gps1pps_tick   ! last 1pps tick from gps
      integer*4 brraw_ctdclock       ! ctd 40MHz clock tick

c Hardware version info
      integer*4 brraw_ctd_version
      integer*4 brraw_tf_version
      integer*4 brraw_sdf_version

c selected TF data
      integer*4 brraw_trig_code      ! tf trigger code:                  
                                     !   0 = not a primary trigger       
                                     !   1 = primary trigger             
                                     !   2 = joint trigger               
                                     !   3, 4 = very large signals       
      integer*4 brraw_second         ! camera store time rel. to 0:00 UT 
      integer*4 brraw_microsec       ! microsec of store time          
      integer*4 brraw_clkcnt         ! camera 40 MHz clock tick          
 
      integer*2 brraw_mir_num        ! mirror id number (0-11)           
      integer*2 brraw_num_chan       ! number of channels with FADC data 

      integer*4 brraw_tf_mode
      integer*4 brraw_tf_mode2

c array of triggered tubes by camera (idx 0-255 are tubes, 256 is empty) 
      integer*2 brraw_hit_pt

c selected SDF data
c channel ID number 
      integer*2 brraw_channel

c peak timing of input pulse 
      integer*2 brraw_sdf_peak
c internal value for trigg 
      integer*2 brraw_sdf_tmphit
c run mode 
      integer*2 brraw_sdf_mode
c communication mode 
      integer*2 brraw_sdf_ctrl
c for threshold and trigger mode 
      integer*2 brraw_sdf_thre 

c average of bkgnd. 0ms, 6ms, 53ms, and 79ms ago. 
      integer*2 brraw_mean 

c rms of bkgnd. 0ms, 6ms, 53ms, and 79ms ago. 
      integer*2 brraw_disp 

c raw 14-bit fadc data 
      integer*2 brraw_m_fadc

