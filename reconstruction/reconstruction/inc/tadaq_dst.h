/*
 *     New Bank for raw fadc 1 values
 *     MRM July 18
*/

#ifndef _TADAQ_
#define _TADAQ_

#define TADAQ_BANKID  12134
#define TADAQ_BANKVERSION   000

#ifdef __cplusplus
extern "C" {
#endif
integer4 tadaq_common_to_bank_();
integer4 tadaq_bank_to_dst_(integer4 *NumUnit);
integer4 tadaq_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 tadaq_bank_to_common_(integer1 *bank);
integer4 tadaq_common_to_dump_(integer4 *opt1) ;
integer4 tadaq_common_to_dumpf_(FILE* fp,integer4 *opt2);
/* get (packed) buffer pointer and size */
integer1* tadaq_bank_buffer_ (integer4* tadaq_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif

integer4 tadaq_time_print_(integer4 *second, integer4 *clkcnt);
integer4 tadaq_time_fprint(FILE* fp, integer4 *second, integer4 *clkcnt);

#define tadaq_ncmax 12             /* number of cameras per site        */
#define tadaq_ntmax 256           /* number of tubes per camera        */
#define tadaq_nfadc 512         /* number of time bins per channel   */

typedef union {
  integer1 byteStream[24];

  struct {
    integer2 year;
    integer2 month;
    integer2 day;
    integer2 hour;
    integer2 minute;
    integer2 second;
    integer4 microsecond;
    integer4 run_id;
    integer4 trig_id;
  } entry;
} PCData;

typedef struct {
  PCData pc;

  /* ctd data */
  integer4                frame_ID;
  integer4                  trg_ID;
  integer4                sync_err;//synchronization error
  integer4               storefull;//status of camera's buffer (full or not)
  integer4                   code1;//array of trigger code 1
  integer4                   code2;//array of trigger code 2
  integer4                   code3;//array of trigger code 3 
  integer4                   code4;//array of trigger code 4
  integer4                trg_mask;//trigger mask for data suppress
  integer4                   swadr;//block ram's high address
  integer4                 tf_mask;//tf mask depending on trigger rate
  integer4                   store;//status of CTD's buffer
  integer4                frm_mdyy;//date(month day year)
  integer4                 frm_hms;//date(hour minute second)
  integer4             frm_time_ID;//number of GPS 1PPS from reset pulse
  integer4             CTD_version;//firmware virsion
  integer4         frm_PPS_counter;//timing of GPS1PPS pulse
  integer4             rst_counter;//timing of reset pulse
  integer4             trg_counter;//timing of final trigger pulse
  integer4           extrg_counter;//timing of external trigger pulse
  integer4              dt_counter;//number of deadtime frames

  /* ctd info */
  integer4       info_gps1pps[256];//gps 1pps counteres array
  integer4   triggered_frame[1024];//triggered frames including deadtime
  integer4 triggered_tf_mask[1024];//tf mask with triggered frames including deadtime
  integer4                rate[12];//trigger rate
  integer4                latitude;//position of site
  integer4               longitude;//position of site
  integer4                   hight;//position of site
  integer4             initial_hms;//hour minute second at reset pulse
  integer4             deadtime_st;//time ID      at the start of deadtime
  integer4           deadPPScnt_st;//1pps counter at the start of deadtime
  integer4              deadfrm_st;//frame ID     at the start of deadtime
  integer4            deadtime_end;//time ID      at the start of deadtime
  integer4          deadPPScnt_end;//1pps counter at the start of deadtime
  integer4             deadfrm_end;//frame ID     at the start of deadtime

} CTDData;

typedef struct {
  integer2 waveform[tadaq_nfadc];
  integer2 hit[16];
  integer2 pmt_id;  //ID of PMT
  integer2 version; //firmware version
  integer2 mean[4]; //average of background. m6ms ago, mean[2] is 53ms ago, mean[3] is 79ms ago.
  integer2 disp[4]; //average of background. d6ms ago, disp[2] is 53ms ago, disp[3] is 79ms ago.
  integer2 trgmnt;  //timing of trigger pulse
  integer2 peak;    //peak timing of input pul
  integer2 frmmnt;  //timing of frame pulse
  integer2 tmphit;  //internal value for trigg
  integer2 trgcnt;  //number of trigger from TF
  integer2 mode;    //run mode
  integer2 ctrl;    //communication mode
  integer2 thre;    //for threshold and trigger mode
  integer2 rstmnt;  //timing of reset pulse
  integer2 rate[7];

  integer2 state;   //buffer state
  real4 hitthre;    //hit threshold
  real4 ncthre;     //nc threshold
  real4 average;
  real4 stdev;
  integer2 tf_hit;
} SDFData;

typedef struct {
  PCData pc;

  integer2 id;   // Camera number id
  integer2 ntube;

  integer4 frame_ID;
  integer4 sync_monitor;
  integer4 trg_ID;
  integer4 trg_mask;
  integer4 store;
  integer4 sec_trig_code;
  integer4 tftrg_ID;
  integer4 swadr;
  integer4 board_mask;
  integer4 mode;
  integer4 mode2;
  integer4 hit_pt[tadaq_ntmax];  // modified
  integer4 nc_bit[16];
  integer4 version;
  integer4 reset_counter;
  integer4 trigger_counter;
  integer4 extrigger_counter;

  SDFData sdf[tadaq_ntmax];

} TFData;

typedef struct {
  /*
   * critical stats
   */
  integer2 event_code;                 /* 1= normal, others to be added     */
  integer2 site;
  integer4 runid;
  integer4 trigid;
  integer2 ncameras;                   /* number of participating cameras   */

  integer4 year;
  integer4 month;
  integer4 day;
  integer4 hour;
  integer4 minute;
  integer4 second;
  integer4 nanosecond;

  /*
   * CTD data
   */
  CTDData ctd;

  /*
   * TF data
   */
  TFData tf[tadaq_ncmax];

} tadaq_dst_common;

extern tadaq_dst_common tadaq_;

#endif


