/*
 *
 * hpkt1_dst.h
 *
 * $Source: /hires_soft/uvm2k/bank/hpkt1_dst.h,v $
 * $Log: hpkt1_dst.h,v $
 * Revision 1.8  2000/09/13 23:19:25  jeremy
 * Added HR_TYPE_CUTEVENT packet type
 *
 * Revision 1.7  1998/11/06 01:06:45  thomas
 * *** empty log message ***
 *
 * Revision 1.6  1998/07/17  20:21:07  jeremy
 * Added HR_CALIB_FQDC[A|B] definitions
 *
 * Revision 1.5  1997/06/27  19:48:23  jeremy
 * Added calibPktType and noticePktType enumerations.
 *
 * Revision 1.4  1997/05/09  23:41:00  vtodd
 * changed event structure to a packed list, eliminated duplicate
 * snapshot structure so that only event is used
 *
 * Revision 1.3  1997/04/26  01:28:34  jui
 * added hpkt1_unpack_bank_hdr_, hpkt1_unpack_packet_hdr_
 * and hpkt1_unpack_packet_body_ plus removed hpkt1_bank_to_packet_
 * in conjunction with modification from rev 1.2 --> 1.3 for hpkt1_dst.c
 *
 * Revision 1.2  1997/04/24  18:18:26  jui
 * added hpkt1_bank_to_packet_ and hpkt1_packet_to_common_ prototypes
 * in conjunction with modification from rev 1.1 --> 1.2 for hpkt1_dst.c
 *
 * Revision 1.1  1997/04/16  22:37:16  vtodd
 * Initial revision
 *
 *
*/

#ifndef _HPKT1_
#define _HPKT1_

#define HPKT1_BANKID        15000
#define HPKT1_BANKVERSION   1

/*************************************************/
/* Define Packet Types--extra packet types are   */
/* required for alignment with hal.h definitions */

typedef enum packetType {
  HR_TYPE_PARTIAL, HR_TYPE_HIRES,     HR_TYPE_ACKNL,     HR_TYPE_CNTRL,
  HR_TYPE_MSTAT,   HR_TYPE_TIME,      HR_TYPE_EVENT,     HR_TYPE_SNAPSHOT,
  HR_TYPE_MINUTE,  HR_TYPE_THRESHOLD, HR_TYPE_COUNTRATE, HR_TYPE_VOLTS,
  HR_TYPE_NOTICE,  HR_TYPE_COMMAND,   HR_TYPE_REMOTE,
  HR_TYPE_CALIB,   HR_TYPE_BOARDID,   HR_TYPE_CUTEVENT
} packetType;

typedef enum calibPktType {
  HR_CALIB_END,  HR_CALIB_TDC,  HR_CALIB_QDC,   HR_CALIB_QDCA,
  HR_CALIB_QDCB, HR_CALIB_AQDC, HR_CALIB_AQDCA, HR_CALIB_AQDCB,
  HR_CALIB_FQDC, HR_CALIB_FQDCA, HR_CALIB_FQDCB
} calibPktType;

typedef enum noticePktType {
  HR_NOTE_PANNIC,  HR_NOTE_ALARM,  HR_NOTE_WARNING,  HR_NOTE_REBOOT,  HR_NOTE_STATUS,
  HR_NOTE_INHIBIT, HR_NOTE_PERMIT, HR_NOTE_STOP,     HR_NOTE_START,   HR_NOTE_CALIB,
  HR_NOTE_TTY,     HR_NOTE_SYSTEM, HR_NOTE_OPERATOR, HR_NOTE_COMMAND, HR_NOTE_GPS,
  HR_NOTE_WEATHER
} noticePktType;

/* #define  HPKT1_MAX_PKT_LEN   2588   Bug. Fixed 11-5-1998 by Stan Thomas.
                                       Size was too small. Should be 4096 
                                       to match DACQ maximum packet size. 
                                       Was clobbering timeQ[0].last when 
                                       large time packets were processed.
                                       */

#define  HPKT1_MAX_PKT_LEN   4096    /* presumes time packet is largest */
#define  HPKT1_MAX_TXT_LEN    512    /* maximum byte size of text array */
#define  HPKT1_MAX_MIR_EVENT  320    /* maximum number of events for a
					mirror between time packets     */
#define  REV3_VOLT_CHNLS      256    /* # of voltage channels for Rev3  */
#define  REV4_VOLT_CHNLS       16    /* # of voltage channels for Rev4  */



#ifdef __cplusplus
extern "C" {
#endif
integer4 hpkt1_common_to_bank_(void);
integer4 hpkt1_bank_to_dst_(integer4 *NumUnit);
integer4 hpkt1_common_to_dst_(integer4 *NumUnit);  /* combines above 2  */
integer4 hpkt1_bank_to_common_(integer1 *bank);
integer4 hpkt1_unpack_bank_hdr_(integer1 *bank);
integer4 hpkt1_unpack_packet_hdr_(integer1 *packet);
integer4 hpkt1_unpack_packet_body_(integer1 *packet);
integer4 hpkt1_packet_to_common_(integer1 *packet);
integer4 hpkt1_common_to_dump_(integer4 *long_output) ;
integer4 hpkt1_common_to_dumpf_(FILE* fp, integer4 *long_output);
/* get (packed) buffer pointer and size */
integer1* hpkt1_bank_buffer_ (integer4* hpkt1_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif

integer4 hpkt1_common_to_hraw1_(void);             /*  fills hraw1 bank */


/***********************************************/
/* Define common block structures beginning    */
/* with a structure to preserve the raw packet */

typedef struct
{
  integer4 bank_id;         /* bank id */
  integer4 bank_ver;        /* bank version */
  integer2 pktHdr_type;     /* packet type taken from packet header */
  integer2 pktHdr_crate;    /* crate id in packet header */
  integer2 pktHdr_id;       /* packet id in packet header */
  integer2 pktHdr_size;     /* packet size minus the packet header (8 bytes) */
  integer1 raw_pkt[HPKT1_MAX_PKT_LEN-8];  /* the raw packet minus the header */

} hpkt1_dst_raw;

extern hpkt1_dst_raw hpkt1_raw_;


/***********************************************/
/* Structure definition for hpkt1_dst_event    */

typedef struct
{
  integer4  event;                         /* event # */
  integer4  version;                       /* reflects electronics version */
  integer4  minute;                        /* minute into run of event */
  integer4  msec;                          /* millisecond into the minute */
  integer4  ntubes;                        /* number of tubes in event */
  integer4  tube_num[HR_UNIV_MIRTUBE];     /* array of tube numbers */
  integer2  thA[HR_UNIV_MIRTUBE];          /* threshold channel A */
  integer2  thB[HR_UNIV_MIRTUBE];          /* threshold channel B */
  integer2  qdcA[HR_UNIV_MIRTUBE];         /* chrg. to digital converter A */
  integer2  qdcB[HR_UNIV_MIRTUBE];         /* chrg. to digital converter B */
  integer2  tdc[HR_UNIV_MIRTUBE];          /* time of digital conversion */
  
} hpkt1_dst_event;

extern hpkt1_dst_event hpkt1_event_;


/*******************************************/
/* Structure definition for hpkt1_dst_time */

typedef struct
{
  integer2  year;                        /* year */
  integer2  day;                         /* day of year */
  integer4  sec;                         /* second into day < 86400 */
  integer4  freq;                        /* oscillator frequency */
  integer2  mark_error;                  /* time mark error (ns) */
  integer2  minute_offset;               /* mirror minute offset (seconds) */
  integer2  error_flags;                 /* timing error flags */
  integer2  events;                      /* number of mirror events */
  integer2  mirror[HPKT1_MAX_MIR_EVENT]; /* array of mirror number */
  integer2  msec[HPKT1_MAX_MIR_EVENT];   /* latched millisecond */
  integer4  nsec[HPKT1_MAX_MIR_EVENT];   /* nanoseconds after second */

} hpkt1_dst_time;

extern hpkt1_dst_time hpkt1_time_;


/*********************************************/
/* Structure definition for hpkt1_dst_minute */

typedef struct
{
  integer4  minute;         /* minute into part */
  integer4  trigs;          /* # of triggers in the minute */
  integer4  msec;           /* milliseconds in the minute */
  integer4  dead;           /* milliseconds of dead time during the minute */
     
} hpkt1_dst_minute;

extern hpkt1_dst_minute hpkt1_minute_;


/************************************************/
/* Structure definition for hpkt1_dst_countrate */

typedef struct
{
  integer4  min;                           /* minute into part */
  integer4  cntRateA[HR_UNIV_MIRTUBE];     /* count rate channel A */
  integer4  cntRateB[HR_UNIV_MIRTUBE];     /* count rate channel B */

} hpkt1_dst_countrate;

extern hpkt1_dst_countrate hpkt1_countrate_;


/************************************************/
/* Structure definition for hpkt1_dst_threshold */

typedef struct
{
  integer4  min;                        /* minute into part */
  integer4  thA[HR_UNIV_MIRTUBE];       /* threshold channel A */
  integer4  thB[HR_UNIV_MIRTUBE];       /* threshold channel B */

} hpkt1_dst_threshold;

extern hpkt1_dst_threshold hpkt1_threshold_;


/***********************************************/
/* Structure definition for hpkt1_dst_notice   */

typedef struct
{
  integer4  type;                    /* note/notice type */
  integer2  year;                    /* year */
  integer2  day;                     /* day */
  integer2  hour;                    /* hour */
  integer2  min;                     /* minute */
  integer2  sec;                     /* second */
  integer2  msec;                    /* millisecond */
  integer1  text[HPKT1_MAX_TXT_LEN]; /* note/notice text */

} hpkt1_dst_notice;

extern hpkt1_dst_notice hpkt1_notice_;


/***********************************************/
/* Structure definition for hpkt1_dst_remote   */

typedef struct
{
  integer1  tag[8];          /* source of command, destination of remote */
  integer1  text[HPKT1_MAX_TXT_LEN];    /* note/notice text */

} hpkt1_dst_remote;

extern hpkt1_dst_remote hpkt1_remote_;


/********************************************/
/* Structure definition for hpkt1_dst_calib */

typedef struct
{
  integer4  type;                       /* calibration type */
  integer4  count;                      /* # of samples */
  integer4  ampl;                       /* PPG pulse amplitude (mV) */
  integer4  width;                      /* PPG pulse width (nS) */
  integer4  period;                     /* PPG pulse period (mS) */
  integer4  delay;                      /* holdoff delay (50 nS) */
  integer2  mean[HR_UNIV_MIRTUBE];      /* mean */
  integer2  sdev[HR_UNIV_MIRTUBE];      /* standard deviation */

} hpkt1_dst_calib;

extern hpkt1_dst_calib hpkt1_calib_;


/********************************************/
/* Structure definition for hpkt1_dst_mstat */

typedef struct
{
  integer4  sent;      /* # of packets sent */
  integer4  resent;    /* # of packets resent */
  integer4  rcvd;      /* # of packets received */
  integer4  rercvd;    /* # of packets re-received */
  integer4  lost;      /* # of packets lost */
  integer4  errs;      /* # of dacq errors */
  integer4  warns;     /* # of dacq warnings */
  integer4  halts;     /* # of dacq halts */
  integer4  maxMsgs;   /* maximum number of buffered packets */

} hpkt1_dst_mstat;

extern hpkt1_dst_mstat hpkt1_mstat_;


/********************************************/
/* Structure definition for hpkt1_dst_volts */

typedef struct
{
  integer4  minute;          /* minute into part */
  integer2  obVer;           /* ommatidial board version (3 or 4) */
  integer2  hvChnls;         /* # hv channels  256 for Rev3 cluster, 16 Rev4 */
  integer2  ob_p12v[16];     /* ommatidial board positive 12 volts */
  integer2  ob_p05v[16];     /* ommatidial board positive  5 volts */
  integer2  ob_n12v[16];     /* ommatidial board negative 12 volts */
  integer2  ob_n05v[16];     /* ommatidial board negative  5 volts */
  integer2  ob_tdcRef[16];   /* ommatidial board tdc reference */
  integer2  ob_temp[16];     /* ommatidial board temperature reading */
  integer2  ob_thRef[16];    /* ommatidial board threshold reference */
  integer2  ob_gnd[16];      /* ommatidial board ground reading */
  integer2  garb_temp;       /* garbage board temperature */ 
  integer2  garb_p12v;       /* garbage board positive 12 volts */ 
  integer2  garb_n12v;       /* garbage board negative 12 volts */ 
  integer2  garb_p05v;       /* garbage board positive  5 volts */ 
  integer2  garb_s05v;       /* garbage board supply    5 volts */ 
  integer2  garb_lemo1;      /* garbage board */ 
  integer2  garb_anlIn;      /* garbage board */ 
  integer2  garb_clsVolts;   /* garbage board cluster voltage */ 
  integer2  garb_clsTemp;    /* garbage board cluster temperature*/ 
  integer2  garb_mirX;       /* garbage board */ 
  integer2  garb_mirY;       /* garbage board */ 
  integer2  garb_clsX;       /* garbage board */ 
  integer2  garb_clsY;       /* garbage board */ 
  integer2  garb_ns;         /* garbage board */ 
  integer2  garb_hvSup;      /* garbage board */ 
  integer2  garb_hvChnl;     /* garbage board */ 
  integer2  cluster[16];     /* cluster mux ADC channels */
  integer2  hv[256];         /* high voltages, 256 for Rev3, 16 for Rev4 */

} hpkt1_dst_volts;

extern hpkt1_dst_volts hpkt1_volts_;


/**********************************************/
/* Structure definition for hpkt1_dst_boardid */

typedef struct
{
  integer4  version;    /* 3=Rev3, 4=Rev4 */
  integer4  cpu;        /* cpu id */
  integer4  spare;      /* spare slot id */
  integer4  ppg;        /* ppg id */     
  integer4  trig;       /* trigger board id */
  integer4  ob[16];     /* ommatidial board */    
  integer4  garb;       /* garbage board */

} hpkt1_dst_boardid;

extern hpkt1_dst_boardid hpkt1_boardid_;


#endif









