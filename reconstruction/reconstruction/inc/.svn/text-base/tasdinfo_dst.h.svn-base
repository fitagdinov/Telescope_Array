/*
 *     Bank for each SD information
 *     written by a student
 *     Time-stamp: Sun Apr 12 00:18:40 2009 JST
*/

#ifndef _TASDINFO_
#define _TASDINFO_

#define TASDINFO_BANKID  13017
#define TASDINFO_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdinfo_common_to_bank_();
int tasdinfo_bank_to_dst_(int *NumUnit);
int tasdinfo_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdinfo_bank_to_common_(char *bank);
int tasdinfo_common_to_dump_(int *opt1) ;
int tasdinfo_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdinfo_bank_buffer_ (integer4* tasdinfo_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdinfo_nhmax   3   /* maximum number of host */
#define tasdinfo_ndmax 256   /* maximum number of detector */
#define tasdinfo_npmax  50   /* number of bins for led data */


typedef struct {
  int lid;
  unsigned int wlanidmsb; /* WLAN ID (MSB side) */
  unsigned int wlanidlsb; /* WLAN ID (LSB side) */
  int upmt_id;		  /* upper PMT serial number */
  int lpmt_id;		  /* lower PMT serial number */
  int ccid;		  /* charge controller ID */
  char box_id[20];	  /* scintillator box ID */
  char elecid[8];	  /* electronics ID */
  char gpsid[8];	  /* GPS ID */
  char cpldver[8];	  /* CPLD firmware version */
  char ccver[8];	  /* charge controller version */
  char firm_version[32];  /* CPU firmware version */

  short trig_mode0;       /* level-0 trigger mode */
  short trig_mode1;	  /* level-1 trigger mode */
  short uthre_lvl0;	  /* threshold of level-0 trigger of
			     upper layer*/
  short lthre_lvl0;	  /* threshold of level-0 trigger of
			     lower layer*/
  short uthre_lvl1;	  /* threshold of level-1 trigger of
			     upper layer*/
  short lthre_lvl1;	  /* threshold of level-1 trigger of
			     lower layer*/

  float uhv;		  /* applied voltage of upper PMT */
  float lhv;		  /* applied voltage of lower PMT */
  float upmtgain;	  /* upper PMT gain */
  float lpmtgain;	  /* lower PMT gain */
  float upmtgainError;    /* estimated error of upper PMT gain */
  float lpmtgainError;    /* estimated error of lower PMT gain */

  int lonmas;		  /* longitude [mas] */
  int latmas;		  /* latitude [mas] */
  int heicm;		  /* height [cm] */
  int lonmasSet;	  /* input longitude [mas] */
  int latmasSet;	  /* input latitude [mas] */
  int heicmSet;		  /* input height [cm] */
  float lonmasError;	  /* error of longitude */
  float latmasError;	  /* error of latitude */
  float heicmError;	  /* error of height */
  float delayns;	  /* signal cable delay */
  float ppsofs;		  /* PPS ofset */
  float ppsfluPH;	  /* PPS fluctuation in position hold
			     mode */
  float ppsflu3D;	  /* PPS fluctuation in position 3D fix
			     mode */

  float ucrrx[tasdinfo_npmax];   /* FADC scale for correction */
  float ucrry[tasdinfo_npmax];   /* correction factor - 1.0*/
  float ucrrsig[tasdinfo_npmax]; /* error of correction factor */
  float uhv_led;  /* applied voltage of upper PMT */
  int udec5p;		  /* maximun lineality range of upper layer
			     [FADC count] */

  float lcrrx[tasdinfo_npmax];   /* FADC scale for correction */
  float lcrry[tasdinfo_npmax];   /* correction factor - 1.0*/
  float lcrrsig[tasdinfo_npmax]; /* error of correction factor */
  float lhv_led;  /* applied voltage of lower PMT */
  int ldec5p;		  /* maximun lineality range of lower layer
			     [FADC count] */

  int error_flag;

} SDInfoSubData;


typedef struct {
  int ndet;  /* the number of detectors */
  int site;
  int dateFrom;
  int dateTo;
  int timeFrom;
  int timeTo;
  int first_run_id;
  int last_run_id;
  int year;

  SDInfoSubData sub[tasdinfo_ndmax];

  int footer;

} tasdinfo_dst_common;

extern tasdinfo_dst_common tasdinfo_;

#endif


