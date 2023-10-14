/*
 *     Bank for each SD constrmation
 *     written by a student
 *     Time-stamp: Sun May 03 11:06:17 2009 JST
*/

#ifndef _TASDCONST_
#define _TASDCONST_

#define TASDCONST_BANKID  13023
#define TASDCONST_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdconst_common_to_bank_();
int tasdconst_bank_to_dst_(int *NumUnit);
int tasdconst_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdconst_bank_to_common_(char *bank);
int tasdconst_common_to_dump_(int *opt1) ;
int tasdconst_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdconst_bank_buffer_ (integer4* tasdconst_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdconst_nledp  50  /* number of bins for led data */
#define tasdconst_nmiph 128  /* number of bins for mip histogram */
#define tasdconst_nmipl  33  /* number of bins for mip linearity */

#define taSiteCentorLongitude -112.90875 /* degree */
#define taSiteCentorLatitude    39.29693 /* degree */
#define taSiteCentorElevation    1.382e5 /* cm */


typedef struct {
  int lid;

  int site;
  int dateFrom;
  int dateTo;
  int timeFrom;
  int timeTo;
  int first_run_id;
  int last_run_id;


  unsigned int wlanidmsb; /* WLAN ID (MSB side) */
  unsigned int wlanidlsb; /* WLAN ID (LSB side) */
  int upmt_id;		  /* upper PMT serial number */
  int lpmt_id;		  /* lower PMT serial number */

  int trig_mode0;	  /* level-0 trigger mode */
  int trig_mode1;	  /* level-1 trigger mode */
  int uthre_lvl0;	  /* threshold of level-0 trigger of
			     upper layer*/
  int lthre_lvl0;	  /* threshold of level-0 trigger of
			     lower layer*/
  int uthre_lvl1;	  /* threshold of level-1 trigger of
			     upper layer*/
  int lthre_lvl1;	  /* threshold of level-1 trigger of
			     lower layer*/

  float uhv;		  /* applied voltage of upper PMT */
  float lhv;		  /* applied voltage of lower PMT */
  float upmtgain;	  /* upper PMT gain */
  float lpmtgain;	  /* lower PMT gain */

  float posX;
  float posY;
  float posZ;
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
  float ppsfluPH; /* PPS fluctuation in position hold mode */
  float ppsflu3D; /* PPS fluctuation in position 3D fix mode */


  int udec5pled;  /* maximun lineality range of upper layer
		     [FADC count] */
  int ldec5pled;  /* maximun lineality range of lower layer
		     [FADC count] */
  float uhv_led;  /* applied voltage of upper PMT */
  float lhv_led;  /* applied voltage of lower PMT */
  float ucrrx[tasdconst_nledp];   /* FADC scale for correction */
  float ucrry[tasdconst_nledp];   /* correction factor - 1.0*/
  float ucrrs[tasdconst_nledp]; /* error of correction factor */
  float lcrrx[tasdconst_nledp];   /* FADC scale for correction */
  float lcrry[tasdconst_nledp];   /* correction factor - 1.0*/
  float lcrrs[tasdconst_nledp]; /* error of correction factor */



  int livetime;		/* total livetime */
  float uavr;
  float lavr;
  float upltot;
  float lpltot;
  float ucltot;
  float lcltot;
  float nuplx;
  float nlplx;
  float nuply;
  float nlply;
  int udec5pmip;  /* maximun lineality range of upper layer
		     [FADC count] */
  int ldec5pmip;  /* maximun lineality range of lower layer
		     [FADC count] */
  float uphx[tasdconst_nmiph]; /* upper peak histogram */
  float lphx[tasdconst_nmiph]; /* lower peak histogram */
  float uphy[tasdconst_nmiph]; /* upper peak histogram */
  float lphy[tasdconst_nmiph]; /* lower peak histogram */
  float uphs[tasdconst_nmiph]; /* upper peak histogram */
  float lphs[tasdconst_nmiph]; /* lower peak histogram */
  float uchx[tasdconst_nmiph]; /* upper charge histogram */
  float lchx[tasdconst_nmiph]; /* lower charge histogram */
  float uchy[tasdconst_nmiph]; /* upper charge histogram */
  float lchy[tasdconst_nmiph]; /* lower charge histogram */
  float uchs[tasdconst_nmiph]; /* upper charge histogram */
  float lchs[tasdconst_nmiph]; /* lower charge histogram */
  float uplx[tasdconst_nmiph]; /* upper peak liniarity */
  float lplx[tasdconst_nmiph]; /* lower peak liniarity */
  float uply[tasdconst_nmiph]; /* upper peak liniarity */
  float lply[tasdconst_nmiph]; /* lower peak liniarity */
  float upls[tasdconst_nmiph]; /* upper peak liniarity */
  float lpls[tasdconst_nmiph]; /* lower peak liniarity */


  int error_flag;

  int footer;

} tasdconst_dst_common;

extern tasdconst_dst_common tasdconst_;

#endif


