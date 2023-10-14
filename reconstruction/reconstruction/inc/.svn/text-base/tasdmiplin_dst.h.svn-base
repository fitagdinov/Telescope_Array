/*
 *     Bank for SD MIP analysis
 *     written by a student
 *     Time-stamp: Fri Apr 10 23:38:08 2009 JST
*/

#ifndef _TASDMIPLIN_
#define _TASDMIPLIN_

#define TASDMIPLIN_BANKID  13013
#define TASDMIPLIN_BANKVERSION   002

#ifdef __cplusplus
extern "C" {
#endif
int tasdmiplin_common_to_bank_();
int tasdmiplin_bank_to_dst_(int *NumUnit);
int tasdmiplin_common_to_dst_(int *NumUnit); /* combines above 2 */
int tasdmiplin_bank_to_common_(char *bank);
int tasdmiplin_common_to_dump_(int *opt1) ;
int tasdmiplin_common_to_dumpf_(FILE* fp, int *opt2);
/* get (packed) buffer pointer and size */
integer1* tasdmiplin_bank_buffer_ (integer4* tasdmiplin_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif




#define tasdmiplin_nhmax   3  /* maximum number of host */
#define tasdmiplin_npmax 128  /* maximum number of point for peak liniarity */
#define tasdmiplin_ndmax 512    /* maximum number of detector per event      */


typedef struct {
  int lid;
  int livetime;		/* livetime in 10 min */

  float uavr;
  float lavr;
  float upltot;
  float lpltot;
  float ucltot;
  float lcltot;
  float uplx[tasdmiplin_npmax]; /* upper peak liniarity */
  float uply[tasdmiplin_npmax]; /* upper peak liniarity */
  float upls[tasdmiplin_npmax]; /* upper peak liniarity */
  float lplx[tasdmiplin_npmax]; /* lower peak liniarity */
  float lply[tasdmiplin_npmax]; /* lower peak liniarity */
  float lpls[tasdmiplin_npmax]; /* lower peak liniarity */
  float uclx[tasdmiplin_npmax]; /* upper charge liniarity */
  float ucly[tasdmiplin_npmax]; /* upper charge liniarity */
  float ucls[tasdmiplin_npmax]; /* upper charge liniarity */
  float lclx[tasdmiplin_npmax]; /* lower charge liniarity */
  float lcly[tasdmiplin_npmax]; /* lower charge liniarity */
  float lcls[tasdmiplin_npmax]; /* lower charge liniarity */

} SDMiplinData;


typedef struct {
  int num_det;   /* the number of detectors       */
  int dateFrom;     /* year month day */
  int dateTo;       /* year month day */
  int timeFrom;     /* year month day */
  int timeTo;       /* year month day */

  int first_run_id[tasdmiplin_nhmax];
  int last_run_id[tasdmiplin_nhmax];

  SDMiplinData sub[tasdmiplin_ndmax+1];

  int footer;

} tasdmiplin_dst_common;

extern tasdmiplin_dst_common tasdmiplin_;

#endif


