/*
 * $Source: /hires_soft/uvm2k/bank/ftrg1_dst.h,v $
 * $Log: ftrg1_dst.h,v $
 * Revision 1.9  2003/05/28 15:30:57  hires
 * update to nevis version to eliminate tar file dependency (boyer)
 *
 *
 * 2002/10/04  changed meaning of ftrg1_.mir_num
 * from triggered_mirid to 100*mirid + triggered_mirid
 *
 * Revision 1.7  1996/09/27  16:53:50  boyer
 * add discarded triggers to bank
 *
 * Revision 1.6  1996/06/07  15:41:43  boyer
 * new trig packet structure
 *
 * Revision 1.4  1996/02/23  18:26:19  boyer
 * change common block a bit to conform with real data
 *
 * Revision 1.3  1995/10/04  19:40:06  boyer
 * modify bank size
 *
 * Revision 1.2  1995/08/22  21:37:09  vtodd
 * provided for conditional inclusion:  #ifndef _FTRG1_
 *
 * Revision 1.1  1995/08/09  20:17:02  boyer
 * Initial revision
 *
 * Created by CPH 10:30 7/19/95
 * *** empty log message ***
 *
*/

/* ftrg1_dst is available if there was a DSP scan of the mirror
 *   during the secondary trigger (trig_code > 0)
 * A hit indicates that the digital filter was above threshold for 1 channel
*/

#ifndef _FTRG1_
#define _FTRG1_

#define FTRG1_BANKID 12002 
#define FTRG1_BANKVERSION 0 

#define ftrg1_nhit_dsp_max 320
#define ftrg1_nmir_max 20
#define ftrg1_discard_max 100

#ifdef __cplusplus
extern "C" {
#endif
integer4 ftrg1_common_to_bank_();
integer4 ftrg1_bank_to_dst_(integer4 *NumUnit);
integer4 ftrg1_common_to_dst_(integer4 *NumUnit); /* combines above 2 */
integer4 ftrg1_bank_to_common_(integer1 *bank);
integer4 ftrg1_common_to_dump_(integer4 *long_option) ;
integer4 ftrg1_common_to_dumpf_(FILE* fp, integer4 *long_option);
/* get (packed) buffer pointer and size */
integer1* ftrg1_bank_buffer_ (integer4* ftrg1_bank_buffer_size);
#ifdef __cplusplus
} //end extern "C"
#endif


typedef struct  {
	integer2 num_mir;

        integer2 num_discard;  /* number of triggers thrown away in raw data stream 
                                  between last end-of-event and this end-of-event */

        integer2 trig_code [ftrg1_nmir_max];   /* trigger code word */
	integer2 mir_num [ftrg1_nmir_max]; /* id of triggered mirror */
  /* changed 10/04/2002 to 100*mirror_id + triggered_mirror_id */
        integer2 nhit_dsp [ftrg1_nmir_max]; /* # of DSP found hits */
	integer2 ichan_dsp [ftrg1_nmir_max][ftrg1_nhit_dsp_max];
	                                              /* channel # (1-256)   */
	integer2 it0_dsp [ftrg1_nmir_max][ftrg1_nhit_dsp_max];   
	                                              /* hit start in units
				               * of 100 nsec since store start */
	integer2 nt_dsp [ftrg1_nmir_max][ftrg1_nhit_dsp_max];    
	                                              /* width of hit in nsec*/
	integer2 tav_dsp [ftrg1_nmir_max][ftrg1_nhit_dsp_max];   
	                                             /* mean time of hit in 
				         * nsec since start of hit (it0_dsp) */
	integer2 sigt_dsp [ftrg1_nmir_max][ftrg1_nhit_dsp_max];  
	                                              /* stddev of hit(nsec) */
	integer2 nadc_dsp [ftrg1_nmir_max][ftrg1_nhit_dsp_max];  
	                                              /* int cnts above ped  */
        integer2 t_pld_start [ftrg1_nmir_max];   /* primary trigger
			      * start in units of 100 nsec since store start */
	integer2 t_pld_end [ftrg1_nmir_max];     /* primary trigger
			      * end in units of 100 nsec since store start   */
 	integer2 row_pattern [ftrg1_nmir_max];   /* 14-bit pattern
					     * of 3-row-hit coincidences
					     * 1st bit is rows 1,2,3 coinc. */
	integer2 col_pattern [ftrg1_nmir_max];  /* 14-bit pattern of
						     *  3-col-hit coincidences 
					     * 1st bit is cols 1,2,3 coinc. */
        integer2 wp[ftrg1_nmir_max][16];  /* write pointers (a diagnostic)  */
        integer2 delay[ftrg1_nmir_max];  /* time diff btw trig processing time and mark */

        integer2 discard_code[ftrg1_discard_max];  /* discarded trigger codes  */
        integer2 discard_mirid[ftrg1_discard_max];
        integer4 discard_clkcnt[ftrg1_discard_max];  /* time of discarded trigger  */
        integer2 discard_vpattern[ftrg1_discard_max];  /* V-pattern of discarded trigger */
        integer2 discard_hpattern[ftrg1_discard_max];  /* H-pattern of discarded trigger */
        integer2 discard_nbyte[ftrg1_discard_max];  /* store size of discarded trigger */

} ftrg1_dst_common ;

extern ftrg1_dst_common ftrg1_ ; 

typedef struct  {

  integer2 id[100];
  integer2 code[100];
  integer2 second[100];
  integer4 fracsec[100];   /*  microsec after GPS sec  */
  integer2 nbin[100];
  integer2 ntrig;

} ftrg1_info_common ;

extern ftrg1_info_common ftrg1_info_ ; 

#endif














