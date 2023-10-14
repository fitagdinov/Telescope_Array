/*
 * $Source: /hires_soft/uvm2k/uti/event.h,v $
 * $Log: event.h,v $
 *
 * Revision 2.00  2008/02/29           seans
 * cleaned out old bank list for new TA project.  Keeping only base Hires 
 *   analysis banks.
 *
 * Revision 1.1  1995/05/09  00:54:08  jeremy
 * Initial revision
 *
*/

#include "dst_std_types.h"
#include "dst_err_codes.h"
#include "dst_bank_proto.h"

#include "univ_dst.h"
#include "start_dst.h"
#include "stop_dst.h"

/* fadc banks */
/* hires2 monte carlo */
#include "fmc1_dst.h"
#include "fpho1_dst.h"
#include "fraw1_dst.h"
#include "fscn1_dst.h"
#include "ftrg1_dst.h"

/* TA data banks */
#include "azgh_showlib_dst.h"
#include "azgh_showscale_dst.h"
#include "tadaq_dst.h"
#include "brped_dst.h"
#include "brpho_dst.h"
#include "brplane_dst.h"
#include "brprofile_dst.h"
#include "brraw_dst.h"
#include "brtime_dst.h"
#include "brtubeprofile_dst.h"
#include "bsdinfo_dst.h"

#include "etrack_dst.h"

#include "fdped_dst.h"
#include "fdplane_dst.h"
#include "fdprofile_dst.h"
#include "fdraw_dst.h"
#include "fdtime_dst.h"
#include "fdtubeprofile_dst.h"
#include "trumpmc_dst.h"
#include "geofd_dst.h"
#include "geobr_dst.h"
#include "geolr_dst.h"
#include "geotl_dst.h"
#include "lrped_dst.h"
#include "lrpho_dst.h"
#include "lrplane_dst.h"
#include "lrprofile_dst.h"
#include "lrraw_dst.h"
#include "lrtime_dst.h"
#include "lrtubeprofile_dst.h"

/* For SD */

#include "rusdraw_dst.h"
#include "rusdcal_dst.h"
#include "rusdmc_dst.h"
#include "rusdmc1_dst.h"
#include "rusdmcd_dst.h"
#include "rufptn_dst.h"
#include "rusdgeom_dst.h"
#include "rufldf_dst.h"
#include "sdgealib_dst.h"
#include "sdmon_dst.h"
#include "sdtrgbk_dst.h"

#include "tasdcalib_dst.h"
#include "tasdcalibev_dst.h"
#include "tasdcond_dst.h"
#include "tasdconst_dst.h"
#include "tasdelecinfo_dst.h"
#include "tasdevent_dst.h"
#include "tasdgps_dst.h"
#include "tasdidhv_dst.h"
#include "tasdinfo_dst.h"
#include "tasdledlin_dst.h"
#include "tasdmiplin_dst.h"
#include "tasdmonitor_dst.h"
#include "tasdpedmip_dst.h"
#include "tasdpmtgain_dst.h"
#include "tasdtemp_dst.h"
#include "tasdtrginfo_dst.h"
#include "tasdtrgmode_dst.h"

/* MD banks */
#include "hmc1_dst.h"
#include "hcal1_dst.h"
#include "hped1_dst.h"
#include "hpkt1_dst.h"
#include "hraw1_dst.h"
#include "hrxf1_dst.h"
#include "hsum4_dst.h"
#include "mc04_dst.h"
#include "mcraw_dst.h"
#include "mcsdd_dst.h"
#include "prfd_dst.h"
#include "stnpe_dst.h"
#include "stpln_dst.h"
#include "stps2_dst.h"
#include "sttrk_dst.h"
#include "tslew_dst.h"

/* HR banks used by MD */
#include "hbar_dst.h"
#include "hctim_dst.h"
#include "hnpe_dst.h"
#include "prfc_dst.h"
#include "hcbin_dst.h"
#include "hv2_dst.h"
#include "hsum_dst.h"
#include "ontime2_dst.h"

/* Hybrid banks derived from D. Ikeda java code */
#include "hybridreconfd_dst.h"
#include "hybridreconbr_dst.h"
#include "hybridreconlr_dst.h"

/* Weather and atmospheric condition DST banks */
#include "atmpar_dst.h"
#include "fdatmos_param_dst.h"
#include "gdas_dst.h"
#include "clfvaod_dst.h"  // 2020/01/28
#include "tafdweather_dst.h"
#include "mdweat_dst.h"
#include "tlweat_dst.h"

/* Additinal FD banks (mostly calibration) */
#include "brfft_dst.h"
#include "fdatmos_trans_dst.h"
#include "fdbg3_trans_dst.h"
#include "fdcamera_temp_dst.h"
#include "fdctd_clock_dst.h"
#include "fdfft_dst.h"
#include "fdmirror_ref_dst.h"
#include "fdparaglas_trans_dst.h"
#include "fdpmt_gain_dst.h"
#include "fdpmt_qece_dst.h"
#include "fdpmt_uniformity_dst.h"
#include "fdscat_dst.h"
#include "fdshowerparameter_dst.h"
#include "hspec_dst.h"
#include "irdatabank_dst.h"
#include "lrfft_dst.h"
#include "showlib_dst.h"
#include "showscale_dst.h"
#include "showpro_dst.h"
#include "stplane_dst.h"
#include "sttubeprofile_dst.h"

// TALE SD Banks
#include "talex00_dst.h"
#include "tale_db_uvled_dst.h"
#include "tlfptn_dst.h"
#include "tlmon_dst.h"

// New TALE FD banks that aren't present in HR2
#include "tlmsnp_dst.h"
#include "tl4rgf_dst.h"

#include "bank_list.h"

integer4 n_banks_total_(); /* total number of DST banks */
integer4 event_all_banks_(integer4 *list);
integer4 new_empty_bank_list_(); /* create and return a new empty bank list of maximal size */
integer4 new_bank_list_with_all_banks_(); /* create and return a new bank list with all banks added to it */
/* event buffering routine
 * INPUT: *n_banks - pointer to the variable that contains the number of banks specified by the caller
 *        *bank_types - array that contains the bank IDs banks for which to fill the buffers, also specified by the caller
 * OUTPUT: for each bank found in the event bank list:
 *  bank_buffer_sizes is an integer array that stores the size of each bank's buffer
 *  (should be allocated by caller to *n_banks)
 *  bank_buffers is the array of pointers to the filled buffers of each bank
 *  (should be allocated by caller to *n_banks)
 * RETURN: SUCCESS if everything worked, error code in case of error. */
integer4 event_commons_to_buffers_(const integer4* n_banks, const integer4* bank_types, 
				   integer4 *bank_buffer_sizes, integer1** bank_buffers);
/* event buffering routine.
 * INPUT: *bank_list - pointer to alocatted (and presumably filled) bank list
 * OUTPUT: for each bank found in the event bank list:
 *  n_banks is the number of successfully buffered banks
 *  bank_types is an integer array that stores the IDs of all banks 
 *  (should be allocated by caller to preferrably n_banks_total_())
 *  bank_buffer_sizes is an integer array that stores the size of each bank's buffer
 *  (should be allocated by caller to preferrably n_banks_total_())
 *  bank_buffers is the array of pointers to the filled buffers of each bank
 *  (should be allocated by caller to preferrably n_banks_total_())
 * RETURN: SUCCESS if everything worked, error code in case of error. */
integer4 event_commons_to_buffers_from_bank_list_(integer4* bank_list, integer4* n_banks, 
						  integer4* bank_types, integer4 *bank_buffer_sizes, 
						  integer1** bank_buffers);
/* Fill out structures that correspond to each bank id found in bank_types (n_banks total) from the corresponding bank 
 * buffer whose pointers are provided by the bank_buffers array
 * INPUT: n_banks: number of banks
 *        bank_types: IDs of the banks, array should be at least n_banks in size
 *        bank_buffers: array of pointers to dst bank buffers of the banks, should be at least n_banks in size
 * OUTPUT: corresponding dst structures (aka common blocks) are filled with information found in the bank buffers.
 * RETURN: SUCCESS if everything worked, error code in case of errors. */
integer4 event_buffers_to_commons_(const integer4* n_banks, const integer4* bank_types, integer1** bank_buffers);
integer4 event_id_from_name_(const integer1* name);
integer4 event_name_from_id_(integer4 *bank_id, integer1 *name, integer4 *len);
integer4 event_version_from_id_(integer4 *bank_id);
integer4 event_set_dump_format_(integer4 *list, integer4 *format);
integer4 event_dumpf_(FILE *fp, integer4 *list);
integer4 event_dump_(integer4 *list);

integer4 event_read_(integer4 *unit, integer4 *want_banks, integer4 *got_banks, integer4 *event);
/* Read bank from DST unit until bank type in want_bank list or start bank.
 * If start bank read, read all banks until stop bank. Returns number of bank
 * types in want_banks read (got_banks). Returns zero if end of DST.
 * Event set to one if read event banks, set to zero otherwise.
 */

integer4 event_write_(integer4 *unit, integer4 *banks, integer4 *event);
/* Write banks in bank list 'banks' to DST unit ID 'unit'. If event is
 * TRUE, write a start bank before any other banks and a stop bank after.
 * Return number of banks (not including start & stop) written or error.
 */

/* C / C++ - simplified versions of above routines */
#ifdef __cplusplus
extern "C" {
#endif
  integer4 nBanksTotal();
  integer4 eventAllBanks(integer4 list);
  integer4 newEmptyBankList();
  integer4 newBankListWithAllBanks();
  integer4 eventBuffersToCommons(integer4 n_banks, const integer4* bank_types, integer1** bank_buffers);
  integer4 eventCommonsToBuffers(integer4 n_banks, const integer4* bank_types, integer4 *bank_buffer_sizes, integer1** bank_buffers);
  integer4 eventCommonsToBuffersFromBankList(integer4 bank_list, integer4* n_banks, 
					     integer4* bank_types, integer4 *bank_buffer_sizes, 
					     integer1** bank_buffers);
  integer4 eventIdFromName(const integer1* name);
  integer4 eventNameFromId(integer4 bank_id, integer1* name, integer4 len);
  integer4 eventVersionFromId(integer4 bank_id);
  integer4 eventSetDumpFormat(integer4 list, integer4 format);
  integer4 eventDumpf(FILE *fp, integer4 list);
  integer4 eventDump(integer4 list);
  integer4 eventRead(integer4 unit, integer4 want_banks, integer4 got_banks,integer4 *event);
  integer4 eventWrite(integer4 unit, integer4 banks, integer4 event);
  integer4 dstOpenUnit(integer4 unit, const integer1* file, integer4 mode);
  integer4 dstCloseUnit(integer4 unit);
#ifdef __cplusplus
} /* end extern "C" */
#endif
