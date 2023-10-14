/*
 * $Source: /hires_soft/uvm2k/uti/event.c,v $
 * $Log: event.c,v $
 *
 * Last modified: DI 20191204
 *
 * DI 20171206
 *
 * Revision 2.00  2008/02/29           seans
 * removed old banks for cleanup
 *
 * Revision 1.95  2008/02/28           seans
 * added tafraw dst bank
 *
 * Revision 1.1  1995/05/09  00:54:08  jeremy
 * Initial revision
 *
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "event.h"

/* Macro that simplifies addition of DST banks to the data base, maximum */
/* bank size can be specified if the bank is not of a standard form */
#define BANKDB_ENTRY_SIZE(dstbank,DSTBANK,banksize)			\
  {   #dstbank,								\
      DSTBANK##_BANKID,							\
      DSTBANK##_BANKVERSION,						\
      banksize,								\
      dstbank##_bank_to_common_,					\
      dstbank##_common_to_bank_,					\
      dstbank##_common_to_dst_,						\
      dstbank##_bank_buffer_,						\
      dstbank##_common_to_dumpf_, 1 }
/* Regular DST bank entry (most banks are this way) */
#define BANKDB_ENTRY(dstbank,DSTBANK)			\
  BANKDB_ENTRY_SIZE(dstbank,DSTBANK,sizeof(dstbank##_))

static struct
{
  integer1  *name;                                /* ASCII name for bank type */
  integer4  bank_id;                              /* Bank id from _dst.h file */
  integer4  bank_version;                         /* Bank version from _dst.h file */
  integer4  bank_size;                            /* Maximum size of bank record */
  integer4  (*bank_to_common)(integer1*);         /* Pointer to function to unpack bank */
  integer4  (*common_to_bank)();                  /* Pointer to function to pack bank */
  integer4  (*common_to_dst)(integer4*);          /* Pointer to function to pack and store bank */
  integer1* (*bank_buffer)(integer4*);            /* Pointer to function that returns packed bank buffer */
  integer4  (*common_to_dumpf)(FILE*, integer4*); /* Pointer to function to generate ASCII dump of bank */
  integer4  dump_format;                          /* ASCII dump format. Short:0, Long:1, Hide:2 */
} bankDb[] = {
  /* add records here for new bank types */
  BANKDB_ENTRY(azgh_showlib,AZGH_SHOWLIB),
  BANKDB_ENTRY(azgh_showscale,AZGH_SHOWSCALE),
  BANKDB_ENTRY(brped,BRPED),
  BANKDB_ENTRY(brpho,BRPHO),
  BANKDB_ENTRY(brplane,BRPLANE),
  BANKDB_ENTRY(brprofile,BRPROFILE),
  BANKDB_ENTRY(brraw,BRRAW),
  BANKDB_ENTRY(brtime,BRTIME),
  BANKDB_ENTRY(brtubeprofile,BRTUBEPROFILE),
  BANKDB_ENTRY(bsdinfo,BSDINFO),
  BANKDB_ENTRY(etrack,ETRACK),
  BANKDB_ENTRY(fdped,FDPED),
  BANKDB_ENTRY(fdplane,FDPLANE),  
  BANKDB_ENTRY(fdprofile,FDPROFILE),  
  BANKDB_ENTRY(fdtime,FDTIME),
  BANKDB_ENTRY(fdtubeprofile,FDTUBEPROFILE),
  BANKDB_ENTRY(trumpmc,TRUMPMC),
  BANKDB_ENTRY(fdraw,FDRAW),
  BANKDB_ENTRY(fmc1,FMC1),
  BANKDB_ENTRY(fpho1,FPHO1),
  BANKDB_ENTRY(fraw1,FRAW1),
  BANKDB_ENTRY(fscn1,FSCN1),
  BANKDB_ENTRY(ftrg1,FTRG1),
  BANKDB_ENTRY(geofd,GEOFD),
  BANKDB_ENTRY(geobr,GEOBR),
  BANKDB_ENTRY(geolr,GEOLR),
  BANKDB_ENTRY(geotl,GEOTL),
  BANKDB_ENTRY(lrped,LRPED),
  BANKDB_ENTRY(lrpho,LRPHO),
  BANKDB_ENTRY(lrplane,LRPLANE),
  BANKDB_ENTRY(lrprofile,LRPROFILE),
  BANKDB_ENTRY(lrraw,LRRAW),
  BANKDB_ENTRY(lrtime,LRTIME),
  BANKDB_ENTRY(lrtubeprofile,LRTUBEPROFILE),
  BANKDB_ENTRY(rusdraw,RUSDRAW),
  BANKDB_ENTRY(rusdcal,RUSDCAL),
  BANKDB_ENTRY(rusdmc,RUSDMC),
  BANKDB_ENTRY(rusdmc1,RUSDMC1),
  BANKDB_ENTRY(rusdmcd,RUSDMCD),
  BANKDB_ENTRY(rufptn,RUFPTN),
  BANKDB_ENTRY(rusdgeom,RUSDGEOM),
  BANKDB_ENTRY(rufldf,RUFLDF),
  BANKDB_ENTRY_SIZE(sdgealib,SDGEALIB,sizeof(sdgealib_head_)+sizeof(sdgealib_hist_)),
  BANKDB_ENTRY(sdmon,SDMON),
  BANKDB_ENTRY(sdtrgbk,SDTRGBK),
  BANKDB_ENTRY(tadaq,TADAQ),
  BANKDB_ENTRY(tasdcalib,TASDCALIB),
  BANKDB_ENTRY(tasdcalibev,TASDCALIBEV),
  BANKDB_ENTRY(tasdcond,TASDCOND),
  BANKDB_ENTRY(tasdconst,TASDCONST),  
  BANKDB_ENTRY(tasdelecinfo,TASDELECINFO),  
  BANKDB_ENTRY(tasdevent,TASDEVENT),
  BANKDB_ENTRY(tasdgps,TASDGPS),  
  BANKDB_ENTRY(tasdidhv,TASDIDHV),  
  BANKDB_ENTRY(tasdinfo,TASDINFO),  
  BANKDB_ENTRY(tasdledlin,TASDLEDLIN),
  BANKDB_ENTRY(tasdmiplin,TASDMIPLIN),   
  BANKDB_ENTRY(tasdmonitor,TASDMONITOR),    
  BANKDB_ENTRY(tasdpedmip,TASDPEDMIP),
  BANKDB_ENTRY(tasdpmtgain,TASDPMTGAIN),
  BANKDB_ENTRY(tasdtemp,TASDTEMP),
  BANKDB_ENTRY(tasdtrginfo,TASDTRGINFO),
  BANKDB_ENTRY(tasdtrgmode,TASDTRGMODE),
  BANKDB_ENTRY(hcal1,HCAL1),
  BANKDB_ENTRY(hmc1,HMC1),
  BANKDB_ENTRY(hped1,HPED1),
  BANKDB_ENTRY_SIZE(hpkt1,HPKT1,sizeof(hpkt1_time_)),  
  BANKDB_ENTRY(hraw1,HRAW1),
  BANKDB_ENTRY(hrxf1,HRXF1),
  BANKDB_ENTRY(mc04,MC04),
  BANKDB_ENTRY(mcraw,MCRAW),
  BANKDB_ENTRY(mcsdd,MCSDD),
  BANKDB_ENTRY(hsum,HSUM),
  BANKDB_ENTRY(hsum4,HSUM4),
  BANKDB_ENTRY(ontime2,ONTIME2),
  BANKDB_ENTRY(prfd,PRFD),  
  BANKDB_ENTRY(stnpe,STNPE),
  BANKDB_ENTRY(stpln,STPLN),
  BANKDB_ENTRY(stps2,STPS2),
  BANKDB_ENTRY(tslew,TSLEW),
  BANKDB_ENTRY(sttrk,STTRK),
  BANKDB_ENTRY(hbar,HBAR),
  BANKDB_ENTRY(hnpe,HNPE),
  BANKDB_ENTRY(hctim,HCTIM),
  BANKDB_ENTRY(prfc,PRFC),
  BANKDB_ENTRY(hcbin,HCBIN),
  BANKDB_ENTRY(hv2,HV2),
  BANKDB_ENTRY(hybridreconfd,HYBRIDRECONFD),  
  BANKDB_ENTRY(hybridreconbr,HYBRIDRECONBR),
  BANKDB_ENTRY(hybridreconlr,HYBRIDRECONLR),
  BANKDB_ENTRY(atmpar,ATMPAR),
  BANKDB_ENTRY(fdatmos_param,FDATMOS_PARAM),
  BANKDB_ENTRY(clfvaod,CLFVAOD),
  BANKDB_ENTRY(gdas,GDAS),
  BANKDB_ENTRY(tafdweather,TAFDWEATHER),
  BANKDB_ENTRY(mdweat,MDWEAT),
  BANKDB_ENTRY(tlweat,TLWEAT),
  BANKDB_ENTRY(fdatmos_trans,FDATMOS_TRANS),  
  BANKDB_ENTRY(fdbg3_trans,FDBG3_TRANS),  
  BANKDB_ENTRY(fdcamera_temp,FDCAMERA_TEMP),  
  BANKDB_ENTRY(fdctd_clock,FDCTD_CLOCK),  
  BANKDB_ENTRY(fdfft,FDFFT),  
  BANKDB_ENTRY(fdmirror_ref,FDMIRROR_REF),  
  BANKDB_ENTRY(fdparaglas_trans,FDPARAGLAS_TRANS),
  BANKDB_ENTRY(fdpmt_gain,FDPMT_GAIN),  
  BANKDB_ENTRY(fdpmt_qece,FDPMT_QECE),  
  BANKDB_ENTRY(fdpmt_uniformity,FDPMT_UNIFORMITY),  
  BANKDB_ENTRY(fdscat,FDSCAT),  
  BANKDB_ENTRY(fdshowerparameter,FDSHOWERPARAMETER),
  BANKDB_ENTRY(hspec,HSPEC),  
  BANKDB_ENTRY(irdatabank,IRDATABANK),
  BANKDB_ENTRY(lrfft,LRFFT),
  BANKDB_ENTRY(showlib,SHOWLIB),
  BANKDB_ENTRY(showpro,SHOWPRO),
  BANKDB_ENTRY(showscale,SHOWSCALE),
  BANKDB_ENTRY(stplane,STPLANE),
  BANKDB_ENTRY(sttubeprofile,STTUBEPROFILE),
  BANKDB_ENTRY(talex00,TALEX00),
  BANKDB_ENTRY(tale_db_uvled,TALE_DB_UVLED),
  BANKDB_ENTRY(tlfptn,TLFPTN),
  BANKDB_ENTRY(tlmon,TLMON),
  BANKDB_ENTRY(tlmsnp,TLMSNP),
  BANKDB_ENTRY(tl4rgf,TL4RGF),
  {0} /* end of list marker */
};

/* clean up the definitions */
#undef BANKDB_ENTRY
#undef BANKDB_ENTRY_SIZE


/* bank data base sorted by bank ID. useful for quick searches */  
static integer4* bank_db_sorted_by_bank_id = 0; /* indices of the bankDb sorted by bank ID */
static integer4* bank_db_bank_id_sorted    = 0; /* keys are integer bank IDs */


/* bank data base sorted by bank ID. useful for quick searches */  
static integer4*  bank_db_sorted_by_bank_name_hash = 0; /* indices of the bankDb sorted by bank name hash */
static uinteger4* bank_db_bank_name_hash_sorted    = 0; /* keys are hash values of bank names */

/* function that compares the bank IDs of two indices, needed for sorting */
static integer4 bank_db_bank_id_cmp_fun(const void * lhs, const void * rhs)
{ return bankDb[*(integer4 *)lhs].bank_id - bankDb[*(integer4 *)rhs].bank_id; }

/* function that compares two integers */
static integer4 event_int_cmp_fun(const void * lhs, const void * rhs)
{ return *(integer4 *)lhs - *(integer4 *)rhs; }

/* function that compares two unsigned integers: careful not to subtract unsigned integers,
 because negative values would get wrapped in the case of unsigned integer type */
static integer4 event_uint_cmp_fun(const void * lhs, const void * rhs)
{ 
  if(*(uinteger4 *)lhs  <  *(uinteger4 *)rhs)
    return -1;
  if(*(uinteger4 *)lhs  >  *(uinteger4 *)rhs)
    return 1;
  return 0;
}

/* hashing bank names (CONVERTS ALL LETTERS TO LOWER CASE FIRST) into integers 
   using djb2 hashing algorithm by Dan Bernstein */
static uinteger4 dst_bank_name_hash_djb2(const char *name)
{
  int c;
  uinteger4 hash = 5381;
  if(name)
    {
      while ((c = *name++))
	hash = ((hash << 5) + hash) + (uinteger4)tolower(c); /* hash * 33 + c */
      return hash;
    }
  return 0;
}

/* function that compares the bank name hashes of two indices, needed for sorting */
static integer4 bank_db_bank_name_hash_cmp_fun(const void * lhs, const void * rhs)
{ 
  if(dst_bank_name_hash_djb2(bankDb[*(integer4 *)lhs].name) < dst_bank_name_hash_djb2(bankDb[*(integer4 *)rhs].name))
    return -1;
  if(dst_bank_name_hash_djb2(bankDb[*(integer4 *)lhs].name) > dst_bank_name_hash_djb2(bankDb[*(integer4 *)rhs].name))
    return 1;
  return 0;
}

/* bank data base initialization routine */ 
static void init_bank_db_()
{
  /* does anything only if the bank ID data base hasn't been sorted by bank ID yet */
  if(!bank_db_sorted_by_bank_id)
    {
      /* quick itegrity check of the data base */
      integer4 i;
      for (i = 0; i < (int)(sizeof(bankDb)/sizeof(bankDb[0])); i++)
	{
	  integer4 j;
	  for (j = i + 1; j < (integer4)(sizeof(bankDb)/sizeof(bankDb[0])); j ++)
	    {
	      if(bankDb[i].bank_id == bankDb[j].bank_id)
		{
		  fprintf(stderr,"ERROR: %s(%d): BANK ID %d USED MORE THAN ONCE FOR BANKS '%s' and '%s'!\n",
			  __FILE__, __LINE__, bankDb[i].bank_id, bankDb[i].name,bankDb[j].name);
		  exit(2);
		}
	      if(dst_bank_name_hash_djb2(bankDb[i].name) == dst_bank_name_hash_djb2(bankDb[j].name))
		{
		  fprintf(stderr,"ERROR: %s(%d): BANK NAMES %s (BANKD ID %d) AND %s (BANKD ID %d)",
			  __FILE__, __LINE__, bankDb[i].name,bankDb[i].bank_id, bankDb[j].name, bankDb[j].bank_id);
		  fprintf(stderr,"YIELD SAME HASH VALUES %u and %u!\n", 
			  dst_bank_name_hash_djb2(bankDb[i].name),dst_bank_name_hash_djb2(bankDb[j].name));
		  exit(2);
		}
	    }
	}
      /* allocate the arrays for the bank data base sorted by bank ID */
      if(!(bank_db_sorted_by_bank_id = (integer4*)calloc(sizeof(bankDb)/sizeof(bankDb[0]), sizeof(integer4))))
	{
	  fprintf(stderr,"ERROR: %s(%d): CALLOC ERROR!\n",
		  __FILE__, __LINE__);
	  exit(2);
	}
      if(!(bank_db_bank_id_sorted = (integer4*)calloc(sizeof(bankDb)/sizeof(bankDb[0]), sizeof(integer4))))
	{
	  fprintf(stderr,"ERROR: %s(%d): CALLOC ERROR!\n",
		  __FILE__, __LINE__);
	  exit(2);
	}
      
      /* allocate the arrays for the bank data base sorted by bank name */
      if(!(bank_db_sorted_by_bank_name_hash = (integer4*)calloc(sizeof(bankDb)/sizeof(bankDb[0]), sizeof(integer4))))
	{
	  fprintf(stderr,"ERROR: %s(%d): CALLOC ERROR!\n",
		  __FILE__, __LINE__);
	  exit(2);
	}
      if(!(bank_db_bank_name_hash_sorted = (uinteger4*)calloc(sizeof(bankDb)/sizeof(bankDb[0]), sizeof(integer4))))
	{
	  fprintf(stderr,"ERROR: %s(%d): CALLOC ERROR!\n",
		  __FILE__, __LINE__);
	  exit(2);
	}

      /* load the bank data base indices */
      for (i = 0; i < (integer4)(sizeof(bankDb)/sizeof(bankDb[0])); i++)
	{
	  bank_db_sorted_by_bank_id[i] = i;
	  bank_db_sorted_by_bank_name_hash[i] = i;
	}
      /* sort the bank data base by bank ID */
      qsort(bank_db_sorted_by_bank_id,
	    sizeof(bankDb)/sizeof(bankDb[0]),
	    sizeof(integer4),bank_db_bank_id_cmp_fun);
      /* sort the bank data base by bank name hash */
      qsort(bank_db_sorted_by_bank_name_hash,
	    sizeof(bankDb)/sizeof(bankDb[0]),
	    sizeof(integer4),bank_db_bank_name_hash_cmp_fun);
      /* load the sorted bank ID and name hash values to use in binary searches later */
      for (i = 0; i < (integer4)(sizeof(bankDb)/sizeof(bankDb[0])); i++)
	{
	  bank_db_bank_id_sorted[i] = bankDb[bank_db_sorted_by_bank_id[i]].bank_id;
	  bank_db_bank_name_hash_sorted[i] = dst_bank_name_hash_djb2(bankDb[bank_db_sorted_by_bank_name_hash[i]].name);
	}
    }
}

/* returns index in the event data base, for a given bank ID. Returns something negative (GET_BANK_UNKWN_BANK) in cases of failure */
static integer4 bankid2dbindex(integer4 bank_id)
{
  integer4 *bank_id_ptr = 0;
  init_bank_db_(); /* make sure that's always initialized (sorted by the bank ID) */
  if((bank_id_ptr = (integer4*)bsearch(&bank_id,bank_db_bank_id_sorted,
				       sizeof(bankDb)/sizeof(bankDb[0]),
				       sizeof(integer4),event_int_cmp_fun)))
    return bank_db_sorted_by_bank_id[(bank_id_ptr - bank_db_bank_id_sorted)];
  return GET_BANK_UNKWN_BANK;
}

/* returns index in the event data base, for a given bank name. Returns something negative (GET_BANK_UNKWN_BANK) in cases of failure */
static integer4 bankname2dbindex(const char* name)
{
  uinteger4 bank_name_hash = dst_bank_name_hash_djb2(name);
  uinteger4 *bank_name_hash_ptr = 0;
  init_bank_db_(); /* make sure that's always initialized (sorted by the bank ID) */
  if((bank_name_hash_ptr = (uinteger4*)bsearch(&bank_name_hash,bank_db_bank_name_hash_sorted,
					       sizeof(bankDb)/sizeof(bankDb[0]),
					       sizeof(integer4),event_uint_cmp_fun)))
    return bank_db_sorted_by_bank_name_hash[(bank_name_hash_ptr - bank_db_bank_name_hash_sorted)];
  return GET_BANK_UNKWN_BANK;
}


integer4 n_banks_total_()
{
  return sizeof(bankDb)/sizeof(bankDb[0]);
}

integer4 nBanksTotal()
{ return n_banks_total_(); }

integer4 event_all_banks_(integer4 *list)
{
  integer4 ibank;
  clr_bank_list_(list);
  for (ibank = 0; bankDb[ibank].name; ibank++)
    add_bank_list_(list, &bankDb[ibank].bank_id);
  return cnt_bank_list_(list);
}

integer4 eventAllBanks(integer4 list)
{ return event_all_banks_(&list); }

integer4 new_empty_bank_list_()
{
  integer4 size = n_banks_total_();
  integer4 list = new_bank_list_(&size);
  if(list == BANK_LIST_ERROR)
    return BANK_LIST_ERROR;
  return list;
}



integer4 newEmptyBankList()
{ return new_empty_bank_list_(); }

integer4 new_bank_list_with_all_banks_()
{
  integer4 size = n_banks_total_();
  integer4 list = new_bank_list_(&size);
  if(list == BANK_LIST_ERROR)
    return BANK_LIST_ERROR;
  if(!event_all_banks_(&list))
    return BANK_LIST_ERROR;
  return list;
}

integer4 newBankListWithAllBanks()
{ return new_bank_list_with_all_banks_(); }


integer4 event_commons_to_buffers_(const integer4* n_banks, const integer4* bank_types, 
				   integer4 *bank_buffer_sizes, integer1** bank_buffers)
{
  integer4 i;
  for (i = 0; i < *n_banks; i++)
    {
      integer4 rc = 0, ibank = bankid2dbindex(bank_types[i]);
      if (ibank < 0)
	{
	  fprintf(stderr, "event_bank_buffers_: error: unknown bank type: %d\n", bank_types[i]);
	  return ibank;
	}
      if((rc = bankDb[ibank].common_to_bank()) != SUCCESS)
	return rc;
      bank_buffers[i] = bankDb[ibank].bank_buffer(&bank_buffer_sizes[i]);
    }
  return SUCCESS;
}

integer4 eventCommonsToBuffers(integer4 n_banks, const integer4* bank_types, integer4 *bank_buffer_sizes, integer1** bank_buffers)
{
  return event_commons_to_buffers_(&n_banks,bank_types,bank_buffer_sizes,bank_buffers);
}


integer4 event_commons_to_buffers_from_bank_list_(integer4* bank_list, integer4* n_banks, 
						  integer4* bank_types, integer4 *bank_buffer_sizes, 
						  integer1** bank_buffers)
{
  integer4 type = 0, itr = 0;
  *n_banks = 0;
  if (cnt_bank_list_(bank_list) <= 0)
    return SUCCESS;
  for (itr = 0; (type = itr_bank_list_(bank_list, &itr)) > 0;)
    {
      integer4 rc = 0, ibank = bankid2dbindex(type);
      if (ibank < 0)
	{
	  fprintf(stderr, "event_bank_buffers_: error: unknown bank type: %d\n", type);
	  return ibank;
	}
      if((rc = bankDb[ibank].common_to_bank()) != SUCCESS)
	return rc;
      bank_types[*n_banks] = type;
      bank_buffers[*n_banks] = bankDb[ibank].bank_buffer(&bank_buffer_sizes[*n_banks]);
      (*n_banks) ++;
    }
  return SUCCESS;
}

integer4 eventCommonsToBuffersFromBankList(integer4 bank_list, integer4* n_banks, 
					   integer4* bank_types, integer4 *bank_buffer_sizes, 
					   integer1** bank_buffers)
{
  return event_commons_to_buffers_from_bank_list_(&bank_list, n_banks, bank_types, bank_buffer_sizes, bank_buffers);
}

integer4 event_buffers_to_commons_(const integer4* n_banks, const integer4* bank_types, integer1** bank_buffers)
{
  integer4 i = 0;
  for (i = 0; i < *n_banks; i++)
    {
      integer4 rc = 0, ibank = bankid2dbindex(bank_types[i]);
      if (ibank < 0)
	{
	  fprintf(stderr, "event_bank_buffers_: error: unknown bank type: %d\n", bank_types[i]);
	  return ibank;
	}
      if((rc = bankDb[ibank].bank_to_common(bank_buffers[i])) != SUCCESS)
	return rc;
    }
  return SUCCESS;
}

integer4 eventBuffersToCommons(integer4 n_banks, const integer4* bank_types, integer1** bank_buffers)
{
  return event_buffers_to_commons_(&n_banks,bank_types,bank_buffers);
}

integer4 event_id_from_name_(const integer1 *name)
{
  integer4 ibank = bankname2dbindex(name);
  if(ibank >= 0)
    return bankDb[ibank].bank_id;
  return 0;
}

integer4 eventIdFromName(const integer1 *name)
{ return event_id_from_name_(name); }

integer4 event_name_from_id_(integer4 *bank_id, integer1 *name, integer4 *len)
{
  integer4 ibank = bankid2dbindex(*bank_id);
  if(ibank >= 0)
    {
      strncpy(name, bankDb[ibank].name ? bankDb[ibank].name : "(null ptr)", *len);
      return *bank_id;
    }
  return 0;
}

integer4 eventNameFromId(integer4 bank, integer1 *name, integer4 len)
{ return event_name_from_id_(&bank, name, &len); }

integer4 event_version_from_id_(integer4 *bank_id)
{
  integer4 ibank = bankid2dbindex(*bank_id);
  if(ibank >= 0)
    return  bankDb[ibank].bank_version;
  return 0;
}

integer4 eventVersionFromId(integer4 bank)
{ return event_version_from_id_(&bank); }

integer4 event_set_dump_format_(integer4 *list, integer4 *format)
{
  integer4 rc = 0;
  if (cnt_bank_list_(list))
    {
      integer4 itr, bank_id;
      for (itr = 0; (bank_id = itr_bank_list_(list, &itr)) > 0; )
	{
	  integer4 ibank = bankid2dbindex(bank_id);
	  if(ibank >= 0)
	    bankDb[ibank].dump_format = *format;
	  else
	    {
	      fprintf(stderr, "event_set_dump_format: warning: unknown bank id %d.\n", bank_id);
	      rc ++;
	    }
	}
    }
  else
    {
      integer4 ibank;
      for (ibank = 0; bankDb[ibank].name; ibank++)
	bankDb[ibank].dump_format = *format;
    }
  return rc;
}

integer4 eventSetDumpFormat(integer4 list, integer4 format)
{ return event_set_dump_format_(&list, &format); }

integer4 event_dumpf_(FILE *fp, integer4 *list)
{
  integer4 rc = 0, itr = 0, bank_id = 0;
  for (itr = 0; (bank_id = itr_bank_list_(list, &itr)) > 0; )
    {
      integer4 ibank = bankid2dbindex(bank_id);
      if (ibank >= 0)
	{
	  integer4 fmt = bankDb[ibank].dump_format;
	  if (fmt != 2)
	    (bankDb[ibank].common_to_dumpf)(fp, &fmt);
	}
      else
	{
	  fprintf(stderr, "event_dumpf: warning: unknown bank id %d.\n", bank_id);
	  rc ++;
	}
    }
  return rc;
}

integer4 eventDumpf(FILE *fp, integer4 list)
{ return event_dumpf_(fp, &list); }

integer4 event_dump_(integer4 *list)
{
  return event_dumpf_(stdout, list);
}

integer4 eventDump(integer4 list)
{ return event_dumpf_(stdout, &list); }

integer4 event_read_(integer4 *unit, integer4 *want_banks, integer4 *got_banks, integer4 *event)
{
  /* Read bank from DST unit until bank type in want_bank list or start bank.
   * If start bank read, read all banks until stop bank. Returns number of bank
   * types in want_banks read (got_banks). Returns -1 if end of DST.
   */
  static integer1 *bank = NULL;
  integer4 ibank, rc, diag, len, type, ver;

  if (bank == NULL)
    {
      /* Determine size of maximum unpacked bank */
      len = 0;
      for (ibank = 0; bankDb[ibank].bank_id; ibank++)
	if (bankDb[ibank].bank_size > len)
	  len = bankDb[ibank].bank_size;
      len += 2 * sizeof(integer4);

      /* Try to make space for it. */
      if ((bank = malloc(len)) == NULL)
	{
	  fprintf(stderr, "event_read: error: failed to allocate memory for read bank.\n");
	  exit(1);
	}
    }

  diag = DIAG_WARN_DST;
  if (clr_bank_list_(got_banks) < 0 || cnt_bank_list_(want_banks) < 0)
    {
      fprintf(stderr, "event_read: error: bad bank list ids: %d, %d\n", *want_banks, *got_banks);
      exit(1);
    }
  *event = 0;

  while ((rc = dst_read_bank_(unit, &diag, bank, &len, &type, &ver)) == SUCCESS)
    {
      if (len <= 0) {
        fprintf(stderr, "event_read: error: bad size returned from dst_read_bank_: %d\n", len);
        exit(1);
      }

      /**
      if (type == FPKT1_BANKID) {
        fpkt1_.version = ver;
	} **/
      if (type == FMC1_BANKID) {
        fmc1_.version = ver;
      } /**
      if (type == FMC2_BANKID) {
        fmc2_.version = ver;
      }
      if (type == FMCDB_BANKID) {
        fmcdb_.version = ver;
      }
      if (type == HR2PROC_BANKID) {
        hr2proc_.version = ver;
      }
      if (type == HRMCPROC_BANKID) {
        hrmcproc_.version = ver;
	} **/
      if (type == START_BANKID)
	{
	  *event = 1;
	  continue;
	}
      if (type == STOP_BANKID)
	{
	  if (*event && cnt_bank_list_(got_banks) > 0) break;
	  *event = 0;
	  continue;
	}
      if (cnt_bank_list_(want_banks) == 0 || tst_bank_list_(want_banks, &type) == 1)
	{
	  if ((rc = add_bank_list_(got_banks, &type)) < 0)
	    {
	      fprintf(stderr, "event_read: error: failed to add bank to banks read list\n");
	      exit(1);
	    }
	  if (rc == 0)
	    fprintf(stderr, "event_read: warning: multiple banks of type %d in event\n", type);
	  if((ibank = bankid2dbindex(type)) >= 0)
	    {
	      if ((rc = bankDb[ibank].bank_to_common(bank)) != SUCCESS)
		return rc;
	    }
	  else
	    fprintf(stderr, "event_read: warning: unknown bank type: %d\n", type);
	  if (*event == 0)
	    break;
	}
    }
  return (rc == SUCCESS) ? cnt_bank_list_(got_banks) : rc;
}   

integer4 eventRead(integer4 unit, integer4 want_banks, integer4 got_banks, integer4 *event)
{ return event_read_(&unit, &want_banks, &got_banks, event); }

integer4 event_write_(integer4 *unit, integer4 *banks, integer4 *event)
{
  /* Write banks in bank list 'banks' to DST unit ID 'unit'.
   * Return number of banks written or error.
   */
  integer4 rc = 0, type = 0, itr = 0;

  if (cnt_bank_list_(banks) <= 0) return 0;

  if (*event && (rc = start_to_dst_(unit)) != SUCCESS)
    return rc;

  for (itr = 0; (type = itr_bank_list_(banks, &itr)) > 0;)
    {
      integer4 ibank = bankid2dbindex(type);
      if(ibank >= 0)
	{
	  if ((rc = (bankDb[ibank].common_to_dst)(unit)) != SUCCESS)
	    return rc;
	}
      else
	{
	  fprintf(stderr, "event_write: error: unknown bank type: %d\n", type);
	  exit(1);
	}
    }

  if (*event && (rc = stop_to_dst_(unit)) != SUCCESS)
    return rc;
  return cnt_bank_list_(banks);
}

integer4 eventWrite(integer4 unit, integer4 banks, integer4 event)
{ return event_write_(&unit, &banks, &event); }

integer4 dstOpenUnit(integer4 unit, const integer1 *file, integer4 mode)
{
  integer1* file_name_writable = (integer1*)malloc(strlen(file)+1);
  integer4 rc = 0;
  if(!file_name_writable)
    {
      fprintf(stderr,"error: dstOpenUnit: can't allocate memory for the file name\n");
      return MALLOC_ERR;
    }
  memcpy(file_name_writable,file,strlen(file)+1);
  rc = dst_open_unit_(&unit,file_name_writable, &mode);
  free(file_name_writable);
  return rc;
}

integer4 dstCloseUnit(integer4 unit)
{ return dst_close_unit_(&unit); }
