#include "sdanalysis_icc_settings.h"
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include "sduti.h"
#include "tafd10info.h"
#include "sddstio.h"
#include "dst_size_limits.h"

using namespace std;

// static variables for keeping track of which dst I/O units have been opened
// by this class; allocate them here.
static bool sddstio_dst_units_initialized = false;
static bool sddstio_dst_unit_used[MAX_DST_FILE_UNITS];


static bool get_dst_event_time(integer4 gotBanks, int *yymmdd, int *hhmmss, int *usec)
{
  // get event time from SD banks
  if (tstBankList(gotBanks,TASDCALIBEV_BANKID))
    {
      (*yymmdd) = tasdcalibev_.date;
      (*hhmmss) = tasdcalibev_.time;
      (*usec) = tasdcalibev_.usec;
      return true;
    }
  if (tstBankList(gotBanks,RUSDRAW_BANKID))
    {
      (*yymmdd) = rusdraw_.yymmdd;
      (*hhmmss) = rusdraw_.hhmmss;
      (*usec) = rusdraw_.usec;
      return true;
    }
  // get event date and time from BR/LR FD fdplane banks
  if (tstBankList(gotBanks,FDPLANE_BANKID))
    {
      tafd10info::get_brlr_time(fdplane_.julian, fdplane_.jsecond, yymmdd, hhmmss);
      (*usec) = (integer4) floor(((real8) fdplane_.jsecfrac) * 1.0e-3 + 0.5);
      return true;
    }
  if (tstBankList(gotBanks,BRPLANE_BANKID))
    {
      tafd10info::get_brlr_time(brplane_.julian, brplane_.jsecond, yymmdd, hhmmss);
      (*usec) = (integer4) floor(((real8) brplane_.jsecfrac) * 1.0e-3 + 0.5); 
      return true;
    }
  if (tstBankList(gotBanks,LRPLANE_BANKID))
    {      
      tafd10info::get_brlr_time(lrplane_.julian, lrplane_.jsecond, yymmdd, hhmmss);
      (*usec) = (integer4) floor(((real8) lrplane_.jsecfrac) * 1.0e-3 + 0.5);
      return true;
    }
  // get event date and time from MD banks
  if (tstBankList(gotBanks,HRAW1_BANKID))
    {
      tafd10info::get_md_time(hraw1_.jday, hraw1_.jsec, yymmdd, hhmmss);
      (*usec) = (integer4) floor((real8) hraw1_.mirtime_ns[0] / 1.0e3 + 0.5);
      return true;
    }
  if (tstBankList(gotBanks,MCRAW_BANKID))
    {
      tafd10info::get_md_time(mcraw_.jday, mcraw_.jsec, yymmdd, hhmmss);
      (*usec) = (integer4) floor((real8) mcraw_.mirtime_ns[0] / 1.0e3 + 0.5);
      return true;
    }
  // get event date and time from TALE FD bank
  if(tstBankList(gotBanks,FRAW1_BANKID))
    {
      (*yymmdd) = fraw1_.julian-2000*10000;
      (*hhmmss) = (fraw1_.jsecond/3600) * 10000 + 
	((fraw1_.jsecond/60)%60) * 100 + fraw1_.jsecond%60;
      (*usec) = (integer4)floor(((real8) fraw1_.jclkcnt + (100.0/6.0) * (real8)fraw1_.clkcnt[0]) / 1e3 + 0.5);
      return true;
    }
  // get event date and time from GENERALIZED SD bank
  if (tstBankList(gotBanks,TALEX00_BANKID))
    {
      (*yymmdd) = talex00_.yymmdd;
      (*hhmmss) = talex00_.hhmmss;
      (*usec)   = talex00_.usec;
      return true;
    }
  // get event date and time from ETRACK bank
  if (tstBankList(gotBanks,ETRACK_BANKID))
    {
      (*yymmdd) = etrack_.yymmdd;
      (*hhmmss) = etrack_.hhmmss;
      (*usec)   = (integer4)floor(etrack_.t0+0.5);
      return true;
    }
  // could not get event time - no banks with time information
  return false;
}

static bool get_dst_event_time(integer4 gotBanks, int *event_yymmdd, double *event_second)
{
  int hhmmss, usec;
  if (!get_dst_event_time(gotBanks,event_yymmdd, &hhmmss, &usec))
    return false;
  (*event_second) = (double) (3600 * (hhmmss / 10000) + 60 * (((hhmmss % 10000)) / 100) + hhmmss % 100) + (1E-6)
    * (double) usec;
  return true;
}


sddstio_class::sddstio_class(int verbosity_level)
{
  // initialize dst unit tracking variables, if necessary
  if (!sddstio_dst_units_initialized)
    {
      for (int i = 0; i < MAX_DST_FILE_UNITS; i++)
	sddstio_dst_unit_used[i] = false;
      sddstio_dst_units_initialized = true;
    }
  // assign input and output DST units from those that are not being currently used by
  // other instances of this class
  inUnit  = -1;
  outUnit = -1;
  for (int i = 0; i < MAX_DST_FILE_UNITS; i++)
    {
      if(!sddstio_dst_unit_used[i] && inUnit == -1)
	{
	  inUnit = i;
	  sddstio_dst_unit_used[i] = true;
	}
      if(!sddstio_dst_unit_used[i] && outUnit == -1)
	{
	  outUnit = i;
	  sddstio_dst_unit_used[i] = true;
	}
      if (inUnit != -1 && outUnit != -1)
	break;
    }
  if (inUnit == -1 || outUnit == -1)
    {
      printErr("fatal error: not enough free dst units, maximum %d instances of sddstio are allowed",
	       (int)(MAX_DST_FILE_UNITS/2));
      exit(2);
    }
  inpfile[0] = 0;
  outfile[0] = 0;  
  wantBanks = newBankListWithAllBanks();
  gotBanks =  newEmptyBankList();
  outBanks =  newEmptyBankList();
  clrBankList(outBanks);
  inUnitOpen = false;
  outUnitOpen = false;
  verbose = verbosity_level;
}

sddstio_class::~sddstio_class() 
{
  // de-allocate all bank lists that were allocated for this instance
  clrBankList(wantBanks);
  delBankList(wantBanks);
  clrBankList(gotBanks);
  delBankList(gotBanks);
  clrBankList(outBanks);
  delBankList(outBanks);
  // mark off the dst unit numbers as "not used" that were used by this instance
  sddstio_dst_unit_used[inUnit]  = false;
  sddstio_dst_unit_used[outUnit] = false;
}

void sddstio_class::setWantBanks(integer4 bankList)
{  
  if (bankList <= 0)
    eventAllBanks(wantBanks);
  else
    cpyBankList(wantBanks,bankList);
}


bool sddstio_class::openDSTinFile(const char *dInFile)
{
  if (!SDIO::check_dst_suffix(dInFile))
    return false;
  inpfile = dInFile;
  FILE *fp = fopen(inpfile.c_str(), "r");
  if (!fp)
    {
      printErr("Can't open %s", inpfile.c_str());
      return false;
    }
  fclose(fp);
  if (dstOpenUnit(inUnit,inpfile.c_str(),MODE_READ_DST) != SUCCESS)
    {
      printErr("Can't open %s", inpfile.c_str());
      return false;
    }
  inUnitOpen = true;
  if(verbose > 0)
    {
      fprintf(stdout, "Opened DST file: %s\n", inpfile.c_str());
      fflush(stdout);
    }
  return true;
}

bool sddstio_class::openDSToutFile(const char *dOutFile, bool force_overwrite_mode)
{
  if (!SDIO::check_dst_suffix(dOutFile))
    return false;
  outfile = dOutFile;
  if (!force_overwrite_mode)
    {
      FILE *fp = fopen(outfile.c_str(), "r");
      if (fp)
	{
	  printErr("'%s' - file exists", outfile.c_str());
	  fclose(fp);
	  return false;
	}
    }
  FILE *fp = fopen(outfile.c_str(), "w");
  if (!fp)
    {
      printErr("Can't start %s", outfile.c_str());
      return false;
    }
  fclose(fp);
  if (dstOpenUnit(outUnit, outfile.c_str(), MODE_WRITE_DST) != SUCCESS)
    {
      printErr("Can't start %s", outfile.c_str());
      return false;
    }
  outUnitOpen = true;
  if(verbose > 0)
    {
      fprintf(stdout, "Started DST file: %s\n", outfile.c_str());
      fflush(stdout);
    }
  return true;
}

bool sddstio_class::openDSTfiles(const char *dInFile, const char *dOutFile, bool force_overwrite_mode)
{
  return (openDSTinFile(dInFile) && openDSToutFile(dOutFile,force_overwrite_mode));
}


bool sddstio_class::readEvent()
{
  integer4 event = 0;
  if (eventRead(inUnit, wantBanks, gotBanks, &event) < 0)
    return false; // Can't read any more events, at the end of the file.
  if (!event)
    {
      printErr("Failed to read an event from %s",inpfile.c_str());
      return false;
    }
  return true;
}

bool sddstio_class::writeEvent(integer4 bankList, bool combine_bank_lists)
{
  if (combine_bank_lists)
    {
      cpyBankList(outBanks,gotBanks);
      sumBankList(outBanks,bankList);
      return dst_curevent_write();
    }
  cpyBankList(outBanks,bankList);
  return dst_curevent_write();
}

bool sddstio_class::writeEvent()
{
  cpyBankList(outBanks,gotBanks);
  return dst_curevent_write();
}

bool sddstio_class::dst_curevent_write()
{
  if (eventWrite(outUnit, outBanks, TRUE) < 0)
    {
      printErr("Failed to write an event to %s",outfile.c_str());
      return false; // Can't write events...
    }
  return true;
}

bool sddstio_class::haveBank(integer4 bank_id, bool displayWarning)
{
  if (tstBankList(gotBanks,bank_id))
    return true;    
  if (displayWarning)
    {
      integer1 bname[0x20];
      if(!eventNameFromId(bank_id,bname,(integer4)sizeof(bname)))
	{
	  printErr("haveBank: failed to identify bank name for bank_id=%d",
		   bank_id);
	  return false;
	}
      fprintf(stderr,"WARNING: missing banks: %s\n",bname);
    }
  return false;	
}

bool sddstio_class::haveBanks(integer4 bankList, bool displayWarning)
{
  if(cmpBankList(gotBanks,bankList) == cntBankList(bankList))
    return true;
  if (displayWarning)
    {
      integer4 noBanks=newEmptyBankList(); 
      integer1 bname[0x20];
      cpyBankList(noBanks,bankList);
      difBankList(noBanks,gotBanks);
      fprintf(stderr, "WARNING: mssing banks:");
      integer4 i = 0, bank_id = 0;
      while((bank_id = itrBankList(noBanks,&i)) > 0)
	{
	  if(!eventNameFromId(bank_id,bname,(integer4)sizeof(bname)))
	    {
	      printErr("\nhaveBanks: failed to identify bank name for bank_id=%d",
		       bank_id);
	      clrBankList(noBanks);
	      delBankList(noBanks);
	      return false;
	    }
	  fprintf(stderr," %s",bname);
	}
      fprintf(stderr,"\n");
      clrBankList(noBanks);
      delBankList(noBanks);
    }
  return false;
}


bool sddstio_class::get_event_time(integer4 *yymmdd, integer4 *hhmmss, integer4 *usec)
{
  return get_dst_event_time(gotBanks,yymmdd,hhmmss,usec);
}

bool sddstio_class::get_event_time(integer4 *event_yymmdd, real8 *event_second)
{
  return get_dst_event_time(gotBanks,event_yymmdd,event_second);
}

void sddstio_class::closeDSTinFile()
{
  if (inUnitOpen)
    dstCloseUnit(inUnit);
  inUnitOpen = false;
}
void sddstio_class::closeDSToutFile()
{
  if (outUnitOpen) 
    dstCloseUnit(outUnit);
  outUnitOpen = false;
}
void sddstio_class::closeDSTfiles()
{
  closeDSTinFile();
  closeDSToutFile();
}
void sddstio_class::printErr(const char *form, ...)
{
  va_list args;
  va_start(args, form);
  SDIO::vprintMessage(stderr, "sddstio", form, args);
  va_end(args);
}


sd_dst_handler::sd_dst_handler(const char* dstfile, int mode, bool fOverWriteMode)
{
  dst_file = dstfile;
  if ((mode != MODE_READ_DST) && (mode != MODE_WRITE_DST))
    {
      printErr("fatal error: invalid dst open mode for %s", dst_file.c_str());
      exit(2);
    }
  dst_mode = mode;
  // to keep track of which dst units have been opened
  if (!sddstio_dst_units_initialized)
    {
      for (int iu = 0; iu < MAX_DST_FILE_UNITS; iu++)
        sddstio_dst_unit_used[iu] = false;
      sddstio_dst_units_initialized = true;
    }

  readEntireFile = false; // the entire dst file has not been read out yet
  nevents = 0;
  cur_event = -1;

  wantBanks  = newBankListWithAllBanks();
  gotBanks   = newEmptyBankList();
  writeBanks = newEmptyBankList();

  dst_unit = -1;
  for (int iu = 0; iu < MAX_DST_FILE_UNITS; iu++)
    {
      if (!sddstio_dst_unit_used[iu])
        {
          dst_unit = iu;
          sddstio_dst_unit_used[iu] = true;
          break;
        }
    }
  if (dst_unit == -1)
    {
      printErr("fatal error: no dst units available for %s", dst_file.c_str());
      exit(2);
    }
  if (mode == MODE_READ_DST)
    {
      FILE *fp = fopen(dst_file.c_str(), "r");
      if (!fp)
        {
          printErr("%s doesn't exist", dstfile);
          exit(2);
        }
      fclose(fp);
      if (dstOpenUnit(dst_unit, dst_file.c_str(), mode) != SUCCESS)
        {
          printErr("can't dst-open %s for reading", dst_file.c_str());
          exit(2);
        }
    }
  else
    {
      if (!fOverWriteMode)
        {
	  FILE *fp = fopen(dst_file.c_str(), "r");
          if (fp)
            {
              printErr("error: %s exists; use force overwrite mode option", dst_file.c_str());
              fclose(fp);
	      exit(2);
            }
        }
      FILE *fp = fopen(dst_file.c_str(), "w");
      if (!fp)
        {
          printErr("can't start %s", dst_file.c_str());
          exit(2);
        }
      fclose(fp);
      if (dstOpenUnit(dst_unit, dst_file.c_str(), mode) != SUCCESS)
        {
          printErr("can't dst-open %s for writing", dst_file.c_str());
          exit(2);
        }
    }
}

void sd_dst_handler::SetDstUnitUsed(integer4 unit)
{
  if (unit < 0 || unit >= MAX_DST_FILE_UNITS)
    printErr("SetDstUnit: invalid dst unit");
  else
    sddstio_dst_unit_used[unit] = true;
}

void sd_dst_handler::get_event(int ievent)
{
  if (dst_mode != MODE_READ_DST)
    {
      printErr("fatal error: attempting to read an event from a file %s opened for writing", dst_file.c_str());
      exit(2);
    }
  if (ievent < 0)
    {
      printErr("ivalid event index in the dst file %s", dst_file.c_str());
      exit(2);
    }
  if (ievent >= nevents)
    {
      printErr("ERROR: attempting to read an event in %s with index %d larger than the current maximum %d",
	       dst_file.c_str(), ievent, nevents - 1);
      if (!readEntireFile)
        printErr("%s has not been entirely read out yet so don't know what's the real maximum index", dst_file.c_str());
      exit(2);
    }
  if (cur_event > ievent)
    {
      cur_event = -1;
      dstCloseUnit(dst_unit);
      FILE *fp = fopen(dst_file.c_str(), "r");
      if (!fp)
        {
          printErr("ERROR: failed to re-open %s", dst_file.c_str());
          exit(2);
        }
      fclose(fp);
      if (dstOpenUnit(dst_unit, dst_file.c_str(), dst_mode) != SUCCESS)
        {
          printErr("ERROR: failed to dst re-open %s for reading", dst_file.c_str());
          exit(2);
        }
    }
  while (cur_event != ievent)
    {
      if (!read_event())
        {
          printErr("get_event: ERROR: event %d is supposed to be in %s but was unable to reach it", ievent,
		   dst_file.c_str());
          exit(2);
        }
    }
}

bool sd_dst_handler::read_event()
{
  if (dst_mode != MODE_READ_DST)
    {
      printErr("fatal error: attempting to read an event from a file %s opened for writing", dst_file.c_str());
      exit(2);
    }
  integer4 event = 0;
  // if can't read any more events, at the end of the file
  if (eventRead(dst_unit, wantBanks, gotBanks, &event) < 0)
    {
      // the file has been read out entirely at least once
      readEntireFile = true;
      cur_event = -1;
      dstCloseUnit(dst_unit);
      FILE *fp = fopen(dst_file.c_str(), "r");
      if (!fp)
        {
          printErr("fatal error: failed to re-open %s", dst_file.c_str());
          exit(2);
        }
      fclose(fp);
      if (dstOpenUnit(dst_unit, dst_file.c_str(), dst_mode) != SUCCESS)
        {
          printErr("fatal error: failed to dst re-open %s for reading", dst_file.c_str());
          exit(2);
        }
      return false;
    }
  if (!event)
    {
      printErr("fatal error: failed to read an event from %s", dst_file.c_str());
      exit(2);
    }
  if (!readEntireFile)
    nevents++;
  cur_event++;
  return true;
}

bool sd_dst_handler::write_event()
{
  if (dst_mode != MODE_WRITE_DST)
    {
      printErr("ERROR: attempting to write an event into a file %s opened for reading", dst_file.c_str());
      exit(2);
    }
  if (eventWrite(dst_unit, writeBanks, TRUE) < 0)
    {
      printErr("ERROR: failed to write an event to %s", dst_file.c_str());
      exit(2);
    }
  nevents++;
  cur_event++;
  return true;
}

bool sd_dst_handler::get_event_time(int *yymmdd, int *hhmmss, int *usec)
{
  return get_dst_event_time(gotBanks,yymmdd,hhmmss,usec);
}

bool sd_dst_handler::get_event_time(int *event_yymmdd, double *event_second)
{
  return get_dst_event_time(gotBanks,event_yymmdd,event_second);
}

sd_dst_handler::~sd_dst_handler()
{
  clrBankList(wantBanks);
  delBankList(wantBanks);
  clrBankList(gotBanks);
  delBankList(gotBanks);
  clrBankList(writeBanks);
  delBankList(writeBanks);
  dstCloseUnit(dst_unit);
  sddstio_dst_unit_used[dst_unit] = false;
  dst_unit = -1;
}

void sd_dst_handler::printErr(const char *form, ...)
{
  va_list args;
  va_start(args, form);
  SDIO::vprintMessage(stderr, "sd_dst_handler", form, args);
  va_end(args);
}
