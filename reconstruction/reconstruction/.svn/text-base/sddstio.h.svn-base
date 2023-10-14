
//////////// CLASSES FOR HANDLING THE DST FILES IN SD ANALYSIS ////////////
////// Dmitri Ivanov, <dmiivanov@gmail.com>, Mar 05, 2019 ////////
/////////////////////////////////////////////////////////////////////////

#ifndef _sddstio_h_
#define _sddstio_h_

#include <cstdio>
#include <cstdlib>
#include <string>
#include "event.h"
#include "sduti.h"

class sddstio_class
{

public:
    
  // pass a bank list that needs to be read out.
  // if zero is passed, will attempt to read all possible event banks
  void setWantBanks(integer4 bankList = 0);
  
  // open a DST file for reading
  bool openDSTinFile(const char *dInFile);

  // open a DST file for writing ( if force_overwrite_mode is true then
  // existing files will be overwritten ) 
  bool openDSToutFile(const char *dOutFile, bool force_overwrite_mode = false);
  
  // simultaneously open intput and output DST files
  bool openDSTfiles(const char *dInFile, const char *dOutFile, bool force_overwrite_mode = false);
  bool readEvent();                    // to read next event in the DST file
  bool readEvent(integer4 bankList);  // to read next event in the DST file
                                      // but only using banks defined in the 'bankList'
  
  bool writeEvent();  // write event with same bank list as was in the input file
  
  // write out the event with bank list of the input dst file + additional banks in bank list
  // if combine_bank_lists is true.  If combine_bank_lists is false,
  // write out the evnet with banks in bankList only
  bool writeEvent(integer4 bankList, bool combine_bank_lists = true);
  
  void closeDSTinFile();  // close current input DST file
  void closeDSToutFile(); // close currrent output DST file
  void closeDSTfiles();   // close current input and output DST files
  
  // check if a bank is present in the event has been read out
  bool haveBank(integer4 bank_id, bool displayWarning=false); 
  
  // check if bank in bankList are present in the event that has been read out
  bool haveBanks(integer4 bankList, bool displayWarning=false); 
  
  // tells if the input or output streams have been opened
  bool inFileOpen()  { return inUnitOpen; }
  bool outFileOpen() { return outUnitOpen; }

  // to find out what DST I/O units are used by this instance of class
  int getInUnit()    { return inUnit;  }
  int getOutUnit()   { return outUnit; }

  integer4 getWantBanks()     { return wantBanks; }
  integer4 getGotBanks()      { return gotBanks;  }
  integer4 getOutBanks()      { return outBanks;  }
  const char* getDSTinFile()  { return inpfile.c_str(); }
  const char* getDSToutFile() { return outfile.c_str(); }

  // to get the read out event date and time from the available DST banks
  bool get_event_time(integer4 *yymmdd, integer4 *hhmmss, integer4 *usec);
  bool get_event_time(integer4 *event_yymmdd, real8 *event_second);
  
  // no extraneous printing by default
  sddstio_class(int verbosity_level = 0);
  ~sddstio_class();
  
private:
  int verbose;                     // verbosity level
  integer4 wantBanks,gotBanks,outBanks,inUnit,outUnit;
  bool inUnitOpen, outUnitOpen;
  std::string inpfile; // current input file
  std::string outfile; // current output file
  bool dst_curevent_write(); // writes out what's currently in outBanks
  void printErr(const char *form, ...) SDUTI_FORMAT_CHECK(2,3);
  
};

// Another class for handling DST files with pseudo-random access
class sd_dst_handler
{
public:

  integer4 GetWantBanks()
  {
    return wantBanks;
  }
  integer4 GetGotBanks()
  {
    return gotBanks;
  }
  integer4 GetWriteBanks()
  {
    return writeBanks;
  }
  int GetNevents()
  {
    if ((dst_mode == MODE_READ_DST) && (!readEntireFile))
      printErr("GetNevents: warning: %s has not been entirely read out yet", dst_file.c_str());
    return nevents;
  }
  // useful if a file has been read out before, and then it has been re-initialized
  // but the user wishes to use previously known information about the numbers of events
  void SetNevents(int set_nevents)
  {
    readEntireFile = true;
    nevents = set_nevents;
  }
  int GetCurEvent()
  {
    return cur_event;
  }
  const char* GetFileName()
  {
    return dst_file.c_str();
  }
  void SetWriteBanks(integer4 bankList)
  {
    clrBankList(writeBanks);
    cpyBankList(writeBanks, bankList);
  }
  void AddWriteBanks(integer4 bankList)
  {
    sumBankList(writeBanks, bankList);
  }
  // if the dst unit is used outside of the class by different routines
  // then one should register it within the class so that the class
  // doesn't try to assign it to different files
  void SetDstUnitUsed(integer4 unit);
 

  // retrieve event with (C-style) index from the dst file
  void get_event(int ievent);

  // read event from the dst file;
  // read in banks that are in wantBank which
  // is all known banks
  bool read_event();

  // write out bank in writeBanks into the dst file
  // if was open for writing.
  bool write_event();

  // initialize the class and opend the dst file for
  // either reading or writing
  sd_dst_handler(const char* dstfile, int mode, bool fOverWriteMode = false);
  virtual ~sd_dst_handler();


  // get simple event time information from the dst:
  // yymmdd = date, hhmmss = time, usec = event micro second
  bool get_event_time(int *yymmdd, int *hhmmss, int *usec);

  // get event time information from the read out banks
  // event_yymmdd = event date
  // event_second = event second since midnight plus the second fraction
  // return: true = success, false = failure
  bool get_event_time(int *event_yymmdd, double *event_second);

private:
  
  std::string dst_file;
  integer4 dst_unit;
  integer4 dst_mode;
  integer4 wantBanks;
  integer4 gotBanks;
  integer4 writeBanks;

  bool readEntireFile;
  int nevents;
  int cur_event; // current event number (C-style)



  void printErr(const char *form, ...) SDUTI_FORMAT_CHECK(2,3);

};

#endif
