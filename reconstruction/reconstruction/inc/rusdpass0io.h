/*
 *      CLASS TO HANDLE RAW SD DATA I/O 
 */

#ifndef RUSDPASS0IO_H_
#define RUSDPASS0IO_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include "event.h"
#include "sduti.h"
#include "filestack.h"
#include "rusdpass0_const.h"


class listOfOpt
{
 public:
  int  yymmdd;                        // date
  char dout[DOUT_NAME_LEN];           // output directory
  char prodir[ASCII_NAME_LEN];        // processing directory
  char evtfile[ASCII_NAME_LEN];       // event output DST file
  char monfile[ASCII_NAME_LEN];       // monitoring output DST file
  int  wevt;                          // write out the event information flag
  int  wmon;                          // write out the monitoring information flag
  int  rem_tmp;                       // remove any temporary files at the end of the processing
  bool fOverwrite;                    // force overwrite mode (for the output DST files)
  bool fIncomplete;                   // allow incomplete data sets in processing
  int  tmatch_usec;                   // events within this many usec are considered same event 
                                      // and are merged
  int  dup_usec;                      // events occuring withing these many us are considered duplicated 
                                      // and are cleaned out 
  int  verbose;                       // verbose mode flag
    
  bool getFromCmdLine(int argc, char **argv);
  void printOpts(); // print out the arguments
  listOfOpt();
  ~listOfOpt();
 private:
  bool checkOpt();  // check & make sure that the options make sense
  void printErr(const char *form, ...);
};


class rusdpass0io
{

 public:
  // Load the list of BR,LR,SK run-files from a want-file
  // wantdate is in yymmdd format
  // Open the first 3 files, 1 for BR, 1 for LR, 1 for SK
  // If there are no raw files for some tower, then exit the program
  // dout - output directory for DST files
  rusdpass0io(listOfOpt& passed_opt);
  
  virtual ~rusdpass0io();

  // Get the total number of data files for a given tower
  int GetNfiles(int itower);
  
  // Return the date for which events need to be read out
  int GetSetDate();
  
  // Get the curent file index number
  int GetReadFileNum(int itower);

  // Get file name of the current file that's being read out for a given tower
  const char* GetReadFile(int itower);
  
  // Get the run ID of the current file that's being read out for a given tower
  int GetReadFileRunID(int itower);
  
  // Get file name corresponding to given file index and tower
  const char* GetReadFile(int ifile, int itower);
  
  // Get the run id for a given file index and tower
  int GetReadFileRunID(int ifile, int itower);

  // Get current read line for the current file
  int GetReadLine(int itower);
  
  // true:  successfuly read a line from an ascii file
  // false: no more ascii files and the last ascii file has finished 
  bool get_line(int itower, char *line);
  bool writeEvent(rusdraw_dst_common *event);
  bool writeMon(sdmon_dst_common *mon);


  // these are usefull for the recovery of timing
  
  // saves the current position in the data stream
  bool save_current_pos(int itower);
  
  // recovers the saved position in the raw data stream
  // and re-sets the corresponding variables which indicate that the 
  // current file position was saved
  bool goto_saved_pos(int itower);

 private:

  const listOfOpt& opt; // reference to the list of options passed to the analysis from the command line
  
  // 0 - BR, 1 - LR, 2 - SK
  int nrawfiles[3];
  uint64_t full_run_id[NRAWFILESPT][3]; // RUN ID's: file run ID + YEAR * 1000000
  int run_jd[NRAWFILESPT][3]; // RUN DATE in JD since Jan 1, 2000
  bool has_date[NRAWFILESPT][3]; // True if the run has a valid date in it
  bool run_needed[NRAWFILESPT][3]; // True means tower run is needed for parsing the set date
  char rawfname[NRAWFILESPT][3][ASCII_NAME_LEN];
  bool is_rawfname_tmp[NRAWFILESPT][3]; // true if the file was temporarily created by unpacking routine
  int irawfile[3]; // index of the current open raw file
  FILE* currawfile[3]; // pointer to current raw file for each tower
  int iline[3];        // current read line in a current opened ascii file
  
  // For writing events and monitoring information into DST files
  integer4 evt_outUnit;
  integer4 evt_outBanks;
  integer4 mon_outUnit;
  integer4 mon_outBanks;
  
  // unpack the .tar.bz2 file and return the path to the unpacked .Y???? - file
  // works only on unix systems.  On other systems, will return null and print
  // an error messsage. Returns a pointer to the unpacked file name string in
  // case of success.
  const char* unpack_run_file(const char* bz2file, const char* prodir);
  
  // Simple function to check towe id validity and print error message
  bool chk_tower_id(int itower);

  // Obtain run information: tower id and run number
  bool getRunInfo(const char* fname, int *itower, int *irun_id);
  
  // determine the earliest date and time found in the .Y???? - file (fname)
  bool getRunStartDate(const char* fname, int *yymmdd_start,int *hhmmss_start);
  
  // Sort the files so that they are in order of increasing run_id for BR, LR, and SK
  void swap_run_files(int itower, int index1, int index2);
  
  void sort_by_full_run_id();
  
  bool check_missing();
  
  int    saved_irawfile[3]; // saved raw ascii file index, -1 if it hasn't been saved
  fpos_t saved_fpos[3];  // saved position in the file stream
  int    saved_iline[3];    // saved line number in the ascii file
  
  bool evt_file_open;  // flag to indicate that the event DST file has been opened
  bool mon_file_open;  // flag to indicate that the monitoring info DST file has been opened
  
  void printErr(const char *form, ...);
};

#endif /*RUSDPASS0IO_H_*/
