/*
 * CLASS WHICH PARSES THE RAW FILES FOR BR,LR,SK TOWERS.  RETURNS EVENTS OR MONITORING CYCLES,
 * AND THEN PARSER MANAGER CLASS DECIDES WHAT TO DO WITH THEM.
 */

#ifndef TOWER_PARSER_H_
#define TOWER_PARSER_H_
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include "event.h"
#include "rusdpass0_const.h"
#include "rusdpass0io.h"
#include "sdindexing.h"


// For saving some repetition headers for determining event trig times
typedef struct
{
  int yymmdd;
  int hhmmss;
  int secnum;
} gpsTime_struct;

// structure to temporarily store the information 
// for recovering the GPS time
typedef struct
{
  int yymmdd;    // date
  int hhmmss;    // time
  int j2000sec;  // second since Jan 1, 2000
  int secnum;    // mod 600 second
  // calculate the rest of the timing variables
  // from the date and time
  void calc_the_rest()
  {
    j2000sec = SDGEN::time_in_sec_j2000(yymmdd,hhmmss);
    secnum   = SDGEN::timeAftMNinSec(hhmmss) % 600;
  }
} tline_recovery_struct;



class towerParser
{
 public:
  
  // Class is initialized by passing tower id and INITIALIZED pointer to
  // I/O handler
  towerParser(const listOfOpt& passed_opt, int itower, rusdpass0io* rusdpass0io_pointer);
  virtual ~towerParser();

  rusdpass0io* p0io; // To read inputs from ascii files
  int tower_id; // 0 - BR, 1-LR, 2-SK
  void clean_event(); // Clean up current event structure
  void clean_mon(); // Clean up current monitoring cycle


  // Parses the raw data for a given tower;

  /*
   * Returns when either occurs:
   * READOUT_EVENT: Read out an event
   * READOUT_MON: Read out a monitoring cycle
   * READOUT_ENDED: Data files ended (not enough data for a given date ?)
   * READOUT_FATAL_ERROR in case something goes very bad with data and the program knows about it
   */
  int Parse();

  void get_curEvent(rusdraw_dst_common *event); // Obtian the current event
  int get_curEventDate(); // Return date of the event in yymmdd format
  void get_curMon(sdmon_dst_common *mon); // Obtain the current mon. cycle
  int get_curMonDate(); // Return date at the beginning of the monitoring cycle, yymmdd format
  void cleanEvent(rusdraw_dst_common *event); // Clean event buffer structure
  void cleanEvent(); // clean the readout buffer for the event
  void cleanMonCycle(sdmon_dst_common *mon); // Clean a given monitoring cycle (set variables to -1)
  void cleanMonCycle(); // clean the readout buffer for the monitoring cycle
  // To fix mon. cycle start date and time so that it's mod 600 sec since midnight
  void fixMonCycleStart(int hhmmss_act, int *hhmmss_fix, int *deltaSec); 

  // Parsing statistics for the given tower
  int nTlines, nLlines, nElines, nEendLines, nWlines, nwLines;
  int total_readout_problems; // counts DAQ problems
  int event_readout_problems; // counta the event readout problems.
  int secnum_mismatches; // (minor) problems at getting the monitoring information
  int mon_readout_problems; // counts problems with reading out the monitoring cycle
  int pps_1sec_problems; // how many times PPS was missed and had to increase the GPS second by 1
  void printStats(FILE *fp); // prints out the parsing statistics for the given tower
  void printStats(); // Simple print out of parsing stats into stdout
  int fSuccess; // 1 - success, 0 - failure

 private:
  
  const listOfOpt& opt; // reference to the list of options passed to the analysis from the command line
  int start_yymmdd; // date from which we start parsing
  int maxIND; // max. value of monitoring index
  int on_time; // detector on-time in seconds for the given date
  sdindex_class *sdi; // internal indexing handler for monitoring information
  rusdraw_dst_common eventBuf; // Event read out buffer
  sdmon_dst_common monBuf; // Monitoring cycle read out buffer
  gpsTime_struct gpstbuf[MAXNGPST];
  int igpstime; // internal GPS time indexing variable
  void addGpsTime(int yymmdd, int hhmmss, int secnum);
  bool getGpsTime(int secnum, int *yymmdd, int *hhmmss);
  // Computes the off-time for the given set_yymmdd date.  
  // jte, jto are the expected and obtained times in seconds since midnight of Jan 1, 2000
  int getOffTime(int jte, int jto);
  int whatLine(char *aline);
  void printErr(const char *form, ...); // To print parsing error messages

  bool firstTline; // True when reading out the 1st T-line by Parse() method
  bool firstMonCycle; // True when reading out the 1st monitoring cycle by Parse() method
  bool onEvent; // set to true after every event header and to false after every event end line
  bool onMon; // is always set to true unless just returned a monitoring cycle


  // Some parsing variables which should be preserved for in each tower case in the program run-time
  int yymmdd_cur, // Current read out date
    hhmmss_cur, // Current read out time  
    yymmdd_exp, // Expected read out date
    hhmmss_exp, // Expected read out time
    secnum_pps_cur, // Previous second number from PPS header
    secnum_pps_last, // Current second number from PPS header
    secnumlast, // Previous secnum from L-line
    secnumcur, // Current secnum from L-line
    wfCount[2]; // to count the waveforms.  First one ([0]) is b/w the W-lines, and 2nd one ([1]) is over the entire event
  
  bool time_recovery_mode; // Is true when we are recovering the GPS time flow. 
  
  int n_tline_recovery; // number of elemts in the time recovery structure
  tline_recovery_struct tline_recovery[N_TLINE_RECOVERY];
};

#endif /*TOWER_PARSER_H_*/
