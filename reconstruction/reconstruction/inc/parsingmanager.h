/*
 * CLASS WHICH WILL MANAGE PARSED EVENTS AND MONITORING CYCLES. DECIDES HOW TO TIME MATCH THEM
 * AND WRITE OUT INTO DST FILES.
 */

#ifndef PARSINGMANAGER_H_
#define PARSINGMANAGER_H_

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rusdpass0io.h"
#include "rusdpass0_const.h"
#include "tower_parser.h"
#include "sdindexing.h"
#include "event.h"
#include <vector>

//// Needed to prepare 1MIP calibration /////
#include "TMath.h"
#include "TF1.h"
#include "TH1F.h"

using namespace std;



class parsingManager
{
 public:
  // Accepts pointer to I/O handler.  The pointer should be properly initialized 
  // by main program.
  parsingManager(const listOfOpt& opt, rusdpass0io* rusdpass0io_pointer);

  // Do the parsing
  void Start();

  // Print out the statistics
  void printStats(FILE *fp);

  void printStats();

  // Buffers for events that were read out
  int n_readout_events[3];
  rusdraw_dst_common *readout_events[3];
  int readout_event_index[3][NRUSDRAW_TOWER];

  // Buffers for monitoring cycles that were readout
  int n_readout_mon[3];
  sdmon_dst_common *readout_mon[3]; // buffers for read out monitoring cycles
  int readout_mon_index[3][NSDMON_TOWER];

  // Buffer for time-matched events
  int n_tmatched_events;
  rusdraw_dst_common *tmatched_events;
  int tmatched_event_index[NRUSDRAW_TMATCHED];

  // Buffer for time-matched monitoring cycles
  int n_tmatched_mon;
  sdmon_dst_common *tmatched_mon;
  int tmatched_mon_index[NSDMON_TMATCHED];

  sdmon_dst_common last_good_mon; // Always keep in a buffer latest good monitornig cycle
  bool have_good_mon; // is true when have a good monitoring cycle in the buffer

  // true when  a given time-matched event is calibrated
  bool is_event_calibrated[NRUSDRAW_TMATCHED];

  rusdraw_dst_common curEvent; // Event buffer for time-matching

  sdmon_dst_common curMon; // Monitoring cycle buffer for time-matching

 private:
  
  const listOfOpt& opt; // reference to the list of options passed to the analysis from the command line
  rusdpass0io* p0io;
  sdindex_class *sdi; // indexing on monitoring cycles
  towerParser *raw[3]; // Parsers for BR,LR,SK


  int nevents_tower_total[3]; // total number of events from each tower on a given date
  int nmon_tower_total[3]; // total number of monitoring cycles from each tower on a given date
  int nevents_tmatched_total; // total number of time-matched events on a given date
  int nmon_tmatched_total; // total number of time-matched monitoring cycles on a given date

  bool need_data[3]; // To indicate that the data from a given tower is needed at the moment
  bool have_data[3]; // Set to true if data files for given tower are not finished
  bool read_data[3]; // True if need and have data for a given tower

  int curevt_num; // Number of the current combined event
  int curmon_num; // Number of the current combined monitoring cycle


  int cur_moncycle_second; // current second for monitoring cycles
  int min_moncycle_second; // monitoring cycle second must be greater of equal to this value
  int max_moncycle_second; // monitoring cycle second must be less than this value


  // Sometimes, tower events are not in time order.  Before processing them, we should sort them
  // and make them be in the time order.  This routine sorts events in all towers.
  void sort_tower_events();

  // After time-matching, it is possible that some events are not in time-order because time-matching
  // routine loops over each tower and looks for time-matching candidates in remaining 2 towers.
  void sort_tmatched_events();

  bool save_event(int itower); // Save tower event to a buffer
  bool save_event(); // Save time-matched event to a buffer
  void remove_event(int itower, int ievent); // remove tower event from the buffer
  void remove_event(int ievent); // remove time-matched event from the buffer
  rusdraw_dst_common* obtain_event(int itower, int ievent); // gets tower event from the buffer
  rusdraw_dst_common* obtain_event(int ievent); // gets time matched event from the buffer

  bool save_mon(int itower); // saves tower monitoring cycle to a buffer
  bool save_mon(); // saves time-matched monitoring cycle to a buffer
  void remove_mon(int itower, int imon); // remove mon. cycle from the buffer
  void remove_mon(int imon); // removing time matched mon. cycle from the buffer
  sdmon_dst_common* obtain_mon(int itower, int imon); // gets the tower mon. cycle from the buffer
  sdmon_dst_common* obtain_mon(int imon); // gets the time matched mon. cycle from the buffer


  bool process_data();

  // For combinig events
  bool addEvent(rusdraw_dst_common *comevent, rusdraw_dst_common *event);

  // For combining monitoring cycles
  bool addMonCycle(sdmon_dst_common *comcycle, sdmon_dst_common *mon);

  // Attach calibration to a given event from a given monitoring cycle
  void calibrate_event(rusdraw_dst_common *event, sdmon_dst_common *mon);
  bool calibrate_events(); // Calibrates time matched that are in the writing buffer
  bool write_out_events(); // For writing out events into DST files
  bool write_out_mon(); // For writing out monitoring cycles into DST files
  bool tower_id_com(int towid_evt, int towid_mon ); // check if event tower ID is compatible with montitoring cycle 
  
  bool chk_tower_id(int itower); // to check tower id
  void printErr(const char *form, ...);

  ///// To prepare the 1MIP calibration /////////

  TH1F *hmip[2]; // Histograms for 1MIP fitting
  TF1 *ffit; // Fit-function for 1MIP fitting
  // For fitting mip histograms into gaussian*pol1 function
  void mipFIT(integer4 ind, sdmon_dst_common *mon);
  // To calculate the peak values of the histograms and find their half-peak channels.
  void compMonCycle(sdmon_dst_common *mon);

  //////////////////////////////////////////////
  
  // remove the duplicated trigger events, if any
  // this should be called after the time matching is
  // done
  bool remove_duplicated_events();
  int n_duplicated_removed; // number of duplicated events removed
  
  int n_bad_calib; // number of badly calibrated events
  
  // to store times of all written events to help with
  // removing the duplicated events
  vector<double> written_event_times;
  
  int fSuccess; // overall success flag
};

#endif /*PARSINGMANAGER_H_*/
