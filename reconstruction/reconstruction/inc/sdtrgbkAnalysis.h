#ifndef _sdtrgbkAnalysis_
#define _sdtrgbkAnalysis_

#include "sdanalysis_icc_settings.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "event.h"
#include "sdtrgbk_constants.h"
#include "sdinfo_class.h"
#include <vector>
#include <map>
#include "sdtrgbk.h"

using namespace std;

class sdtrgbkAnalysis
{

public:

  // Space pattern recognition for a triplet of counters
  // If a triplet of SDs is not spatially contiguous, then returns zero
  // If a triplet of SDs is spatially contiguous, then it returns the maximum
  // square separation distance between a pair of counters in the triplet:
  // 2 for simple triangle triggers, and 4 for the line triggers
  int space_pat_recog(int xxyy1, int xxyy2, int xxyy3);

  // Returns true if times t1,t2,t3 (second) are in the same time window
  bool in_time(double t1, double t2, double t3, double time_window = L2TWND);

  // Returns true if there is a permutation of indices i,j,k
  // such that t1[i], t2[j], t3[k] are in time (i,j,k go from 0 to 1)
  bool in_time(double t1[2], double t2[2], double t3[2], double time_window = L2TWND);

  // Analyze events using either Rutgers of ICRR banks
  void analyzeEvent(tasdevent_dst_common *tasdeventp, bsdinfo_dst_common *bsdinfo);
  void analyzeEvent(tasdcalibev_dst_common *tasdcalibevp, bsdinfo_dst_common *bsdinfo);
  void analyzeEvent(rusdraw_dst_common *rusdrawp, bsdinfo_dst_common *bsdinfo);
  
  sdtrgbkAnalysis(const listOfOpt& cmd_line_opt): opt(cmd_line_opt)
  { init(); }

  ~sdtrgbkAnalysis()
  {
    init();
  }

  int GetNSD()
  {
    return n_sdinfo_buffer;
  }
  int getNgoodPedSD()
  {
    return sd_good_ped.size();
  }
  int GetNbadSD()
  {
    return n_bad_sds;
  }
  int GetNspatContSD()
  {
    return sd_spat_cont.size();
  }
  int GetNisolSD()
  {
    return sd_spat_isol.size();
  }
  bool hasTriggered()
  {
    return has_triggered;
  }

private:
  const listOfOpt& opt; // command line options
  void load_sds(tasdevent_dst_common *p); // Load Sds using ICRR raw event bank
  void load_sds(tasdcalibev_dst_common *p); // Load Sds using ICRR calibrated event bank
  void load_sds(rusdraw_dst_common *p); // Load SDs using Rutgers bank
  void pick_good_ped(); // pick out SDs with pedestals >= 1 fadc count / fadc time slice
  // pick out SDs that are spatially contiguous (form triples) and also possible time contiguous
  // (according to their earliest and latest possible times)
  // picks out SDs that are spatially isolated
  // assigns potentially space-time contiguous SDs to trigger search table
  void pick_cont_sd();

  // finds a triplet of space and time contiguous SDs
  // that are also level-1 trigger SD. Supports raising or lowering the
  // pedestals.
  // DeltaPed < 0: pedestals are lowered (inside the sliding window or the entire waveform depending on the version of level-1 trigger finder)
  // DeltaPed > 0: pedestals are raised (inside the sliding window or the entire waveform depending on the version of level-1 trigger finder)
  bool find_level2_trigger(int DeltaPed = 0);

  // finds a level-2 trigger while changing the pedestals
  // (if not successful using real pedestal, then it will try to decrease
  // the pedestal until is successful)
  // returns true if successful, false if
  // no trigger found after trying out a range of pedestals.
  bool find_level2_trigger_lower_ped();


  int find_level2_trigger_raise_ped();

  void put2sdtrgbk_triginfo();  // put the variables into sdtrgbk DST bank

  void analyze_event(); // analyze events once the SD information has been loaded
  vector<sdinfo_class *> sd_good_ped; // vector of pointers to sds with good pedestals
  vector<sdinfo_class *> sd_spat_cont; // vector of pointers to sds that are in triplets
  vector<sdinfo_class *> sd_spat_isol; // vector of pointers to spatially isolated SDs
  vector<sdinfo_class *> sd_pot_spat_tim_cont; // SDs that are potentially in time, using their earliest and latest times
  sdinfo_class sdinfo_buffer[NSDMAX]; // buffer for SD information
  vector<sdinfo_class *> l1sd; //  vector of pointers to level-1 trigger SDs out of potentiall space-time contiguous SDs
  vector<sdinfo_class *> l2sd; //  vector of pointers to SDs that caused the level-2 trigger

  int n_sdinfo_buffer; // number of SDs in sdinfo_buffer
  int n_bad_sds; // number of bad SDs encountered ( bad pedestal and/or mip, etc )
  int space_pattern; // For triggered events, 2 - triangle, 4 - line trigger
  bool has_triggered; // true if event has triggered, false otherwise

  // Event date and time
  int yymmdd;
  int hhmmss;
  int usec;

  
  map<int,int> bsd_list; // map list of bad SDs from bsdinfo DST bank, indexed by xxyy
  void load_bad_sds(bsdinfo_dst_common *bsdinfo); // load bad SDs from bsdinfo into the map
  bool is_in_bsd_list(int xxyy); // checks if the xxyy is in the list of bad SDs
  
  void init()
  {
    bsd_list.clear();
    sd_good_ped.clear();
    sd_spat_cont.clear();
    sd_spat_isol.clear();
    sd_pot_spat_tim_cont.clear();
    l1sd.clear();
    l2sd.clear();
    n_sdinfo_buffer = 0;
    n_bad_sds = 0;
    space_pattern = 0;

    has_triggered = false;
    yymmdd = 0;
    hhmmss = 0;
    usec = 0;
  }

  void printErr(const char *form, ...);
  void printWarn(const char *form, ...);
};

#endif
