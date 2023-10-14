#ifndef SDINFO_CLASS_H_
#define SDINFO_CLASS_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "sdtrgbk_constants.h"

class sdinfo_class
{
public:
  int sdindex; // SD index in the array that holds SDs information
  int raw_bankid; // BANK ID of the raw bank used
  int xxyy; // counter position ID
  int wfindex_cal; // waveform index used for calibration
  int nwf; // number of waveforms
  int ped[2]; // pedestal in 8 fadc time slices
  int ped1[2]; // pedestal in 128 fadc time slices
  double mip[2]; // number of fadc counts in 1MIP for lower and upper
  int clkcnt[NWFPSD]; // clock count at the beginning of each waveform
  int mclkcnt[NWFPSD]; // max. clock count
  int wfindex[NWFPSD]; // index of the waveform within the given raw bank used ( rusdraw or tasdcalibev)
  int fadc[NWFPSD][2][rusdraw_nchan_sd]; // fadc trace of a given waveform
  double tlim[2]; // time limits for each SD, [0] - minimum possible time, [1] - maximum possible time


  // by how much pedestal inside
  // the sliding window was changed
  // if d_ped < 0, pedestals were lowered,
  // if d_ped > 0, pedestals were raised
  int d_ped;

  // number of (level-1) signals that are above 150 fadc counts
  int nl1;

  // If the SD is participating in level-2 trigger then
  // this will be pointing to the signal that's used
  // in level-2 trigger
  int il2sig;

  // second fractions for the level-1 signals
  // (corresponds to the left edge of the sliding window)
  double secf[NSIGPSD];

  // fadc time slices for the level-1 signals
  // (left edge of the sliding window)
  int ich[NSIGPSD];

  // signal size (fadc counts) for the level-1 signals,
  // [0] = lower, [1] - upper
  int q[NSIGPSD][2];

  // Waveform index inside this class
  // to indicate the waveforms from
  // which each level-1 signal comes
  int iwfsd[NSIGPSD];

  // Finds level-1 trigger SDs, using either original pedestals
  // (DeltaPed = 0) or altered pedestals by the amount DeltaPed
  // (in FADC counts inside the 8 fadc time slice sliding window)
  // returns the number of level-1 trigger signals found
  int find_l1_sig(int DeltaPed = 0);

  // Finds level-1 trigger SDs, using either original pedestals
  // (DeltaPed = 0) or altered pedestals by the amount DeltaPed
  // (in FADC counts inside the entire waveform)
  // returns the number of level-1 trigger signals found
  // NOW USES CORRECT DESCRIPTION OF LEVEL-1 SIGNAL
  int find_l1_sig_correct(int DeltaPed = 0);

  // initialize SD before adding the waveforms
  // specify the storage index of the SD (isd))
  // use a certain waveform index in the raw banks (iwf))
  // to get the calibration information
  bool init_sd(tasdevent_dst_common *p, int isd, int iwf);
  bool init_sd(tasdcalibev_dst_common *p, int isd, int iwf);
  bool init_sd(rusdraw_dst_common *p, int isd, int iwf);

  // to add a waveform with certain index in the raw banks
  bool add_wf(tasdevent_dst_common *p, int iwf);
  bool add_wf(tasdcalibev_dst_common *p, int iwf);
  bool add_wf(rusdraw_dst_common *p, int iwf);

  sdinfo_class();
  virtual ~sdinfo_class();
private:
  void Clean();
  void printWarn(const char *form, ...);
};

#endif /* SDINFO_CLASS_H_ */
