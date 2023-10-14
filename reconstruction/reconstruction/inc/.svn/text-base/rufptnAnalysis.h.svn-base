#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "event.h"

// constants in rufptn analysis
#include "rufptn_constants.h"
#include "rufptn.h"

// gen. sd utilities
#include "sduti.h"

// Root routines
// #include "TObject.h"
#include "TGraphErrors.h"
#include "TF1.h"
#include "TMath.h"
#include "TSpectrum.h"

// Needed for fitting geometry using Minuit
#include "p1geomfitter.h"

// People who wrote TSpectrum class did not ensure backward
// compatiblity: in the older versions of ROOT the Deconvolution method
// of the class used Float_t type but for the newer versions
// it uses only Double_t type.
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,0,0)
#   define TSPECTRUM_DECONVOLUTION_TYPE Double_t
#else
#   define TSPECTRUM_DECONVOLUTION_TYPE Float_t
#endif


class rufptnAnalysis
{

public:
  
  const listOfOpt& opt; // reference to the list of options passed to the analysis from the command line
  
  // 0: lower, 1: upper, 3: both
  real8 charge [RUSDRAWMWF][3], // hit pulse area in VEM
    chargeErr [RUSDRAWMWF][3], // error on hit pulse area in VEM
    relTime [RUSDRAWMWF][3], // relative to earliest hit
    timeErr [RUSDRAWMWF][3]; // error on time

  real8 sdorigin_xy[2];  // XY coordinates of SD Origin in CLF Frame, [1200m] units

  // Graph for fitting t vs u plot into a straight line.
  TGraphErrors *gLaxisFit[3];

  p1geomfitter_class *p1geomfitter; // Needed for geometry fitting that uses Minuit.


  TSPECTRUM_DECONVOLUTION_TYPE muresp_shape[128]; // Shape of 1mu response
  TSpectrum *sNpart;       // For deconvolving FADC traces (when checking for saturation)
  
  // when analysis initializes, it takes a reference to the list of options
  // variable to obtain the command line arguments.
  rufptnAnalysis(listOfOpt& opt);
  virtual ~ rufptnAnalysis();

  inline void xycoor(int xy, int *c)
  {
    c[0] = xy / 100;
    c[1] = xy % 100;
  }

  void processFADC();   /* Process fadc traces */

  //  void processFADC_new(); /* Another method of processing FADC */

  // calculate the calib. information
  // needed for event analysis.
  void compEventMon();

  // Find the pulse areas in units of VEM.
  void findCharge();

  /* Determines whether the two hits are next to each other spacially
     Assume adjacent if they are separated by a squared distances less than
     or equal to spaceadj.  spaceadj should be an integer.

     INPUTS:
     1) ih1 - index of the 1st hit
     2) ih2 - index of the 2nd hit
     3) spaceadj: maximum squared separation distance
     4) xxyy: array of counter positions, indext by waveform indecies,
     i.e. iwf1,iwf2 are indecies for xxyy array.

     RETURNS:
     0: NO
     1: YES
  */
  integer4 areSadjacent(integer4 ih1, integer4 ih2, integer4 spaceadj,
			integer4 *xxyy);

  /*
    Determines whether the two spatially adjacent hits can belong to the shower front
    plane apart to some tolerance . This is for the "time pattern recognition".
    INPUTS:
    1) ih1, 2) ih2: hit indecies
    2) xyz[][3] - hit positions with respect to CLF
    3) hit time array (in units of counter separation distance, 1200m)
    4) timeadjtol: how much off can the time be

    RETURSN:
    0: NO
    1: YES
  */
  integer4 areInTime(integer4 ih1, integer4 ih2, real8 xyz[][3],
		     real8 *xxyyt, real8 timeadjtol);

  /*
    Given Pass0 event, this determines the
    largest space clusterings in the event, scluster array
    contains intra-event ordering indecies of the LID's
    in the largest space cluster and nScluster
    contains the number of LID's in such cluster

    INPUTS:
    1) nofwf: number of waveforms
    2) xxyy:  detector integer positions which registered waveforms

    OUTPUT:
    1) nclust: number of hits in space cluster
    2) clust:  indecies of hits that are in space cluster

  */

  void spacePatRecog(integer4 nofwf, integer4 *xxyy, integer4 *nclust,
		     integer4 *clust);



  /*
    To recognize clusters of counters which are contiguous in time

    INPUTS:
    1) xxyy: array of integer positions with respect to SD origin
    2) xyz[][3]: array of positions with respect to CLF
    3) xxyyt: array of times, in units of counter
    separation distance (1200m)
    4) nsclust: number of hits in space cluster
    5) sclust: hit indecies for hits that are in space cluster
    OUTPUTS:
    1) nclust: number of hits in the space-time cluster
    2) clust:  hit indecies for hits that are in space-time cluster

  */
  void timePatRecog(integer4 *xxyy,real8 xyz[][3],
		    real8 *xxyyt, integer4 nsclust,integer4 *sclust, integer4 *nclust,
		    integer4 *clust);



  // This routine is for combining separate same-counter signals into one
  // multi-fold signal.  Signal start channel and signal time will be that
  // of the 1st signal, and signal stop channel will be that of the last signal.
  void combineMfHits(integer4 ihit1, integer4 ihit2);

  // This routine is called after the time pattern recognition and after the
  // hits that pass time pattern recognition are labeled. It will look for same counter
  // signals, find the ones that time pattern recognition picks as signal from shower,
  // and combine all signals (in the same counter) in SIGNAL_INCLUSION_RANGE after the
  // chosen signal.
  void combineSignals();

  // Tyro geometry reconstruction
  bool tyroGeom();

  // Fit to plane and Modified Linsley function
  bool geomFit();

  // After geometry fitting is done, some counters from space-time cluster are removed
  // using chi2 cleaning procedure.  We want to apply these changes to all rufptn variables,
  // i.e. we want to find signals which pass the chi2- cleaning procedure and increase
  // their goodness variable.
  void changeStclust();


  // Determines weather a given hit saturates the counter
  bool isSaturated(integer4 ihit);


  // This routine goes over all signals that are a part of the shower and lables
  // the hits which saturate the counters
  void labelSaturatingHits();


  // Put variables into rusdgeom DST bank
  bool put2rusdgeom();

  // Contains all the analysis performed on an individual event
  bool analyzeEvent();

  // Calculates additional MC variables from thrown MC variables
  void comp_rusdmc1();

  // Calculates additional MC variables from thrown MC variables
  // w/o using the reference time of the earliest SD. Useful in the
  // cases when there are no information available on which SDs were
  // hit ( or that no SDs were hit )
  void comp_rusdmc1_no_tref();

  // return true if the last processed event has triggered
  bool hasTriggered() { return fHasTriggered; }

  integer4 nspaceclust; // # of hits in space cluster
  integer4 spaceclust[RUFPTNMH]; // indeces of hits in space cluster
  integer4 nspacetimeclust; // # of hits in space-time cluster
  integer4 spacetimeclust[RUFPTNMH]; // indices of hits in space-time cluster

private:
  //  void get_fadc_signals();   // To process a combined fadc trace
  //  integer4 fadc[2][1280];    // combined fadc trace for upper and lower

  // true if the last analyzed event has triggered, false otherwise
  bool fHasTriggered;
  
  // information on bad SDs
  bool printBadSDInfo(int iwf, char *problem_description, FILE *fp=0);

};
