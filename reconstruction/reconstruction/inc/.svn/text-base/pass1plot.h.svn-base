#ifndef _pass1plot_h_
#define _pass1plot_h_

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <cstring>

#include "TObject.h"
#include "TChain.h"
#include "TTree.h"
#include "TFile.h"
#include "TH1F.h"
#include "TF1.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TSpectrum.h"
#include "TGraphErrors.h"

#include "sdrt_class.h"
#include "fdrt_class.h"
#include "sduti.h"
#include "tafd10info.h"

#include "event.h"
#include "TMath.h"
#include "tacoortrans.h"
#include "sdxyzclf_class.h"
#include "sdenergy.h"
#include "TTimeStamp.h"

class md_pcgf_class
{
 public:
  md_pcgf_class();
  virtual ~md_pcgf_class();
  
  Int_t imin[3];                   // the minimum chi-squares of combined, profile, and timing
  Int_t igfit[HCTIM_MAXFIT];       // determination of a good fit (fit good)
  Int_t ig;                        // determination of a good fit (at least one mirror)
  Double_t avgcfc[HCTIM_MAXFIT];   // determination of a good fit (avg. correction fact.)
  Double_t ndft_fit[HCTIM_MAXFIT]; // # degrees of freedom in timing fit
  Double_t tot_npe;                // total number of npe's for the event
  Int_t nbins, ng;                 // counters: avgcfc, ndft
  Double_t sum;                    // adders: avgcfc
  Int_t FIT_START;                 // where the constrained profile fits start in the utafd DST banks
  Double_t c2c;                    // best combined profile and time fit chi2
  Double_t c2p;                    // best chi2 of the profile fit
  Double_t c2t;                    // best chi2 of the time fit
  Double_t ndfp;                   // profile fit number of degrees of freedom
  Double_t ndft;                   // time fit number of degrees of freedom
  Double_t normp;                  // normalization factor for profile fit chi2
  Double_t normt;                  // normalization factor for time fit chi2
  ClassDef(md_pcgf_class,1);
};


class pass1plot
{

 public:
  
  pass1plot(const char *listfile);
  virtual  ~pass1plot();


  
  // Runs pass1tree->GetEntry on the pass1tree chain and runs certain commands on every event
  Int_t GetEntry(Long64_t entry = 0, Int_t getall = 0);
  
  // Runs pass1tree->GetEntries with cut selection
  Long64_t GetEntries(const char* sel)
  {
    if(pass1tree)
      return pass1tree->GetEntries(sel);
    return 0;
  }
  // Runs pass1tree->GetEntries to get the total number of events in the chain
  Long64_t GetEntries()
  {
    if(pass1tree)
      return pass1tree->GetEntries();
    return 0;
  }
  // Runs GetReadEvent on pass1tree
  Long64_t GetReadEvent()
  {
    if(pass1tree)
      return pass1tree->GetReadEvent();
    return 0;
  }
  TTimeStamp event_time_stamp; // Time information of the event that's currently looked at
  // Set the event time stamp in a best way possible
  Bool_t set_event_time_stamp(Bool_t set_to_default_value = false);

  void event_hist(); /* to histogram the FADC counters for
			all the counters which were hit, and
			find the corresponding muon equivalents.
			This is for an individual event */


  void get_sded_coordinates_from_gps(Double_t lat_degree, 
                                     Double_t lon_degree, 
                                     Double_t alt_meters,
                                     Double_t *sded_x_1200m_units,
                                     Double_t *sded_y_1200m_units)
  {
    Double_t xyz_clf_m[3]={0,0,0};
    tacoortrans::latlonalt_to_xyz_clf_frame(lat_degree,lon_degree,alt_meters,xyz_clf_m);
    (*sded_x_1200m_units) = xyz_clf_m[0]/1.2e3 - RUSDGEOM_ORIGIN_X_CLF;
    (*sded_y_1200m_units) = xyz_clf_m[1]/1.2e3 - RUSDGEOM_ORIGIN_Y_CLF;    
  }


  /* To make a fadc plot over all waveforms in a multi-folded hit */
  bool fadc_hist(Int_t ihit = 0);


  bool npart_hist(Int_t ihit); // to make histogram of # of particles as a function of time

  

  /* to set on a given event */
  /* -1 means look at the current event again */
  bool lookat(Int_t eventNumber = -1);




  // To histogram the pulse height area sums,
  // azimuthal and zenith angles;
  // bool phys_hist( Int_t nKosher = 2 );


  /* To focus on a next event in a data sample with
     largest number of counters in a ST-cluster greater or
     equal to nmin. Start searching from the current event. */
  bool findCluster(Int_t nmin);


  /* To focus on a next event in a data sample with
     largest number of counters in a ST-cluster greater or
     equal to nmin.eNumber is from where to start searching for such an
     event */
  bool findCluster(Int_t eNumber,Int_t nmin);



  /* To find an event by its trigger id and tower that triggered, starting at
   * the current event */
  bool findTrig(Int_t trig_id,Int_t site);

  /* To find an event by its trigger id and tower that triggered, starting at
   * the specified event */
  bool findTrig(Int_t eNumber,Int_t trig_id, Int_t site);


  Double_t svalue(Int_t ihit); // find dist. from shower axis of ith hit counter

  inline void xycoor(int xxyy, int *xy)
    {
      xy[0] = xxyy / 100;
      xy[1] = xxyy % 100;
    }

  // This routine uses PROTON energy estimation table
  Double_t get_sdenergy(Double_t s800, Double_t theta);
  
  // This routine uses IRON energy estimation table
  Double_t get_sdenergy_iron(double s800, double theta);

  // 0 = largest shower signal counter not surrounded by working counters
  // 1 = largest shower signal counter surrounded by 4 working counters
  // 2 - largest shower signal counter surrounded by 4 working counters that
  //     are its immediate neighbors
  Int_t get_event_surroundedness();
  

  
  // return 1 if something is available on stdin, 0 otherwise
  // poll_timeout_ms is time in milliseconds to wait for data to become
  // available on standard input; if this is -1 then wait until data
  // becomes available through standard input
  Bool_t have_stdin(Int_t poll_timeout_ms = 0)
  { return (Bool_t)SDIO::have_stdin(poll_timeout_ms); }

  // To continue doing some activity every time_interval_seconds
  // until enter is pressed
  Bool_t continue_activity(Double_t time_interval_seconds = 1.0);
  
  void Help();
  
  const double MD_PMT_QE;              // MD PMT quantum efficiency
  
  TChain             *pass1tree;       // pass1 chain of trees
  
  // DECLARE DST BRANCHES AND RELATED VARIABLES
  #ifdef _PASS1PLOT_IMPLEMENTATION_
  #undef _PASS1PLOT_IMPLEMENTATION_
  #endif
  #include "pass1plot_dst_branch_handler.h"
  
  md_pcgf_class *md_pcgf;                   // Not a branch but a class calculated from other branches. 
                                            // Contains fit quality descriptors for MD PCGF analysis.
  
  // events found in the tree
  Int_t eventsRead;                    

  
  ///////////// Relevant for FDs (below) ////////////////////////////////
  // set fdplane to point on brplane or lrplane 
  Bool_t set_fdplane (Int_t siteid, Bool_t chk_ntube=true, Bool_t prinWarn=true);
  // set fdporfile to point on brprofile or lrprofile
  Bool_t set_fdprofile (Int_t siteid, Bool_t chk_ntube=true, Bool_t printWarn=true);
  bool get_fd_tube_pd(int fdsiteid, int mir_num, int tube_num, double *v);
  bool get_fd_time(int fdsiteid, int fdJday, int fdJsec, int *yymmdd, int *hhmmss);
  // fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
  // xfd[3]  - vector in FD frame, [meters] (INPUT)
  // xclf[3] - vector in CLF frame, [meters] (OUTPUT)
  bool fdsite2clf(int fdsiteid, double *xfd, double *xclf);

  // fdsiteid: 0 - BR, 1 - LR, 2 - MD (INPUT)
  // xclf[3] - vector in CLF frame, [meters], (INPUT)
  // xfd[3]  - vector in FD frame, [meters], (OUTPUT)
  bool clf2fdsite(int fdsiteid, double *xclf, double *xfd);
  // This is a routine for obtaining hraw1 class variable from 
  // mcraw class.  Applicable for MD MC events.
  bool get_hraw1();
  
  ///////////// Relevant for FDs (above) ////////////////////////////////


  /************* For every event ************************/
  // 2nd index: 0 - lower, 1 - upper, 2 - average b/w upper and lower


  Int_t
    ngclust,                    // number of hits in good cluster
    gclust     [RUSDRAWMWF][3], // hits in the good cluster
    laxisgclust[RUSDRAWMWF][3], // good cluster hits along the long axis
    saxisgclust[RUSDRAWMWF][3]; // good cluster hits along the short axis


  Double_t
    relTime    [RUSDRAWMWF][3],
    timeErr    [RUSDRAWMWF][3],
    charge     [RUSDRAWMWF][3],
    chargeErr  [RUSDRAWMWF][3];


  Double_t
    tEarliest  [3],               // earliest hit, relative to T-line
    lcharge    [3],               // largest hit pulse area
    qtot [3];                     // Sum of all charge in the event (in space-time cluster)

  Int_t ngaps_plot;                    // count how many time gaps there are b/w adjacent fadc traces
  Int_t nfadcb;                        // number of fadc bins with gaps included
  std::vector <Double_t> gapsize_plot; // record the sizes of the gaps b/w the adjacent waveforms in a multi-fold hit
  TH1F
    *hErelTime[3],                // relative hit times over 1 event
    *hCharge[3],                  // charge in VEM
    *hVEM[2],                     // FADC counts/VEM for given counter in the event (labeled by hit index)
    *hPed[2],                     // pedestals, computed using monitoring information
    *hFadcPed[2],                 // pedestals, from parsing FADC traces
    *hfadc[2],                    // Generalized FADC trace for each event
    *hPhi[3],                     // azimuthal angles
    *hTheta[3],                   // zenith angles
    *hParea[3];                   // pulse areas in ST-clusters





  /// Used mostly for debugging purposes (below) ///////////////////////

  // To histogram the difference b/w cores computed in different ways: using
  // charge and using sqrt of charge.
  TH1F *hCoreDiffX;
  TH1F *hCoreDiffY;
  TH1F *hCoreDiffR;
  // Change in core position squared vs minimum dist. from core and minimum 1/charge cuts.
  TH2F *hDcoreR2vsRmin;
  TProfile *pDcoreR2vsRmin;
  // Change in core position when a saturated counter is removed
  TH1F *hDcoreRnoSat;
  TH2F *hDcoreRnoSatVsR;
  TProfile *pDcoreRnoSatVsR;
  // Time delay, time fluctuation and error on time of saturated counters
  TH1F *hTdSat;
  TH1F *hTsSat;
  TH1F *hTrSat;
  TH2F *hNremVsRmin; // number of counters remaining by these cuts
  TProfile *pNremVsRmin;
  TH2F *hDcoreR2vsOneOverQmin;
  TProfile *pDcoreR2vsOneOverQmin;
  TH2F *hNremVsOneOverQmin;
  TProfile *pNremVsOneOverQmin;
  // To check pattern recognition
  TH1F *hTdiff1;     // Same counter hits
  TH2F *hQVsdT;
  TH2F *hdQVsdT;
  TH2F *hQrVsdT;
    // Try two different methods of calculating the core
  void tryCoreCalc();
  // To recalculate the core using max. charge cut
  bool recalcCore(Double_t qmax,Double_t *coreXY);
  // Recalculate the core w/o the i'th counter
  bool recalcCore(Int_t iexclude,Double_t *coreXY);
  // Compute new core location with given cuts:
  // rmin: minimum distance from the Old core cut
  // minOneOverQ: minimum 1/charge cut
  bool calcNewCore(Double_t rmin, Double_t minOneOverQ,
      Double_t *oldCoreXY, Double_t *newCoreXY, Int_t *nremoved);
  // Fill histograms of change in core vs cuts
  void histDcore();
  // To check if the counter corresponding to i'th hit is saturated
  bool isSaturated(Int_t ihit);
  // Histograms when counters are saturated (change in core when such counter
  // is removed, it's time delay, etc
  void histSat();
  // to determine whether given detector is in space cluster
  bool inSclust(Int_t xxyy);
  TH2F *hTdiff2VsR;  // Hits separated by 1200m vs distance from core
  TH2F *hTdiff3VsR;  // Hits separated by sqrt (2) * 1200m vs distance from core
  bool areSadjacent(Int_t ih1, Int_t ih2, Double_t *r);
  void chk_precog();
  /// Used mostly for debugging purposes (above) ///////////////////////


  TGraphErrors *gTvsUall[3];      // time vs dist. along long axis, for lower,upper, and both (all)
  TGraphErrors *gTvsUsclust[3];   // time vs dist. along long axis, for lower,upper, and both (space clust)
  TGraphErrors *gTvsUclust[3];    // time vs dist. along long axis, for lower,upper, and both (clust)

  TGraphErrors *gTvsVall[3];      // time vs dist. along short axis, for lower,upper, and both (all)
  TGraphErrors *gTvsVsclust[3];   // time vs dist. along short axis, for lower,upper, and both (space clust)
  TGraphErrors *gTvsVclust[3];    // time vs dist. along short axis, for lower,upper, and both (clust)


  TGraphErrors *gQvsSall[3];      // charge vs distance from shower axis graph (all)
  TGraphErrors *gQvsSsclust[3];   // charge vs distance from shower axis graph (space clust)
  TGraphErrors *gQvsSclust[3];    // charge vs distance from shower axis graph (clust)

  TGraphErrors *gTvsRall[3];      // time vs distance from the core graph (all)
  TGraphErrors *gTvsRsclust[3];   // time vs distance from the core graph (space clust)
  TGraphErrors *gTvsRclust[3];    // time vs distance from the core graph (clust)



  // Scatter/profile plots for charge in lower and upper
  TH2F *hQscat;
  TProfile *pQscat;
  TH1F *hQupQloRat;
  TH2F *hQupQloRatScat;
  TProfile *pQupQloRatScat;

  // Scatter/profile plots of time in lower and upper
  TH2F *hTscat;
  TProfile *pTscat;

  // variables to analyze multi-fold hits
  TH1F *hNfold;

  // For multiple waveform - hits, these are the ratios of
  // the two adjacent pulse areas: earlier divided by the later.
  TH1F *hQrat;


  TH1F *hLargePheight; // Largest pulse height in FADC


  TH1F *h1MIPResp;    // FADC response to 1MIP signal


  // 1 mu response shape, obtained from looking at many 1mu fadc traces. This shape is normalized to 1.
  Double_t muresp_shape[12]; // normalized so that the largest bin is 1

  TH1F *hNpart[2]; // # of particles vs time
  TH1F *hNfadc[2]; // fadc with pedestals subtracted
  TH1F *hResp[2];
  TSpectrum *sNpart;


  bool isKosher[3]; // if event passes the geometry fits

  // initialize all the histograms
  void init_hist();

  // initialize variable bin number histograms
  void init_varbnum_hist();

  // analyze multi-fold hits
  void analyze_Mf_hits(Bool_t fsclust);



  // Fills various signal histograms
  void histSignal();

  Int_t get_tasdcalibev_iwf(Int_t rusdraw_iwf); // obtains tasdcalibev waveform index
  

  // useful misc. functions
  
  // Convert year, month, day to julian days since 1/1/2000
  int greg2jd(int year, int month, int day);
  
  // Convert julian days corresponding to midnight since Jan 1, 2000 to gregorian date
  void jd2greg(double julian, int *year, int *month, int *day);
  
  // Get time in seconds since midnight of Jan 2000
  int time_in_sec_j2000(int year,int month,int day,
			int hour,int minute,int second);
  double get_sd_secfrac();

  Int_t get_yymmdd(); // return the date of the current event
  Int_t get_hhmmss(); // return the time of the current event
  Int_t get_usec();   // return the second fraction of the current even
  
  // dst dump any branch that is present in the root tree
  bool dump_dst_class(dstbank_class* dstbank, Int_t format = 0, FILE* fp = stdout);
  

  // saves root tree entries from imin to imax (inclusively) to a dst file
  // if dst_file_name pointer is zero then it creates a generic file name 
  bool events_to_dst_file(const char* dst_file_name = 0, 
                          Int_t imin=0, Int_t imax=0);

  

  // This is a routine for initializing and checking variables needed
  // to obtain md plane fit results: if there is no hraw1 or hraw1 is
  // empty then it will try to obtain hraw1 from mcraw and mc04
  // If hraw1 is missing and can't be obtained from mc04 and mcraw, or 
  // if stpln is missing or empty, this routine returns false.
  // Otherwise, returns true.  When returns true, one has
  // at working and non-emtpy hraw1 and stpln branches at least.
  bool init_md_plane_geom(bool printWarning = true);
  
  // return false if the required banks are absent
  bool get_md_pcgf();

  // Get counter xyz coordinates in CLF frame in [1200m] units
  // on any given date
  bool get_xyz(int yymmdd, int xxyy, double *xyz);
  
private:
  
  // returns a chain of input root files found in the list file
  TChain* chain_input_root_files(const char* listfile);
  
  // reset only individual event histograms
  void clean_event_hist();

  TFile *f; // file that contains the tree

  // if tasdcalibev branch is present, then fill the arrray for rusdraw
  // index mapping from rusdraw waveforms to tasdcalibev waveforms
  void fill_tasdcalibev_wf_indices();
  
  // for each rusdraw waveform index we have a corresponding 
  // tasdcalibev waveform index 
  Int_t rusdraw2tasdcalibev[RUSDRAWMWF];

  // contains the addresses of all root tree branches
  std::vector<dstbank_class**> dst_branches;

  // acceptible object names for the root trees that one may encounter
  // if a root tree in the file doesn't match any of these then exit with an error.
  // also, when one chains the root tree files, all files must have root trees
  // with of the same name, otherwise will exit with an error.
  std::vector<TString> tree_names;
  
  // various ROOT definitions
  ClassDef(pass1plot,1)


};


#endif
