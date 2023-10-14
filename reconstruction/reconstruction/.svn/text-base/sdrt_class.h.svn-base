#include "sdfdrt_class.h"
//    Class for SD talex00_ DST bank.
//    Last modified: Jan 15, 2020
//    Dmitri Ivanov <dmiivanov@gmail.com>
//


#ifndef _talex00_class_
#define _talex00_class_


using namespace std;


// if the number of communication towers is not defined
// then define that number to the best knowledge of what
// corresponds to each version of TALEX00 bank
#ifndef TALEX00_NCT
#if TALEX00_BANKVERSION < 1
#define TALEX00_NCT 3
#else
#define TALEX00_NCT 10
#endif
#endif

class talex00_class: public dstbank_class
{

public:
  Int_t event_num;		            // event number
  Int_t event_code;                         // 1=data, 0=Monte Carlo
  Int_t site;                               // site bitflag index (bit0=BR,1=LR,2=SK,[3-8]=[BF,DM,KM,SC,SN,SR],bit9=MD
  vector<Int_t> run_id;                     // [TALEX00_NCT] run IDs of the raw data files from each tower, -1 if irreleveant
  vector<Int_t> trig_id;		    // [TALEX00_NCT] trigger IDs for each tower, -1 if irrelevant
  Int_t errcode;                            // should be zero if there were no readout problems
  Int_t yymmdd;		                    // year, month, day
  Int_t hhmmss;		                    // hour minut second
  Int_t usec;		                    // usecond
  Int_t monyymmdd;                          // yymmdd at the beginning of the mon. cycle used in this event
  Int_t monhhmmss;                          // hhmmss at the beginning of the mon. cycle used in this event
  Int_t nofwf;		                    // number of waveforms in the event

  
  // These arrays contain the waveform information
  vector<Int_t> nretry;                     // number of retries to get the waveform
  vector<Int_t> wf_id;                      // waveform id in the trigger
  vector<Int_t> trig_code;                  // level1 trigger code
  vector<Int_t> xxyy;	                    // detector that was hit (XXYY)
  vector<Int_t> clkcnt;	                    // GPS clock count at the waveform beginning
  vector<Int_t> mclkcnt;	            // max clock count for detector, around 50E6 
  vector<vector<Int_t> > fadcti;            // fadc trace integral, for upper and lower
  vector<vector<Int_t> > fadcav;            // average of the FADC trace
  vector<vector<vector<Int_t> > > fadc;	    // fadc trace for upper and lower
  
  // Useful calibration information 
  vector<vector<Int_t> > pchmip;            // peak channel of 1MIP histograms
  vector<vector<Int_t> > pchped;            // peak channel of pedestal histograms
  vector<vector<Int_t> > lhpchmip;          // left half-peak channel for 1mip histogram
  vector<vector<Int_t> > lhpchped;          // left half-peak channel of pedestal histogram
  vector<vector<Int_t> > rhpchmip;          // right half-peak channel for 1mip histogram
  vector<vector<Int_t> > rhpchped;          // right half-peak channel of pedestal histograms

  // Results from fitting 1MIP histograms
  vector <vector <Int_t> > mftndof;         // number of degrees of freedom in 1MIP fit
  vector<vector<Double_t> > mip;            // 1MIP value (ped. subtracted)
  vector<vector<Double_t> > mftchi2;        // chi2 of the 1MIP fit
  
  // 1MIP Fit function: 
  // [3]*(1+[2]*(x-[0]))*exp(-(x-[0])*(x-[0])/2/[1]/[1])/sqrt(2*PI)/[1]
  // 4 fit parameters:
  // [0]=Gauss Mean
  // [1]=Gauss Sigma
  // [2]=Linear Coefficient
  // [3]=Overall Scalling Factor
  vector<vector<vector<Double_t> > > mftp;  // 1MIP fit parameters
  vector<vector<vector<Double_t> > > mftpe; // Errors on 1MIP fit parameters
  
  vector<vector<Double_t> >lat_lon_alt;     // GPS coordinates: latitude, longitude, altitude
                                            // [0] - latitude in degrees,  positive = North
                                            // [1] - longitude in degrees, positive = East
                                            // [2] - altitude is in meters
  
  vector<vector<Double_t> >xyz_cor_clf;     // XYZ coordinates in CLF frame:
                                            // origin=CLF, X=East,Y=North,Z=Up, [meters]

  talex00_class();
  virtual ~talex00_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();

  ClassDef(talex00_class,3)
};



#endif
//    Class for SD rusdraw_ bank.
//    Last modified: Jan 15, 2020
//    Dmitri Ivanov <dmiivanov@gmail.com>
//


#ifndef _rusdraw_class_
#define _rusdraw_class_


using namespace std;

class rusdraw_class: public dstbank_class
{

public:
  Int_t event_num;		                  // event number
  Int_t event_code;                               // 1=data, 0=Monte Carlo
  Int_t site;                                     // BR=0,LR=1,SK=2,BRLR=3,BRSK=4,LRSK=5,BRLRSK=6
  vector<Int_t> run_id;                           // run number(s), -1 if irrelevant
  vector<Int_t> trig_id;        	          // event number in the raw ascii data file 
  Int_t errcode;                                  // should be zero if there were no readout problems
  Int_t yymmdd;		                          // year, month, day
  Int_t hhmmss;		                          // hour minut second
  Int_t usec;		                          // usecond
  Int_t monyymmdd;                                // yymmdd at the beginning of the mon. cycle used in this event
  Int_t monhhmmss;                                // hhmmss at the beginning of the mon. cycle used in this event
  Int_t nofwf;		                          // number of waveforms in the event

  
  // These arrays contain the waveform information
  vector <Int_t> nretry;                          // number of retries to get the waveform
  vector <Int_t> wf_id;                           // waveform id in the trigger
  vector <Int_t> trig_code;                       // level1 trigger code
  vector <Int_t> xxyy;	                          // detector that was hit (XXYY)
  vector <Int_t> clkcnt;	                  // GPS clock count at the waveform beginning
  vector <Int_t> mclkcnt;	                  // max clock count for detector, around 50E6 
  vector <vector <Int_t> > fadcti;	          // fadc trace integral, for upper and lower
  vector <vector <Int_t> > fadcav;                // average of the FADC trace
  vector <vector <vector <Int_t> > > fadc;        // fadc trace for upper and lower
  
  // Useful calibration information 
  vector <vector <Int_t> > pchmip;                // peak channel of 1MIP histograms
  vector <vector <Int_t> > pchped;                // peak channel of pedestal histograms
  vector <vector <Int_t> > lhpchmip;              // left half-peak channel for 1mip histogram
  vector <vector <Int_t> > lhpchped;              // left half-peak channel of pedestal histogram
  vector <vector <Int_t> > rhpchmip;              // right half-peak channel for 1mip histogram
  vector <vector <Int_t> > rhpchped;              // right half-peak channel of pedestal histograms

  // Results from fitting 1MIP histograms
  vector <vector <Int_t> > mftndof;               // number of degrees of freedom in 1MIP fit
  vector <vector <Double_t> > mip;                // 1MIP value (ped. subtracted)
  vector <vector <Double_t> > mftchi2;            // chi2 of the 1MIP fit
  // 
  // 1MIP Fit function: 
  // [3]*(1+[2]*(x-[0]))*exp(-(x-[0])*(x-[0])/2/[1]/[1])/sqrt(2*PI)/[1]
  // 4 fit parameters:
  // [0]=Gauss Mean
  // [1]=Gauss Sigma
  // [2]=Linear Coefficient
  // [3]=Overall Scalling Factor
  vector < vector <vector <Double_t> > > mftp;   // 1MIP fit parameters
  vector < vector <vector <Double_t> > > mftpe;  // Errors on 1MIP fit parameters
  
  rusdraw_class();
  virtual ~rusdraw_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(rusdraw_class,5)
};



#endif
//    Class for SD rusdmc_ bank.
//    Last modified: Jan 16, 2020
//    Dmitri Ivanov <dmiivanov@gmail.com>
//


#ifndef _rusdmc_class_
#define _rusdmc_class_


using namespace std;

class rusdmc_class : public dstbank_class
{

public:
  Int_t event_num;         // event number
  Int_t parttype;          // Corsika particle code [proton=14, iron=5626, for others, consult Corsika manual]
  Int_t corecounter;       // counter closest to core
  Int_t tc;                // clock count corresponding to shower front passing through core position
  Float_t energy;          // total energy of primary particle [EeV]
  Float_t height;          // height of first interation [cm]
  Float_t theta;           // zenith angle [rad]
  Float_t phi;             // azimuthal angle (N of E) [rad]
  vector<Float_t> corexyz; // [3] 3D core position in CLF reference frame [cm]
  rusdmc_class();
  virtual ~rusdmc_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  ClassDef(rusdmc_class,5)
};
#endif
//    Class for SD rusdmc1_ bank.
//    Last modified: Jan 16, 2020
//    Dmitri Ivanov <dmiivanov@gmail.com>
//


#ifndef _rusdmc1_class_h_
#define _rusdmc1_class_h_


using namespace std;

class rusdmc1_class : public dstbank_class
{
    
public:
  
  // Thrown MC core position, CLF frame, [1200m] units, 
  // SD origin subtracted off (RUSDMC1_SD_ORIGIN_(X,Y)_CLF)
  Double_t xcore;
  Double_t ycore;  
  Double_t t0;      // Time of the core hitting the ground, [1200m], 
                    // with respect to the earliest SD time
  Double_t bdist;   // Distance of the core from the edge of the array.
                    // If negative, then the core is outside.  
  Double_t tdistbr; // Distance of the core position from BR T-Shaped boundary,
                    // negative if not in BR
  Double_t tdistlr; // same for LR
  Double_t tdistsk; // same for SK
  Double_t tdist;   // Closest distance to any T-shaped boundary (BR,LR,SK)
  
  rusdmc1_class();
  virtual ~rusdmc1_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(rusdmc1_class,5)
};
#endif
//    Class for showlib_ bank.
//    Last modified: Oct 21, 2010
//    Dmitri Ivanov <ivanov@physics.rutgers.edu>
//


#ifndef _showlib_class_
#define _showlib_class_


using namespace std;

class showlib_class: public dstbank_class
{

public:

  Int_t code; 
  /* first number  = species (1 for proton, 2 for iron)
   * second number = energy (second number of log(energy): 7 = 10^17, 0 = 10^20 ....)
   * third number  = not used (for finer energy grid) 
   * fourth+fifth  = zenith angle in degree (rounded value)
   * last number   = hadronic interaction code
   *                 0 = QGSJET
   *                 1 = SIBYLL 1.6
   *                 2 = SIBYLL 2.1
   */       
  Int_t       number;    // Number of shower in CORSIKA run
  Float_t     angle;     // Generation angle (radians)
  Int_t       particle;  // Primary particle (CORSIKA convention)
  Float_t     energy;    // Energy of primary particle in GeV
  Float_t     first;     // (Slant) Depth of first (actual) interaction in g/cm^2
  Float_t     nmax;      // GH fit parameter, maximum shower size divided by 1e9
  Float_t     x0;        // GH fit parameter, shower starting depth in g/cm^2
  Float_t     xmax;      // GH fit parameter, depth of shower max in g/cm^2
  Float_t     lambda;    // GH fit parameter, shower development rate in g/cm^2
  Float_t     chi2;      // Chi-square of CORSIKA's fit of the actual shower to the GH profile

  
  showlib_class();
  virtual ~showlib_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(showlib_class,1)
};



#endif
//    Class for hsum_ bank.
//    Last modified: Aug 8, 2017
//    Dmitri Ivanov <dmiivanov@gmail.com>


#ifndef _bsdinfo_class_
#define _bsdinfo_class_


using namespace std;

#ifndef BSDINFO_BANKID
#define BSDINFO_NBSDS 1024
#define BSDINFO_NBITS 16
#endif

class bsdinfo_class: public dstbank_class
{

public:
  
  Int_t yymmdd;       // date, YYMMDD format
  Int_t hhmmss;       // time, HHMMSS format
  Int_t usec;         // micro second
  Int_t nbsds;        // number of SDs that are part of event but not working properly
  vector<Int_t> xxyy; // [BSDINFO_NBSDS]  position IDs of bad SDs
  vector<Int_t> bitf; // [BSDINFO_NBSDS]  bit flag that describes what's wrong with the SD, 
  // if either of the 16 bits is set, there is a problem:
  // Checks that are done during TA SD Monte Carlo generation.  If either
  // of these conditions failed. then the SD is not used in the Monte Carlo.
  // bit 0:  ICRR calibration issue, failed ICRR don't use criteria
  // bit 1:  ICRR calibration issue, Mev2pe problem
  // bit 2:  ICRR calibration issue, Mev2cnt problem
  // bit 3:  ICRR calibration issue, bad pedestal mean values
  // bit 4:  ICRR calibration issue, bad pedestal standard deviation
  // bit 5:  ICRR calibration issue, saturation information not available  
  // bit 6:  Rutgers calibration issue, bad mip values
  // bit 7:  Rutgers calibration issue, bad pedestal peak channel
  // bit 8:  Rutgers calibration issue, bad pedestal right half peak channel
  // bit 9:  Rutgers calibration issue, bad 1-MIP peak fit number of degrees of freedom
  // bit 10: Rutgers calibration issue, bad 1-MIP peak fit chi2
  // Checks done the during event reconstruction.  If either of these fail then the counter
  // is not used in the event reconstruction.
  // bit 11: Rutgers calibration issue, peak channel of pedestal histogram
  // bit 12: Rutgers calibration issue, peak channel of 1-MIP histogram
  // bit 13: Rutgers calibration issue, 1-MIP histogram fit number of degrees of freedom
  // bit 14: Rutgers calibration issue, 1-MIP histogram chi2 / dof
  // bit 15: Rutgers calibration issue, FADC counts per VEM
  Int_t nsdsout;         // number of SDs either completely out (absent in the live detector list during event)
  vector<Int_t> xxyyout; // SDs that are completely out (can't participate in event readout)
  vector<Int_t> bitfout; // Bit flags of SDs that are considered as completely out, if not available then 0xFFFF
  bsdinfo_class();
  virtual ~bsdinfo_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(bsdinfo_class,2)
};



#endif
//    Class for SD sdtrgbk_ bank.
//    Last modified: Jan 20, 2020
//    Dmitri Ivanov <dmiivanov@gmail.com>
//

#ifndef _sdtrgbk_class_
#define _sdtrgbk_class_


using namespace std;

class sdtrgbk_class: public dstbank_class
{
public:

  // second fractions for the level-1 signals
  // (corresponds to the left edge of the sliding window)
  // [S]
  vector <vector<Double_t> >secf; // [SDTRGBK_NSD][SDTRGBK_NSIGPSD]

  // time limits for each SD,
  // [0] - minimum possible time,
  // [1] - maximum possible time [S]
  vector <vector<Double_t> >tlim; // [SDTRGBK_NSD][2]

  // BANK ID of the raw SD data bank used
  // Two possibilities:
  // RUSDRAW_BANKID
  // TASDCALIBEV_BANKID
  Int_t raw_bankid;

  // FADC time slices for the level-1 signals
  // (left edge of the sliding window)
  vector<vector<Short_t> > ich; // [SDTRGBK_NSD][SDTRGBK_NSIGPSD]

  // signal size (FADC counts) for the level-1 signals,
  // [0] = lower, [1] - upper
  vector<vector<vector<Short_t> > >q; //[SDTRGBK_NSD][SDTRGBK_NSIGPSD][2]

  // Waveform index inside the raw SD data bank
  // to tell from what waveform a given level-1 trigger
  // signal comes.
  vector<vector<Short_t> > l1sig_wfindex; //[SDTRGBK_NSD][SDTRGBK_NSIGPSD]

  vector<Short_t> xxyy; //[SDTRGBK_NSD] // counter position ID

  // index inside raw SD data bank
  // of the waveform that was used for
  // calibrating this SD
  vector<Short_t> wfindex_cal; //[SDTRGBK_NSD]

  // number of (level-1) signals that are above 150 FADC counts
  vector<Short_t> nl1; //[SDTRGBK_NSD]

  Short_t nsd; // number of SDs

  Short_t n_bad_ped; // number of SDs with bad pedestals

  Short_t n_spat_cont; // number of spatially contiguous SDs

  Short_t n_isol; // number of isolated SDs - SDs not taking part in space trigger patterns
  
  // number of SDs that take part in space trigger pattern
  // and that have waveforms whose parts connect in time with
  // parts of waveforms in other SDs in a space trigger pattern
  Short_t n_pot_st_cont;

  // number of SDs out of n_pot_spat_time_cont that also have level-1 trigger signals in them
  Short_t n_l1_tg;

  // by how much pedestal inside
  // the sliding window had to be lowered
  // to get the level-2 trigger
  Short_t dec_ped;

  // if the event triggers fine w/o 
  // lowering the pedestals, then
  // this variable tells
  // by how much pedestal inside
  // the sliding window can be reaised
  // and still have the event trigger
  Short_t inc_ped;

  vector<Short_t> il2sd;     // [3] indices of SDs that caused level-2 trigger (within this bank)
  vector<Short_t> il2sd_sig; // [3] indices of signals in each SD causing level-2 trigger

  // SD goodness flag.
  //    0 - SD has bad pedestals
  //    1 - SD has good pedestals but doesn't participate in space trigger patterns
  //    2 - SD participates in spatial trigger patterns
  //    3 - SD participates in spatial trigger patterns and parts of its waveforms 
  //        connect with parts of waveforms in other SDs in a space trigger pattern 
  //        with a given SD
  //    4 - SD satisfies the ig[isd] = 3 criteria and has level-1 signals in it
  //    5 - SD satisfies the ig[isd] = 4 criteria and happened to be picked for the event trigger
  //    6 - SD satisfies the ig[isd] = 4 criteria and happened to be picjed for the event trigger
  //        with raised pedestals but was not chosen in the event trigger 
  //        w/o raising the pedestal (ig[isd] = 5 criteria)
  //    7 - SD satisfies the ig[isd] = 5 criteria and also participates in the
  //        event trigger with raised pedestals
  vector <Char_t> ig; // [SDTRGBK_NSD]

  // trigger pattern
  //    IF TRIGGERED:
  //       0 - triangle
  //       1 - line
  //    IF DID NOT TRIGGER:
  //       0 - don't have at least 3 space contiguous SDs with good pedestals
  //       1 - don't have at least 3 SDs in the set of space-contiguous SDs 
  //           with waveforms whose parts connect in time with parts of waveforms in other
  //           SDs that formed a spatial trigger pattern with a given SD
  //       2 - don't have at least 3 level-1 signal SDs in the set of potentially space-time
  //           contiguous SDs
  //       3 - don't have at least 3 SDs with level-1 signals that are in time
  //           (in the set of potentially space-time contiguous SDs)
  Char_t trigp;
  
  // event goodness flag
  //    0 - event doesn't pass SD trigger backup even with lowered pedestals
  //    1 - event passes SD trigger backup with lowered pedestals
  //    2 - event passes SD trigger backup without lowering the pedestals
  //    3 - event passes SD trigger backup with raised pedestals
  Char_t igevent;

  sdtrgbk_class();
  virtual ~sdtrgbk_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(sdtrgbk_class,5)
};
#endif
//   Root class for raw ICRR event dst bank
//
//   tasdevent dst bank was written by  Akimichi Taketa,
//   Fri Apr 10 23:19:45 2009 JST
//  
//   converter from tasdevent DST bank to Root class was added by 
//   Dmitri Ivanov, Sun Mar 7, 2010.  Last modified: Oct 21, 2010
//


#ifndef _tasdevent_class_
#define _tasdevent_class_

#ifndef TASDEVENT_BANKID
#define tasdevent_nfadc 128
#endif

using namespace std;

class SDEventSubData_class : public dstbank_class
{
  
public:
  Int_t clock;		// clock count at the trigger timing	
  Int_t max_clock;	// maximum clock count between 1PPS	  
  Short_t lid;		// logical ID				
  Short_t usum;		// summation value of the upper layer	
  Short_t lsum;		// summation value of the lower layer	
  Short_t uavr;		// average of the FADC of the upper layer
  Short_t lavr;		// average of the FADC of the lower layer
  Short_t wf_id;	// waveform id in the trigger		
  Short_t num_trgwf;	// number of triggered waveforms 
  Short_t bank;		// ID of the triggered waveform		
  Short_t num_retry;	// the number of the retry		
  Short_t trig_code;	// level-1 trigger code			
  Short_t wf_error;	// broken waveform data			
  vector<Short_t> uwf;	// waveform of the upper layer 
  vector<Short_t> lwf;	// waveform of the lower layer 
  SDEventSubData_class();
  virtual ~SDEventSubData_class();
  void loadFromDST(Int_t iwf);
  void loadToDST(Int_t iwf);
  ClassDef(SDEventSubData_class,5)
};

class tasdevent_class : public dstbank_class
{
public:
  Int_t event_code;	 // 1=data, 0=Monte Carlo	
  Int_t run_id;		 // run id			
  Int_t site;		 // site id			
  Int_t trig_id;	 // trigger ID			
  
  Int_t trig_code;	 // level-2 trigger code,0 is internal,
			 // others are external		
  Int_t code;		 // internal trigger code	
  Int_t num_trgwf;	 // number of triggered waveform	
  Int_t num_wf;		 // number of aquired waveforms	
  
  Int_t bank;		 // bank id			
  Int_t date;		 // triggered date		
  Int_t time;		 // triggered time		
  Int_t date_org;	 // original triggered date	
  Int_t time_org;	 // original triggered time	
  Int_t usec;		 // triggered usec		
  Int_t gps_error;
  Int_t pos;		 // triggered position		
  vector<Int_t> pattern; // trigger pattern		
  
  vector<SDEventSubData_class> sub;   // [tasdevent_ndmax];
    
  Int_t pos2xxyy(Int_t pos_flag) { return ((pos_flag&0x3f)+(100*((pos_flag>>6)&0x3f))); }
  Int_t pos2xxyy() { return pos2xxyy(pos); }
  void trigp2xxyyt(Int_t trigp, Int_t *trigp_xxyy, Int_t *trigp_usec);
  void itrigp2xxyyt(Int_t itrigp, Int_t *trigp_xxyy, Int_t *trigp_usec);
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  tasdevent_class();
  virtual ~tasdevent_class();
  ClassDef(tasdevent_class,5)
};


#endif
//    Root tree class for SD tasdcalib_ bank. 
//    DST bank written by A. Taketa Sun May 03 15:45:25 2009 JST
//    This Root tree class made by Dmitri Ivanov <ivanov@physics.rutgers.edu>
//    Last modified: Oct 23, 2010


#ifndef _tasdcalib_class_
#define _tasdcalib_class_


#ifndef TASDCALIBEV_BANKID
#define tasdcalib_nhmax   3    // maximum number of host 
#define tasdcalib_nwmax   3    // maximum number of weather station 
#define tasdcalib_ndmax 512    // maximum number of detector/event 
#define tasdcalib_ntmax 100    // maximum number of trigger in 10 min. 
#endif

using namespace std;



class SDCalibHostData_class : public dstbank_class 
{
public:
  Int_t           site;        // site id 
  Int_t           numTrg;
  vector<Int_t>   trgBank;     // Trigger bank ID 
  vector<Int_t>   trgSec;      // Trigger sending time in 10 min.
  //                              [0-599]
  vector<Short_t> trgPos;      // Triggered position 
  vector<Int_t>   daqMode;     // daq code from central PC 
  vector<Char_t>  miss;        // DAQ error or GPS timestamp error
  //                              0 means OK, 1 means error.
  //                              LSB   : DAQ stop
  //                              bit-1 : DAQ timeout
  //                              bit-2 : timestamp miss 1 sec
  //                              bit-3 : timestamp miss more than 1 sec
  //                              bit-4 : timestamp miss more than 10 min.
  //                              bit-5 : critical error
  vector<Short_t> run_id;
  
  SDCalibHostData_class();
  virtual ~SDCalibHostData_class();
  void loadFromDST(Int_t ihost);
  void loadToDST(Int_t ihost);
  
  ClassDef(SDCalibHostData_class,1)
};


class  SDCalibSubData_class : public dstbank_class 
{
public:
  Int_t site;	               // site id 
  Int_t lid;	               // position id 
  Int_t livetime;              // livetime in 10 min
  Int_t warning;               // condition of sensors and trigger rate.
  //		                  0 means OK, 1 means error.
  //	                          LSB    : level-0 trigger rate
  //	                          bit-1  : level-1 trigger rate
  //	                          bit-2  : temperature sensor on scinti.
  //	                          bit-3  : temperature sensor on elec.
  //	                          bit-4  : temperature sensor on battery
  //	                          bit-5  : temperature sensor on charge cont.
  //	                          bit-6  : humidity sensor on scinti.
  //	                          bit-7  : battery voltage
  //	                          bit-8  : solar panel voltage
  //	                          bit-9  : LV value of charge cont.
  //	                          bit-10 : current from solar panel
  //	                          bit-11 : ground level
  //	                          bit-12 : 1.2V
  //	                          bit-13 : 1.8V
  //	                          bit-14 : 3.3V
  //	                          bit-15 : 5.0V
  //	                          bit-16 : clock count vs pedestal
	       

  Char_t dontUse;              // bad detector flag.
  //	                          0 means OK, 1 means error.
  //	                          LSB   : gps
  //	                          bit-1 : clock
  //	                          bit-2 : upper pedestal
  //	                          bit-3 : lower pedestal
  //	                          bit-4 : upper mip info
  //	                          bit-5 : lower mip info
  //	                          bit-6 : trigger rate
  //	                          bit-7 : temperature
		    
  Char_t dataQuality;          // condtion of data
  //	                          0 means exist, 1 means interpolated
  //	                          LSB   : gps
  //	                          bit-1 : clock
  //	                          bit-2 : upper pedestal
  //	                          bit-3 : lower pedestal
  //	                          bit-4 : upper mip info
  //	                          bit-5 : lower mip info
  //	                          bit-6 : trigger rate
  //	                          bit-7 : temperature
		    

  Char_t gpsRunMode;           // 1 is 3D fix, 2 is position hold
  vector<Char_t> miss;         // comm. error bit field
  Float_t clockFreq;           // clock frequency [Hs] 
  Float_t clockChirp;          // time deviation of clock frequency [Hs/s] 
  Float_t clockError;          // fluctuation of clock [ns]
  Float_t upedAvr;             // average of pedestal (upper layer) 
  Float_t lpedAvr;             // average of pedestal (lower layer) 
  Float_t upedStdev;           // standard deviation of pedestal
  //		                  (upper layer) 
  Float_t lpedStdev;           // standard deviation of pedestal
  //		                  (lower layer) 
  Float_t upedChisq;           // Chi square value (upper layer) 
  Float_t lpedChisq;           // Chi square value (lower layer)
  Float_t umipNonuni;          // Non-uniformity (upper layer) 
  Float_t lmipNonuni;          // Non-uniformity (lower layer) 
  Float_t umipMev2cnt;         // Mev to count conversion factor
  //		                  (upper layer) 
  Float_t lmipMev2cnt;         // Mev to count conversion factor
  //		                  (lower layer) 
  Float_t umipMev2pe;          // Mev to photo-electron conversion factor
  //		                  (upper layer) 
  Float_t lmipMev2pe;          // Mev to photo-electron conversion factor
  //		                  (lower layer) 
  Float_t umipChisq;           // Chi square value (upper layer) 
  Float_t lmipChisq;           // Chi square value (lower layer)
  Float_t lvl0Rate;            // level-0 trigger rate 
  Float_t lvl1Rate;            // level-1 trigger rate 
  Float_t scinti_temp;


  // [0] - lower layer, [1] - upper layer
  vector<Int_t> pchmip;       // peak channel of 1MIP histograms 
  vector<Int_t> pchped;       // peak channel of pedestal histograms 
  vector<Int_t> lhpchmip;     // left half-peak channel for 1mip histogram 
  vector<Int_t> lhpchped;     // left half-peak channel of pedestal histogram 
  vector<Int_t> rhpchmip;     // right half-peak channel for 1mip histogram 
  vector<Int_t> rhpchped;     // right half-peak channel of pedestal histograms 
  
  //Results from fitting 1MIP histograms 
  vector<Int_t> mftndof;      // number of degrees of freedom in 1MIP fit 
  vector<Float_t> mip;        // 1MIP value (ped. subtracted) 
  vector<Float_t> mftchi2;    // chi2 of the 1MIP fit 
  
  //   1MIP Fit function: 
  //   [3]*(1+[2]*(x-[0]))*exp(-(x-[0])*(x-[0])/2/[1]/[1])/sqrt(2*PI)/[1]
  //   4 fit parameters:
  //   [0]=Gauss Mean
  //   [1]=Gauss Sigma
  //   [2]=Linear Coefficient
  //   [3]=Overall Scalling Factor
  vector<vector<Float_t> > mftp;          //1MIP fit parameters 
  vector<vector<Float_t> > mftpe;         //Errors on 1MIP fit parameters 
  
  SDCalibSubData_class();
  virtual ~SDCalibSubData_class();
  void loadFromDST(Int_t idet);
  void loadToDST(Int_t idet);
  
  ClassDef(SDCalibSubData_class,1);
};


class SDCalibWeatherData_class 
{
public:
  Int_t   site;                // 0 is BRFD, 1 is LRFD, 4 will be CLF
  Float_t averageWindSpeed;    // [m/s]
  Float_t maximumWindSpeed;    // [m/s]
  Float_t windDirection;       // 0 is north, 90 is east [deg]
  Float_t atmosphericPressure; // [hPa]
  Float_t temperature;	       // [C]
  Float_t humidity;	       // [%RH]
  Float_t rainfall;	       // [mm/hour]
  Float_t numberOfHails;       // [hits/cm^2/hour]
  
  SDCalibWeatherData_class();
  virtual ~SDCalibWeatherData_class();
  void loadFromDST(Int_t iweat);
  void loadToDST(Int_t iweat);
  
  ClassDef(SDCalibWeatherData_class,1)
};




class tasdcalib_class: public dstbank_class
{

public:
  
  Int_t  num_host;             // the number of hosts
  Int_t  num_det;              // the number of detectors
  Int_t  num_weather;          // the number of weather stations
  Int_t  date;                 // year month day
  Int_t  time;                 // hour minute second
  vector<Char_t> trgMode;      // Trigger Mode
  vector<SDCalibHostData_class>    host;
  vector<SDCalibSubData_class>     sub;
  vector<SDCalibWeatherData_class> weather;
  Int_t footer;
  
  tasdcalib_class();
  virtual ~tasdcalib_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(tasdcalib_class,1)
};



#endif
//   Root class for calibrated event dst bank
//
//   tasdcalibev dst bank was written by  Akimichi Taketa,
//   Wed May 06 20:20:42 2009 JST
//  
//   converter from tasdcalibev DST bank to Root class was added by 
//   Dmitri Ivanov, Tue Feb 9, 2010; Lat Modified: Oct 21, 2010
//


#ifndef _tasdcalibev_class_
#define _tasdcalibev_class_

#ifndef TASDCALIBEV_BANKID
#define tasdcalibev_nhmax 3
#define tasdcalibev_nwmax 3
#define tasdcalibev_ndmax 512
#define tasdcalibev_nfadc 128
#endif

using namespace std;

class SDCalibevData_class : public dstbank_class
{
public:
  
  
  Short_t site;         // site id 
  Short_t lid;          // position id 
  
  Int_t clock;          // clock count at the trigger timing	
  Int_t maxClock;       // maximum clock count between 1PPS	
  Char_t wfId;          // waveform id in the trigger		
  Char_t numTrgwf;      // number of triggered waveforms 
  Char_t trgCode;       // level-1 trigger code			
  Char_t wfError;       // broken waveform data	or saturate
                        // LSB   : upper PMT saturate (with LED data)
                        // bit-1 : lower PMT saturate (with LED data)
                        // bit-2 : upper PMT saturate (with MIP data)
                        // bit-3 : lower PMT saturate (with MIP data)
                        // bit-4 : upper FADC saturate
                        // bit-5 : lower FADC saturate
                        // bit-6 : daq error (FPGA to SDRAM) 
  vector<Short_t> uwf;  // waveform of the upper layer 
  vector<Short_t> lwf;  // waveform of the lower layer 
  
  
  Float_t clockError;   // fluctuation of maximum clock count [ns] 
  Float_t upedAvr;      // average of pedestal (upper) 
  Float_t lpedAvr;      // average of pedestal (lower) 
  Float_t upedStdev;    // standard deviation of pedestal (upper) 
  Float_t lpedStdev;    // standard deviation of pedestal (lower) 
  
  Float_t umipNonuni;   // Non-uniformity (upper layer) 
  Float_t lmipNonuni;   // Non-uniformity (lower layer) 
  Float_t umipMev2cnt;  // Mev to count conversion factor (upper) 
  Float_t lmipMev2cnt;  // Mev to count conversion factor (lower) 
  Float_t umipMev2pe;   // Mev to photo-electron conversion factor
                        // (upper layer) 
  Float_t lmipMev2pe;   // Mev to photo-electron conversion factor
                        // (lower layer) 
  Float_t lvl0Rate;     // level-0 trigger rate 
  Float_t lvl1Rate;     // level-1 trigger rate 
  Float_t scintiTemp;
  
  
  Int_t warning;  // condition of sensors and trigger rate.
                  // 0 means OK, 1 means error.
                  // LSB    : level-0 trigger rate
                  // bit-1  : level-1 trigger rate
                  // bit-2  : temperature sensor on scInt_ti.
                  // bit-3  : temperature sensor on elec.
                  // bit-4  : temperature sensor on battery
                  // bit-5  : temperature sensor on charge cont.
                  // bit-6  : humidity sensor on scInt_ti.
                  // bit-7  : battery voltage
                  // bit-8  : solar panel voltage
                  // bit-9  : LV value of charge cont.
                  // bit-10 : current from solar panel
                  // bit-11 : ground level
                  // bit-12 : 1.2V
                  // bit-13 : 1.8V
                  // bit-14 : 3.3V
                  // bit-15 : 5.0V
                  // bit-16 : clock count vs pedestal 
  
  Char_t dontUse; // bad detector flag.
                  // 0 means OK, 1 means error.
                  // LSB   : gps
                  // bit-1 : clock
                  // bit-2 : upper pedestal
                  // bit-3 : lower pedestal
                  // bit-4 : upper mip info
                  // bit-5 : lower mip info
                  // bit-6 : trigger rate
                  // bit-7 : communication
  
  Char_t dataQuality; // condtion of data
                      //   0 means exist, 1 means Int_terpolated
                      //   LSB   : gps
                      //   bit-1 : clock
                      //   bit-2 : upper pedestal
                      //   bit-3 : lower pedestal
                      //   bit-4 : upper mip info
                      //   bit-5 : lower mip info
                      //   bit-6 : trigger rate
                      //   bit-7 : temperature 
  
  
  Char_t trgMode0;    // level-0 trigger mode 
  Char_t trgMode1;    // level-1 trigger mode 
  Char_t gpsRunMode;  // 1 is 3D fix, 2 is position hold
  Short_t uthreLvl0;  // threshold of level-0 trigger (upper) 
  Short_t lthreLvl0;  // threshold of level-0 trigger (lower) 
  Short_t uthreLvl1;  // threshold of level-1 trigger (upper) 
  Short_t lthreLvl1;  // threshold of level-1 trigger (lower) 
  
  Float_t posX;	      // relative position [m], positive is east 
  Float_t posY;	      // relative position [m], positive is north 
  Float_t posZ;	      // relative position [m], positive is up 
  Float_t delayns;    // signal cable delay 
  Float_t ppsofs;     // PPS ofset 
  Float_t ppsflu;     // PPS fluctuation 
  Int_t lonmas;	      // longitude [mas] 
  Int_t latmas;	      // latitude [mas] 
  Int_t heicm;	      // height [cm] 

  Short_t udec5pled;  // maximun lineality range of upper layer
                      // [FADC count] 
  Short_t ldec5pled;  // maximun lineality range of lower layer
                      // [FADC count] 
  Short_t udec5pmip;  // maximun lineality range of upper layer
                      // [FADC count] 
  Short_t ldec5pmip;  // maximun lineality range of lower layer
                      // [FADC count] 


  // [0] - lower layer, [1] - upper layer
  vector<Int_t> pchmip;       // peak channel of 1MIP histograms 
  vector<Int_t> pchped;       // peak channel of pedestal histograms 
  vector<Int_t> lhpchmip;     // left half-peak channel for 1mip histogram 
  vector<Int_t> lhpchped;     // left half-peak channel of pedestal histogram 
  vector<Int_t> rhpchmip;     // right half-peak channel for 1mip histogram 
  vector<Int_t> rhpchped;     // right half-peak channel of pedestal histograms 
  
  //Results from fitting 1MIP histograms 
  vector<Int_t> mftndof;      // number of degrees of freedom in 1MIP fit 
  vector<Float_t> mip;        // 1MIP value (ped. subtracted) 
  vector<Float_t> mftchi2;    // chi2 of the 1MIP fit 
  
  //   1MIP Fit function: 
  //   [3]*(1+[2]*(x-[0]))*exp(-(x-[0])*(x-[0])/2/[1]/[1])/sqrt(2*PI)/[1]
  //   4 fit parameters:
  //   [0]=Gauss Mean
  //   [1]=Gauss Sigma
  //   [2]=Linear Coefficient
  //   [3]=Overall Scalling Factor
  vector<vector<Float_t> > mftp;          //1MIP fit parameters 
  vector<vector<Float_t> > mftpe;         //Errors on 1MIP fit parameters 

  void loadFromDST(Int_t iwf); // iwf is the index of the triggered waveform
                               // (tasdcalibev_.sub[iwf])
  void loadToDST(Int_t iwf); 
  SDCalibevData_class();
  virtual ~SDCalibevData_class() ;
  ClassDef(SDCalibevData_class,5)
};


class SDCalibevWeatherData_class : public dstbank_class
{
public:
  
  Int_t site; // 0 is BRFD, 1 is LRFD, 4 will be CLF 
  Float_t atmosphericPressure; // [hPa] 
  Float_t temperature;	       // [C]   
  Float_t humidity;            // [%RH] 
  Float_t rainfall;	       // [mm/hour] 
  Float_t numberOfHails;       // [hits/cm^2/hour]
  
  void loadFromDST(Int_t iweat); // iweat is the index of the weather station
  void loadToDST(Int_t iweat);
  SDCalibevWeatherData_class();
  virtual ~SDCalibevWeatherData_class();
  ClassDef(SDCalibevWeatherData_class,5)
};


class SDCalibevSimInfo_class : public dstbank_class
{
public:
  vector<Char_t> interactionModel;
  vector<Char_t> primaryParticleType;
  Float_t primaryEnergy;
  Float_t primaryCosZenith;
  Float_t primaryAzimuth;
  Float_t primaryFirstIntDepth;
  Float_t primaryArrivalTimeFromPps;
  Float_t primaryCorePosX;
  Float_t primaryCorePosY;
  Float_t primaryCorePosZ;
  Float_t thinRatio;
  Float_t maxWeight;
  Int_t trgCode;
  Int_t userInfo;
  vector<Float_t> detailUserInfo;
  
  void loadFromDST();
  void loadToDST();
  SDCalibevSimInfo_class();
  virtual ~SDCalibevSimInfo_class();
  ClassDef(SDCalibevSimInfo_class,5)
};

class tasdcalibev_class : public dstbank_class
{
public:
  Int_t eventCode;  // 1=data, 0=Monte Carlo 
  Int_t date;       // triggered date 
  Int_t time;       // triggered time 
  Int_t usec;       // triggered usec 
  Int_t trgBank;    // Trigger bank ID 
  Int_t trgPos;     // triggered position (detector ID) 
  Int_t trgMode;    // LSB:BR, bit-1:LR, bit-2:SK 
  Int_t daqMode;    // LSB:BR, bit-1:LR, bit-2:SK 
  Int_t numWf;      // number of aquired waveforms	
  Int_t numTrgwf;   // number of triggered waveform	
  Int_t numWeather; // the number of weather stations 
  Int_t numAlive;
  Int_t numDead;

  vector<Int_t> runId;  // run id
  vector<Int_t> daqMiss;// DAQ error or GPS timestamp error
  // 0 means OK, 1 means error.
  // LSB   : DAQ stop
  // bit-1 : DAQ timeout
  // bit-2 : timestamp miss 1 sec
  // bit-3 : timestamp miss more than 1 sec
  // bit-4 : timestamp miss more than 10 min.
  // bit-5 : critical error
  vector <SDCalibevData_class>  sub; // max. size is tasdcalibev_ndmax
  vector <SDCalibevWeatherData_class> weather; // max. size is tasdcalibev_nwmax
  SDCalibevSimInfo_class sim;
    
  // the alive detectors information (max. size is tasdcalibev_ndmax)
  vector <Short_t>  aliveDetLid;
  vector <Short_t>  aliveDetSite;
  vector <Float_t>  aliveDetPosX;
  vector <Float_t>  aliveDetPosY;
  vector <Float_t>  aliveDetPosZ;

  // the dead detectors information (max. size is tasdcalibev_ndmax)
  vector <Short_t>  deadDetLid;
  vector <Short_t>  deadDetSite;
  vector <Float_t>  deadDetPosX;
  vector <Float_t>  deadDetPosY;
  vector <Float_t>  deadDetPosZ;

  Int_t footer;
  
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  tasdcalibev_class();
  virtual ~tasdcalibev_class();
  
  ClassDef(tasdcalibev_class,5)
};

#endif
//    Class for SD rufptn_ bank.
//    Last modified: Jan 16, 2020
//    Dmitri Ivanov <dmiivanov@gmail.com>
//


#ifndef _rufptn_class_
#define _rufptn_class_


using namespace std;

class rufptn_class : public dstbank_class
{

public:
  
  Int_t nhits;     // number of independent signals (hits) in the trigger
  Int_t nsclust;   // number of hits in the largest space cluster
  // Only one hit from each SD can be a part of space-time cluster
  Int_t nstclust; // number of SDs in the largest plane cluster
  
  // number of SDs which passed time pattern recognition and lie on border of
  // either BR or LR or SK
  Int_t nborder;

  // isgood - variable:
  // isgood[i] = 0 : the counter to which i'th hit corresonds was not working properly
  // isgood[i] = 1 : i'th hit is not a part of any clusters
  // isgood[i] = 2 : i'th hit is a part of space cluster
  // isgood[i] = 3:  i'th hit passed a rough time pattern recognition
  // isgood[i] = 4:  i'th hit is a part of the event
  // isgood[i] = 5:  i'th hit saturates the counter
  vector <Int_t> isgood;
  vector <Int_t> wfindex; // indicate to what 1st rusdraw waveform each hit corresponds
  vector <Int_t> xxyy;    // position of the hit
  vector <Int_t> nfold;   // foldedness of the hit (over how many 128 fadc widnows this signal extends)

  // Here,2nd index is interpreted as follows:
  // [*][0]: for lower counters
  // [*][1]: for upper counters
  vector <vector <Int_t> > sstart; // channel where the signal starts
  vector <vector <Int_t> > sstop; // channel where the signal stops

  // This is the channel since the signal start channel where the first derivative peak occurs
  // (first point of inflection). After that, the derivative will hit zero and become negative.*/
  vector <vector <Int_t> > lderiv; // Channel after which FADC makes a big jump

  // Record the channel, since the first point of inflection,
  // after which derivative is negative.
  vector <vector <Int_t> > zderiv; // channel after which derivative hits zero

  // fadc trace and monitoring, double precision data
  // SD coordinates with respect to CLF frame of reference, in units of [1200m]
  vector <vector <Double_t> > xyzclf;
  vector<Double_t> qtot; // Total charge in the event (sum over counters in space-time cluster)

  // Time of the earliest waveform in the trigger in seconds since midnight.
  //   To find time in seconds since midnight of any hit, do
  //   tearliest + (reltime of a given hit) * 1200m / (c*t).
  vector<Double_t> tearliest;
  
  vector <vector <Double_t> > reltime; // hit time, relative to to EARLIEST hit, in units of counter sep. dist
  vector <vector <Double_t> > timeerr; // error on time, in counter separation units
  vector <vector <Double_t> > fadcpa; // pulse area, in fadc counts, peds subtracted
  vector <vector <Double_t> > fadcpaerr; // errror on (pulse area - peds) in fadc counts
  vector <vector <Double_t> > pulsa; // pulse area in VEM (pedestals subtracted)
  vector <vector <Double_t> > pulsaerr; // error on pulse area in VEM (pedestals subtracted)
  vector <vector <Double_t> > ped; // pedestals taken from monitoring 
  vector <vector <Double_t> > pederr; // pedestal errors computed from the monitoring information (FWHM/2.33)
  vector <vector <Double_t> > vem; // FADC counts/VEM, from monitoring
  vector <vector <Double_t> > vemerr; // errors on the FADC counts/VEM (FWHM/2.33), using monitoring

  // Tyro geometry reconstruction (double precision data)

  // first index interpreted as follows:
  // [*][0]: using lower counters
  // [*][1]: using upper counters
  // [*][2]: using upper and lower counters (avaraged over upper/lower)
  vector<vector<Double_t> > tyro_cdist; // distances from the core for all counters that were hit
  // <x>,<y> are with respect to CLF origin in [1200m] units


  vector<vector<Double_t> > tyro_xymoments;  // <x>,<y>, and <x**2>,<xy>,<y**2> about (<x>,<y>) using charge
  vector<vector<Double_t> > tyro_xypmoments; // pricipal moments (eigenvalues) of 2nd moments matrix
  vector<vector<Double_t> > tyro_u;          // long axis, corresponding to larger eigenvalue
  vector<vector<Double_t> > tyro_v;          // short axis, corresponding to smaller eigenvalue
  
  // Time fit to a straight line of a t (rel. time) vs u plot, for points
  // in the st-cluster such that t<u (demand physicsally plausible timing)
  // [][0]-constant offset, [][1]-slope
  vector<vector<Double_t> > tyro_tfitpars;
  vector<Double_t> tyro_chi2;  // chi2/d.o.f. for T vs U fit
  vector<Double_t> tyro_ndof;  // # of d.o.f. for T vs U fit
  vector<Double_t> tyro_theta; // event zenith angle
  vector<Double_t> tyro_phi;   // event azimuthal angle

  rufptn_class();
  virtual ~rufptn_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(rufptn_class,5)
};



#endif
//    Class for SD rusdgeom_ bank.
//    Last modified: Jan 16, 2020
//    Dmitri Ivanov <dmiivanov@gmail.com>
//


#ifndef _rusdgeom_class_
#define _rusdgeom_class_
using namespace std;

class rusdgeom_class : public dstbank_class
{

public:

  vector <vector <Double_t> > sdsigq;  /* charge in VEM of the given signal 
					  in the given counter */
  
  
  /* Relative time of each signal in each counter in [1200m] units. 
     To convert this into time after midnight, do
     'time = tearliest + sdsigt[i][j]/RUSDGEOM_TIMDIST * (1e-6)' */
  vector <vector <Double_t> > sdsigt;

  /* Time resolution of each signal in each counter in [1200m] units. 
     To convert this into time after midn
     ight, do 'time = tearliest + sdsigt[i][j]/RUSDGEOM_TIMDIST * (1e-6)' */
  vector <vector <Double_t> > sdsigte;
  
  
  vector <vector <Double_t> > xyzclf; /* clf-frame xyz coordinates of each 
					 sd, [1200m] units */
  /* RUSDGEOM_ORIGIN_X_CLF,RUSDGEOM_ORIGIN_Y_CLF are subtracted off 
     from each X and Y value by the fitter, so that the core that comes out 
     is with respect to the origin of SD array */
  
  /* The following two variables are using sd signals which are part of the 
     event. If the sd itself is not
     a part of the event (see igsd - variable), then these variables are 
     calculated using the first signal seen by this sd */
  vector <Double_t> pulsa;  /* charge of the i'th counter in VEM */

  /* To convert this to time after midnight in seconds, do 
     'time = tearliest + sdtime[i]/RUSDGEOM_TIMDIST* (1e-6)'   */
  vector <Double_t> sdtime;   /* relative time of the i'th counter, 
				 [1200m] units */ 
    
  vector <Double_t> sdterr;   /* time resolution of the i'th counter, 
				 [1200m] units */ 
  
  /* 
     Results of geometry fits (double precision data)
     [0] - for plane fit
     [1] - for Modified Linsley's fit
     [2] - final values of the geometry fit (left blank for now)
  */

  /* To find the core position in CLF frame in meters with respect to 
     CLF origin, do
     coreX = (xcore+RUSDGEOM_ORIGIN_X_CLF)*1200.0, 
     coreY = (ycore+RUSDGEOM_ORIGIN_Y_CLF)*1200.0
     CLF XY plane is used as 'SD ground plane', that's why coreZ is absent. */
  vector<Double_t> xcore; /* core X and Y, in 1200m units, with respect to CLF, 
				SD origin subtracted */
  vector<Double_t> dxcore; /* uncertainty on xcore */
  vector<Double_t> ycore;
  vector<Double_t> dycore;
  vector<Double_t> t0;    /* time when the core hits the CLF plane, sec. aft. 
				midnight */
  vector<Double_t> dt0;
  vector<Double_t> theta; /* event zenith angle, degrees */
  vector<Double_t> dtheta;
  vector<Double_t> phi;   /* event azimuthal angle, degrees */
  vector<Double_t> dphi;
  vector<Double_t> chi2;  /* chi2 of the fit */
  Double_t a;       /* Curvature parameter */
  Double_t da;
  /* Earliest signal time in the trigger in seconds after midnight. 
     All other quoted times are relative to this time, 
     and are converted to [1200m] units for convenience. */
  Double_t tearliest;   /* Earliest signal time in the trigger in 
			   seconds after midnight */

  
  /* igsig[sd_index][signal_index]:
     0 - given SD was not working properly
     1 - given SD is not a part of any clusters 
     2 - given SD is a part of space cluster
     3 - given SD signal passed a rough time pattern recognition
     4 - given SD is a part of the event
     5 - given SD signal saturates the counter
  */
  
  vector <vector <Int_t> > igsig;
  
  /* irufptn[sd_index][signal_index]: points to the signal in rufptn dst bank, 
     which is a signal-based list of variables */
  vector <vector <Int_t> > irufptn;


  
  /* igsd[sd index]:
     0 - sd was not working properly (bad 1MIP fit,etc)
     1 - sd was working but is none of its signals is a part of event
     2 - sd is a part of event
     3 - sd is saturated 
  */
  vector <Int_t>  igsd;
  vector <Int_t>  xxyy;  /* sd position IDs */
  vector <Int_t>  nsig;  /* number of independent signals (hits) in each SD */
  
  /* 
     For each counter that's a part of the event there is only one signal 
     chosen.
     sdirufptn[sd_index] contains rufptn index (rufptn is a signal-based 
     list of variables) of the chosen signal.  
     If this sd is not a part of the event (see igsd variable), 
     then we quote here the rufptn index of the first singnal seen by the sd.
  */
  vector <Int_t>  sdirufptn;
  
  /* # of d.o.f. for geom. fitting, [0] - plane fitting, 
     [1] - Modified Linsley fit, [3] - Final result (b
     lank for now). 
     Calculated as  
     (# of signals in the event) - (# of fit parameters) - 1 + 2. 
     The (-1) term is because closest to the core counter is removed out in 
     the fitter. The 2 is added bec
     ause core X, Y are found
     using C.O.G. of charge and are treated as 2 additional data points in 
     the fit. */
  vector<Int_t> ndof;

  Int_t nsds;   /* number of sds in the trigger */  

  rusdgeom_class();
  virtual ~rusdgeom_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(rusdgeom_class,5)
};



#endif
//    Class for SD rufldf_ bank.
//    Last modified: Jan 16, 2020
//    Dmitri Ivanov <dmiivanov@gmai.com>
//

#ifndef _rufldf_class_h_
#define _rufldf_class_h_


using namespace std;

class rufldf_class : public dstbank_class
{

public:

  // [0] - LDF alone fit, [1] - Combined LDF and geom. fit
  vector<Double_t> xcore;
  vector<Double_t> dxcore;
  vector<Double_t> ycore;
  vector<Double_t> dycore;
  vector<Double_t> sc;
  vector<Double_t> dsc;
  vector<Double_t> s600;
  vector<Double_t> s600_0;
  vector<Double_t> s800;
  vector<Double_t> s800_0;
  vector<Double_t> aenergy;
  vector<Double_t> energy;
  vector<Double_t> atmcor;
  vector<Double_t> chi2;
  Double_t theta;
  Double_t dtheta;
  Double_t phi;
  Double_t dphi;
  Double_t t0;
  Double_t dt0;

  // Distance of the shower core from a closes SD array edge boundary.
  // If it is negative, then the core is outside of the array
  Double_t bdist;

  // Distance of the shower core from the closest T-shape bounday for BR,LR,SK
  // At most only one such distance is non-negagtive, as the shower core can
  // hit only one of the subarrays.  
  // If all distances are negative, this means that
  // the shower core either hits outside of the array or outside of 
  // BR,LR,SK subarrays
  Double_t tdistbr;
  Double_t tdistlr;
  Double_t tdistsk;
  Double_t tdist;   // Actual distance for T-shape boundary for a subarray
  vector<Int_t> ndof;

  rufldf_class();
  virtual ~rufldf_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(rufldf_class,7)
};

#endif
//    Class for etrack bank
//    Last modified: Apr 23, 2011
//    Dmitri Ivanov <ivanov@physics.rutgers.edu>
//


#ifndef _etrack_class_
#define _etrack_class_



using namespace std;

class etrack_class : public dstbank_class
{
  
public:
  
  // Privide all information in CLF frame: X=East, Y=North, Z=Up, Origin = CLF
  // CLF origin: (LATITUDE=39.29693 degrees, LONGITUDE=-112.90875 degrees).  
  // For most recent numbers, visit: 
  // http://www.telescopearray.org/tawiki/index.php/CLF/FD_site_locations
  Float_t    energy;    // event energy in EeV units (1 EeV = 10^18 eV), 0 if not available
  Float_t    xmax;      // slant depth where maximum number of charged particles occurs, [g/cm^2], 0 if not available
  
  // event direction in CLF frame
  // <sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)> points to where event comes from.
  Float_t    theta;     // zenith angle [radians]
  Float_t    phi;       // azimuthal angle [radians], counter-clock-wise from X=East
  
  
  // shower core in CLF frame
  // NOT NEEDED: a 3D point where event axis crosses Z=0 plane in some FD frame
  // WANT: just a 2D point where shower axis crosses the CLF Z=0 plane
  Double_t    t0;       // time when the shower axis crosses CLF Z=0 plane, [uS], with respect to GPS second
  Float_t    xycore[2]; // CLF XY point where event axis croses CLF Z=0 plane [meters], with respect to CLF origin
  
  
  vector<Float_t> udata; // [ETRACK_NUDATA]; non-essential, user-specific pieces of information
  Int_t nudata;    // number of user-specific pieces of information
  Int_t yymmdd;    // UTC date, yy=year since 2000, mm=month, dd=day
  Int_t hhmmss;    // UTC time, hh=hour,mm=minute,ss=second
  
  // optionally, one can also put in a label that tells if the event passes
  // the quality cuts of the reconstruction that produced these results.  This can be anything
  // starting from simple 1=YES, 0=NO to more elaborate flag that tells which cuts were passed and 
  // which ones failed.  For simplicity and clarity, it is best if one uses just 1=pass, 0=fail
  Int_t qualct;    // flag to indicate whether events passes the quality cuts
  
  etrack_class();
  virtual ~etrack_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(etrack_class,1)
};
#endif
/**
   Root tree class for atmpar DST bank.
   Last modified: Mar 09, 2018
   Dmitri Ivanov <dmiivanov@gmail.com>
**/
#ifndef _atmpar_class_h_
#define _atmpar_class_h_


using namespace std;

class atmpar_class : public dstbank_class
{

public:
  
  UInt_t dateFrom;    // sec from 1970/1/1
  UInt_t dateTo;      // sec from 1970/1/1
  Int_t  modelid;     // number of models 
  Int_t  nh;          // number of heights that distinquish layers
  vector<Double_t> h; // layer transition heights [cm]
  vector<Double_t> a; // parameters of the T(h) = a_i + b_i * exp(h/c_i) model determined by the fit
  vector<Double_t> b;
  vector<Double_t> c;
  Double_t chi2;      // quality of the fit
  Int_t    ndof;      // number of degees of freedom in the fit = number of points - number of fit parameter
  
  atmpar_class();
  virtual ~atmpar_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(atmpar_class,1)
};

#endif
