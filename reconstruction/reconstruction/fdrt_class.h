#include "sdfdrt_class.h"
/**
   Root tree class for fdraw DST bank.
   Last modified: Oct 21, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

#ifndef _fdraw_class_h_
#define _fdraw_class_h_


using namespace std;

class fdraw_class : public dstbank_class
{

public:
  
  Short_t event_code;                       // 1=normal, 0=monte carlo            
  Short_t part;                             // = run_id % 100                    
  Int_t num_mir;                            // number of participating cameras   
  Int_t event_num;                          // trigger id number                 

  // CTD trigger time 
  Int_t julian;                             // julian day                        
  Int_t jsecond;                            // second into julian day            
  Int_t gps1pps_tick;                       // last 1pps tick from gps           
  Int_t ctdclock;                           // ctd 40MHz clock tick              

  // Hardware version info
  Int_t ctd_version;
  Int_t tf_version;
  Int_t sdf_version;

  // selected TF data
   
    
  vector<Int_t> trig_code;                  // tf trigger code:
                                            // 0 = not a primary trigger
                                            // 1 = primary trigger
                                            // 2 = joint trigger
                                            // 3, 4 = very large signals
  
  vector<Int_t> second;                     // camera store time rel. to 0:00 UT 
  vector<Int_t> microsec;                   // microsec of store time          
  vector<Int_t> clkcnt;                     // camera 40 MHz clock tick          
  
  vector<Short_t> mir_num;                  // mirror id number (0-11)            
  vector<Short_t> num_chan;                 // number of channels with FADC data 
  
  vector<Int_t> tf_mode;
  vector<Int_t> tf_mode2;







  vector<vector<Short_t> > hit_pt;          // array of triggered tubes by camera (idx 0-255 are tubes, 256 is empty)

  // selected SDF data
  vector<vector<Short_t> > channel;         // channel ID number 
  vector<vector<Short_t> > sdf_peak;        // peak timing of input pulse 
  vector<vector<Short_t> > sdf_tmphit;      // internal value for trigg 
  vector<vector<Short_t> > sdf_mode;        // run mode 
  vector<vector<Short_t> > sdf_ctrl;        // communication mode 
  vector<vector<Short_t> > sdf_thre;        // for threshold and trigger mode 

  
  vector<vector<vector<UShort_t> > > mean;   // average of bkgnd. 0ms, 6ms, 53ms, and 79ms ago.
  vector<vector<vector<UShort_t> > > disp;   // rms of bkgnd. 0ms, 6ms, 53ms, and 79ms ago. 
  vector<vector<vector<Short_t> > > m_fadc; // raw 14-bit fadc data 

  
  fdraw_class();
  virtual ~fdraw_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
 
protected:
  // to used_bankid to tell the difference b/w BR/LR
  // or just the generic FD
  int used_bankid;
  ClassDef(fdraw_class,4)
};

#endif

/**
   Root tree class for brraw DST bank.
   Last modified: Oct 21, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

#ifndef _brraw_class_h_
#define _brraw_class_h_

class brraw_class : public fdraw_class
{
public:
  brraw_class();
  virtual ~brraw_class();
  ClassDef(brraw_class,4)
};

#endif

/**
   Root tree class for lrraw DST bank.
   Last modified: Oct 21, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

#ifndef _lrraw_class_h_
#define _lrraw_class_h_

class lrraw_class : public fdraw_class
{
public:
  lrraw_class();
  virtual ~lrraw_class();
  ClassDef(lrraw_class,4)
};

#endif
/**
   Root tree class for fdplane DST bank.
   Last modified: May 17, 2014
   Dmitri Ivanov <dmiivanov@gmail.com>
**/

#ifndef _fdplane_class_h_
#define _fdplane_class_h_



using namespace std;

class fdplane_class : public dstbank_class
{

public:
  Int_t part;                        // part number
  Int_t event_num;                   // event number
  Int_t julian;                      // run start day
  Int_t jsecond;                     // run start second (from start of julian)
  Int_t jsecfrac;                    // run start nanosecond (from start of jsecond)
  Int_t second;                      // event start second (from run start)
  Int_t secfrac;                     // event start nanosecond (from start of second)
  Int_t ntube;                       // number of tubes in event

  Int_t uniqID;                     // New to bank version 001: uniqID and fmode
  Int_t fmode;                      // Fmode used: 0=default (triangle) 1=MCRU


  vector<Double_t> npe;              // integrated pulse above pedestal in NPE
  vector<Double_t> adc;              // integrated pulse above pedestal in FADC counts
  vector<Double_t> ped;              // pedestal value under the pulse in FADC counts
  vector <Double_t> time;            // weighted average pulse time
  vector <Double_t> time_rms;        // weighted average pulse time rms
  vector <Double_t> sigma;           // tube significance

  Double_t sdp_n[3];                 // shower-detector plane normal (SDPN)
  Double_t sdp_en[3];                // uncertainty on SDPN fit
  Double_t sdp_n_cov[3][3];          // covariance matrix of SDPN fit
  Double_t sdp_the;                  // shower-detector plane theta angle
  Double_t sdp_phi;                  // shower-detector plane phi angle
  Double_t sdp_chi2;                 // SDPN fit chi2

  vector <Double_t> alt;             // altitude of tube
  vector <Double_t> azm;             // azimuth of tube
  vector <Double_t> plane_alt;       // altitude of tube rotated into SDP coordinate system
  vector <Double_t> plane_azm;       // azimuth of tube rotated into SDP coordinate system

  Double_t linefit_slope;            // linear fit to time vs. angle slope (ns / degree)
  Double_t linefit_eslope;           // linear fit to time vs. angle slope uncertainty (ns / degree)
  Double_t linefit_int;              // linear fit to time vs. angle intercept (ns)
  Double_t linefit_eint;             // linear fit to time vs. angle intercept uncertainty (ns)
  Double_t linefit_chi2;             // linear fit chi2
  Double_t linefit_cov[2][2];        // linear fit covariance
  vector <Double_t> linefit_res;     // linear fit tube residual (ns)
  vector <Double_t> linefit_tchi2;   // linear fit tube chi2 contribution

  Double_t ptanfit_rp;               // pseudo-tangent fit rp (meters)
  Double_t ptanfit_erp;              // pseudo-tangent fit rp uncertainty (meters)
  Double_t ptanfit_t0;               // pseudo-tangent fit t0 (ns)
  Double_t ptanfit_et0;              // pseudo-tangent fit t0 uncertainty (ns)
  Double_t ptanfit_chi2;             // pseudo-tangent fit chi2
  Double_t ptanfit_cov[2][2];        // pseudo-tangent fit covariance
  vector <Double_t> ptanfit_res;     // pseudo-tangent fit tube residual contribution (ns)
  vector <Double_t> ptanfit_tchi2;   // pseudo-tangent fit tube chi2 contribution

  Double_t rp;                       // tangent-fit rp (meters)
  Double_t erp;                      // tangent-fit rp uncertainty (meters)
  Double_t psi;                      // tangent-fit psi (radians)
  Double_t epsi;                     // tangent-fit psi uncertainty (radians)
  Double_t t0;                       // tangent-fit t0 (ns)
  Double_t et0;                      // tangent-fit t0 uncertainty (ns)
  Double_t tanfit_chi2;              // tangent-fit chi2
  Double_t tanfit_cov[3][3];         // pseudo-tangent fit covariance
  vector <Double_t> tanfit_res;      // pseudo-tangent fit tube residual (ns)
  vector <Double_t> tanfit_tchi2;    // pseudo-tangent fit tube chi2 contribution

  Double_t azm_extent;               // azimuthal extent of good tubes rotated into SDP coordinate system (radians)
  Double_t time_extent;              // time extent of good tubes (ns)

  Double_t shower_zen;               // Shower zenith angle (radians)
  Double_t shower_azm;               // Shower azimuthal angle (pointing back to source, radians, E=0, N=PI/2)
  Double_t shower_axis[3];           // Shower axis vector (along direction of shower propagation)
  Double_t rpuv[3];                  // Rp unit vector
  Double_t core[3];                  // Shower core location (meters)

  vector <Int_t> camera;             // camera number
  vector <Int_t> tube;               // tube number
  vector <Int_t> it0;                // FADC index of start of pulse
  vector <Int_t> it1;                // FADC index of end of pulse
  vector <Int_t> knex_qual;          // 1 = good connectivity, 0 = bad connectivity
  vector <Int_t> tube_qual;          // total tube quality
                                     // good = 1
                                     // bad  = decimal (-[bad_knex][bad_sdpn][bad_tvsa])

  Int_t ngtube;                      // number of good tubes in event
  Int_t seed;                        // original knex seed
  Int_t type;                        // type of event (down=2, up=3, intime=4, noise=5)
  Int_t status;                      // decimal time fit status ([good linear][good pseudotan][good tangent])
  Int_t siteid;                      // site ID (BR = 0, LR = 1)


  fdplane_class();
  virtual ~fdplane_class();
  virtual void loadFromDST();
  virtual void loadToDST();
  virtual void clearOutDST();

protected:
  // to used_bankid to tell the difference b/w BR/LR
  // or just the generic FD
  int used_bankid;
  ClassDef(fdplane_class,6)
};

#endif

/**
   Root tree class for brplane DST bank.
   Last modified: May 17, 2015
   Dmitri Ivanov <dmiivanov@gmail.com>
**/

#ifndef _brplane_class_h_
#define _brplane_class_h_

class brplane_class : public fdplane_class
{
public:
  brplane_class();
  virtual ~brplane_class();
  ClassDef(brplane_class,5)
};

#endif

/**
   Root tree class for lrplane DST bank.
   Last modified: May 17, 2015
   Dmitri Ivanov <dmiivanov@gmail.com>
**/

#ifndef _lrplane_class_h_
#define _lrplane_class_h_

class lrplane_class : public fdplane_class
{
public:
  lrplane_class();
  virtual ~lrplane_class();
  ClassDef(lrplane_class,5)
};

#endif
/**
   Root tree class for fdprofile DST bank.
   Last modified: Oct 21, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

#ifndef _fdprofile_class_h_
#define _fdprofile_class_h_



using namespace std;

class fdprofile_class : public dstbank_class
{

public:
 
  Int_t siteid;			     // site ID (BR = 0, LR = 1)
  Int_t ntslice;		     // number of time slices (FADC bins)
  Int_t ngtslice[3];		     // number of good time slices (acceptance)
  Int_t status[3];                   // status[0] is for fdplane_.psi
                                     // status[1] is for fdplane_.psi - fdplane_.epsi
                                     // status[2] is for fdplane_.psi + fdplane_.epsi
                                     // (-2 if bank is not filled
				     //  -1 if bad geometry fit
                                     //   0 if bad profile fit
                                     //   1 if good profile fit)
  
  vector<Int_t> timebin;             // [FDPROF_MAXTSLICE];   // FADC bin time slice
  
  Double_t rp[3];                    // Impact parameter (meters)
  Double_t psi[3];                   // Shower-detector plane angle (radians)
  Double_t t0[3];                    // Detection time at Rp, less Rp travel time (ns)

  Double_t Xmax[3];		     // Shower maximum (g/cm2)
  Double_t eXmax[3];		     // uncertainty on xmax
  Double_t Nmax[3];		     // Number of charged particles at shower maximum
  Double_t eNmax[3];		     // uncertainty on nmax
  Double_t Energy[3];		     // Initial cosmic-ray energy
  Double_t eEnergy[3];		     // uncertainty on energy
  Double_t chi2[3];		     // Total chi2 of fit
  
  vector<Double_t> npe;              // [FDPROF_MAXTSLICE];	// number of photoelectrons by time slice
  vector<Double_t> enpe;             // [FDPROF_MAXTSLICE];	// uncertainty on npe

  vector<vector<Double_t> > x;       // [3][FDPROF_MAXTSLICE];	// slant depth at middle of time slice (g/cm2)

  vector<vector<Double_t> > dtheta;  // [3][FDPROF_MAXTSLICE];  // angular size of bin (radians)
  vector<vector<Double_t> > darea;   // [3][FDPROF_MAXTSLICE];	// cosine-corrected active area of mirror (sq. meter)

  vector<vector<Double_t> > acpt;    // [3][FDPROF_MAXTSLICE];	// PMT acceptance by time slice
  vector<vector<Double_t> > eacpt;   // [3][FDPROF_MAXTSLICE];	// binomial uncertainty on acceptance

  vector<vector<Double_t> > flux;    // [3][FDPROF_MAXTSLICE];	// flux at the mirror [photons / (m2 * radian)]
  vector<vector<Double_t> > eflux;   // [3][FDPROF_MAXTSLICE];	// uncertainty on flux

  vector<vector<Double_t> > nfl;     // [3][FDPROF_MAXTSLICE];  // Flux of simulated fluorescence photons
  vector<vector<Double_t> > ncvdir;  // [3][FDPROF_MAXTSLICE];  // Flux of simulated direct cerenkov photons
  vector<vector<Double_t> > ncvmie;  // [3][FDPROF_MAXTSLICE];  // Flux of simulated Mie scattered cerenkov photons
  vector<vector<Double_t> > ncvray;  // [3][FDPROF_MAXTSLICE];  // Flux of simulated Rayleigh scattered cerenkov photons
  vector<vector<Double_t> > simflux; // [3][FDPROF_MAXTSLICE];  // Total flux of simluated photons

  vector<vector<Double_t> > tres;    // [3][FDPROF_MAXTSLICE];	// Time-slice fit residual
  vector<vector<Double_t> > tchi2;   // [3][FDPROF_MAXTSLICE];	// Time-slice fit chi2 contribution

  vector<vector<Double_t> > ne;      // [3][FDPROF_MAXTSLICE];	// Number of charged particles
  vector<vector<Double_t> > ene;     // [3][FDPROF_MAXTSLICE];	// uncertainty on ne

  Int_t mc;			     // [0 = don't use trumpmc bank info, 1 = use trumpmc bank]
  

  fdprofile_class();
  virtual ~fdprofile_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
protected:
  // to used_bankid to tell the difference b/w BR/LR
  // or just the generic FD
  int used_bankid;
  ClassDef(fdprofile_class,4)
};

#endif

/**
   Root tree class for brprofile DST bank.
   Last modified: Oct 21, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

#ifndef _brprofile_class_h_
#define _brprofile_class_h_

class brprofile_class : public fdprofile_class
{
public:
  brprofile_class();
  virtual ~brprofile_class();
  ClassDef(brprofile_class,4)
};

#endif

/**
   Root tree class for lrprofile DST bank.
   Last modified: Oct 21, 2010
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

#ifndef _lrprofile_class_h_
#define _lrprofile_class_h_

class lrprofile_class : public fdprofile_class
{
public:
  lrprofile_class();
  virtual ~lrprofile_class();
  ClassDef(lrprofile_class,4)
};

#endif
/**
   Root tree class for fdtubeprofile DST bank.
   Last modified: May 1, 2015
   Dmitri Ivanov <dmiivanov@gmail.com>
**/

#ifndef _fdtubeprofile_class_h_
#define _fdtubeprofile_class_h_


using namespace std;

// maximum number of fits in fdtubeprofile
#define fdtubeprofile_maxfit 3

class fdtubeprofile_class : public dstbank_class
{

public:

  
  Int_t ntube;                               // total number of tubes
  Int_t ngtube[fdtubeprofile_maxfit];        // number of good tubes
  
  Double_t rp[fdtubeprofile_maxfit];         // Impact parameter (meters)
  Double_t psi[fdtubeprofile_maxfit];        // Shower-detector plane angle (radians)
  Double_t t0[fdtubeprofile_maxfit];         // Detection time at Rp, less Rp travel time (ns)

  Double_t Xmax[fdtubeprofile_maxfit];       // Shower maximum (g/cm2)
  Double_t eXmax[fdtubeprofile_maxfit];      // uncertainty on xmax
  Double_t Nmax[fdtubeprofile_maxfit];       // Number of charged particles at shower maximum
  Double_t eNmax[fdtubeprofile_maxfit];      // uncertainty on nmax
  Double_t Energy[fdtubeprofile_maxfit];     // Initial cosmic-ray energy
  Double_t eEnergy[fdtubeprofile_maxfit];    // uncertainty on energy
  Double_t chi2[fdtubeprofile_maxfit];       // Total chi2 of fit
  
  real8 X0[fdtubeprofile_maxfit];            // effective depth of 1st inter.
  real8 eX0[fdtubeprofile_maxfit];           // uncertainty in X0
  real8 Lambda[fdtubeprofile_maxfit];        // profile width parameter
  real8 eLambda[fdtubeprofile_maxfit];       // uncertainty in lambda

  vector<vector<Double_t> > x;               // slant depth at middle of tube (g/cm2)

  vector<vector<Double_t> > npe;             // number of photo-electrons in tube
  vector<vector<Double_t> > enpe;            // uncertainty on NPE, including uncertainty from acceptance
  vector<vector<Double_t> > eacptfrac;       // fraction of uncertainty due to acceptance.

  vector<vector<Double_t> > acpt;            // PMT acceptance
  vector<vector<Double_t> > eacpt;           // binomial uncertainty on acceptance

  vector<vector<Double_t> > flux;            // flux at the mirror [detectable npe / (m2 * radian)]
  vector<vector<Double_t> > eflux;           // uncertainty on flux

  vector<vector<Double_t> > simnpe;          // simulated photo-electrons in tube

  vector<vector<Double_t> > nfl;             // Flux of simulated fluorescence photons
  vector<vector<Double_t> > ncvdir;          // Flux of simulated direct cerenkov photons
  vector<vector<Double_t> > ncvmie;          // Flux of simulated Mie scattered cerenkov photons
  vector<vector<Double_t> > ncvray;          // Flux of simulated Rayleigh scattered cerenkov photons
  vector<vector<Double_t> > simflux;         // Total flux of simluated photons

  vector<vector<Double_t> > ne;	             // Number of charged particles
  vector<vector<Double_t> > ene;	     // uncertainty on ne
  
  vector<vector<Double_t> > tres;	     // Time-slice fit residual
  vector<vector<Double_t> > tchi2;	     // Time-slice fit chi2 contribution

  vector <Int_t> camera;                     // Camera number for this tube
  vector <Int_t> tube;                       // Tube ID
  vector<vector <Int_t> >tube_qual;          // tube quality (good = 1, bad = 0, added = copy of fdplane tube status (EXPERIMENTAL, TENTATIVE))
  Int_t status[fdtubeprofile_maxfit];        // status[0] is for fdplane_.psi
                                             // status[1] is for fdplane_.psi - fdplane_.epsi
                                             // status[2] is for fdplane_.psi + fdplane_.epsi
                                             // (-2 if bank is not filled                    
                                             //  -1 if bad geometry fit
                                             //   0 if bad profile fit
                                             //   1 if good profile fit)

  
  Int_t siteid;                             // site ID (BR = 0, LR = 1)
  Int_t mc;                                 // [0 = don't use trumpmc bank info, 1 = use trumpmc 
  // available in version 3 or higher of fdtubeprofile bank 
  vector<vector<Double_t> >simtime;  // [3][FDTUBEPROF_MAXTUBE] time of simulated signal from waveform
  vector<vector<Double_t> >simtrms;  // [3][FDTUBEPROF_MAXTUBE] RMS of time
  vector<vector<Double_t> >simtres;  // [3][FDTUBEPROF_MAXTUBE] waveform time residual with fdplane
  vector<vector<Double_t> >timechi2; // [3][FDTUBEPROF_MAXTUBE] chi2 of above qty with fdplane
  fdtubeprofile_class();
  virtual ~fdtubeprofile_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
protected:
  // to used_bankid to tell the difference b/w BR/LR
  // or just the generic FD
  int used_bankid;
  ClassDef(fdtubeprofile_class,6)
};

#endif

/**
   Root tree class for brtubeprofile DST bank.
   Last modified: Sep 25, 2011
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

#ifndef _brtubeprofile_class_h_
#define _brtubeprofile_class_h_

class brtubeprofile_class : public fdtubeprofile_class
{
public:
  brtubeprofile_class();
  virtual ~brtubeprofile_class();
  ClassDef(brtubeprofile_class,5)
};

#endif

/**
   Root tree class for lrtubeprofile DST bank.
   Last modified: Sep 25, 2011
   Dmitri Ivanov <ivanov@physics.rutgers.edu>
**/

#ifndef _lrtubeprofile_class_h_
#define _lrtubeprofile_class_h_

class lrtubeprofile_class : public fdtubeprofile_class
{
public:
  lrtubeprofile_class();
  virtual ~lrtubeprofile_class();
  ClassDef(lrtubeprofile_class,5)
};

#endif
//    Class for MD hbar_ bank.
//    Last modified: Oct 21, 2010
//    Dmitri Ivanov <ivanov@physics.rutgers.edu>
//

#ifndef _hbar_class_h_
#define _hbar_class_h_


using namespace std;



class hbar_class : public dstbank_class
{

public:
  /* 2440000 subtracted from the julian day to give room for millisecond 
     precison */
  /* checks: "1/1/1985,0.0hr UT" gives 6066 + 0.5; jday=0 gives 1968 May 23.5 
     UT*/
  
  Int_t jday;        /* mean julian day - 2.44e6 */
  Int_t jsec;        /* second into Julian day */
  Int_t msec;        /* milli sec of julian day (NOT since UT0:00) */
 

  /* Filled with an enum source type */  
  Int_t source;
  
  Int_t nmir;        /* number of mirrors for this event */
  Int_t ntube;       /* total number of tubes for this event */
  
  
  
  /* 
     Jday of hnpe dst bank from which data for this bank is extracted. 
     One jday for each mirror.
  */
  vector <Double_t>  hnpe_jday; 
  
  /*---------------mirror info, one for each of nmir mirrors-------------*/
  
  vector <Int_t>  mir;	  /* mirror # (id), one for each of nmir
			     mirrors */
  
  /* Mirror reflectivity coefficient, one for each mirror */
  
  vector <Double_t>  mir_reflect;

  /*---------------tube info, one for each of ntube tubes----------------*/

  vector <Int_t>  tubemir;  /* mirror #, saved with tube as short */
  vector <Int_t>  tube;    /* tube # */
  vector <Int_t>  qdcb;    /* digitized channel B charge integral */

  /*
    Number of photo-electrons that is calculated by taking the qdcb from 
    above and multipling it by the first order gain from the hnpe dst bank
  */
  
  vector <Double_t>  npe;

  /*
    Sigma of photo-electrons from above calculated by taking the sigma of the 
    first order gain and multiplying it by the qdcb from above
  */

  vector <Double_t>  sigma_npe;  
 
  /*
    flag used to determine if a given tube has had problems in the fitting 
    process.
  */
  
  vector <Byte_t>  first_order_gain_flag;
  
 
  /*
    Second order gain obtained from the electronic calibration and relates a
    nanovolt-second pulse width to photo-electrons.

    nVs(QDCB[i][t], width) * second_order_gain[i][t] = NPE[i][t] 
  */
     
  vector <Double_t>  second_order_gain; 
						
  /*
    The second order fit goodness is a "goodness of fit" estimate for the 
    NVs(QDCB, w) vs NPE fit.
  */
  
  vector <Double_t>  second_order_gain_sigma;
  
  /*
    flag used to determine if a given tube has had problems in the second 
    order gain fitting process.
  */
  
  vector <Byte_t>  second_order_gain_flag;
  
  /*
    These values are the quantum efficiency factor for each tube at 337 nm.
    The values are to be used in scaling a standard quantum curve. 
  */
  
  vector <Double_t>  qe_337;
  vector <Double_t>  sigma_qe_337;
  
  /*
    These values modify the given HiRes UV filter curve for every tube.
    They are applied as an exponent to the curve being used.
  */

  vector <Double_t>  uv_exp;

  hbar_class();
  virtual ~hbar_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(hbar_class,4)
};



#endif
//    Class for MD hraw1_ bank.
//    Last modified: Oct 21, 2010
//    Dmitri Ivanov <ivanov@physics.rutgers.edu>
//


#ifndef _hraw1_class_
#define _hraw1_class_


using namespace std;

class hraw1_class : public dstbank_class
{

public:
  /* 2440000 subtracted from the julian day to give room for millisecond precision */
  /* checks: "1/1/1985,0.0hr UT" gives 6066 + 0.5; jday=0 gives 1968 May 23.5 UT */
  Int_t jday;        /* mean julian day - 2.44e6 */
  Int_t jsec;        /* second into Julian day */
  Int_t msec;        /* milli sec of julian day (NOT since UT0:00) */
  Int_t status;      /* set bit 0 to 1 when converts from .pln file */
  /* set bit 1 to 1 for Monte Carlo events */
  Int_t nmir;        /* number of mirrors for this event */
  Int_t ntube;			/* total number of tubes for this event */
	
  /* -------------- mir info, one for each of nmir mirrors  ---------------*/
  vector <Int_t>  mir ;	    /* mirror # (id), saved as short */
  vector <Int_t>  mir_rev;	    /* mirror version (rev3 or rev4) */
  vector <Int_t>  mirevtno ;	    /* event # from mirror packet */
  vector <Int_t>  mirntube ;	    /* # of tubes for that mir, saved as short */
  vector <Int_t>  miraccuracy_ns;  /* clock accuracy (gps or wwvb) in nsec */
  vector <Int_t>  mirtime_ns;	    /* time of mirror holdoff in nsec from second */
  
  /* -------------- tube info, one for each of ntube tubes  ---------------*/
  vector <Int_t>  tubemir ;	    /* mirror #, saved with tube as short */
  vector <Int_t>  tube ;    /* tube # */
  vector <Int_t>  qdca ;    /* digitized channel A charge integral */
  vector <Int_t>  qdcb ;    /* digitized channel B charge integral */
  vector <Int_t>  tdc ;    /* digitized tube trigger to holdoff time */
  vector <Int_t>  tha ;    /* trigger threshold in millivolts on minute ch A */
  vector <Int_t>  thb ;    /* trigger threshold in millivolts on minute ch B */
  /* thb[] = 0 for hr1 mirrors */
  vector <Float_t> prxf ;      /* # of photons according to RXF calib. */
  vector <Float_t> thcal1 ;    /* time according to HCAL1 calib. */
    
  hraw1_class();
  virtual ~hraw1_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(hraw1_class,4)
};

#endif
//    Class for MD mc04_ bank.
//    Last modified: May 5, 2015
//    Dmitri Ivanov <dmiivanov@gmail.com>
//

#ifndef _mc04_class_h_
#define _mc04_class_h_


using namespace std;



class mc04_class : public dstbank_class
{

public:

  // define geometry parameters

  Double_t    energy;                  // shower energy (EeV), (mJ) if laser
  Double_t    csmax;                   // shower size at shower max. / 1e9
  Double_t    x0;                      // G-H fit paramater (g/cm^2)
  Double_t    x1;                      // depth of first interaction (g/cm^2)
  Double_t    xmax;                    // depth of shower max        (g/cm^2)
  Double_t    lambda;                  // G-H fit paramater          (g/cm^2)
  Double_t    xfin;                    // depth at rfin (from rini)  (g/cm^2)
  Double_t    rini[3];                 // vector position of x0 point
  Double_t    rfin[3];                 // vector position of final point

  Double_t    uthat[3];                // track direction unit vector
  Double_t    theta;                   // shower track zenith angle
  Double_t    Rpvec[3];                // Rp vector to track (m) from origin
  Double_t    Rcore[3];                // shower core (from origin),z=0 (m)
  Double_t    Rp;                      // magnitude of Rpvec

  vector<vector<Double_t> >  rsite;    // [MC04_MAXEYE][3] site location with respect to origin
  vector<vector<Double_t> >  rpvec;    // [MC04_MAXEYE][3] Rp vector to track (meters)
  vector<vector<Double_t> >  rcore;    // [MC04_MAXEYE][3] shower core vector,z=0 (meters)
  vector<vector<Double_t> >  shwn;     // [MC04_MAXEYE][3] shower-detector plane
  vector<Double_t> rp;                 // [MC04_MAXEYE] magnitude of rpvec
  vector<Double_t> psi;                // [MC04_MAXEYE] psi angle in SD plane

  Double_t    aero_vod;                // aerosols vertical optical depth
  Double_t    aero_hal;                // aerosols horiz. attenuation length (m)
  Double_t    aero_vsh;                // aerosols vertical scale height (m)
  Double_t    aero_mlh;                // aerosols mixing layer height

  Double_t    la_site[3];              // laser or flasher site (meters)
  Double_t    la_wavlen;               // laser wave length (nm)
  Double_t    fl_totpho;               // total number of photons
  Double_t    fl_twidth;               // flasher pulse width (ns)

  Int_t iprim;                         // primary particle: 1=proton, 2=iron

  Int_t eventNr;                       // event number in mc file (set)
  Int_t setNr;                         // set identifier YYMMDDPP

  Int_t iseed1;                        // iseed before event
  Int_t iseed2;                        // iseed after event

  Int_t detid;                         // detector id ( Hires, TA, TALE, ... )
  Int_t maxeye;                        // number of sites in detector
  vector<Int_t> if_eye;                // [MC04_MAXEYE] if (site[ieye] != 1) ignore site

  Int_t neye;                          // number of sites triggered
  Int_t nmir;                          // number of mirrors in event
  Int_t ntube;                         // total number of tubes in event

  vector<Int_t> eyeid;                 // [MC04_MAXEYE] triggered site id
  vector<Int_t> eye_nmir;              // [MC04_MAXEYE] number of triggered mirrors in eye
  vector<Int_t> eye_ntube;             // [MC04_MAXEYE] number of triggered tube in eye

  vector<Int_t> mirid;                 // [MC04_MAXMIR] triggered mirrors id
  vector<Int_t> mir_eye;               // [MC04_MAXMIR] triggered mirrors id
  vector<Int_t> thresh;                // [MC04_MAXMIR] mir. average tube threshold in mV

  vector<Int_t> tubeid;                // [MC04_MAXTUBE] tube id
  vector<Int_t> tube_mir;              // [MC04_MAXTUBE] mirror id for each tube
  vector<Int_t> tube_eye;              // [MC04_MAXTUBE] eye id for each tube
  vector<Int_t> pe;                    // [MC04_MAXTUBE] pe's received by tube from shower
  vector<Int_t> triggered;             // [MC04_MAXTUBE] 1 if tube is part of triggered event, 0 otherwise
  vector<Float_t> t_tmean;             // [MC04_MAXTUBE] pe's mean arrival time
  vector<Float_t> t_trms;              // [MC04_MAXTUBE] pe's RMS of arrival times
  vector<Float_t> t_tmin;              // [MC04_MAXTUBE] pe's min. arrival time
  vector<Float_t> t_tmax;              // [MC04_MAXTUBE] pe's max. arrival time
  
  
  mc04_class();
  virtual ~mc04_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(mc04_class,5)
};

#endif
//    Class for MD mcraw_ bank.
//    Last modified: Oct 21, 2010
//    Dmitri Ivanov <ivanov@physics.rutgers.edu>
//

#ifndef _mcraw_class_h_
#define _mcraw_class_h_


using namespace std;


class mcraw_class : public dstbank_class
{

public:

  
  /* 2440000 subtracted from the julian day to give room for millisecond 
     precision.
     checks: "1/1/1985,0.0hr UT" gives 6066 + 0.5;
     jday=0 gives 1968 May 23.5 UT
  */
  Int_t jday;        /* mean julian day - 2.44e6 */
  Int_t jsec;        /* second into Julian day */
  Int_t msec;        /* milli sec of julian day (NOT since UT0:00) */

  Int_t neye;        /* number of sites triggered */
  Int_t nmir;        /* number of mirrors for this event */
  Int_t ntube;       /* total number of tubes for this event */
	
  /* -------------- eye info  ---------------*/
  
  vector <Int_t>  eyeid;

  /* -------------- mir info, one for each of nmir mirrors  ---------------*/

  vector <Int_t>  mirid;  /* mirror # (id), saved as short */
  vector <Int_t>  mir_eye;  /* eye # (id), saved as short */
  vector <Int_t>  mir_rev;  /* mirror version (rev3 or rev4) */
  vector <Int_t>  mirevtno;  /* event # from mirror packet */
  vector <Int_t>  mir_ntube;  /* # of tubes for that mir */
  vector <Int_t>  mirtime_ns;  /* time of mir. holdoff in ns from sec. */

  /* -------------- tube info, one for each of ntube tubes  ---------------*/
  
  vector <Int_t>  tube_eye; /* eye #, saved with tube as short */
  vector <Int_t>  tube_mir; /* mirror #, saved with tube as short */
  vector <Int_t>  tubeid; /* tube # */
  vector <Int_t>  qdca; /* digitized channel A charge integral */
  vector <Int_t>  qdcb; /* digitized channel B charge integral */
  vector <Int_t>  tdc; /* digitized tube trigger to holdoff time */
  vector <Int_t>  tha; /* trigger threshold (mV) on minute ch A */
  vector <Int_t>  thb; /* trigger threshold (mV) on minute ch B */
  vector <Float_t>  prxf; /* # of photons according to RXF calib. */
  vector <Float_t>  thcal1 ; /* time according to HCAL1 calib. */
  
  mcraw_class();
  virtual ~mcraw_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(mcraw_class,4)
};


#endif
//    Class for MD stps2_ bank.
//    Last modified: Oct 21, 2010
//    Dmitri Ivanov <ivanov@physics.rutgers.edu>
//

#ifndef _stps2_class_
#define _stps2_class_


using namespace std;

class stps2_class : public dstbank_class
{

public:

  
  Int_t maxeye;
  
  /* 
     plog is the negative log10 probability (0.0 - 1.0) that an event is noise.
     It is calculated by the function rc1_ray0(), and equals: 
     rvec^2 / log10(npair), where rvec is the Rayleigh Vector computed by
     rc1_ray0() and npair is the number of neighboring tubes in an event.
  */
  vector <Float_t>  plog;
  
  /* 
     rvec is, as describes above, the Rayleigh Vector magnitude calculated by
     the function rc1_ray0() for the event.
  */
  vector <Float_t>  rvec;
  
  /* 
     rwalk is the Rayleigh Vector magnitude that would be due to a random 
     scattering of a given number of pairs of neighboring tubes (npair)..
  */
  vector <Float_t>  rwalk;
  
  /* 
     ang is the angle between the y-axis and the Rayleigh Vector  =
     arccos( rvec_y / rvec_mag ). It is used to calculate the upward bit..
  */
  vector <Float_t>  ang;
  
  /* 
     aveTime, and sigmaTime are the mean and standard deviation of the 
     calibrated trigger times of all the in-time tubes in an event. In-time
     tubes are described below. The calibrated values for the tubes are
     in the thcal1[] array of the hraw1 common block.
  */
  vector <Float_t>  aveTime;
  vector <Float_t>  sigmaTime;
  
  /* 
     The mean and standard deviaiton of the calibrated photon count (prxf[])
     for all in-time tubes in the event.
  */
  vector <Float_t>  avePhot;
  vector <Float_t>  sigmaPhot;

  /*
    inTimeTubes is the number of tubes whose standard deviation from the mean
    of all the tubes falls inside 3x the standard deviation for all the tubes.

    total_lifetime is the total amount of time between the first tube that
    fired and the last tube that fired.

    lifetime is the maximum in-time tube trigger time minus the minimum 
    in-time tube trigger time. It gives some idea of the temporal spread of
    the in-time tubes in an event.
  */
  vector <Float_t>  lifetime;
  vector <Float_t>  totalLifetime;
  vector <Int_t>  inTimeTubes;

  /*
    if ( if_eye[ieye] != 1) ignore site
  */
  vector <Int_t>  if_eye;
  
  /* 
     upward is either 1 or 0 depending on whether the stps2 filter thought an
     event was upward going or downward going, respectively. 
  */
  vector <Char_t>  upward;
  
  
  stps2_class();
  virtual ~stps2_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(stps2_class,4)
};


#endif
//    Class for MD stpln_ bank.
//    Last modified: Feb 26, 2016
//    Dmitri Ivanov <dmiivanov@gmail.com>
//

#ifndef _stpln_class_
#define _stpln_class_


using namespace std;


class stpln_class : public dstbank_class
{

public:

  /* 2440000 subtracted from the julian day to give room for millisecond 
     precision.
     checks: "1/1/1985,0.0hr UT" gives 6066 + 0.5;
     jday=0 gives 1968 May 23.5 UT
  */
  Int_t jday;        /* mean julian day - 2.44e6 */
  Int_t jsec;        /* second into Julian day */
  Int_t msec;        /* milli sec of julian day (NOT since UT0:00) */

  Int_t neye;        /* number of sites triggered */
  Int_t nmir;        /* number of mirrors for this event */
  Int_t ntube;       /* total number of tubes for this event */
	
  /*
    if ( if_eye[ieye] != 1) ignore site
  */
  Int_t maxeye;

  
  /* -------------- eye info  ---------------*/
  vector <Int_t>  if_eye;
  vector <Int_t>  eyeid;
  vector <Int_t>  eye_nmir;        /* number of mirrors for this event */
  vector <Int_t>  eye_ngmir;        /* number of mirrors for this event */
  vector <Int_t>  eye_ntube;       /* total number of tubes for this event */
  vector <Int_t>  eye_ngtube;

  vector < vector <Float_t> >  n_ampwt;    /* amplitude weighted plane normal */
  vector < vector <Float_t> >  errn_ampwt; /* error in n_ampwt[]              */
  
  vector <Float_t>  rmsdevpln;   /* rms deviation in offplane angle (rad)  */
  vector <Float_t>  rmsdevtim;   /* rms deviation in tube trigger time from
				    time fit to a quadratic (microseconds) */

  vector <Float_t>  tracklength;  /* tracklength in degrees */
  vector <Float_t>  crossingtime;  /* time difference between last and first
				      good tubes to trigger (microseconds) */
  vector <Float_t>  ph_per_gtube;  /* average number of photons per good tube */

  /* -------------- mir info, one for each of nmir mirrors  ---------------*/

  vector <Int_t>  mirid;  /* mirror # (id), saved as short */
  vector <Int_t>  mir_eye;  /* eye # (id), saved as short */
  vector <Int_t>  mir_type;  /* Hires/TA/ToP */
  vector <Int_t>  mir_ngtube;  /* # of tubes for that mir */
  vector <Int_t>  mirtime_ns;  /* time of mir. holdoff in ns from sec. */

  /* -------------- tube info, one for each of ntube tubes  ---------------*/

  vector <Int_t>  ig;        /* ig = 1 is for good tubes, 0 for noise tubes */
  vector <Int_t>  tube_eye;  /* eye #, saved with tube as short */
  vector <Int_t>  saturated;   /* tube saturation flag */
  vector <Int_t>  mir_tube_id; /* mir_tube_id = (mirror_id*1000+tube_id) */

  
  stpln_class();
  virtual ~stpln_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(stpln_class,5)
};


#endif
//    Class for MD hctim_ bank.
//    Last modified: May 29, 2019
//    Dmitri Ivanov <dmiivanov@gmail.com>
//

#ifndef _hctim_class_h_
#define _hctim_class_h_


using namespace std;


class hctim_class : public dstbank_class
{
  
public:

#define HCTIM_MAXFIT 16

  vector<Double_t> mchi2; 
  vector<Double_t> rchi2; 
  vector<Double_t> lchi2;  
  vector<Double_t> mrp; 
  vector<Double_t> rrp ; 
  vector<Double_t> lrp ;
  vector<Double_t> mpsi; 
  vector<Double_t> rpsi; 
  vector<Double_t> lpsi;
  vector<Double_t> mthe; 
  vector<Double_t> rthe; 
  vector<Double_t> lthe;
  vector<Double_t> mphi; 
  vector<Double_t> rphi; 
  vector<Double_t> lphi;
  vector<Int_t> failmode;
  vector<Int_t> timinfo;
  vector<Int_t> jday;   /* mean julian day - 2.44e6 */
  vector<Int_t> jsec;   /* second into Julian day */
  vector<Int_t> msec;   /* milli sec of julian day (NOT since UT0:00) */
  vector<Int_t> ntube; // number of tubes in each fit
  vector<Int_t> nmir; // numbers of mirrors in each fit
  
  vector<vector<Double_t> > mtkv; // [HCTIM_MAXFIT][3]
  vector<vector<Double_t> > rtkv; // [HCTIM_MAXFIT][3] 
  vector<vector<Double_t> > ltkv; // [HCTIM_MAXFIT][3]
  vector<vector<Double_t> > mrpv; // [HCTIM_MAXFIT][3] 
  vector<vector<Double_t> > rrpv; // [HCTIM_MAXFIT][3] 
  vector<vector<Double_t> > lrpv; // [HCTIM_MAXFIT][3]
  vector<vector<Double_t> > mrpuv; // [HCTIM_MAXFIT][3] 
  vector<vector<Double_t> > rrpuv; // [HCTIM_MAXFIT][3] 
  vector<vector<Double_t> > lrpuv; // [HCTIM_MAXFIT][3]
  vector<vector<Double_t> > mshwn; // [HCTIM_MAXFIT][3] 
  vector<vector<Double_t> > rshwn; // [HCTIM_MAXFIT][3] 
  vector<vector<Double_t> > lshwn; // [HCTIM_MAXFIT][3]
  vector<vector<Double_t> > mcore; // [HCTIM_MAXFIT][3] 
  vector<vector<Double_t> > rcore; // [HCTIM_MAXFIT][3] 
  vector<vector<Double_t> > lcore; // [HCTIM_MAXFIT][3]

  /* tube info */
  vector<vector<Int_t > > tubemir; // mirror number of the tube
  vector<vector<Int_t > > tube;    // tube number
  vector<vector<Int_t > > ig;      // tube flag
  vector<vector<Double_t > > time; /* tube time */
  vector<vector<Double_t > > timefit; /* time from best fit */
  vector<vector<Double_t > > thetb; /* viewing angle */
  vector<vector<Double_t > > sgmt; /* sigma time */ 
  vector<vector<Double_t > > asx;
  vector<vector<Double_t > > asy;
  vector<vector<Double_t > > asz;

  /* mirror info */
  vector<vector<Int_t > > mir; /* mir number */
  vector<vector<Int_t > > mirntube;


 

  
  hctim_class();
  virtual ~hctim_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(hctim_class,5)
};

#endif
//    Class for MD hcbin_ bank.
//    Last modified: Oct 21, 2010
//    Dmitri Ivanov <ivanov@physics.rutgers.edu>
//

#ifndef _hcbin_class_h_
#define _hcbin_class_h_


using namespace std;


class hcbin_class : public dstbank_class
{
  
public:
#define HCBIN_MAXFIT 16  
#define HCBIN_MAXBIN 300
  vector <vector <Double_t> > bvx;
  vector <vector <Double_t> > bvy;
  vector <vector <Double_t> > bvz;
  vector <vector <Double_t> > bsz; /* bin size in degrees */   
  vector <vector <Double_t> > sig; /* signal in pe/degree/m^2 */
  vector <vector <Double_t> > sigerr; /* error on the signal */
  vector <vector <Double_t> > cfc; /* correction factor or exposure
				      of the bin in degree*MRAREA */
  vector <vector <Int_t> > ig;   /*  ig=  1: good bin */
  vector <Int_t> nbin;                  /* number of bins */
  vector <Int_t> failmode;                  /* 0 ==> Success */
  vector <Int_t> bininfo;

  vector <Int_t> jday;   /* mean julian day - 2.44e6 */
  vector <Int_t> jsec;   /* second into Julian day */
  vector <Int_t> msec;   /* milli sec of julian day 
			    (NOT since UT0:00) */

  hcbin_class();
  virtual ~hcbin_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
  
  ClassDef(hcbin_class,4)
};

#endif
//    Class for MD prfc_ bank.
//    Last modified: Oct 21, 2010
//    Dmitri Ivanov <ivanov@physics.rutgers.edu>
//

#ifndef _prfc_class_h_
#define _prfc_class_h_


using namespace std;


class prfc_class : public dstbank_class
{
  
public:

#define PRFC_MAXFIT 16
#define PRFC_MAXBIN 300
#define PRFC_MAXMEL 10   /* Maximum number of error matrix elements */
  /* There are several arrays ( fit[], bininfo[], nbin[] ) at the end 
     that the user must set correctly. I have put them at the end to 
     avoid fortran common alignment problems */

  /* Nonreduced chi2 value */
  vector <Double_t> chi2;

  /* Profile parameters from each eye. The dparam is the statistical
     error from the fit to the profile. The rparam and lparam are the
     results assuming the 1 sigma away geometry on either side of the
     best fit. tparam is a total geometrical error. This applies when
     the geometry fit is so well constrained that it is no longer
     reasonable to talk about "left" and "right" trajectories in the
     Rp, Psi chi^2 "trench" */

  vector <Double_t> szmx;   /* Shower size. Number of charged */
  vector <Double_t> dszmx;   /* particles at Shower Maximum */
  vector <Double_t> rszmx;   
  vector <Double_t> lszmx;   
  vector <Double_t> tszmx;   

  vector <Double_t> xm;   /* Xmax = Shower Maximum   g/cm^2 */
  vector <Double_t> dxm;
  vector <Double_t> rxm;
  vector <Double_t> lxm;
  vector <Double_t> txm;

  vector <Double_t> x0;   /* X0   = Shower initial point  g/cm^2 */
  vector <Double_t> dx0;
  vector <Double_t> rx0;
  vector <Double_t> lx0;
  vector <Double_t> tx0;

  vector <Double_t> lambda;   /* Elongation parameter  g/cm^2 */
  vector <Double_t> dlambda;
  vector <Double_t> rlambda;
  vector <Double_t> llambda;
  vector <Double_t> tlambda;

  vector <Double_t> eng;   /* Shower energy  EeV  */
  vector <Double_t> deng;
  vector <Double_t> reng;
  vector <Double_t> leng;
  vector <Double_t> teng;


  /* Information about the grammage at each bin along the shower 
     trajectory. Keep this information because it depends on the
     atmospheric model used. The BIN1 bank gives the unit vector 
     for the bin centers */

  vector <vector <Double_t> > dep;  /* slant grammage g/cm^2 */
  vector <vector <Double_t> > gm;  /* vertical grammage g/cm^2 */


  /* Computed light contributions from each source along the shower 
     trajectory. This is the unmormalized result in photoelectrons 
     received at the phototube. Actual result is obtained by normalizing
     by the shower size, szmx.  Note: The BIN1 bank gives the unit vectors 
     for the bin centers. */

  vector <vector <Double_t> > scin;   /* Scintillation */
  vector <vector <Double_t> > rayl;   /* Rayleigh Scattered */
  vector <vector <Double_t> > aero;   /* Aerosol Scattered */
  vector <vector <Double_t> > crnk;   /* Direct Cherenkov */
  vector <vector <Double_t> > sigmc;   /* Total MC signal */
  vector <vector <Double_t> > sig;   /* Signal measured */


  /* The idea here is since the error matrix is symmetric only 
     mor * (mor + 1) /2 elements need to be packed, where mor
     is the matrix order. Both nel and mor are packed as integer2 */

  vector <vector <Double_t> > mxel;   /* Error Matrix Elements */
  vector <Int_t> nel;                /* Number of Matrix Elements */
  vector <Int_t> mor;                /* Matix order */

  vector <vector <Int_t> > ig;   /* flag of the bin */



  /* 
     User must set these. Each fit:

     pflinfo[]  - should be set to PRFC_PFLINFO_USED or PRFC_PFLINFO_UNUSED
     depending on whether or not the fit contains valid profile 
     results. N.B. The fit contains still contains valid
     fit information when failmode!=SUCCESS, i.e. it contains
     information about how the fit failed.
        
     bininfo[]  - should be set to PRFC_BININFO_USED or PRFC_BININFO_UNUSED
     depending on whether or not the fit contains profile bin
     information, i.e. the light contributions in each bin.

     mtxinfo[]  - should be set to PRFC_MTXINFO_USED or PRFC_MTXINFO_UNUSED
     depending on whether or not the fit contains an error
     matrix.

     N.B. It is possible for a member of bininfo[] == PRFC_BININFO_UNUSED
     while the corresponding member of pflinfo[] == PRFC_PFLINFO_USED
     This may occur when different physics assumptions are being
     tested and one is only checking for differences in quantities
     like Xmax and Energy.


     failmode[] - Must be set if corresponding pflinfo[] == PRFC_PFLINFO_USED
     nbin[]     - Must be set if corresponding bininfo[] == PRFC_BININFO_USED
     This will normally be equal to the number of bins in 
     the source bank, i.e. BIN1 or presumably BIN2 when the 
     hires2 analysis gets that far. This value must not
     exceed PRFC_MAXBIN.
  */
 
  vector <Int_t> pflinfo;
  vector <Int_t> bininfo;
  vector <Int_t> mtxinfo;

  vector <Int_t> failmode;
  vector <Int_t> nbin; 


  /* Trajectory source. This should be filled with the bankid
     of the bank used as the trajectory source */

  vector <Int_t> traj_source;

  /* Status of errors. Usually this will be filled with SUCCESS,
     but if there is some problem computing the "left" or "right"
     trajectories, an errstat[] value could be fill with something
     like PRFC_LEFT_ERROR_FAILURE */

  vector <Int_t> errstat; 
  
  /* Number of degrees of freedom for the chi2 fit. Reduced chi2
     is chi2[]/ndf[] */

  vector <Int_t> ndf;

  prfc_class();
  virtual ~prfc_class();
  void loadFromDST();
  void loadToDST();
  void clearOutDST();

  ClassDef(prfc_class,5)
};

#endif
//    Root tree class for fdatmos_param_ dst bank
//    Dmitri Ivanov <ivanov@physics.rutgers.edu>
//    Last modified: Dec 17, 2010


#ifndef _fdatmos_param_class_
#define _fdatmos_param_class_


using namespace std;

#ifndef FDATMOS_PARAM_BANKID
#define FDATMOS_PARAM_MAXITEM 500
#endif


class fdatmos_param_class: public dstbank_class
{
  
public:
  
  Int_t uniqID;                     // uniq ID
  Int_t dateFrom;                   // available date from sec from 1970/1/1
  Int_t dateTo;                     // available date to sec from 1970/1/1
  Int_t nItem;                      // number of data line
  vector<Float_t> height;           // height [km]
  vector<Float_t> pressure;         // pressure [hPa] 
  vector<Float_t> pressureError;    // pressure error [hPa]
  vector<Float_t> temperature;      // temperature [degree]
  vector<Float_t> temperatureError; // temperature error [degree]
  vector<Float_t> dewPoint;         // dew point [degree]
  vector<Float_t> dewPointError;    // dew point error [degree]

  fdatmos_param_class();
  virtual ~fdatmos_param_class();
  virtual void loadFromDST();
  virtual void loadToDST();
  virtual void clearOutDST();
  
protected:
  
  // to used_bankid to tell the difference b/w fdatmos and gdas
  Int_t used_bankid;

  ClassDef(fdatmos_param_class,3)
};


#endif

/**
   Root tree class for gdas DST bank.
   Last modified: June 23, 2016
   Dmitri Ivanov <dmiivanov@gmail.com>
**/

#ifndef _gdas_class_h_
#define _gdas_class_h_

class gdas_class : public fdatmos_param_class
{
public:
  gdas_class();
  virtual ~gdas_class();
  // Overwriting  the virtual functions of fdatmos_param_class
  // if gdas file has not been included.
#ifndef GDAS_BANKID
  void loadFromDST();
  void loadToDST();
  void clearOutDST();
#endif
  ClassDef(gdas_class,2)
};

#endif
