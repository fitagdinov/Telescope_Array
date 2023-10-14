#include "dst2rt_sd.h"
#include "stdarg.h"
#include "sdrt_class.h"
#include "fdrt_class.h"
#include "TTree.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TMath.h"
#include "TProfile.h"
#include "TString.h"
#include "TMath.h"
#include "tacoortrans.h"
#include "tafd10info.h"
#include "sdparamborder.h"
#include <map>
#include "sduti.h"
#include "sdxyzclf_class.h"

static sdxyzclf_class dst2rt_sd_sdxyzclf;


using namespace TMath;



static void printWarn(const char *form, ...)
{
  char mess[0x400];
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fprintf(stderr, "warning: dst2rt_sd: %s\n",mess);
}

int main(int argc, char **argv)
{

  
  listOfOpt opt; // to handle the program arguments
  
  // parses cmd line and checks that it's possible to start the RZ file
  if (!opt.getFromCmdLine(argc, argv))
    return 2;
  opt.printOpts();

  ////////////// DST BRANCHES INITIALIZED AND ALLOCATED //////////

  // this macro initializes and allocates the detailed branches
#define dst2rt_sd_idb(branch_var)					\
  branch_var##_class *branch_var = new branch_var##_class
  
  dst2rt_sd_idb(atmpar);
  dst2rt_sd_idb(etrack);
  dst2rt_sd_idb(rusdraw);
  dst2rt_sd_idb(rufptn);
  dst2rt_sd_idb(rusdgeom);
  dst2rt_sd_idb(rufldf);
  dst2rt_sd_idb(tasdevent);
  dst2rt_sd_idb(bsdinfo);
  dst2rt_sd_idb(tasdcalibev);
  dst2rt_sd_idb(sdtrgbk);
  dst2rt_sd_idb(rusdmc);
  dst2rt_sd_idb(rusdmc1);
  dst2rt_sd_idb(brplane);
  dst2rt_sd_idb(lrplane);
  dst2rt_sd_idb(brprofile);
  dst2rt_sd_idb(lrprofile);
  dst2rt_sd_idb(hraw1);
  dst2rt_sd_idb(mcraw);
  dst2rt_sd_idb(mc04);
  dst2rt_sd_idb(stps2);
  dst2rt_sd_idb(stpln);
  dst2rt_sd_idb(hctim);
  dst2rt_sd_idb(hcbin);
  dst2rt_sd_idb(prfc);
  
#undef dst2rt_sd_idb
  
  ////////////// SIMPLIFIED VARIABLES DEFINED (BELOW) ////////////
  
  Int_t irec;
  Int_t ihit;

  Int_t towid;       // tower ID: 0=BR, 1=LR, 2=SK, 3=BRLR, 4=BRSK, 5=LRSK, 6=BRLRSK
  Int_t yymmdd;      // date
  Int_t hhmmss;      // time
  Int_t usec;        // trigger micro-second

  Int_t nsds;        // number of SDs in the readout
  Int_t nofwf;       // total number of waveforms in the readout

  // Thrown values

  Int_t parttype;   // Corsika particle type, proton=14, iron=5626
  Double_t height;  // height of the first interaction, cm
  
  Double_t mctheta;
  Double_t mcphi;
  Double_t mcxcore;
  Double_t mcycore;
  Double_t mcenergy;
  Int_t    mctc;     // Clock count courresponding to core hitting the ground, as in Ben's MC
  Double_t mct0;     // Time of the core hitting the ground, [1200m], with respect to the earliest SD hit time

  Double_t mcbdist;  // distances from the borders for thrown cores
  Double_t mctdist;

  // Reconstructed values
  Int_t    nstclust;       // number of counters in space-time cluster after geometry clean-up
  Int_t    nstclusts;      // number of counters in space-time cluster after a simple pattern recognition
  Double_t qtot;           // total charge in the event over good SDs in the timing fit, [VEM]
  Double_t qtots;          // total charge in the space-time cluster after a simple pattern recognition, [VEM]
  Int_t    srndedness;     // 0 if largest signal counter that's part of the event not surrounded by 4 working counters
  //                          1 if largest signal counter that's part of the event surrounded by 4 working counters
  //                          2 if largest signal counter that's part of the event surrounded by 4 working counters
  //                            and all 4 working counters are immediate neighbors of the largest signal counter 
  /*
    Theta, phi:
    [0] - plane fit
    [1] - Linsley fit
    [2] - Linsley with curvature and development
    [3] - LDF + Linsley
  */
  Double_t theta[4];
  Double_t dtheta[4];
  Double_t phi[4];
  Double_t dphi[4];
  Double_t pderr[4];
  /*
    Curvature parameter from Linsley with curv. fit
  */
  Double_t a;
  Double_t da;

  /*
    Core Position:
    [0] - plane fit
    [1] - Linsley fit
    [2] - Linsley with curvature and propagation
    [3] - LDF alone
    [4] - LDF + Linsley
  */
  Double_t xcore[5];
  Double_t dxcore[5];
  Double_t ycore[5];
  Double_t dycore[5];
  
  /* time of the core hit, with respect to time of the earliest hit, 
     [1200m] units
     [0] - plane fit
     [1] - Linsely fit
     [2] - Linsley with curvature and propagation
     [3] - LDF + Linsley 
  */
  Double_t t0[4]; 
  Double_t dt0[4];
  
  /* Core position from old tyro analysis */
  Double_t tyro_xcore;
  Double_t tyro_ycore;


  /*
    Scaling factor in fron of LDF:
    [0] - LDF alone fit
    [1] - LDF + Linsely fit
  */

  Double_t sc[2];
  Double_t dsc[2];

  /*
    s800:
    [0] - LDF alone fit
    [1] - LDF + Linsley fit
  */
  Double_t s800[2];

  /*
    energy using the old formula.
    [0] - LDF alone fit
    [1] - LDF + Linsley fit
  */

  Double_t energy[2];

  /* atmospheric correction factor that was applied, if any */
  Double_t atmcor[2];

  // Distance from the borders for reconstructed core
  Double_t bdist;
  Double_t tdist;

  /*
    "Geom. Fit chi2/dof: ( when ndof <=0, this is just chi2"
    [0] - plane fit
    [1] - Linsley fit
    [2] - Linsley + Curvature + development fit
  */
  Double_t gfchi2[3];
  /*
    "LDF fit chi2/dof ( when ndof <=0, this is just chi2"
    [0] - LDF alone fit
    [1] - LDF + Linsley fit
  */
  Double_t ldfchi2[2];

  // Some additional useful FD variables
  
  Char_t   ifdplane[2];   // [0] = 1: have brplane, [1] = 1: have lrplane
  Char_t   ifdprofile[2]; // [0] = 1: have brprofile, [1] = 1: have lrprofile
  Double_t fdtheta[2];
  Double_t fdphi[2];
  Double_t fdpsi[2];
  Double_t fdepsi[2];
  Double_t fdxcore[2];
  Double_t fdycore[2];
  Double_t fdtdist[2];
  Double_t fdbdist[2];
  Double_t fdenergy[2];
  Double_t fdxmax[2];
  Double_t fdnmax[2];
  
  // Simple Trigger backup variables
  Char_t igevent; // 0 - don't trigger event with lowered ped, 
                  // 1 - had to lower ped, 2 - triggers fine, 
                  // 3 - triggers with increased ped
  Char_t trigp;   // if triggers: 0 - line, 1 - triangle
                  // if doesn't trigger: 0 - no 3 SDs with good peds, 
                  // 1 - no 3 spatially connected SDs
                  // 2 - no 3 spatially and time connected SDs
                  // 3 - no 3 spatially and time connected SDs with level1 signals
  Int_t dec_ped;  // ammount by which had to decrease the pedestals
  
  Int_t inc_ped;  // ammount by which can increase the pedestals and still have the event triggered

  Int_t nbsds;   // Number of mi-scalibrated SDs (usually present in the data, MC usually treats them
  //                as completely out because simulating mis-calibrated SDs is difficult)
  Int_t nsdsout; // number of SDs that are completely out

  Double_t gsd_av_temp; // average temperature of good SDs, must have tasdcalibev bank

  // Some simplified tasdevent variables ( goes into pass1tree only )
  Int_t trigp_n;
  Int_t trigp_pos;
  Int_t trigp_usec[16];
  Int_t trigp_xxyy[16];
  

  // Anisotropy variables
  Double_t jday; // Full julian time in days
  Double_t ha;   // Hour Angle, Degree, (-180..180 range for convenience)
  Double_t lmst; // Local Mean Sidereal Time, degree
  Double_t ra;   // Right Ascension, degree
  Double_t dec;  // Declination, degree
  Double_t l;    // Galactic longitude, degree
  Double_t b;    // Galactic latitude, degree
  Double_t sgl;  // Supergalactic longitude, degree
  Double_t sgb;  // Supergalactic latitude, degree

  Int_t    atmpar_nh = 0; // number of heights that distinquish layers
  Double_t atmpar_h[ATMPAR_NHEIGHT]; // layer transition heights [cm]
  // parameters of the T(h) = a_i + b_i * exp(h/c_i) model determined by the fit
  Double_t atmpar_a[ATMPAR_NHEIGHT];
  Double_t atmpar_b[ATMPAR_NHEIGHT];
  Double_t atmpar_c[ATMPAR_NHEIGHT];
  Double_t atmpar_chi2; // quality of the fit
  Int_t    atmpar_ndof; // number of degees of freedom in the fit = number of points - number of fit parameters

  
  // Calculated numerically without doing fit to atmospheric model
  Double_t gdas_mo_18;    // mass overburden [g/cm2] at 18 km
  Double_t gdas_mo_15;    // at 15 km
  Double_t gdas_mo_12;
  Double_t gdas_mo_09;
  Double_t gdas_mo_06;
  Double_t gdas_mo_03;
  Double_t gdas_mo_01_4;  // at 1.4 km where TA SD is
  Double_t gdas_rho_01_4; // density [g/cm3] at 1.4km
  Double_t gdas_temp_01_4; // temperature [ k ] at 1.4 km
  
  ////////////// SIMPLIFIED VARIABLES DEFINED (ABOVE) ////////////
  

  // dummy variables
  int year,month,day;
  int hour,minute,second;
  double ani_phi, ani_theta;
  double clflat,clflon;
  double xclf[3];
  fdplane_dst_common    *fdplane_ptr   = 0;
  fdprofile_dst_common  *fdprofile_ptr = 0;

  clflat = tacoortrans_CLF_Latitude  * DegToRad();
  clflon = tacoortrans_CLF_Longitude * DegToRad();
  
  ////////  ROOT TREE WITH DETAILED INFORMATION INITIALIZED (BELOW) /////
  TFile* detailedOut = 0;
  TTree* pass1tree   = 0;
  if (opt.wt==0 || opt.wt ==2)
    {
      if (!opt.fOverwrite)
	{
	  FILE* fp = fopen(opt.dtof,"r");
	  if(fp)
	    {
	      fprintf(stderr,"error: %s exists; use '-f' option to overwrite files\n",opt.dtof);
	      fclose(fp);
	      return 2;
	    }
	}
      detailedOut = new TFile(opt.dtof,"recreate");
      pass1tree   = new TTree ("pass1tree", "Detailed Information");
      if (detailedOut->IsZombie()) 
	return 2;
    }
  ////////  ROOT TREE WITH DETAILED INFORMATION INITIALIZED (ABOVE) /////
  

  //////// ROOT TREE WITH SIMPLIFIED VARIABLES INITIALIZED (BELOW) //////
  TFile* simplifiedOut = 0;
  TTree* resTree       = 0;
  if (opt.wt==0 || opt.wt == 1)
    {
      if (!opt.fOverwrite)
	{
	  FILE* fp = fopen(opt.rtof,"r");
	  if(fp)
	    {
	      fprintf(stderr,"error: %s exists; use '-f' option to overwrite files\n",opt.rtof);
	      fclose(fp);
	      return 2;
	    }
	}
      simplifiedOut = new TFile(opt.rtof,"recreate");
      if (simplifiedOut->IsZombie()) 
	return 2;
      resTree = new TTree("resTree","SD variables");
    }
  //////// ROOT TREE WITH SIMPLIFIED VARIABLES INITIALIZED (ABOVE) //////

  
  //////////////// BRANCHES ADDED TO ROOT TREES (BELOW) ////////////////////
  
#define _dst2rt_sd_stringinize_(x) #x
  // this macro adds the detailed branch to the (detailed) root tree
#define dst2rt_sd_adb(branch_var)						\
  pass1tree->Branch(#branch_var,_dst2rt_sd_stringinize_(branch_var##_class),&branch_var,64000,0)
  
  // option to add information from atmpar dst bank
  if(opt.atmparopt)
    {
      // detailed tree
      if(pass1tree)
	dst2rt_sd_adb(atmpar);
      if(resTree)
	{
	  resTree->Branch   ( "atmpar_nh",     &atmpar_nh,       "atmpar_nh/I"                    );
	  resTree->Branch   ( "atmpar_h",      atmpar_h,         "atmpar_h[atmpar_nh]/D"          );
	  resTree->Branch   ( "atmpar_a",      atmpar_a,         "atmpar_a[atmpar_nh]/D"          );
	  resTree->Branch   ( "atmpar_b",      atmpar_b,         "atmpar_b[atmpar_nh]/D"          );
	  resTree->Branch   ( "atmpar_c",      atmpar_c,         "atmpar_c[atmpar_nh]/D"          );
	  resTree->Branch   ( "atmpar_chi2",   &atmpar_chi2,     "atmpar_chi2/D"                  );
	  resTree->Branch   ( "atmpar_ndof",   &atmpar_ndof,     "atmpar_ndof/I"                  );
	}
    }
  if(opt.gdasopt)
    {
      if(resTree)
	{
	  
	  resTree->Branch   ( "gdas_mo_18",      &gdas_mo_18,         "gdas_mo_18/D"              );
	  resTree->Branch   ( "gdas_mo_15",      &gdas_mo_15,         "gdas_mo_15/D"              );
	  resTree->Branch   ( "gdas_mo_12",      &gdas_mo_12,         "gdas_mo_12/D"              );
	  resTree->Branch   ( "gdas_mo_09",      &gdas_mo_09,         "gdas_mo_09/D"              );
	  resTree->Branch   ( "gdas_mo_06",      &gdas_mo_06,         "gdas_mo_06/D"              );
	  resTree->Branch   ( "gdas_mo_03",      &gdas_mo_03,         "gdas_mo_03/D"              );
	  resTree->Branch   ( "gdas_mo_01_4",    &gdas_mo_01_4,       "gdas_mo_01_4/D"            );
	  resTree->Branch   ( "gdas_rho_01_4",   &gdas_rho_01_4,      "gdas_rho_01_4/D"           );
	  resTree->Branch   ( "gdas_temp_01_4",  &gdas_temp_01_4,     "gdas_temp_01_4/D"          );
	  
	}
    }
  

  // option to write information from etrack dst bank
  if(opt.etrackopt)
    {
      // detailed tree
      if(pass1tree)
	dst2rt_sd_adb(etrack);
      
      // nothing added to simplified tree - etrack is simple enough already
    }
  
  // SD passes branches added
  if (opt.sdopt)
    {
      // detailed tree
      if(pass1tree)
	{
	  dst2rt_sd_adb(rusdraw);
	  dst2rt_sd_adb(rufptn);
	  dst2rt_sd_adb(rusdgeom);
	  dst2rt_sd_adb(rufldf);
	  pass1tree->Branch ( "towid",      &towid,       "towid/I"      );
	  pass1tree->Branch ( "yymmdd",     &yymmdd,      "yymmdd/I"     );
	  pass1tree->Branch ( "hhmmss",     &hhmmss,      "hhmmss/I"     );
	  pass1tree->Branch ( "usec",       &usec,        "usec/I"       );
	  pass1tree->Branch ( "nsds",       &nsds,        "nsds/I"       );
	  pass1tree->Branch ( "nofwf",      &nofwf,       "nofwf/I"      );
	  pass1tree->Branch ( "nstclust",   &nstclust,    "nstclust/I"   );
	  pass1tree->Branch ( "nstclusts",  &nstclusts,   "nstclusts/I"  );
	  pass1tree->Branch ( "qtot",       &qtot,        "qtot/D"       );
	  pass1tree->Branch ( "qtots",      &qtots,       "qtots/D"      );
	  pass1tree->Branch ( "srndedness", &srndedness,  "srndedness/I" );
	  pass1tree->Branch ( "theta",      theta,        "theta[4]/D"   );
	  pass1tree->Branch ( "dtheta",     dtheta,       "dtheta[4]/D"  );
	  pass1tree->Branch ( "phi",        phi,          "phi[4]/D"     );
	  pass1tree->Branch ( "dphi",       dphi,         "dphi[4]/D"    );
	  pass1tree->Branch ( "pderr",      pderr,        "pderr[4]/D"   );
	  pass1tree->Branch ( "a",          &a,           "a/D"          );
	  pass1tree->Branch ( "da",         &da,          "da/D"         );
	  pass1tree->Branch ( "t0",         t0,           "t0[4]/D"      );
	  pass1tree->Branch ( "dt0",        dt0,          "dt0[4]/D"     );
	  pass1tree->Branch ( "tyro_xcore", &tyro_xcore,  "tyro_xcore/D" );
	  pass1tree->Branch ( "tyro_ycore", &tyro_ycore,  "tyro_ycore/D" );
	  pass1tree->Branch ( "xcore",      xcore,        "xcore[5]/D"   );
	  pass1tree->Branch ( "dxcore",     dxcore,       "dxcore[5]/D"  );
	  pass1tree->Branch ( "ycore",      ycore,        "ycore[5]/D"   );
	  pass1tree->Branch ( "dycore",     dycore,       "dycore[5]/D"  );
	  pass1tree->Branch ( "sc",         sc,           "sc[2]/D"      );
	  pass1tree->Branch ( "dsc",        dsc,          "dsc[2]/D"     );
	  pass1tree->Branch ( "s800",       s800,         "s800[2]/D"    );
	  pass1tree->Branch ( "energy",     energy,       "energy[2]/D"  );
	  pass1tree->Branch ( "atmcor",     atmcor,       "atmcor[2]/D"  );
	  pass1tree->Branch ( "bdist",      &bdist,       "bdist/D"      );
	  pass1tree->Branch ( "tdist",      &tdist,       "tdist/D"      );
	  pass1tree->Branch ( "gfchi2",     gfchi2,       "gfchi2[3]/D"  );
	  pass1tree->Branch ( "ldfchi2",    ldfchi2,      "ldfchi2[2]/D" );
	}
      // simplified tree
      if(resTree)
	{
	  resTree->Branch   ( "towid",      &towid,       "towid/I"      );
	  resTree->Branch   ( "yymmdd",     &yymmdd,      "yymmdd/I"     );
	  resTree->Branch   ( "hhmmss",     &hhmmss,      "hhmmss/I"     );
	  resTree->Branch   ( "usec",       &usec,        "usec/I"       );
	  resTree->Branch   ( "nsds",       &nsds,        "nsds/I"       );
	  resTree->Branch   ( "nofwf",      &nofwf,       "nofwf/I"      );     
	  resTree->Branch   ( "nstclust",   &nstclust,    "nstclust/I"   );
	  resTree->Branch   ( "nstclusts",  &nstclusts,   "nstclusts/I"  );
	  resTree->Branch   ( "qtot",       &qtot,        "qtot/D"       );
	  resTree->Branch   ( "qtots",      &qtots,       "qtots/D"      );
	  resTree->Branch   ( "srndedness", &srndedness,  "srndedness/I" );
	  resTree->Branch   ( "theta",      theta,        "theta[4]/D"   );
	  resTree->Branch   ( "dtheta",     dtheta,       "dtheta[4]/D"  );
	  resTree->Branch   ( "phi",        phi,          "phi[4]/D"     );
	  resTree->Branch   ( "dphi",       dphi,         "dphi[4]/D"    );
	  resTree->Branch   ( "pderr",      pderr,        "pderr[4]/D"   );
	  resTree->Branch   ( "a",          &a,           "a/D"          );
	  resTree->Branch   ( "da",         &da,          "da/D"         );
	  resTree->Branch   ( "t0",         t0,           "t0[4]/D"      );
	  resTree->Branch   ( "dt0",        dt0,          "dt0[4]/D"     );
	  resTree->Branch   ( "tyro_xcore", &tyro_xcore,  "tyro_xcore/D" );
	  resTree->Branch   ( "tyro_ycore", &tyro_ycore,  "tyro_ycore/D" );
	  resTree->Branch   ( "xcore",      xcore,        "xcore[5]/D"   );
	  resTree->Branch   ( "dxcore",     dxcore,       "dxcore[5]/D"  );
	  resTree->Branch   ( "ycore",      ycore,        "ycore[5]/D"   );
	  resTree->Branch   ( "dycore",     dycore,       "dycore[5]/D"  );
	  resTree->Branch   ( "sc",         sc,           "sc[2]/D"      );
	  resTree->Branch   ( "dsc",        dsc,          "dsc[2]/D"     );
	  resTree->Branch   ( "s800",       s800,         "s800[2]/D"    );
	  resTree->Branch   ( "energy",     energy,       "energy[2]/D"  );
	  resTree->Branch   ( "atmcor",     atmcor,       "atmcor[2]/D"  );
	  resTree->Branch   ( "bdist",      &bdist,       "bdist/D"      );
	  resTree->Branch   ( "tdist",      &tdist,       "tdist/D"      );
	  resTree->Branch   ( "gfchi2",     gfchi2,       "gfchi2[3]/D"  );
	  resTree->Branch   ( "ldfchi2",    ldfchi2,      "ldfchi2[2]/D" );
	  resTree->Branch   ( "jday",       &jday,        "jday/D"       );
	  resTree->Branch   ( "ha",         &ha,          "ha/D"         );
	  resTree->Branch   ( "lmst",       &lmst,        "lmst/D"       );
	  resTree->Branch   ( "ra",         &ra,          "ra/D"         );
	  resTree->Branch   ( "dec",        &dec,         "dec/D"        );
	  resTree->Branch   ( "l",          &l,           "l/D"          );
	  resTree->Branch   ( "b",          &b,           "b/D"          );
	  resTree->Branch   ( "sgl",        &sgl,         "sgl/D"        );
	  resTree->Branch   ( "sgb",        &sgb,         "sgb/D"        );
	}
    }
  
  // tasdevent (ICRR) branch added
  if (opt.tasdevent)
    {
      if(pass1tree)
	{
	  dst2rt_sd_adb(tasdevent);
	  pass1tree->Branch(  "trigp_n",    &trigp_n,      "trigp_n/I"         );
	  pass1tree->Branch(  "trigp_pos",  &trigp_pos,    "trigp_pos/I"       );
	  pass1tree->Branch(  "trigp_xxyy", trigp_xxyy,    "trigp_xxyy[16]/I"  );
	  pass1tree->Branch(  "trigp_usec", trigp_usec,    "trigp_usec[16]/I"  );
	}
    }
  
  // bsdinfo branch added
  if (opt.bsdinfo)
    {
      if(pass1tree)
	{
	  dst2rt_sd_adb(bsdinfo);
	  pass1tree->Branch(  "nbsds",      &nbsds,         "nbsds/I"          );
	  pass1tree->Branch(  "nsdsout",    &nsdsout,       "nsdsout/I"        );
	}
      if(resTree)
	{
	  resTree->Branch   ( "nbsds",      &nbsds,         "nbsds/I"          );
	  resTree->Branch   ( "nsdsout",    &nsdsout,       "nsdsout/I"        );
	}
    }

  // tasdcalibev (ICRR) branch added
  if (opt.tasdcalibev)
    {
      if(pass1tree)
	dst2rt_sd_adb(tasdcalibev);
      if(resTree)
	{
	  resTree->Branch   ( "gsd_av_temp",    &gsd_av_temp,      "gsd_av_temp/D"   );	  
	}
    }
  // Trigger backup branches added
  if (opt.tbopt)
    {
      // detailed tree
      if(pass1tree)
	{
	  dst2rt_sd_adb(sdtrgbk);
	  pass1tree->Branch ( "igevent",    &igevent,      "igevent/B"   );
	  pass1tree->Branch ( "trigp",      &trigp,        "trigp/B"     );
	  pass1tree->Branch ( "dec_ped",    &dec_ped,      "dec_ped/I"   );
	  pass1tree->Branch ( "inc_ped",    &inc_ped,      "inc_ped/I"   );
	}
      // simplified tree
      if(resTree)
	{
	  resTree->Branch   ( "igevent",    &igevent,      "igevent/B"   );
	  resTree->Branch   ( "trigp",      &trigp,        "trigp/B"     );
	  resTree->Branch   ( "dec_ped",    &dec_ped,      "dec_ped/I"   );
	  resTree->Branch   ( "inc_ped",    &inc_ped,      "inc_ped/I"   );
	}
    }

  // SD MC branches added
  if (opt.mcopt && opt.sdopt)
    {
      // detailed tree
      if(pass1tree)
	{
	  pass1tree->Branch( "rusdmc",       "rusdmc_class",      &rusdmc,      64000,  0  );
	  pass1tree->Branch( "rusdmc1",      "rusdmc1_class",     &rusdmc1,     64000,  0  );
	  pass1tree->Branch ( "parttype",   &parttype,    "parttype/I"   );
	  pass1tree->Branch ( "height",     &height,      "height/D"     );
	  pass1tree->Branch ( "mctheta",    &mctheta,     "mctheta/D"    );
	  pass1tree->Branch ( "mcphi",      &mcphi,       "mcphi/D"      );
	  pass1tree->Branch ( "mcxcore",    &mcxcore,     "mcxcore/D"    );
	  pass1tree->Branch ( "mcycore",    &mcycore,     "mcycore/D"    );
	  pass1tree->Branch ( "mcenergy",   &mcenergy,    "mcenergy/D"   );
	  pass1tree->Branch ( "mctc",       &mctc,        "mctc/I"       );
	  pass1tree->Branch ( "mct0",       &mct0,        "mct0/D"       );
	  pass1tree->Branch ( "mcbdist",    &mcbdist,     "mcbdist/D"    );
	  pass1tree->Branch ( "mctdist",    &mctdist,     "mctdist/D"    );
	}
      // simplified tree
      if(resTree)
	{
	  resTree->Branch   ( "parttype",   &parttype,    "parttype/I"   );
	  resTree->Branch   ( "height",     &height,      "height/D"     );
	  resTree->Branch   ( "mctheta",    &mctheta,     "mctheta/D"    );
	  resTree->Branch   ( "mcphi",      &mcphi,       "mcphi/D"      );
	  resTree->Branch   ( "mcxcore",    &mcxcore,     "mcxcore/D"    );
	  resTree->Branch   ( "mcycore",    &mcycore,     "mcycore/D"    );
	  resTree->Branch   ( "mcenergy",   &mcenergy,    "mcenergy/D"   );
	  resTree->Branch   ( "mctc",       &mctc,        "mctc/I"       );
	  resTree->Branch   ( "mct0",       &mct0,        "mct0/D"       );
	  resTree->Branch   ( "mcbdist",    &mcbdist,     "mcbdist/D"    );
	  resTree->Branch   ( "mctdist",    &mctdist,     "mctdist/D"    );
	}
    }
  // MD MC branches added
  if (opt.mcopt && opt.mdopt)
    {
      // detailed tree
      if(pass1tree)
	dst2rt_sd_adb(mc04);
    }

  // General MD branches added
  if (opt.mdopt)
    {
      // detailed tree
      if(pass1tree)
	{
	  dst2rt_sd_adb(hraw1);
	  dst2rt_sd_adb(mcraw);
	  dst2rt_sd_adb(stps2);
	  dst2rt_sd_adb(stpln);
	  dst2rt_sd_adb(hctim);
	  dst2rt_sd_adb(hcbin);
	  dst2rt_sd_adb(prfc);
	}
    }
  
  // BR,LR fdplane mono branches added
  if (opt.fdplane_opt)
    {
      // detailed tree
      if(pass1tree)
	{
	  dst2rt_sd_adb(brplane);
	  dst2rt_sd_adb(lrplane);
	  pass1tree->Branch ( "ifdplane",   ifdplane,    "ifdplane[2]/B"   );
	  pass1tree->Branch ( "fdtheta",    fdtheta,     "fdtheta[2]/D"    );
	  pass1tree->Branch ( "fdphi",      fdphi,       "fdphi[2]/D"      );
	  pass1tree->Branch ( "fdpsi",      fdpsi,       "fdpsi[2]/D"      );
	  pass1tree->Branch ( "fedpsi",     fdepsi,      "fdepsi[2]/D"     );
	  pass1tree->Branch ( "fdxcore",    fdxcore,     "fdxcore[2]/D"    );
	  pass1tree->Branch ( "fdycore",    fdycore,     "fdycore[2]/D"    );
	  pass1tree->Branch ( "fdbdist",    fdbdist,     "fdbdist[2]/D"    );
	  pass1tree->Branch ( "fdtdist",    fdtdist,     "fdtdist[2]/D"    );
	}
      // simplified tree
      if(resTree)
	{
	  resTree->Branch   ( "ifdplane",   ifdplane,   "ifdplane[2]/B"    );
	  resTree->Branch   ( "fdtheta",    fdtheta,    "fdtheta[2]/D"     );
	  resTree->Branch   ( "fdphi",      fdphi,      "fdphi[2]/D"       );
	  resTree->Branch   ( "fdpsi",      fdpsi,      "fdpsi[2]/D"       );
	  resTree->Branch   ( "fedpsi",     fdepsi,     "fdepsi[2]/D"      );
	  resTree->Branch   ( "fdxcore",    fdxcore,    "fdxcore[2]/D"     );
	  resTree->Branch   ( "fdycore",    fdycore,    "fdycore[2]/D"     );
	  resTree->Branch   ( "fdbdist",    fdbdist,    "fdbdist[2]/D"     );
	  resTree->Branch   ( "fdtdist",    fdtdist,    "fdtdist[2]/D"     );
	}
    }
  // BR,LR fdprofile branches added
  if (opt.fdprofile_opt)
    {
      // detailed tree
      if(pass1tree)
	{
	  dst2rt_sd_adb(brprofile);
	  dst2rt_sd_adb(lrprofile);
	  pass1tree->Branch ( "ifdprofile", ifdprofile,  "ifdprofile[2]/B" );
	  pass1tree->Branch ( "fdenergy",   fdenergy,    "fdenergy[2]/D"   );
	  pass1tree->Branch ( "fdxmax",     fdxmax,      "fdxmax[2]/D"     );
	  pass1tree->Branch ( "fdnmax",     fdnmax,      "fdnmax[2]/D"     );
	}
      // simplified tree
      if(resTree)
	{
	  resTree->Branch   ( "ifdprofile", ifdprofile,  "ifdprofile[2]/B" );
	  resTree->Branch   ( "fdenergy",   fdenergy,    "fdenergy[2]/D"   );
	  resTree->Branch   ( "fdxmax",     fdxmax,      "fdxmax[2]/D"     );
	  resTree->Branch   ( "fdnmax",     fdnmax,      "fdnmax[2]/D"     );
	}
    }
  
#undef _dst2rt_sd_stringinize_
#undef dst2rt_sd_adb
  
  //////////////// BRANCHES ADDED TO ROOT TREES (ABOVE) ////////////////////
  
  
  ///////////////////// READING THE DST FILES (BELOW) /////////////////////
  
  char *dstfile;
  sddstio_class *dstio = new sddstio_class;   // DST I/O handler
  int ievent = 0;

  // This macro loads the corresponding (detailed) root tree branches from the DST
#define dst2rt_sd_ldb(branch_var,bank_id)					\
  if(pass1tree)								\
    {									\
      if (dstio->haveBank(bank_id))					\
	branch_var->loadFromDST();					\
      else								\
	{								\
	  branch_var->clearOutDST();					\
	  if (opt.verbose)							\
	    printWarn("event %d from '%s' doesn't have %s",ievent,dstfile,#branch_var); \
	}								\
    }
  
  while ((dstfile=pullFile()))
    {
      ievent = 0;
      
      // skip bad files
      if (!dstio->openDSTinFile(dstfile))
	continue;
      
      while (dstio->readEvent())
	{
	  
	  ////////////// BRANCHES ASSIGNED (BELOW) ////////////
	  
	  // atmospheric parameter option
	  if(opt.atmparopt)
	    {
	      dst2rt_sd_ldb(atmpar,ATMPAR_BANKID);
	      if(resTree)
		{
		  atmpar_nh  = atmpar_.nh;
		  memcpy(atmpar_h,atmpar_.h,atmpar_nh*sizeof(Double_t));
		  memcpy(atmpar_a,atmpar_.a,atmpar_nh*sizeof(Double_t));
		  memcpy(atmpar_b,atmpar_.b,atmpar_nh*sizeof(Double_t));
		  memcpy(atmpar_c,atmpar_.c,atmpar_nh*sizeof(Double_t));
		  atmpar_chi2 = atmpar_.chi2;
		  atmpar_ndof = atmpar_.ndof;
		}
	    }
	  if(opt.gdasopt)
	    {
	      if(gdas_.nItem > 0)
		{
		  gdas_mo_18     = SDGEN::get_gdas_mo_numerically(1.8e6);
		  gdas_mo_15     = SDGEN::get_gdas_mo_numerically(1.5e6);
		  gdas_mo_12     = SDGEN::get_gdas_mo_numerically(1.2e6);
		  gdas_mo_09     = SDGEN::get_gdas_mo_numerically(9.0e5);
		  gdas_mo_06     = SDGEN::get_gdas_mo_numerically(6.0e5);
		  gdas_mo_03     = SDGEN::get_gdas_mo_numerically(3.0e5);
		  gdas_mo_01_4   = SDGEN::get_gdas_mo_numerically(1.4e5);
		  gdas_rho_01_4  = SDGEN::get_gdas_rho_numerically(1.4e5);
		  gdas_temp_01_4 = SDGEN::get_gdas_temp(1.4e5);
		}
	      else
		{
		  gdas_mo_18     = 0;
		  gdas_mo_15     = 0;
		  gdas_mo_12     = 0;
		  gdas_mo_09     = 0;
		  gdas_mo_06     = 0;
		  gdas_mo_03     = 0;
		  gdas_mo_01_4   = 0;
		  gdas_rho_01_4  = 0;
		  gdas_temp_01_4 = 0;
		}
	      
	    }
	  // event track branch
	  if (opt.etrackopt)
	    dst2rt_sd_ldb(etrack,ETRACK_BANKID);
	  
	  // SD MC Branches
	  if (opt.mcopt && opt.sdopt)
	    {    
	      dst2rt_sd_ldb(rusdmc,RUSDMC_BANKID);
	      dst2rt_sd_ldb(rusdmc1,RUSDMC1_BANKID);
	      
	      parttype = rusdmc_.parttype;
	      height   = rusdmc_.height;
	      mctheta  = RadToDeg() * (rusdmc_.theta);
	      mcphi    = RadToDeg() * (rusdmc_.phi);
	      mcenergy = rusdmc_.energy;
	      mctc     = rusdmc_.tc;
	      mct0     = rusdmc1_.t0;
	      mcxcore  = rusdmc1_.xcore;
	      mcycore  = rusdmc1_.ycore;
	      mcbdist  = rusdmc1_.bdist;
	      mctdist  = rusdmc1_.tdist;
	    }
	  
	  // MD MC option
	  if (opt.mdopt && opt.mcopt)
	    {
	      dst2rt_sd_ldb(mc04,MC04_BANKID);
	    }
	  
	  if (opt.mdopt)
	    {	      
	      dst2rt_sd_ldb(hraw1,HRAW1_BANKID);
	      dst2rt_sd_ldb(mcraw,MCRAW_BANKID);
	      dst2rt_sd_ldb(stps2,STPS2_BANKID);
	      dst2rt_sd_ldb(stpln,STPLN_BANKID);
	      dst2rt_sd_ldb(hctim,HCTIM_BANKID);
	      dst2rt_sd_ldb(hcbin,HCBIN_BANKID);
	      dst2rt_sd_ldb(prfc,PRFC_BANKID);
	    }
	  
	  // trigger backup branches
	  if (opt.tbopt)
	    {
	      dst2rt_sd_ldb(sdtrgbk,SDTRGBK_BANKID);
	      igevent = sdtrgbk_.igevent;
	      trigp   = sdtrgbk_.trigp;
	      dec_ped = sdtrgbk_.dec_ped;
	      inc_ped = sdtrgbk_.inc_ped;
	    }
	  if(opt.tasdcalibev)
	    {
	      std::map<Int_t,Int_t> xxyy_to_tasdcalibev_index;
	      for (Int_t i=0; i<tasdcalibev->numTrgwf; i++)
		xxyy_to_tasdcalibev_index[tasdcalibev_.sub[i].lid] = i;
	      gsd_av_temp = 0.0;
	      Int_t ngsds = 0;
	      for (Int_t i=0; i<rusdgeom_.nsds; i++)
		{
		  if(rusdgeom_.igsd[i]>=2)
		    {
		      gsd_av_temp += tasdcalibev_.sub[xxyy_to_tasdcalibev_index[rusdgeom_.xxyy[i]]].scintiTemp;
		      ngsds ++;
		    }
		}
	      if(ngsds > 0)
		gsd_av_temp /= (Double_t)ngsds;
	    }
	  
	  // SD reconstruction branches
	  if (opt.sdopt)
	    {
	      dst2rt_sd_ldb(rusdraw,RUSDRAW_BANKID);
	      dst2rt_sd_ldb(rufptn,RUFPTN_BANKID);
	      dst2rt_sd_ldb(rusdgeom,RUSDGEOM_BANKID);
	      dst2rt_sd_ldb(rufldf,RUFLDF_BANKID);
	     
	      // house keeping variables
	      towid    =  rusdraw_.site;
	      yymmdd   =  rusdraw_.yymmdd;
	      hhmmss   =  rusdraw_.hhmmss;
	      usec     =  rusdraw_.usec;
	      nsds     =  rusdgeom_.nsds;
	      nofwf    =  rusdraw_.nofwf;
	      
	      // reconstructed variables
	      nstclust  = 0;
	      nstclusts = 0;
	      qtot  = 0.0;
	      qtots = 0.0;
	      for (ihit=0; ihit < rufptn_.nhits; ihit++)
		{
		  if (rufptn_.isgood[ihit] >= 4)
		    {
		      nstclust++;
		      qtot += 0.5 * (rufptn_.pulsa[ihit][0]+rufptn_.pulsa[ihit][1]);
		    }
		  if (rufptn_.isgood[ihit] >= 3)
		    {
		      nstclusts++;
		      qtots += 0.5 * (rufptn_.pulsa[ihit][0]+rufptn_.pulsa[ihit][1]);
		    }
		}

	      // check if the largest charge SD is surrounded by working SDs 
	      if(dstio->haveBank(BSDINFO_BANKID))
		srndedness = dst2rt_sd_sdxyzclf.get_event_surroundedness(&rusdraw_,&rusdgeom_,&bsdinfo_);
	      else
		srndedness = -1;
	      // 3 geometry reconstructions of theta,phi
	      for (irec=0; irec < 3; irec++)
		{
		  theta[irec]    =  rusdgeom_.theta[irec];
		  dtheta[irec]   =  rusdgeom_.dtheta[irec];
		  phi[irec]      =  rusdgeom_.phi[irec];
		  dphi[irec]     =  rusdgeom_.dphi[irec];
		  pderr[irec]    = sqrt(sin(DegToRad()*theta[irec])*sin(DegToRad()*theta[irec])*
					dphi[irec]*dphi[irec] + dtheta[irec]*dtheta[irec]);
		}
	      
	      // 1 geom+ldf reconstruction of theta,phi
	      irec=3;
	      theta[irec] = rufldf_.theta;
	      dtheta[irec] = rufldf_.dtheta;
	      phi  [irec] = rufldf_.phi;
	      dphi [irec] = rufldf_.dphi;
	      pderr[irec]    = sqrt(sin(DegToRad()*theta[irec])*sin(DegToRad()*theta[irec])*
				    dphi[irec]*dphi[irec] + dtheta[irec]*dtheta[irec]);
	      
	      // Curvature parameter from Linsley+curv.+development fit
	      a  = rusdgeom_.a;
	      da = rusdgeom_.da;
	      
	      // Tyro core position
	      tyro_xcore = rufptn_.tyro_xymoments[2][0];
	      tyro_ycore = rufptn_.tyro_xymoments[2][1];
	      
	      // 3 geom. reconstructions of core
	      for (irec=0; irec < 3; irec++)
		{
		  xcore[irec]    = rusdgeom_.xcore[irec];
		  dxcore[irec]   = rusdgeom_.dxcore[irec];
		  ycore[irec]    = rusdgeom_.ycore[irec];
		  dycore[irec]   = rusdgeom_.dycore[irec];
		}
	      // 2 ldf reconstructions of core
	      for (irec=3; irec<5; irec++)
		{
		  xcore[irec]    = rufldf_.xcore[irec-3];
		  dxcore[irec]   = rufldf_.dxcore[irec-3];
		  ycore[irec]    = rufldf_.ycore[irec-3];
		  dycore[irec]   = rufldf_.dycore[irec-3];
		}
	      // 3 geom. reconstructions of t0
	      for (irec=0; irec<3; irec++)
		{
		  t0[irec]       = rusdgeom_.t0[irec];
		  dt0[irec]      = rusdgeom_.dt0[irec];
		}
	      // 1 LDF reconstruction of t0
	      t0[3]              = rufldf_.t0;
	      dt0[3]             = rufldf_.dt0;
	      
	      // 2 ldf reconstructions of energy and s800, and LDF scaling factor
	      for (irec=0; irec < 2; irec++)
		{
		  sc[irec]       = rufldf_.sc[irec];
		  dsc[irec]      = rufldf_.dsc[irec];
		  s800[irec]     = rufldf_.s800[irec];
		  energy[irec]   = rufldf_.energy[irec];
#if RUFLDF_BANKVERSION >= 1
		  atmcor[irec]   = rufldf_.atmcor[irec];
#else
		  atmcor[irec]   = 1.0;
#endif
		}
	      
	      bdist    = rufldf_.bdist;
	      tdist    = rufldf_.tdist;
	      
	      for (irec=0; irec<3; irec++)
		gfchi2[irec]   = (rusdgeom_.ndof[irec] < 1 ? rusdgeom_.chi2[irec] :
				  (rusdgeom_.chi2[irec]/(double)rusdgeom_.ndof[irec]));
	      
	      for (irec=0; irec < 2; irec++)
		ldfchi2[irec]  = (rufldf_.ndof[irec] < 1 ? rufldf_.chi2[irec] :
				  (rufldf_.chi2[irec]/(double)rufldf_.ndof[irec]));
	      
	      // anisotropy
	      year   = 2000 + (yymmdd / 10000);
	      month  = (yymmdd % 10000) / 100;
	      day    = yymmdd % 100;
	      hour   = hhmmss / 10000;
	      minute = (hhmmss %10000) / 100;
	      second = hhmmss % 100;
	      
	      
	      // Event angles in the local sky that are pointing back to the source
	      ani_theta = (theta[2]+0.5) * DegToRad();
	      ani_phi   = (phi[2]+180.0) * DegToRad();
	      
	      // Full Julian time in days
	      jday = tacoortrans::utc_to_jday(year,month,day,(double)(3600*hour+60*minute+second));
	      
	      // Hour Angle, in radians
	      ha   = tacoortrans::get_ha(ani_theta,ani_phi,clflat);
	      
	      // Local Mean Sidereal Time, in radians
	      lmst = tacoortrans::jday_to_LMST(jday,clflon);
	      
	      // Right Ascension, in radians
	      ra = lmst - ha;
	      while(ra<0.0) 
		ra += TwoPi();
	      while(ra>=TwoPi()) 
		ra -= TwoPi();
	      
	      // Declination, in radians
	      dec   = tacoortrans::get_dec(ani_theta,ani_phi,clflat);
	      
	      // Galactic coordinates, radians
	      l  = tacoortrans::gall(ra,dec);
	      b  = tacoortrans::galb(ra,dec);
	      
	      // Supergalactic coordinates, radians
	      sgl = tacoortrans::sgall(ra,dec);
	      sgb = tacoortrans::sgalb(ra,dec);

	      // Convert the anisotropy angles into degrees
	      ha   *= RadToDeg();
	      lmst *= RadToDeg();
	      ra   *= RadToDeg();
	      dec  *= RadToDeg();
	      l    *= RadToDeg();
	      b    *= RadToDeg();
	      sgl  *= RadToDeg();
	      sgb  *= RadToDeg();
	      
	    }
	  
	  // tasdevent branch
	  if (opt.tasdevent)
	    {
	      if(dstio->haveBank(TASDEVENT_BANKID))
		{
		  trigp_n = 0;
		  trigp_pos = tasdevent->pos2xxyy();
		  for (int ipat = 0; ipat<16; ipat++)
		    {
		      if (tasdevent_.pattern[ipat] != -1)
			{
			  tasdevent->itrigp2xxyyt(ipat,&trigp_xxyy[trigp_n],&trigp_usec[trigp_n]);
			  trigp_n ++;
			}
		    }
		  tasdevent->loadFromDST();
		}
	      else
		{
		  tasdevent->clearOutDST();
		  trigp_n = 0;
		  trigp_pos = 0;
		  if (opt.verbose)
		    printWarn("-tasdevent used but tasdevent not found in '%s'",dstfile);
		}

	    }

	  // bsdinfo branch
	  if (opt.bsdinfo)
	    {
	      if(dstio->haveBank(BSDINFO_BANKID))
		{
		  bsdinfo->loadFromDST();
		  nbsds = bsdinfo->nbsds;
		  nsdsout = bsdinfo->nsdsout;
		}
	      else
		{
		  bsdinfo->clearOutDST();
		  nbsds = 0;
		  nsdsout = 0;
		  if (opt.verbose)
		    printWarn("-bsdinfo used but bsdinfo not found in '%s'",dstfile);
		}
	      
	    }

	  // tasdcalibev branch
	  if (opt.tasdcalibev)
	    dst2rt_sd_ldb(tasdcalibev,TASDCALIBEV_BANKID);
	  
	  
	  // fdplane branch
	  if (opt.fdplane_opt)
	    {
	      
	      for (irec=0; irec < 2; irec++)
		ifdplane[irec] = 0;
	      
	      if (dstio->haveBank(BRPLANE_BANKID))
		ifdplane[0] = 1;	      
	      if (dstio->haveBank(LRPLANE_BANKID))
		ifdplane[1] = 1;
	      
	      if(pass1tree)
		{
		  if(ifdplane[0])
		    brplane->loadFromDST();
		  else
		    brplane->clearOutDST();
		  
		  if(ifdplane[1])
		    lrplane->loadFromDST();
		  else
		    lrplane->clearOutDST();
		}
	      
	      if(!ifdplane[0] && !ifdplane[1])
		{
		  if(opt.verbose)
		    printWarn("event %d from '%s' doesn't have {br,lr}plane",ievent,dstfile);
		}
	      
	      for (irec=0; irec<2; irec++)
		{
		  if (!ifdplane[irec])
		    continue;
		  
		  if(irec==0)
		    fdplane_ptr = &brplane_;
		  if (irec==1)
		    fdplane_ptr = &lrplane_;
		  
		  tafd10info::get_brlr_time(fdplane_ptr->julian,fdplane_ptr->jsecond,&yymmdd,&hhmmss);
		  usec = (int)Floor(((double)fdplane_ptr->jsecfrac)/1e3 - 25.6 + 0.5);
		  
		  fdtheta[irec] = fdplane_ptr->shower_zen * RadToDeg();
		  fdphi[irec]   = fdplane_ptr->shower_azm * RadToDeg();
		  while(fdphi[irec] >= 360.0)
		    fdphi[irec] -= 360.0;
		  while(fdphi[irec] < 0.0)
		    fdphi[irec] += 360.0;
		  fdpsi[irec]    = fdplane_ptr->psi * DegToRad();
		  fdepsi[irec]   = fdplane_ptr->epsi * DegToRad();
		  tacoortrans::fdsite2clf(fdplane_ptr->siteid,fdplane_ptr->core,xclf);
		  fdxcore[irec]   = xclf[0]/1200.0 - RUSDGEOM_ORIGIN_X_CLF;
		  fdycore[irec]   = xclf[1]/1200.0 - RUSDGEOM_ORIGIN_Y_CLF;	      
		  comp_boundary_dist(fdxcore[irec],fdycore[irec],&fdbdist[irec],&fdtdist[irec]);
		}
	    }
	  
	  // fdprofile branch
	  if (opt.fdprofile_opt)
	    {
	      
	      for (irec=0; irec < 2; irec++)
		ifdprofile[irec] = 0;
	      
	      if (dstio->haveBank(BRPROFILE_BANKID))
		ifdprofile[0] = 1;
	      if (dstio->haveBank(LRPROFILE_BANKID))
		ifdprofile[1] = 1;
	      
	      if(pass1tree)
		{
		  if(ifdprofile[0])
		    brprofile->loadFromDST();
		  else
		    brprofile->clearOutDST();
		  
		  if(ifdprofile[1])
		    lrprofile->loadFromDST();
		  else
		    lrprofile->clearOutDST();
		}
	      
	      if(!ifdprofile[0] && !ifdprofile[1])
		{
		  if(opt.verbose)
		    printWarn("event %d from '%s' doesn't have {br,lr}profile",ievent,dstfile);
		}
	      
	      for (irec=0; irec<2; irec++)
		{
		  if(irec==0)
		    fdprofile_ptr = &brprofile_;
		  if (irec==1)
		    fdprofile_ptr = &lrprofile_;
		  
		  fdenergy[irec] = fdprofile_ptr->Energy[0];
		  fdxmax[irec]   = fdprofile_ptr->Xmax[0];
		  fdnmax[irec]   = fdprofile_ptr->Nmax[0];
		  
		} 
	    }
	  
	  ////////////// BRANCHES ASSIGNED (ABOVE) ///////////
	  
	  if (resTree)
	    resTree->Fill();   // SIMPLIFIED ROOT TREE FILLED
	  if(pass1tree)
	    pass1tree->Fill(); // DETAILED ROOT TREE FILLED
	  
	  ievent ++;
	  
	} // while(dstio->readEvent ...
      
      dstio->closeDSTinFile();
      
    }
  ///////////////////// READING THE DST FILES (ABOVE) /////////////////////
  
#undef dst2rt_sd_ldb
  
  //////////////// FINISHING THE OUTPUTS (BELOW) //////////////////
  
  // detailed root tree file closed
  if(pass1tree)
    {
      detailedOut = pass1tree->GetCurrentFile();
      detailedOut->Write();
      detailedOut->Close();
    }
  // simplified root tree file closed
  if(resTree)
    {
      simplifiedOut  = resTree->GetCurrentFile();
      simplifiedOut->Write();
      simplifiedOut->Close();
    }
  //////////////// FINISHING THE OUTPUTS (ABOVE) //////////////////
  
  fprintf(stdout,"\n\nDone\n");
  return 0;
}
