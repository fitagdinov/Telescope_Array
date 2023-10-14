/////// to study the timing offset b/w FD and SD /////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sddstio.h"
#include "TMath.h"
#include "TFitter.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "tacoortrans.h"
#include "tafd10info.h"

// I/O stuff
static FILE *fdtoffset_fp = 0;
static bool parseCmdLine(int argc, char **argv);

// known FD time offset (uS) : time from FD should be made smaller by this ammount
// 0 = BR, 1 = LR, 2 = MD
static const double FD_TIME_LATENCY[3] = { 0.0, 0.0, 0.0 };

// no known SD time latency.  All latencies have been fixed in the SD analysis.
#define SD_TIME_LATENCY 0.0

// MD tube quantum efficiency
#define MD_QE 0.278

// flags that tell whether to use FDs with corresponding IDs (0 - BR, 1-LR, 2 - MD)
static bool use_fdsiteid[3] = { false, false, false };

// use SD core position constrain
static int fit_sd_cc = 1;

// maximum zenith angle cut [Degree] to remove bias in SD core time estimation
static double max_za_cut = 45.0;

class fdtoffset_var_class
{
public:
  fdtoffset_var_class() : d2r(TMath::DegToRad()), m2uS (1/TMath::C() * 1e6) { ; }
  ~fdtoffset_var_class();

  const double d2r;
  const double m2uS;
  int ntube;                   // number of FD mono points to fit = number of good tubes
  int fdsiteid;                // 0 = BR, 1 = LR
  double tube_time[0xc00];     // PMT times, uS
  double tube_timerr[0xc00];   // PMT time errors, uS
  double tube_sdpangle[0xc00]; // PMT angle in SDP frame, degree (angle that goes to T vs A fit)
  double tube_npe[0xc00];      // PMT npe
  double sdp_n[3];             // FD shower-detecor plane normal vector
  double theta;                // degree
  double phi;                  // degree
  double rp;                   // m
  double psi;                  // degree
  double trp;                  // time when the shower front is at the rp, uS
  double tkuv[3];              // shower track unit vector
  double xycore_clf[2];        // core in CLF in meters (fit)
  double thbcore;              // core time from the fit (FD T vs A + SD core constrained) 
  double sdxycore_clf[2];      // core location by SD detector alone 
  double tsdcore;              // core time by SD detector alone
  double sdcore_fd_dist;       // distance from FD to SD core
  double sdcore_sdp_azm;       // angle in the FD shower-detector plane for the SD core
  double tsdcore_fd_exp;       // expected time of arrival of light from the SD core (time expected from the T vs A fit), uS
  double tsdcore_fd_act;       // actual time of arrival of light from the SD core to the FD
  double chi2;                 // chi2 of the fit
  
  // various variables (track information, core position, core time, etc)
  // are calculated. alse chi2 is calculated for a given T_RP,Rp,PSI set
  // and the chi2 calculation uses the SD core position constraint
  double calc_var()
  {
    double delta;             // dummy
    double fdxyzcore_fd[3];   // core in FD frame in meters, in FD Z=0 plane
    double fdxyzcore_clf[3];  // FD Z=0 core transformed to CLF frame w/o taking into account the height difference 
    double rpvec_fd[3];       // rp vector in FD frame in meters
    double rpvec_clf[3];      // rp vector in CLF frame in meters
    double sdcore_xyz_clf[3]; // SD core in CLF frame for transformation purposes
    double sdcore_xyz_fd[3];  // SD core in FD frame for transformation purposes

    // calculate expected time of arrival of light from the SD core (time expected from the T vs A fit), uS
    // the corresponding distance from the FD to SD core
    // and the actual time of light arrival from the SD core at the FD station
    sdcore_xyz_clf[0] = sdxycore_clf[0];
    sdcore_xyz_clf[1] = sdxycore_clf[1];
    sdcore_xyz_clf[2] = 0.0;
    tacoortrans::clf2fdsite(fdsiteid,sdcore_xyz_clf,sdcore_xyz_fd);
    sdcore_fd_dist = sqrt(tacoortrans::dotProduct(sdcore_xyz_fd,sdcore_xyz_fd));
    tacoortrans::get_alt_azm_in_SDP(sdp_n,sdcore_xyz_fd,&delta,&sdcore_sdp_azm);
    if (sdcore_sdp_azm > 180.0 )
      sdcore_sdp_azm -= 360.0;
    tsdcore_fd_exp = trp + m2uS*rp/tan(d2r*(psi+sdcore_sdp_azm)/2.0);
    tsdcore_fd_act = tsdcore + sdcore_fd_dist * m2uS;

    // shower track information from shower-detector plane normal and psi values.
    delta = sqrt(1.0 - sdp_n[2] * sdp_n[2]);
    tkuv[0] = (cos(psi * d2r) * sdp_n[1] + sin(psi * d2r) * sdp_n[0] * sdp_n[2]) / delta;
    tkuv[1] = (sin(psi * d2r) * sdp_n[1] * sdp_n[2] - cos(psi * d2r) * sdp_n[0]) / delta;
    tkuv[2] = -sin(psi * d2r) * delta;
    theta = acos(-tkuv[2])/d2r; // zenith angle
    phi = tacoortrans::range(atan2(-tkuv[1],-tkuv[0])/d2r,360.0); // azimuthal angle, X=East, 0 to 360 range
    
    
    // shower rp vector in fd frame
    tacoortrans::crossProduct(tkuv,sdp_n,rpvec_fd);
    tacoortrans::unitVector(rpvec_fd,rpvec_fd);
    rpvec_fd[0] *= rp;
    rpvec_fd[1] *= rp;
    rpvec_fd[2] *= rp;

    // shower rp vector in CLF frame
    tacoortrans::fdsite2clf(fdsiteid,rpvec_fd,rpvec_clf);
    
    // shower core information from shower-detector-plane normal, rp, and psi values 
    // (defined as the CLF X-Y point where the shower axis crosses the CLF Z=0 plane)
    
    // know that the core XY vector in FD frame is along SDP normal cross Z direction 
    // (this is by sdp normal convention)
    delta            = 1.0 / sqrt(1-sdp_n[2]*sdp_n[2]) * rp / sin (psi*d2r);
    fdxyzcore_fd[0]  = sdp_n[1]  * delta;
    fdxyzcore_fd[1]  = -sdp_n[0] * delta;
    fdxyzcore_fd[2]  = 0.0;
    tacoortrans::fdsite2clf(fdsiteid,fdxyzcore_fd,fdxyzcore_clf);
    
    // fdxyzcore_clf is the FD core position in CLF frame, it is the point where
    // the shower axis crosses the FD Z=0 plane.  Determine the point where the shower axis crosses 
    // the CLF Z=0 plane.    
    delta = - fdxyzcore_clf[2] / tkuv[2];
    xycore_clf[0] = fdxyzcore_clf[0] + delta * tkuv[0];
    xycore_clf[1] = fdxyzcore_clf[1] + delta * tkuv[1];

    // time of the core equals time of the shower front at the point of closest approach 
    // + time it takes for the shower front to propagate from the point of closest 
    // approach to the core location
    delta = (xycore_clf[0]-rpvec_clf[0]) * tkuv[0] + 
      (xycore_clf[1]-rpvec_clf[1]) * tkuv[1] - rpvec_clf[2] * tkuv[2];
    thbcore = trp + delta * m2uS;
    
    ///////////////////////////////////////////////////////////////////////////
    //_________________________________________________________________________
    // CHI2 calculation
    //_________________________________________________________________________
    ///////////////////////////////////////////////////////////////////////////

    chi2 = 0.0; // init

    // contribution from the tangent fit
    for (int i=0; i < ntube; i++)
      {
	delta=(tube_time[i]-trp-m2uS*rp/tan(d2r*(psi+tube_sdpangle[i])/2.0))/tube_timerr[i];
	chi2 += delta * delta;
      }
    
    // contribution from the SD core constraint
    if(fit_sd_cc)
      {
	delta = (xycore_clf[0] - sdxycore_clf[0]) / 100.0;
	chi2 += delta * delta;
	delta = (xycore_clf[1] - sdxycore_clf[1]) / 100.0;
	chi2 += delta * delta;
      }
    
    return chi2;
    
  }
  
  // calculate the NPE-weighted time
  double get_npew_tm()
  {
    double tav = 0.0, w = 0.0;
    for (int i=0; i<ntube; i++)
      {
	w += tube_npe[i];
	tav += tube_npe[i] * tube_time[i];
      }
    if (w > 1e-3) tav /= w;
    return tav;
  }
  
};

static fdtoffset_var_class* fdtoffset_vars = 0;

// The "sd core constrained time vs angle fit" function-to-minimize
static void sd_cc_tvsa_fcn(Int_t &npar, Double_t *gin, Double_t &f,
			   Double_t *par, Int_t iflag)
{
  (void)(npar);
  (void)(gin);
  (void)(iflag);
  fdtoffset_vars->trp = par[0];    // time of the shower front at the Rp, uS
  fdtoffset_vars->rp  = par[1];    // rp, m
  fdtoffset_vars->psi = par[2];    // psi, degree
  f = fdtoffset_vars->calc_var();  // the calc_var() method returns chi2
}

int main(int argc, char **argv)
{ 
  
  // I/O stuff
  char *infile;
  sddstio_class *dstio = new sddstio_class;  
  
  fdtoffset_vars = new fdtoffset_var_class;  // allocte the global variable holder class instance
  fdtoffset_var_class* var = fdtoffset_vars; // make a shoter name pointer for simplification
  TFitter* fit = new TFitter(3);             // allocate the fitter interface
  fdplane_dst_common  *fdplane  = 0; // points either to brplane_ or lrplane_
  
  bool do_fdsiteid[3] = {false,false,false};
  
  int ievent = 0;

  // these are dummy variables needed for adding the reference times to the final results
  // so that the final results come out as times with respect to the beginning of the second
  // these variables will be assigned below
  double fd_tref = 0.0; // SD reference time, uS
  double sd_tref = 0.0; // FD reference time, uS
  
  
  if(!parseCmdLine(argc, argv))
    return 2;
  while ((infile = pullFile()))
    {      
      if(!dstio->openDSTinFile(infile))
	{
	  fprintf(stderr, "error: can't open %s for reading\n",infile);
	  return 2;
	}
      while(dstio->readEvent())
	{
	  ievent++;
	  
	  if(!dstio->haveBank(RUSDGEOM_BANKID))
	    {
	      fprintf(stderr,"warning: no rusdgeom bank in event %d readout; skipping the event\n", ievent);
	      continue;
	    }
	  if(!dstio->haveBank(RUFLDF_BANKID))
	    {
	      fprintf(stderr,"warning: no rufldf bank in event %d readout; skipping the event\n", ievent);
	      continue;
	    }
	  
	  // first try to do all FDs that were requested; then if the necessary requirements (necessary banks) are not
	  // met for any given SD, exclude it from the do_fdsite list by setting the appropriate element to false
	  for (int i=0; i<3; i++)
	    do_fdsiteid[i] = use_fdsiteid[i];

	  
	  // Check the BR requirements
	  if (do_fdsiteid[0] && !dstio->haveBank(BRPLANE_BANKID))
	    {
	      do_fdsiteid[0] = false;
	      fprintf(stderr,"notice: brplane bank is missing in the event %d readout, not doing BR for this event\n", 
		      ievent);
	    }
	  
	  // Check the LR requirements
	  if (do_fdsiteid[1] && !dstio->haveBank(LRPLANE_BANKID))
	    {
	      do_fdsiteid[1] = false;
	      fprintf(stderr,"notice: lrplane bank is missing in the event %d readout, not doing LR for this event\n", 
		      ievent);
	    }
	  
	  // Check the MD requirements
	  if(do_fdsiteid[2] && !dstio->haveBank(STPLN_BANKID))
	    {
	      do_fdsiteid[2] = false;
	      fprintf(stderr,"notice: stpln bank is missing in the event %d readout, not doing MD for this event\n", 
		      ievent);
	    }
	  if (do_fdsiteid[2])
	    {
	      if(!dstio->haveBank(HRAW1_BANKID))
		{
		  if(!dstio->haveBank(MCRAW_BANKID))
		    {
		      fprintf(stderr,"notice: no hraw1 or mcraw banks in the event %d readout; not doing MD for this event\n",
			      ievent);
		      do_fdsiteid[2] = false;
		    }
		  // convert mcraw to hraw1 in case of MD MC
		  if(do_fdsiteid[2])
		    tafd10info::mcraw2hraw1();
		}
	    }
	  
	  ///////////////////////// LOAD SD VARIABLES (BELOW) ////////////////////////////////////////
	  
	  // SD core position in CLF frame.  SD core is defined to be the point
	  // where the shower axis crosses the CLF Z=0 plane
	  var->sdxycore_clf[0] = 1.2e3 * (rufldf_.xcore[1] + RUSDGEOM_ORIGIN_X_CLF);
	  var->sdxycore_clf[1] = 1.2e3 * (rufldf_.ycore[1] + RUSDGEOM_ORIGIN_Y_CLF);	  
	  
	  // SD core time, taking into account any known latencies, if any
	  
	  // sd core time with respect to SD referernce time
	  var->tsdcore = rufldf_.t0 * 1200.0 / TMath::C() * 1.0e6;
	  
	  // addding SD second fraction (SD reference time)
	  sd_tref = (rusdgeom_.tearliest-TMath::Floor(rusdgeom_.tearliest))*1.0e6;
	  var->tsdcore += sd_tref;
	  
	  // removing latency, if any
	  var->tsdcore -= SD_TIME_LATENCY;
	  
	  ///////////////////////// LOAD SD VARIABLES (ABOVE) ////////////////////////////////////////
	  
	  //////////////// LOOP THRU EACH FD AND DO THE TIME COMPARISON IF REQUIRED (BELOW) //////////
	  
	  for (int fdsiteid = 0; fdsiteid < 3; fdsiteid ++)
	    {
	      
	      if(!do_fdsiteid[fdsiteid])
		continue;
	      
	      ////////////////////////// LOAD FD VARIABLES (BELOW) ///////////////////////////////////	  
	      
	      var->fdsiteid   = fdsiteid;
	      
	      // BR and LR FDs
	      if (fdsiteid  < 2)
		{
		  fdplane = (fdsiteid==0 ? &brplane_ : &lrplane_);
		  
		  for (int ix=0; ix<3; ix++)
		    var->sdp_n[ix] = fdplane->sdp_n[ix];
		  var->ntube = 0; 
		  for (int itube=0; itube < fdplane->ntube; itube ++)
		    {
		      if (fdplane->tube_qual[itube] != 1)
			continue;
		      var->tube_time[var->ntube] = fdplane->time[itube]/1.0e3;
		      var->tube_timerr[var->ntube] = 4.25 * fdplane->time_rms[itube] / 1.0e3;
		      var->tube_sdpangle[var->ntube] = fdplane->plane_azm[itube] * TMath::RadToDeg();
		      var->tube_npe[var->ntube] = fdplane->npe[itube];
		      var->ntube ++ ;
		    }
		  fd_tref = ((double)fdplane->jsecfrac)*1.0e-3; // FD reference time, uS
		}
	      // MD
	      else
		{
		  var->fdsiteid = 2;
		  
		  // get the MD shower-detector plane normal vector in a convenient format
		  for (int ix=0; ix<3; ix++)
		    var->sdp_n[ix] = stpln_.n_ampwt[2][ix];
		  
		  // time from the first triggered mirror
		  int ttrigfirst = hraw1_.mirtime_ns[0];
		  for(int imir=1; imir<hraw1_.nmir; ++imir) 
		    {
		      if (hraw1_.mirtime_ns[imir] < ttrigfirst) 
			ttrigfirst = hraw1_.mirtime_ns[imir];
		    }
		  fd_tref = 1.0e20;
		  var->ntube = 0;
		  double npe_max = 0.0;
		  double tube_v[3] = {0,0,0};
		  double alt_sdp = 0.0;
		  for (int itube=0; itube < hraw1_.ntube; itube ++)
		    {
		      if (stpln_.ig[itube] != 1)
			continue;
		      var->tube_npe[var->ntube] = hraw1_.prxf[itube] * MD_QE;
		      if (var->tube_npe[var->ntube] > npe_max)
			npe_max = var->tube_npe[var->ntube];
		      var->tube_time[var->ntube] = ((double)ttrigfirst)/1.0e3 + hraw1_.thcal1[itube];
		      if (var->tube_time[var->ntube] < fd_tref )
			fd_tref = var->tube_time[var->ntube];
		      tafd10info::get_md_tube_pd(hraw1_.tubemir[itube],hraw1_.tube[itube],tube_v);
		      tacoortrans::get_alt_azm_in_SDP(var->sdp_n,tube_v,&alt_sdp,&var->tube_sdpangle[var->ntube]);
		      var->ntube ++ ;
		    }
		  // subtract the MD second fraction before doing the fitting so that the fitter
		  // works with small numbers for time
		  // also calculate the errors on tube time here (now when the maximum tube npe is known)
		  for (int itube=0; itube < var->ntube; itube++)
		    {
		      var->tube_time[itube] -= fd_tref;
		      var->tube_timerr[itube] = 0.7/sqrt(100.0 * var->tube_npe[itube] / npe_max);
		    }
		}
	      
	      ////////////////////////// LOAD FD VARIABLES (ABOVE) ///////////////////////////////////
	      
	      ////////////////// TIME VS ANGLE FIT WITH THE SD CORE CONSTRAINT (BELOW) ///////////////
	      
	      fit->Clear();
	      double tmp_arglist = -1; // for passing arguments to Minuit; no printing
	      fit->ExecuteCommand("SET PRINT", &tmp_arglist, 1);
	      fit->ExecuteCommand("SET NOWarnings",&tmp_arglist,0);
	      fit->SetFCN(sd_cc_tvsa_fcn);
	      fit->SetParameter(0, "trp", var->get_npew_tm()-(5e3*var->m2uS),3.0,0,0);
	      fit->SetParameter(1, "rp",  10e3, 5e3, 0.5, 1e5);
	      fit->SetParameter(2, "psi", 90.0, 5.0, 0.0, 180.0);
	      
	      
	      // pseudo-tangent fit first
	      tmp_arglist = 100000.0; // max. number of iterations
	      fit->FixParameter(2);
	      fit->ExecuteCommand("MIGRAD",&tmp_arglist,1);
	      
	      // fit with psi allowed to float
	      fit->ReleaseParameter(2);
	      tmp_arglist = 100000.0;
	      fit->ExecuteCommand("MIGRAD",&tmp_arglist,1);
	      
	      // get the best fit parameter values
	      var->trp = fit->GetParameter(0);
	      var->rp  = fit->GetParameter(1);
	      var->psi = fit->GetParameter(2);
	      
	      // recalculate all fit variables using these best fit parameters
	      var->calc_var();
	      
	      // In hybrid fit, only FD times are used, so one needs to subtract any
	      // known FD latencies. Also, the FD reference time needs to be taken into account.
	      
	      // Adding FD second fraction (FD reference time)
	      var->thbcore += fd_tref;
	      var->tsdcore_fd_exp += fd_tref;
	      
	      // Subtracting known FD latency
	      var->thbcore -= FD_TIME_LATENCY[fdsiteid];
	      var->tsdcore_fd_exp -= FD_TIME_LATENCY[fdsiteid];
	   

	      /////////////////// TIME VS ANGLE FIT WITH THE SD CORE CONSTRAINT (ABOVE) //////////////
	      
	      
	      /////////////////// CUTS ///////////////////////////////////////////////////////////////
	      if(rusdgeom_.theta[2] > max_za_cut)
		continue;
	      
	      
	      // print the time of the core by hybrid fit (fit that uses FD time and SD only as core constraint)
	      // and the time of the core by SD along
	      // col1 = FD site ID
	      // col2 = time when the light is expected from the sd core time at FD site, uS
	      // col3 = time when the light arrives from the sd core at the FD site, uS
	      fprintf(fdtoffset_fp,"%d %.9e %.9e\n",fdsiteid,var->tsdcore_fd_act,var->tsdcore_fd_exp);
	      //fprintf(stdout,"PSI: %f %f\n", var->psi,trumpmc_.psi[fdsiteid] * TMath::RadToDeg());
	      //fprintf(stderr, "trp = %f rp = %f psi = %f\n",var->trp,var->rp,var->psi);
	    }
	  
	  //////////////// LOOP THRU EACH FD AND DO THE TIME COMPARISON IF REQUIRED (ABOVE) //////////
	  
	}
      
      dstio->closeDSTinFile();
      
    }
  
  
  // close the output file if the output is other than stdout
  if((fdtoffset_fp) && (fdtoffset_fp != stdout))
    fclose(fdtoffset_fp);
  
  return 0;
}

bool parseCmdLine(int argc, char **argv)
{
  int i;
  FILE *fp;
  char *line;
  char inBuf[0x400];
  char outfile[0x400];
  bool fOverwriteMode;
  int fdsiteid;
  if (argc <= 1)
    goto showCmdLineArg_and_return;
  outfile[0] = 0;
  fdtoffset_fp = stdout;
  fOverwriteMode = false;
  for (i = 1; i < argc; i++)
    {
      // man
      if ( 
	  (strcmp("-h",argv[i]) == 0) || 
	  (strcmp("--h",argv[i]) == 0) ||
	  (strcmp("-help",argv[i]) == 0) ||
	  (strcmp("--help",argv[i]) == 0) ||
	  (strcmp("-?",argv[i]) == 0) ||
	  (strcmp("--?",argv[i]) == 0) ||
	  (strcmp("/?",argv[i]) == 0)
	   )
	goto showCmdLineArg_and_return;
      
      // list file
      else if (strcmp("-i", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      printf("error: -i: specify the list file\n");
	      return false;
	    }
	  else
	    {
	      if ((fp = fopen(argv[i], "r")))
		{
		  while (fgets(inBuf, 0x400, fp))
		    {
		      if (((line = strtok(inBuf, " \t\r\n")))
			  && (strlen(line) > 0))
			{
			  if (pushFile(line) != SUCCESS)
			    return false;
			}
		    }
		  fclose(fp);
		}
	      else
		{
		  fprintf(stderr, "error: can't open %s\n", argv[i]);
		  return false;
		}
	    }
	}
      // standard input
      else if (strcmp("--tty", argv[i]) == 0)
	{
	  while (fgets(inBuf, 0x400, stdin))
	    {
	      if (((line = strtok(inBuf, " \t\r\n")))
		  && (strlen(line) > 0))
		{
		  if (pushFile(line) != SUCCESS)
		    return false;
		}
	    }
	}
      // output ascii file
      else if (strcmp("-o", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: specify the output root file\n");
	      return false;
	    }
	  else
	    sscanf(argv[i], "%1023s", outfile);
	}
      // fd site flag
      else if (strcmp("-fd", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: -fd: specify the fd site flag, (0 for BR, 1 for LR, 2 for MD)\n");
	      return false;
	    }
	  else
	    {
	      sscanf(argv[i], "%d", &fdsiteid);
	      if (fdsiteid < 0 || fdsiteid > 2)
		{
		  fprintf(stderr,"error: -fd: fd site id must be in [0-2] range, 0=BR, 1=LR, 2=MD\n");
		  return false;
		}
	      use_fdsiteid[fdsiteid] = true;
	    }
	}
      // maximum zenith angle cut
      else if (strcmp("-za", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: -za: specify the zenith angle cut\n");
	      return false;
	    }
	  else
	    sscanf(argv[i], "%lf", &max_za_cut);
	}
      // sd core position constrained fit
     else if (strcmp("-sdcc", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: -sdcc: specify the sd core position constraint flag (1/0 = YES/NO)\n");
	      return false;
	    }
	  else
	    {
	      sscanf(argv[i], "%d", &fit_sd_cc);
	      if (fit_sd_cc < 0 || fit_sd_cc > 1)
		{
		  fprintf(stderr,"error: -sdcc: flag must be in [0-1] range, 1=YES, 0=NO\n");
		  return false;
		}
	    }
	}
      // force overwrite mode
      else if (strcmp("-f", argv[i]) == 0)
	fOverwriteMode = true;
      // all arguments w/o the '-' switch should be the input files
      else if (argv[i][0] != '-')
	{
	  if (pushFile(argv[i]) != SUCCESS)
	    return false;
	}
      else
	{
	  fprintf(stderr, "error: %s: unrecognized option\n", argv[i]);
	  return false;
	}
    }
  if (countFiles()==0)
    {
      fprintf(stderr,"error: no input files\n");
      return false;
    }
  if (outfile[0])
    {      
      if(!fOverwriteMode)
	{
	  if((fdtoffset_fp = fopen(outfile,"r")))
	    {
	      fprintf(stderr,"error: %s exist; use -f option to overwrite the file\n",outfile);
	      return false;
	    }
	}
      if (!(fdtoffset_fp = fopen(outfile,"w")))
	{
	  fprintf(stderr,"error: can't open %s for writing\n",outfile);
	  return false;
	}
    }
  
  if (!(use_fdsiteid[0] || use_fdsiteid[1] || use_fdsiteid[2]))
    {
      fprintf(stderr,"use -fd option to specify the FD site flag.  You may do more than one FD site by repeatedly using -fd option\n");
      return false;
    }
  
  return true;
  
 showCmdLineArg_and_return:
  fprintf(stderr,"\nCompare the core light arrival times at the FD of the SD-core-constrained hybrid time vs angle fit and SD-only\n");
  fprintf(stderr,"print out variables (second fractions, uS):\n");
  fprintf(stderr,"col1: fdsiteid (0=BR, 1=LR, 2=MD)\n");
  fprintf(stderr,"col2: thbcore_fd_act (time when the light is arriving from the sd core at the FD site, uS)\n");
  fprintf(stderr,"col3: tsdcore_fd_exp (time when the light is expected to arrive from the sd core at the FD site, uS)\n");
  fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list file]  -o [output ascii file]\n",argv[0]);
  fprintf(stderr,"pass input dst file names as arguments without any prefixes or switches\n");
  fprintf(stderr, "-i <string>    : specify the want file (with dst files)\n");
  fprintf(stderr, "--tty <string> : or get input dst file names from stdin\n");
  fprintf(stderr, "-o <string>    : output ascii file name; by default output goes to stdout\n");
  fprintf(stderr, "-fd  <int>     : FD site flag, 0 is BR, 1 is LR, 2 is MD\n");
  fprintf(stderr, "-sdcc <int>    : use SD core constraint in time vs angle fits (1/0=YES/NO) default is %d\n",fit_sd_cc);
  fprintf(stderr, "-za <float>    : maximum zenith angle [degree], default is %.1f (SD core time is biased for inclined events)\n",max_za_cut);
  fprintf(stderr, "-f             : overwrite the output files if they exist\n");
  fprintf(stderr,"\n");
  return false;
}
