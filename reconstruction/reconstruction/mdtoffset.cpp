/////// to study the timing offset b/w FD and SD /////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sddstio.h"
#include "TMath.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "tacoortrans.h"
#include "tafd10info.h"

// I/O stuff
static FILE *mdtoffset_fp = 0;
static bool parseCmdLine(int argc, char **argv);

// no known MD latencies at this time
#define MD_TIME_LATENCY 0.0

// no known SD time latency.  All latencies have been fixed in the SD analysis.
#define SD_TIME_LATENCY 0.0

// MD tube quantum efficiency
#define MD_QE 0.278


// make hraw1 bank from mcraw, mc04, if needed
static bool mcraw2hraw1();

int main(int argc, char **argv)
{ 


  // I/O stuff
  char* infile;
  sddstio_class* dstio = new sddstio_class;  
  
  // the tangent fit function
  // t0  in uS
  // rp  in km
  // psi in degree
  TF1 *fTvsAfd = new TF1("fTvsAfd","[0]+[1]/299.792458/tan(([2]+x)/2.0/57.296)",-180.0,180.0);
  fTvsAfd->SetParNames("t0","rp","psi");
  
  
  int itube;
  int npts;                    // number of FD mono points to fit = number of good tubes
  double tube_time[0xc00];     // PMT times, uS
  double tube_timerr[0xc00];   // PMT time errors, uS
  double tube_sdpangle[0xc00]; // PMT angle in SDP frame, degree (angle that goes to T vs A fit)
  TGraphErrors *gTvsAfd = 0;   // graph used for fitting T vs A 
  
  double sdcore_xyz_clf[3];     // SD core in CLF frame, meters
  double sdcore_xyz_fd[3];      // SD core in FD frame, meters
  double sdcore_sdp_alt;        // SD core altitude in SDP frame
  double sdcore_sdp_azm;        // SD core azimuth in SDP frame
  double sdcore_dist;           // Distance from FD to SD core
  double sdcore_act_time_fd;    // SD core time at the FD (when the light from it would reach the FD)
  double sdcore_exp_time_fd;    // SD core time expected by the FD Mono T vs A Fit

  double md_secfrac_uS;         // MD second fraction, uS
  double tube_v[3];             // MD tube pointing direction
  double sdp_n[3];              // MD Shower-Detector plane normal vector
  double alt_sdp;               // Tube altitude in SDP, [Degree]
  double npe_max;               // maximum tube npe in the shower
  
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
	  if(!dstio->haveBank(RUSDGEOM_BANKID))
	    {
	      fprintf(stderr,"warning: no rusdgeom bank; skipping the event\n");
	      continue;
	    }
	  if(!dstio->haveBank(RUFLDF_BANKID))
	    {
	      fprintf(stderr,"warning: no rufldf bank; skipping the event\n");
	      continue;
	    }
	  if(!dstio->haveBank(HRAW1_BANKID))
	    {
	      if(!dstio->haveBank(MCRAW_BANKID))
		{
		  fprintf(stderr,"warning: no hraw1 or mcraw banks; skipping the event\n");
		  continue;
		  if(!dstio->haveBank(MC04_BANKID))
		    {
		      fprintf(stderr,"warning: mcraw requires mc04 bank to make hraw1; skipping the event\n");
		      continue;
		    }
		}
	      if(!mcraw2hraw1())
		{
		  fprintf(stderr,"warning: failed to make hraw1 bank from mcraw,hraw1; skipping the event\n");
		  continue;
		}
	    }
	  if(!dstio->haveBank(STPLN_BANKID))
	    {
	      fprintf(stderr,"warning: no stpln bank; skipping the event\n");
	      continue;
	    }
	  
	  ////////////////////////// CALCULATIONS DONE BELOW ////////////////////////////////////////
	  
	  ////////// do the time vs angle fit ///////////
	  
	  // get the MD shower-detector plane normal vector in a convenient format
	  for (int ix=0; ix<3; ix++)
	    sdp_n[ix] = stpln_.n_ampwt[2][ix];
	  
	  // first triggered mirror
	  int t2 = hraw1_.mirtime_ns[0];
	  for(int imir=1; imir<hraw1_.nmir; ++imir) 
	    {
	      if (hraw1_.mirtime_ns[imir] < t2) 
		t2 = hraw1_.mirtime_ns[imir];
	    }
	  
	  md_secfrac_uS = 1.0e20;
	  npts = 0;
	  npe_max = 0.0;
	  for (itube=0; itube < hraw1_.ntube; itube ++)
	    {
	      if (stpln_.ig[itube] != 1)
		continue;
	      
	      if (hraw1_.prxf[itube] * MD_QE > npe_max)
		npe_max = hraw1_.prxf[itube] * MD_QE;
	      
	      tube_time[npts] = ((double)t2)/1.0e3 + hraw1_.thcal1[itube];
	      if (tube_time[npts] < md_secfrac_uS )
		md_secfrac_uS = tube_time[npts];
	      
	      tafd10info::get_md_tube_pd(hraw1_.tubemir[itube],hraw1_.tube[itube],tube_v);
	      tacoortrans::get_alt_azm_in_SDP(sdp_n,tube_v,&alt_sdp,&tube_sdpangle[npts]);
	      npts ++ ;
	    }
	  
	  // subtract the MD second fraction before doing the fitting so that the fitter
	  // works with small numbers for time
	  // also calculate the errors on tube time here (now when the maximum tube npe is known)
	  for (itube=0; itube < npts; itube++)
	    {
	      tube_time[itube] -= md_secfrac_uS;
	      tube_timerr[itube] = 0.7/sqrt(100.0 * hraw1_.prxf[itube] * MD_QE / npe_max);
	    }
	  
	  if (gTvsAfd)
	    {
	      delete gTvsAfd;
	      gTvsAfd = 0;
	    } 
	  gTvsAfd = new TGraphErrors(npts,tube_sdpangle,tube_time,0,tube_timerr);
	  fTvsAfd->FixParameter(2,90.0); // 1st fit with psi fixed at 90 degrees
	  gTvsAfd->Fit(fTvsAfd,"Q,0");
	  fTvsAfd->ReleaseParameter(2);  // make psi a fit parameter again and re-fit
	  gTvsAfd->Fit(fTvsAfd,"0,Q");

	  ////////////// calculate the SD core times at the MD 
	  sdcore_xyz_clf[0] = 1.2e3 * (rufldf_.xcore[1] + RUSDGEOM_ORIGIN_X_CLF);
	  sdcore_xyz_clf[1] = 1.2e3 * (rufldf_.ycore[1] + RUSDGEOM_ORIGIN_Y_CLF);
	  sdcore_xyz_clf[2] = 0.0;
	  tacoortrans::clf2fdsite(2,sdcore_xyz_clf,sdcore_xyz_fd);
	  sdcore_dist = sqrt(tacoortrans::dotProduct(sdcore_xyz_fd,sdcore_xyz_fd));
	  tacoortrans::get_alt_azm_in_SDP(sdp_n,sdcore_xyz_fd,&sdcore_sdp_alt,&sdcore_sdp_azm);
	  if (sdcore_sdp_azm > 180.0 )
	    sdcore_sdp_azm -= 360.0;
	  
	  // when FD should see the light from the SD core, using SD time information only (uS)
	  sdcore_act_time_fd  = (rufldf_.t0) * 1200.0 / TMath::C() * 1.0e6;
	  sdcore_act_time_fd += sdcore_dist / TMath::C() * 1.0e6;
	  sdcore_act_time_fd += (rusdgeom_.tearliest-TMath::Floor(rusdgeom_.tearliest))*1.0e6;
	  sdcore_act_time_fd -= SD_TIME_LATENCY;
	  
	  // when FD expects to see the light from the SD core, using mono T vs A result (uS)
	  sdcore_exp_time_fd  = fTvsAfd->Eval(sdcore_sdp_azm);
	  sdcore_exp_time_fd += md_secfrac_uS;
	  sdcore_exp_time_fd -= MD_TIME_LATENCY;
	  
	  fprintf(mdtoffset_fp,"%.9e %.9e\n",sdcore_act_time_fd,sdcore_exp_time_fd);
	  
	  ////////////////////////// CALCULATIONS DONE ABOVE ////////////////////////////////////////
	  
	}
      dstio->closeDSTinFile();
    }
  

  // close the output file if the output is other than stdout
  if((mdtoffset_fp) && (mdtoffset_fp != stdout))
    fclose(mdtoffset_fp);
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
  if (argc <= 1)
    goto showCmdLineArg_and_return;
  outfile[0] = 0;
  mdtoffset_fp = stdout;
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
	  if((mdtoffset_fp = fopen(outfile,"r")))
	    {
	      fprintf(stderr,"error: %s exist; use -f option to overwrite the file\n",outfile);
	      return false;
	    }
	}
      if (!(mdtoffset_fp = fopen(outfile,"w")))
	{
	  fprintf(stderr,"error: can't open %s for writing\n",outfile);
	  return false;
	}
    }
  return true;
  
 showCmdLineArg_and_return:
  fprintf(stderr,"\nCaclulate the time of would-be-seen light by MD from the SD core and the expectation\n");
  fprintf(stderr,"for such time from MD Time vs Angle fit\n");
  fprintf(stderr,"print out variables (second fractions, uS):\n");
  fprintf(stderr,"col0: sdcore_act_time_fd (SD time, uS)\n");
  fprintf(stderr,"col1: sdcore_exp_time_fd (FD time, uS)\n");
  fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list file]  -o [output ascii file]\n",argv[0]);
  fprintf(stderr,"pass input dst file names as arguments without any prefixes or switches\n");
  fprintf(stderr, "-i <string>    : specify the want file (with dst files)\n");
  fprintf(stderr, "--tty <string> : or get input dst file names from stdin\n");
  fprintf(stderr, "-o <string>    : output ascii file name; by default output goes to sdtdout\n");
  fprintf(stderr,"\n");
  return false;
}
// This is a routine for obtaining hraw1 variable from
// mcraw and mc04.  Applicable for MD MC events.
bool mcraw2hraw1()
{
  int imir, itube;
  if (mcraw_.ntube != mc04_.ntube)
    {
      fprintf(stderr,
	      "Error: mcraw_.ntube = %d is not same as mc04_.ntube = %d\n",
	      mcraw_.ntube, mc04_.ntube);
      return false;
    }
  hraw1_.jday = mcraw_.jday;
  hraw1_.jsec = mcraw_.jsec;
  hraw1_.msec = mcraw_.msec;
  hraw1_.status = 1; // no status variable in mcraw
  hraw1_.nmir = mcraw_.nmir;
  hraw1_.ntube = mcraw_.ntube;
  for (imir = 0; imir < hraw1_.nmir; imir++)
    {
      hraw1_.mir[imir] = mcraw_.mirid[imir];
      hraw1_.mir_rev[imir] = mcraw_. mir_rev[imir];
      hraw1_.mirevtno[imir] = mcraw_.mirevtno[imir];
      hraw1_.mirntube[imir] = mcraw_. mir_ntube[imir];
      hraw1_.miraccuracy_ns[imir] = 100; // no mirror accuracy in mcraw
      hraw1_.mirtime_ns[imir] = mcraw_. mirtime_ns[imir];
    }

  for (itube = 0; itube < hraw1_.ntube; itube++)
    {
      hraw1_.tubemir[itube] = mcraw_.tube_mir[itube];
      hraw1_.tube[itube] = mcraw_.tubeid[itube];
      hraw1_.qdca[itube] = mcraw_.qdca[itube];
      hraw1_.qdcb[itube] = mcraw_.qdcb[itube];
      hraw1_.tdc[itube] = mcraw_.tdc[itube];
      hraw1_.tha[itube] = mcraw_.tha[itube];
      hraw1_.thb[itube] = mcraw_.thb[itube];
      hraw1_.prxf[itube] = mc04_.pe[itube] / MD_QE;
      hraw1_.thcal1[itube] = mcraw_.thcal1[itube];
    }
  return true;
}
