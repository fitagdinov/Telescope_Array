#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sddstio.h"
#include "sduti.h"
#include "TMath.h"
#include <string>
#include <map>

using namespace std;
using namespace TMath;




void apply_density_correction(rufldf_dst_common* rufldf)
{
  const double density_0 = 1.042e-3; // Mean air density at TA (1400m) in g/cm^3
  const double eslope    = 1.7;      // approximate slope of integral flux versus energy above 10 EeV
  const double corslope  = -3.5;     // slope of the correction expression in the denominator
  double density = SDGEN::get_gdas_rho_numerically(1.4e5);
  double cfactor = 1.0 / TMath::Power((1.0 + corslope * (density / density_0 - 1.0)), 1.0/eslope);
  for (Int_t i=0; i<2; i++)
    {
      rufldf->energy[i] *= cfactor;
      rufldf->atmcor[i]  = cfactor;
    }
}


void apply_temperature_correction(rufldf_dst_common* rufldf)
{

  const double temperature_0 = 287.3; // Mean temperature at TA (1400m) in K
  const double eslope    = 1.7;       // approximate slope of integral flux versus energy above 10 EeV
  const double corslope  = 2.9;       // slope of the correction expression in the denominator
  double temperature = SDGEN::get_gdas_temp(1.4e5);
  double cfactor = 1.0 / TMath::Power((1.0 + corslope * (temperature / temperature_0 - 1.0)), 1.0/eslope);
  for (Int_t i=0; i<2; i++)
    {
      rufldf->energy[i] *= cfactor;
      rufldf->atmcor[i]  = cfactor;
    }
}

class sdatm_correction_listOfOpt
{
  
public:
  char   outfile[0x400];
  char   dout[0x400];
  int    cor;
  bool   fOverwrite;
  int    verbosity;
  map <int,string> sdatm_corr_name;
  
  sdatm_correction_listOfOpt()
  {
    outfile[0]          = 0;      // output file initialized
    sprintf(dout,"./");           // default output directory is the current working directory
    cor                 = 0;      // by default, correct using density correction formula
    sdatm_corr_name[0]  = "density correction";
    sdatm_corr_name[1]  = "temperature correction";
    fOverwrite          = false;  // don't overwrite the output files if they exist by default
    verbosity           = 1;      // print minimum stuff by default
  }
  ~sdatm_correction_listOfOpt() { ; }
  

  bool parseCmdLine(int argc, char **argv)
  {
    int i;
    char inBuf[0x400];
    
    if (argc <= 1)
      {
	printMan(argv[0]);
	return false;
      }

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
	  {
	    printMan(argv[0]);
	    return false;
	  }
	
	
	// list file
	else if (strcmp("-i", argv[i]) == 0)
	  {
	    FILE *fp = 0;
	    char* line = 0;
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr,"error: -i: specify the list file\n");
		return false;
	      }
	    if (!(fp = fopen(argv[i], "r")))
	      {
		fprintf(stderr, "error: can't open %s\n", argv[i]);
		return false;
	      }
	    while (fgets(inBuf, 0x400, fp))
	      {
		if (!((line = strtok(inBuf, " \t\r\n"))) || !strlen(line))
		  continue;
		if (pushFile(line) != SUCCESS)
		  return false;
	      }
	    fclose(fp); 
	  }
	// standard input
	else if (strcmp("--tty", argv[i]) == 0)
	  {
	    char* line = 0;
	    while (fgets(inBuf, 0x400, stdin))
	      {
		if (!((line = strtok(inBuf, " \t\r\n"))) || !strlen(line))
		  continue;
		if (pushFile(line) != SUCCESS)
		  return false;
	      }
	  }
	// output dst file
	else if (strcmp("-o1f", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -o1f: specify the output dst file\n");
		return false;
	      }
	    sscanf(argv[i], "%1023s", outfile);
	  }
	// output directory
	else if (strcmp("-o", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -o: specify the output directory\n");
		return false;
	      }
	    sscanf(argv[i], "%1023s", dout);
	  }
	// correction flag
	else if (strcmp("-cor", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -cor: specify the correction flag\n");
		return false;
	      }
	    sscanf(argv[i], "%d", &cor);
	    if (sdatm_corr_name.find(cor) == sdatm_corr_name.end()) 
	      {
		fprintf(stderr,"error: -cor: %d is an ivalid correction flag, use only\n", cor);
		for (map<int,string>::iterator it = sdatm_corr_name.begin(); it != sdatm_corr_name.end(); it++)
		  fprintf(stderr,"             %d (%s)\n", (*it).first, ((*it).second).c_str());
	      }
	  }
	// force overwrite mode
	else if (strcmp("-f", argv[i]) == 0)
	  fOverwrite = true;
	// verbosity flag
	else if (strcmp("-v", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -v: specify the verbosity flag\n");
		return false;
	      }
	    sscanf(argv[i], "%d", &verbosity);
	  }
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
    if (countFiles() < 1)
      {
	fprintf(stderr,"error: no input files\n");
	return false;
      }
    return true;
  }
 
  void printMan(char* progName)
  {
    fprintf(stderr,"\nApply atmospheric corrections to SD energies to fix the time dependence of SD enent rate\n");
    fprintf(stderr,"above 10 EeV.  Required banks: gdas, rufldf.\n");
    fprintf(stderr,"Correction is applied to energy stored in rufldf.\n");
    fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list file]  -o [output directory]\n",progName);
    fprintf(stderr,"pass input event dst file names as arguments without any prefixes or switches\n");
    fprintf(stderr, "-i <string>    : specify the want file (with dst files)\n");
    fprintf(stderr, "--tty <string> : or get input dst file names from stdin\n");
    fprintf(stderr, "-o <string>    : directory for output DST files with automatically\n");
    fprintf(stderr,"                  generated names for each input DST file name\n");
    fprintf(stderr,"                  default output directory is %s\n",dout);
    fprintf(stderr, "-o1f <string>  : single output DST file name, overrides -o option\n");
    fprintf(stderr, "-cor <int>     : correction flag:\n");
    fprintf(stderr,"                  0:       %s\n",sdatm_corr_name[0].c_str()); 
    fprintf(stderr,"                  1:       %s\n",sdatm_corr_name[1].c_str());
    fprintf(stderr,"                  default: %d (%s)\n",cor,sdatm_corr_name[cor].c_str());
    fprintf(stderr, "-f             : overwrite the output files if they exist\n");
    fprintf(stderr, "-v             : verbosity flag, default %d\n",verbosity);
    fprintf(stderr,"\n");
  }
  
};


int main(int argc, char *argv[])
{
   
  // cmd line options
  sdatm_correction_listOfOpt opt;
  if (!opt.parseCmdLine(argc,argv))
    return 2;
  
  // dst iterator
  sddstio_class dstio((opt.verbosity>=1));

  // go over all input files and over all events
  char* infile  = 0;
  char* outfile = 0;
  int eventNo = 0;
  while((infile=pullFile()))
    {
      if(!dstio.openDSTinFile(infile))
	return 2;
      if(!opt.outfile[0])
	{
	  outfile = new char[strlen(infile)+strlen(".sdatm_corr.dst.gz")+1];
	  if (SDIO::makeOutFileName(infile, opt.dout, (char *)".sdatm_corr.dst.gz", outfile) != 1)
	    return 2;
	}
      else
	outfile = opt.outfile;
      eventNo = 0;
      while(dstio.readEvent())
	{
	  eventNo++;

	  // make sure the mandatory banks are present, or
	  // else skip the event
	  if(!dstio.haveBank(RUFLDF_BANKID))
	    {
	      fprintf(stderr,"error: required rufldf bank missing for event %d; skipping event\n",eventNo);
	      continue;
	    }
	  if(!dstio.haveBank(GDAS_BANKID))
	    {
	      fprintf(stderr,"error: required gdas bank missing for event %d; skipping event\n",eventNo);
	      continue;
	    }
	  
	  // apply atmopsheric corerction only if this is a real data (not Monte Carlo) event!
	  if(!dstio.haveBank(RUSDMC_BANKID))
	    {
	      // apply correction to energy in rufldf DST bank
	      if (opt.cor==0)
		apply_density_correction(&rufldf_);
	      else if(opt.cor == 1)
		apply_temperature_correction(&rufldf_);
	      else
		{
		  fprintf(stderr,"error: wrong correction flag, only 0 (%s) or 1 (%s) supported\n",
			  opt.sdatm_corr_name[0].c_str(),opt.sdatm_corr_name[1].c_str());
		  return 2;
		}
	    }
	  else
	    {
	      static bool already_printed_this_messagage=false;
	      if(!already_printed_this_messagage)
		{
		  fprintf(stderr,"Notice: Monte Carlo event, not applying atmospheric correction.\n");
		  already_printed_this_messagage = true;
		}
	    }
	  
	  // open the output DST file if it hasn't been opened yet
	  if(!dstio.outFileOpen() && !dstio.openDSToutFile(outfile,opt.fOverwrite))
	    return 2;
	  
	  // write out the event
	  dstio.writeEvent();
	}
      dstio.closeDSTinFile();
      if(!opt.outfile[0])
	{
	  if(dstio.outFileOpen())
	    dstio.closeDSToutFile();
	  delete[] outfile;
	  outfile = 0;
	}
    }
  
  // finilize the dst output file
  if(dstio.outFileOpen())
    dstio.closeDSToutFile();

  fprintf(stdout,"\nDone\n");
  
  return 0; 
}

