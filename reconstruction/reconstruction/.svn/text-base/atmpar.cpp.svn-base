#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sddstio.h"
#include "atmparfitter.h"
#include "sduti.h"

class atmpar_listOfOpt
{
  
public:
  
  char   outfile[0x400];
  char   dout[0x400];
  bool   fOverwrite;
  bool   verbose;
  
  atmpar_listOfOpt()
  {
    outfile[0] = 0;               // output file initialized
    sprintf(dout,"./");           // default output directory is the current working directory
    fOverwrite = false;           // don't overwrite the output files if they exist by default
    verbose    = false;           // don't print extra stuff by default
  }
  ~atmpar_listOfOpt() { ; }
  

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
		fprintf(stderr, "error: specify the output dst file\n");
		return false;
	      }
	    else
	      sscanf(argv[i], "%1023s", outfile);
	  }
	// output directory
	else if (strcmp("-o", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: specify the output directory\n");
		return false;
	      }
	    else
	      sscanf(argv[i], "%1023s", dout);
	  }
	// force overwrite mode
	else if (strcmp("-f", argv[i]) == 0)
	  fOverwrite = true;
	// verbose mode
	else if (strcmp("-v", argv[i]) == 0)
	  verbose = true;
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
    return true;
  }
 
  void printMan(char* progName)
  {
    fprintf(stderr,"\nFit GDAS bank data into 5 layer model T(h) = a_i + b_i*exp(h/c_i), i=1..5 \n");
    fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list file]  -o [output directory]\n",progName);
    fprintf(stderr,"pass input dst file names as arguments without any prefixes or switches\n");
    fprintf(stderr, "-i <string>    : specify the want file (with dst files)\n");
    fprintf(stderr, "--tty <string> : or get input dst file names from stdin\n");
    fprintf(stderr, "-o <string>    : directory for output DST files with automatically\n");
    fprintf(stderr,"                  generated names for each input DST file name\n");
    fprintf(stderr,"                  default output directory is %s\n",dout);
    fprintf(stderr, "-o1f <string>  : single output DST file name, overrides -o option\n");
    fprintf(stderr, "-f             : overwrite the output files if they exist\n");
    fprintf(stderr, "-v             : verbose mode\n");
    fprintf(stderr,"\n");
  }
  
};


int main(int argc, char *argv[])
{
   
  // cmd line options
  atmpar_listOfOpt opt;
  if (!opt.parseCmdLine(argc,argv))
    return 2;
  
  // dst iterator
  sddstio_class dstio(opt.verbose);
  
  // atmopsheric parameter fitter
  atmparfitter atmparfit;
  
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
	  outfile = new char[strlen(infile)+strlen(".atmpar.dst.gz")+1];
	  if (SDIO::makeOutFileName(infile, opt.dout, (char *)".atmpar.dst.gz", outfile) != 1)
	    return 2;
	}
      else
	outfile = opt.outfile;
      eventNo = 0;
      while(dstio.readEvent())
	{
	  eventNo++;
	  
	  // first set output banks are same as those that were read in
	  integer4 outBanks = dstio.getGotBanks();
	  
	  if(!dstio.haveBank(GDAS_BANKID))
	    {
	      fprintf(stderr,"warning: no GDAS bank in file %s event %d; skipping event\n",infile,eventNo);
	      continue;
	    }
	  if(!dstio.haveBank(ATMPAR_BANKID))
	    addBankList(outBanks,ATMPAR_BANKID);
	  
	  // do the fit
	  if(opt.verbose)
	    fprintf(stdout,"Fitting eventNo %d\n",eventNo);
	  atmparfit.loadVariables(&gdas_);
	  atmparfit.Fit(opt.verbose);
	  atmparfit.put2atmpar(&atmpar_);
	  
	  // open the output DST file if it hasn't been opened yet
	  if(!dstio.outFileOpen() && !dstio.openDSToutFile(outfile,opt.fOverwrite))
	    return 2;
	  
	  // write out the event
	  dstio.writeEvent(outBanks,false);
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
  
  return 0; 
}

