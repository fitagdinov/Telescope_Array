#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sddstio.h"

// binning for making thrown energy histogram
#define elo 17.0
#define ebsize 0.1
#define n_e_bins 40

#define fd_energy_scale (1/1.27) // default energy scale

class reduceDST_listOfOpt
{
  
public:
  
  char   outfile[0x400];
  bool   fOverwrite;
  int    nsd;
  int    nofwf;
  bool   rm_notsd;
  bool   rm_sdrec;
  bool   p_mcehist;
  double enscl;
  bool   verbose;
  
  reduceDST_listOfOpt()
  {
    outfile[0] = 0;               // output file initialized
    fOverwrite = false;           // don't overwrite the output files if they exist by default
    nsd        = 3;               // default minimum number of good SDs
    nofwf      = 3;               // default minimum number of waveforms
    rm_notsd   = false;           // don't remove non-SD banks by default
    rm_sdrec   = false;           // don't remove SD reconstruction banks by default
    p_mcehist  = false;           // don't print the MC thrown energy histogram by default
    enscl      = fd_energy_scale; // default energy scale value for printing MC thrown energy histogram
    verbose    = false;           // don't print extra stuff by default
  }
  ~reduceDST_listOfOpt() { ; }
  

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
	else if (strcmp("-o", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: specify the output dst file\n");
		return false;
	      }
	    else
	      sscanf(argv[i], "%1023s", outfile);
	  }
	// force overwrite mode
	else if (strcmp("-f", argv[i]) == 0)
	  fOverwrite = true;
	// minimum number of SDs cut
	else if (strcmp("-nsd", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -nsd: specify the minimum number of SDs cut!\n");
		return false;
	      }
	    else
	      sscanf(argv[i], "%d", &nsd);
	  }
	// minimum number of waveforms cut
	else if (strcmp("-nofwf", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -nofwf: specify the minimum number of waveforms cut!\n");
		return false;
	      }
	    else
	      sscanf(argv[i], "%d", &nofwf);
	  }
	// remove non-sd dst banks
	else if (strcmp("-rm_notsd", argv[i]) == 0)
	  rm_notsd = true;
	// remove sd reconstruction dst banks
	else if (strcmp("-rm_sdrec", argv[i]) == 0)
	  rm_sdrec = true;
	// print out the MC thrown energy histogram
	else if (strcmp("-p_mcehist", argv[i]) == 0)
	  p_mcehist = true;
	// energy scale
	else if (strcmp("-enscl", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -enscl: specify the energy scale!\n");
		return false;
	      }
	    else
	      sscanf(argv[i], "%lf", &enscl);
	  }
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
    if (!outfile[0])
      sprintf(outfile,"output.dst");
    if(nsd < 0)
      nsd = 0;
    if(nofwf < 0)
      nofwf = 0;
    return true;
  }
 
  void printMan(char* progName)
  {
    fprintf(stderr,"\nReduce SD dst files in size, available options are: \n");
    fprintf(stderr,"a) Cut on minimum number of counters\n");
    fprintf(stderr,"b) Remove extra banks (not related to SD)\n");
    fprintf(stderr,"c) Remove banks with reconstruction information to re-processed the files with SD analysis again\n");
    fprintf(stderr,"all events are required to have either rusdraw or tasdcalibev bank; otherwise they are skipped\n");
    fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list file]  -o [output dst file]\n",progName);
    fprintf(stderr,"pass input dst file names as arguments without any prefixes or switches\n");
    fprintf(stderr, "-i <string>    : specify the want file (with dst files)\n");
    fprintf(stderr, "--tty <string> : or get input dst file names from stdin\n");
    fprintf(stderr, "-o <string>    : output DST file name\n");
    fprintf(stderr, "-nsd <int>     : minimum number of good SDs, rufptn bank required otherwise event is skipped; default: %d\n",nsd);
    fprintf(stderr, "-nofwf <int>   : min. number of waveforms, rusdraw or tasdcalibev required otherwise evnet is skipped; default: %d\n",nofwf);
    fprintf(stderr, "-rm_notsd      : remove all banks that are not related to SD\n");
    fprintf(stderr, "-rm_sdrec      : remove SD reconstruction banks (except rusdraw and tasdcalibev)\n");
    fprintf(stderr, "-p_mcehist     : print thrown energy (MC only) off by default\n");
    fprintf(stderr, "-enscl <float> : energy scale constant for thrown MC energies, default: %.3f\n",enscl);
    fprintf(stderr, "-f             : overwrite the output files if they exist\n");
    fprintf(stderr, "-v             : verbose mode\n");
    fprintf(stderr,"\n");
  }
  
};


int main(int argc, char *argv[])
{
   
  // cmd line options
  reduceDST_listOfOpt opt;
  if (!opt.parseCmdLine(argc,argv))
    return 2;
  
  // dst iterator
  sddstio_class dstio(opt.verbose);
  
  // by default, output banks are same as those that
  // were read in
  integer4 outBanks = dstio.getGotBanks();
  
  // possible SD banks
  integer4 sdAllBanks = newBankList(10);
  addBankList(sdAllBanks,RUSDMC_BANKID);
  addBankList(sdAllBanks,RUSDMC1_BANKID);
  addBankList(sdAllBanks,TASDCALIBEV_BANKID);
  addBankList(sdAllBanks,RUSDRAW_BANKID);
  addBankList(sdAllBanks,RUFPTN_BANKID);
  addBankList(sdAllBanks,RUSDGEOM_BANKID);
  addBankList(sdAllBanks,RUFLDF_BANKID);
  
  integer4 sdRecBanks = newBankList(10);
  addBankList(sdRecBanks,RUFPTN_BANKID);
  addBankList(sdRecBanks,RUSDGEOM_BANKID);
  addBankList(sdRecBanks,RUFLDF_BANKID);
  
  
  // energy histogram
  int energy_hist[n_e_bins];
  int ienergy;
  for (ienergy=0; ienergy<n_e_bins; ienergy++)
    energy_hist[ienergy] = 0;
  
  char* infile = 0;
  bool have_rusdraw  = false, have_tasdcalibev = false;
  while((infile=pullFile()))
    {
      if(!dstio.openDSTinFile(infile))
	return 2;
      
      while(dstio.readEvent())
	{
	  // thrown energy histogram before making any cuts
	  // if it has been requested and event has rusdmc dst bank
	  if(opt.p_mcehist && dstio.haveBank(RUSDMC_BANKID))
	    {
	      ienergy = (int)floor((18.0+log(rusdmc_.energy * opt.enscl)/M_LN10-elo)/ebsize);
	      if (ienergy>=0 && ienergy < n_e_bins)
		energy_hist[ienergy] ++;
	    }
	  
	  // if minimum number of waveforms cut is used then rusdraw or tasdcalibev banks should be present
	  // and check if the minimum number of waveforms satisfies the criteria
	  if(opt.nofwf > 0)
	    {
	      have_rusdraw     = dstio.haveBank(RUSDRAW_BANKID);
	      have_tasdcalibev = dstio.haveBank(TASDCALIBEV_BANKID);
	      if(!have_rusdraw && !have_tasdcalibev)
		continue;
	      if(have_rusdraw)
		{
		  if(rusdraw_.nofwf < opt.nofwf)
		    continue;
		}
	      else
		{
		  if(tasdcalibev_.numTrgwf < opt.nofwf)
		    continue;
		}
	    }
	
	  // if minimum number of good SDs used then rufptn bank must be present
	  if (opt.nsd > 0 && (!dstio.haveBank(RUFPTN_BANKID) || rufptn_.nstclust < opt.nsd ))
	    continue;

	  // remove banks that are not SD-related, if such option is used
	  if(opt.rm_notsd)
	    comBankList(outBanks,sdAllBanks);
	  
	  // remove banks that are SD reconstruction, if such option is used
	  if(opt.rm_sdrec)
	    difBankList(outBanks,sdRecBanks);
	  
	  // open the output DST file if it hasn't been opened yet	  
	  if(!dstio.outFileOpen() && !dstio.openDSToutFile(opt.outfile,opt.fOverwrite))
	    return 2;
	  
	  // write out the event
	  dstio.writeEvent(outBanks,false);
	}
      dstio.closeDSTinFile();
    }
  
  // finilize the dst output file
  if(dstio.outFileOpen())
    dstio.closeDSToutFile();
  
  // print the MC energy histogram
  if(opt.p_mcehist)
    {
      for (ienergy=0; ienergy<n_e_bins; ienergy++)
	fprintf(stdout,"E_BIN: %02d %.2f %d\n",ienergy,elo+ebsize*((double)ienergy+0.5),energy_hist[ienergy]);
      fflush(stdout);
    }
  return 0; 
}

