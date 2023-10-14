#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "event.h"
#include "filestack.h"
#include "sddstio.h"
#include "fdcalib_util.h"
#include "sduti.h"
#include <vector>
#include <algorithm> 
using namespace std;

vector<gdas_dst_common>  gdas_all;
vector<gdas_dst_common*> gdas_sorted;


struct gdas_less_than_key
{
  inline bool operator() (const gdas_dst_common* gdas1, const gdas_dst_common* gdas2)
  {
    return (gdas1->dateFrom < gdas2->dateFrom);
  }
};

static int find_closest_gdas_before_event(int j2000sec)
{
  int n=(int)gdas_sorted.size();
  int i=n/2,ilo=0,iup=n;
  // make sure have GDAS data loaded
  if ( n < 1)
    {
      fprintf(stderr,"error: find_closest_gdas_before_event: no GDAS data loaded\n");
      exit(2);
    }
  // check if there are any GDAS cycles that are earlier than the event
  int j2000sec_rec_earliest = (int) ((time_t)gdas_sorted[0]->dateFrom-convertDate2Sec(2000,1,1,0,0,0));
  if(j2000sec < j2000sec_rec_earliest)
    {
      int yymmdd = SDGEN::j2000sec2yymmdd(j2000sec);
      int sec_since_mn=j2000sec-SDGEN::time_in_sec_j2000(yymmdd,0);
      int hhmmss=10000*(sec_since_mn/3600)+100*((sec_since_mn%3600)/60)+(sec_since_mn%60);
      fprintf(stderr,"WARNING: no GDAS cycle earlier than event %06d:%06d available\n",yymmdd,hhmmss);
      return 0;
    }
  // now find the closest GDAS cycle that's earlier than the event, for the most cases the closest GDAS
  // cycle will have started within 3 hours of the event.  External routines will check whether that's the
  // case.  If it is not the case, then the external routines will print a warning message.
  while (i > 0 && i < n - 1)
    {
      int j2000sec_rec = (int) ((time_t)gdas_sorted[i]->dateFrom-convertDate2Sec(2000,1,1,0,0,0));
      if(j2000sec_rec == j2000sec)
	break;
      if(j2000sec_rec < j2000sec)
	{
	  ilo=i;
	  i=(i+iup)/2;
	  continue;
	}
      int j2000sec_rec1 = (int) ((time_t)gdas_sorted[i-1]->dateFrom - convertDate2Sec(2000,1,1,0,0,0));
      if(j2000sec_rec1 < j2000sec)
	break;
      iup=i;
      i=(ilo+i)/2;
    }
  i -= 1;
  if (i < 0)
    i = 0;
  return i;
}


class sdatm_calib_listOfOpt
{
  
public:

  char   gdasfile[0x400];
  char   outfile[0x400];
  char   dout[0x400];
  bool   fOverwrite;
  double warn_tdiff_h;
  int    verbosity;
  
  sdatm_calib_listOfOpt()
  {
    gdasfile[0] = 0;          // user must specify atmospheric data base DST file
    outfile[0] = 0;           // output file initialized
    sprintf(dout,"./");       // default output directory is the current working directory
    fOverwrite = false;       // don't overwrite the output files if they exist by default
    warn_tdiff_h = 3.0;       // warn if the closest good GDAS cycle is farther than this in time [h]
    verbosity  = 1;           // print minimum stuff by default
  }
  ~sdatm_calib_listOfOpt() { ; }
  

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
	
	// gdas DST file
	else if (strcmp("-a", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -a: specify the atmospheric parameter data base DST file!\n");
		return false;
	      }
	    sscanf(argv[i], "%1023s", gdasfile);
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
	// force overwrite mode
	else if (strcmp("-f", argv[i]) == 0)
	  fOverwrite = true;
	// warning when a good gdas cycle is farther than this in time
	else if (strcmp("-w", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -w: specify the warning time difference in hours\n");
		return false;
	      }
	    sscanf(argv[i], "%lf", &warn_tdiff_h);
	  }
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
    if(!gdasfile[0])
      {
	fprintf(stderr,"error: gdas DST file is not given; use -a option\n");
	fprintf(stderr,"to specify the gdas database DST file\n");
	return false;
      }
    return true;
  }
 
  void printMan(char* progName)
  {
    fprintf(stderr,"\nAssign gdas bank, that corresponds in date and time, to each event in the DST files\n");
    fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list file]  -o [output directory]\n",progName);
    fprintf(stderr,"pass input event dst file names as arguments without any prefixes or switches\n");
    fprintf(stderr, "-i <string>    : specify the want file (with dst files)\n");
    fprintf(stderr, "-a <string>    : (mandatory) DST file that contains gdas bank for the most recent TA period\n");
    fprintf(stderr, "--tty <string> : or get input dst file names from stdin\n");
    fprintf(stderr, "-o <string>    : directory for output DST files with automatically\n");
    fprintf(stderr,"                  generated names for each input DST file name\n");
    fprintf(stderr,"                  default output directory is %s\n",dout);
    fprintf(stderr, "-o1f <string>  : single output DST file name, overrides -o option\n");
    fprintf(stderr, "-f             : overwrite the output files if they exist\n");
    fprintf(stderr, "-w             : warn when no good gdas cycle found within this time [h], default %.2f\n",warn_tdiff_h);
    fprintf(stderr, "-v             : verbosity flag, default %d\n",verbosity);
    fprintf(stderr,"\n");
  }
  
};


int main(int argc, char *argv[])
{
   
  // cmd line options
  sdatm_calib_listOfOpt opt;
  if (!opt.parseCmdLine(argc,argv))
    return 2;
  
  // dst iterator
  sddstio_class dstio((opt.verbosity >= 1));


  // load atmospheric parameter DST file
  if(!dstio.openDSTinFile(opt.gdasfile))
    return 2;

  if(opt.verbosity>=1)
    {
      fprintf(stdout,"Reading GDAS database DST file %s:\n",opt.gdasfile);
      fflush(stdout);
    }
  int recordNo = 0;

  while(dstio.readEvent())
    {
      recordNo ++;
      if(!dstio.haveBank(GDAS_BANKID))
	{
	  fprintf(stderr,"error: gdas DST bank absent for the record number %d\n", recordNo);
	  continue;
	}
      if(!SDGEN::check_gdas())
	{
	  if(opt.verbosity>=1)
	    {
	      char d1[0x100];
	      char d2[0x100];
	      convertSec2DateLine((time_t)gdas_.dateFrom,d1);
	      convertSec2DateLine((time_t)gdas_.dateTo,d2);
	      fprintf(stderr,"Rejecting gdas recordNo %d FROM %s TO %s: bad gdas cycle\n",recordNo,d1,d2);
	    }
	  continue;
	}
      gdas_all.push_back(gdas_);
    }
  dstio.closeDSTinFile();
  gdas_sorted.resize(gdas_all.size());
  for (int i=0; i < (int)gdas_all.size(); i++)
    gdas_sorted[i] = &gdas_all[i];
  sort(gdas_sorted.begin(),gdas_sorted.end(),gdas_less_than_key());
  if(opt.verbosity>=1)
    {
      char d1[0x100];
      char d2[0x100];
      convertSec2DateLine((time_t)gdas_sorted.front()->dateFrom,d1);
      convertSec2DateLine((time_t)gdas_sorted.back()->dateTo,d2);
      fprintf(stdout,"%d gdas data base records loaded, from %s to %s\n",
	      (int)gdas_sorted.size(),d1,d2);
      fflush(stdout);
    }
  
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
	  outfile = new char[strlen(infile)+strlen(".sdatm_calib.dst.gz")+1];
	  if (SDIO::makeOutFileName(infile, opt.dout, (char *)".sdatm_calib.dst.gz", outfile) != 1)
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
	  addBankList(outBanks,GDAS_BANKID);
	  
	  // open the output DST file if it hasn't been opened yet
	  if(!dstio.outFileOpen() && !dstio.openDSToutFile(outfile,opt.fOverwrite))
	    return 2;

	  // find the closest good GDAS cycle that occurs before the event
	  int yymmdd,hhmmss,usec;
	  dstio.get_event_time(&yymmdd,&hhmmss,&usec);
	  int j2000sec = SDGEN::time_in_sec_j2000(yymmdd,hhmmss);
	  int ical = find_closest_gdas_before_event(j2000sec);
	  memcpy(&gdas_,gdas_sorted[ical],sizeof(gdas_dst_common));
	  
	  // check if the closest GDAS cycle is within 3 hours of the event
	  int j2000sec_gdas = (int) ((time_t)gdas_.dateFrom-convertDate2Sec(2000,1,1,0,0,0));
	  if((opt.verbosity>=1) && ((double)abs(j2000sec-j2000sec_gdas) > 3600.0*opt.warn_tdiff_h))
	    {
	      fprintf(stderr,"WARNING: event %06d:%06d: no good GDAS cycle found that is within %.2f hours ",
		      yymmdd,hhmmss,opt.warn_tdiff_h);
	      fprintf(stderr,"and that starts before the event\n");
	    }
	  
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

  fprintf(stdout,"\nDone\n");
  fflush(stdout);
  
  return 0; 
}

