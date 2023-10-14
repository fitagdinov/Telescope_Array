#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include "event.h"
#include <string>
#include <vector>
#include <map>
#include "sddstio.h"
#include "sduti.h"
#include <algorithm>
#include "dst_size_limits.h"

// maximum number of input DST files that can be safely processed
#define N_DST_INFILES_MAX (MAX_DST_FILE_UNITS - 1)

using namespace std;

class sdmc_byday_listOfOpt
{
public:
  vector <sd_dst_handler*> infiles;  // input DST files
  char   outdir[0x400];              // output directory
  char   prefix[0x400];              // output file prefix
  bool   fOverwrite;
  bool   verbose;
  sdmc_byday_listOfOpt();
  ~sdmc_byday_listOfOpt();
  bool parseCmdLine(int argc, char **argv);
  void printMan(char* progName);
};


class sdmc_byday_tstamp_class
{
public:
  int    ifile;    // input dst file number
  int    ievent;   // event number in the dst file
  int    yymmdd;   // date of the event
  double t2000s;   // time of the event (include second fraction) since midnight of Jan 1, 2000
  sdmc_byday_tstamp_class(int file_num, int event_num, int event_yymmdd, int event_hhmmss, int usec)
  {
    ifile      =  file_num;
    ievent     =  event_num;
    yymmdd     =  event_yymmdd;
    t2000s     =  SDGEN::time_in_sec_j2000f(event_yymmdd,event_hhmmss,usec);
    if(yymmdd != get_yymmdd())
      {
	fprintf(stderr,"warning: sdmc_cat_tsort_tstamp_class::store_event_time: inconsistent date and time\n");
	yymmdd = get_yymmdd();
      }
  }
  virtual ~sdmc_byday_tstamp_class() { ; }
  // get yymmdd from the second since midnight of Jan 1, 2000
  int get_yymmdd()
  {
    return SDGEN::j2000sec2yymmdd((int)floor(t2000s));
  }
  // get hhmmss from the second since midnight of Jan 1, 2000
  int get_hhmmss()
  {
    double day_fract_second = t2000s-86400*floor(t2000s/86400);
    int second = (int)floor(day_fract_second);
    int hhmmss = (second/3600)*10000+((second%3600)/60)*100+(second%60);
    return hhmmss;
  }
  // get usec from the second since midnight of Jan 1, 2000
  int get_usec()
  {
    double second_fraction = t2000s-floor(t2000s);
    int usec = (int)floor(second_fraction*1e6+0.5);
    return usec;
  }
};

// needed for sorting the events
bool operator <(const sdmc_byday_tstamp_class & left, const sdmc_byday_tstamp_class& right)
{
  return (left.t2000s < right.t2000s);
}

// end points (inclusive) for each yymmdd in the sorted event_stamps array
class tstamp_endpts_class
{
public:
  tstamp_endpts_class()
  {
    istart              = (int)floor(1e9+0.5);
    iend                = 0;
    outfile_initialized = false;
  }
  int istart;                // date beginning index
  int iend;                  // date ending index
  bool outfile_initialized;  // true if the output file has been initialized for the date
  ~tstamp_endpts_class() {}
};

int main(int argc, char **argv)
{
  sdmc_byday_listOfOpt opt;
  
  // hold the time stamp for every event encountered
  vector<sdmc_byday_tstamp_class> event_tstamps;
  
  // end points (inclusive) for each yymmdd in the sorted event_stamps array
  map<int,tstamp_endpts_class> tstamp_endpts;

 
  if(!opt.parseCmdLine(argc,argv))
    return 2;
  

  // initialize points for the input DST file handlers
  sd_dst_handler *ofl   = 0; // output file, one day at a time
  
  // stats
  int ndays_total = 0, nevents_total = 0;
  int date_from = (int)floor(1e9+0.5);
  int date_to  = 0; 

  // iterate over all input files, acquire dates and times for all events
  for (int ifile=0; ifile < (int)opt.infiles.size(); ifile++)
    {
      while(opt.infiles[ifile]->read_event())
	{
	  int yymmdd = 0, hhmmss = 0, usec = 0;
	  if(!opt.infiles[ifile]->get_event_time(&yymmdd,&hhmmss,&usec))
	    continue;
	  event_tstamps.push_back(sdmc_byday_tstamp_class(ifile, opt.infiles[ifile]->GetCurEvent(),yymmdd,hhmmss,usec));
	}
    }
  
  // sort the event times by their time stamps
  sort(event_tstamps.begin(), event_tstamps.end());
  
  // sort out events by date as well
  for (int i = 0; i < (int)event_tstamps.size(); i++)
    {
      // determining the end-points in the sorted array for each date
      if(i <= tstamp_endpts[event_tstamps[i].yymmdd].istart)
	tstamp_endpts[event_tstamps[i].yymmdd].istart = i;
      if(i >= tstamp_endpts[event_tstamps[i].yymmdd].iend)
	tstamp_endpts[event_tstamps[i].yymmdd].iend = i;
    }
  
  // write out the events into separate DST files for each date
  for(map<int, tstamp_endpts_class>::iterator itr=tstamp_endpts.begin(); 
      itr != tstamp_endpts.end(); ++itr)
    {
      int yymmdd = itr->first;
      if(yymmdd <= date_from)
	date_from = yymmdd;
      if(yymmdd >= date_to)
	date_to = yymmdd;
      ndays_total ++; // counting the total number of days

      int istart = itr->second.istart;
      int iend   = itr->second.iend;

      // if the output file has not been initialized for this date, then
      // open a new file
      if(!itr->second.outfile_initialized)
	{
	  // close the previous output file if it has been opened
	  if(ofl)
	    {
	      if(opt.verbose)
		{
		  fprintf(stdout,"file: %s nevents: %d\n", ofl->GetFileName(),ofl->GetNevents());
		  fflush(stdout);
		}
	      delete 
		ofl;
	      ofl  = 0;
	    }
	  
	  // output file name
	  string fname = opt.outdir;
	  fname += opt.prefix;
	  fname += "_";
	  char yymmdd_str[0x16];
	  sprintf(yymmdd_str,"%06d",yymmdd);
	  fname += yymmdd_str;
	  fname += ".dst.gz";
	  
	  // the program exits if it can't open the output file
	  ofl = new sd_dst_handler(fname.c_str(),MODE_WRITE_DST,opt.fOverwrite);
	}
	  
      for (int i = istart; i<= iend; i++)
	{
	  int ifile  = event_tstamps[i].ifile;
	  int ievent = event_tstamps[i].ievent;
	  
	  nevents_total ++; // counting the total number of events
	  
	  if(!opt.infiles[ifile])
	    {
	      fprintf(stderr,"error: (internal) file %d in queue was supposed to be open!\n",ifile);
	      exit(2);
	    }
	  
	  // read the event program exits if can't read the expected event
	  opt.infiles[ifile]->get_event(ievent);
	  
	  // write out the event
	  ofl->SetWriteBanks(opt.infiles[ifile]->GetGotBanks());
	  if(!ofl->write_event())
	    {
	      fprintf(stderr,"error: failed to write out the event; exiting\n");
	      exit(2);
	    }
	}
    }
  
  // finilize output file if it's open
  if(ofl)
    {
      if(opt.verbose)
	{
	  fprintf(stdout,"file: %s nevents: %d\n", ofl->GetFileName(),ofl->GetNevents());
	  fflush(stdout);
	}
      delete
	ofl;
      ofl = 0;
    }

  // close input files that are open
  opt.infiles.clear();

  fprintf(stdout,"ndays_total=%04d nevents_total=%08d date_from=%06d date_to=%06d\n",
	  ndays_total,nevents_total,date_from,date_to);
  fprintf(stdout,"\n\nDone\n");
  return 0;
}


sdmc_byday_listOfOpt::sdmc_byday_listOfOpt()
{
  infiles.clear();
  sprintf(outdir,"./");     // default output directory
  sprintf(prefix,"sdmc");   // default prefix
  fOverwrite = false;       // don't overwrite the output files if they exist by default
  verbose    = false;       // don't print extra stuff by default
}

sdmc_byday_listOfOpt::~sdmc_byday_listOfOpt() 
{
  
}
  

bool sdmc_byday_listOfOpt::parseCmdLine(int argc, char **argv)
{
  if (argc <= 1)
    {
      printMan(argv[0]);
      return false;
    }

  for (int i = 1; i < argc; i++)
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
	  char inBuf[0x400];
	  if ((++i >= argc) || (!argv[i]) || (argv[i][0] == '-'))
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
	      infiles.push_back(new sd_dst_handler(line, MODE_READ_DST));
	    }
	  fclose(fp); 
	}
      // standard input
      else if (strcmp("--tty", argv[i]) == 0)
	{
	  char inBuf[0x400];
	  char* line = 0;
	  while (fgets(inBuf, 0x400, stdin))
	    {
	      if (!((line = strtok(inBuf, " \t\r\n"))) || !strlen(line))
		continue;
	      infiles.push_back(new sd_dst_handler(line, MODE_READ_DST));
	    }
	}
      // output directory
      else if (strcmp("-o", argv[i]) == 0)
	{
	  if ((++i >= argc) || (!argv[i]) || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: specify the output directory!\n");
	      return false;
	    }
	  sscanf(argv[i], "%1023s", outdir);
	  // make sure that the output directory ends with '/'
	  int l = (int)strlen(outdir);
	  if( l >= 0x400)
	    {
	      fprintf(stderr,"error: output directory name is too long\n");
	      return false;
	    }
	  if(outdir[l-1]!='/')
	    {
	      outdir[l] = '/';
	      outdir[l+1] = '\0';
	    }
	}
      // prefix
      else if (strcmp("-p", argv[i]) == 0)
	{
	  if ((++i >= argc) || (!argv[i]) || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: -p: specify the prefix!\n");
	      return false;
	    }
	  sscanf(argv[i], "%1023s", prefix);
	}
      // force overwrite mode
      else if (strcmp("-f", argv[i]) == 0)
	fOverwrite = true;
      // verbose mode
      else if (strcmp("-v", argv[i]) == 0)
	verbose = true;
      // all arguments w/o the '-' switch should be the input files
      else if (argv[i][0] != '-')
	infiles.push_back(new sd_dst_handler(argv[i], MODE_READ_DST));
      else
	{
	  fprintf(stderr, "error: %s: unrecognized option\n", argv[i]);
	  return false;
	}
    }
  if (infiles.size() < 1)
    {
      fprintf(stderr,"error: no input files\n");
      return false;
    }
  if (infiles.size() > N_DST_INFILES_MAX)
    {
      fprintf(stderr,"error: too many input files! maximum number is %d\n",N_DST_INFILES_MAX);
      infiles.clear();
      return false;
    }
  
  return true;
}
 
void sdmc_byday_listOfOpt::printMan(char* progName)
{
  fprintf(stderr,"\nSplit the SD MC into by-day parts.\n");
  fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list_file]  -o [output_directory]\n",progName);
  fprintf(stderr,"pass input dst file names as arguments without any prefixes or switches\n");
  fprintf(stderr, "-i <string>    : specify the want file (with dst files)\n");
  fprintf(stderr, "--tty <string> : or get input dst file names from stdin\n");
  fprintf(stderr, "-o <string>    : output directory, default is '%s'\n",outdir);
  fprintf(stderr, "-p <string>    : output file name prefix, default is '%s ';\n",prefix);
  fprintf(stderr,"output files are of the form prefix_YYMMDD.dst.gz\n");
  fprintf(stderr, "-f             : overwrite the output files if they exist\n");
  fprintf(stderr, "-v             : verbose mode\n");
  fprintf(stderr,"\n");
}
