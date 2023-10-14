//  Time sort sdmc_spctr events in the DST files.  
//  Dmitri Ivanov <dmiivanov@gmail.com>
//  Last modified: 2019-03-09

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <cmath>
#include "event.h"
#include <algorithm>
#include <vector>
#include <map>
#include <filestack.h>
#include "sduti.h"
#include "sddstio.h"
#include <time.h>
#include "dst_size_limits.h"



// maximum number of input DST files that can be safely processed
#define N_DST_INFILES_MAX (MAX_DST_FILE_UNITS - 1)

static size_t sdmc_tsort_buffer_size = 0;

using namespace std;

class sdmc_tsort_listOfOpt
{
public:
  char   dout[0x400];        // output directory
  char   outfile[0x400];     // output file in case all dst output goes to one file
  bool   fOverwrite;         // overwrite the output files if exist
  double mem_usage_gb;       // event buffer memory usage [Gb]
  int    verbosity;          // verbosity mode
  bool getFromCmdLine(int argc, char **argv);
  void printOpts(); // print out the arguments
  void printMan();  // print out the manual
  sdmc_tsort_listOfOpt();
  ~sdmc_tsort_listOfOpt();
private:
  bool checkOpt();  // check & make sure that the options make sense
  char progName[0x400]; // save the program name
};


class sdmc_dst_event
{
public:
  int64_t j2000_usec; // micro second since midnight of January 1, 2000, to use as basis for time sorting
  vector<integer1> rusdmc_buffer;
  vector<integer1> rusdraw_buffer;
  vector<integer1> bsdinfo_buffer;
  sdmc_dst_event(size_t *buffer_to_increment)
  { 
    j2000_usec = 1000000 * (int64_t)SDGEN::time_in_sec_j2000(rusdraw_.yymmdd,rusdraw_.hhmmss)+(int64_t)rusdraw_.usec;    
    rusdmc_common_to_bank_();
    integer4 rusdmc_bank_buffer_size;
    integer1* rusdmc_bank = rusdmc_bank_buffer_(&rusdmc_bank_buffer_size);
    rusdmc_buffer.assign(rusdmc_bank,rusdmc_bank+rusdmc_bank_buffer_size);
    rusdraw_common_to_bank_();
    integer4 rusdraw_bank_buffer_size;
    integer1* rusdraw_bank = rusdraw_bank_buffer_(&rusdraw_bank_buffer_size);
    rusdraw_buffer.assign(rusdraw_bank,rusdraw_bank+rusdraw_bank_buffer_size);
    bsdinfo_common_to_bank_();
    integer4 bsdinfo_bank_buffer_size;
    integer1* bsdinfo_bank = bsdinfo_bank_buffer_(&bsdinfo_bank_buffer_size);
    bsdinfo_buffer.assign(bsdinfo_bank,bsdinfo_bank+bsdinfo_bank_buffer_size);
    (*buffer_to_increment) += (rusdmc_buffer.size() + rusdraw_buffer.size() + bsdinfo_buffer.size());
  }
  void CopyToDSTbanks()
  {
    rusdmc_bank_to_common_((integer1*)&rusdmc_buffer[0]);
    rusdraw_bank_to_common_((integer1*)&rusdraw_buffer[0]);
    bsdinfo_bank_to_common_((integer1*)&bsdinfo_buffer[0]);
  }
  ~sdmc_dst_event() { ; }
};

// compares the times of the two events represented by pointers to sdmc_dst_event objects
static bool cmp_sdmc_dst_event_ptr(sdmc_dst_event const * ev1, sdmc_dst_event const * ev2)
{
  return ev1->j2000_usec < ev2->j2000_usec;
}

static integer4 sdmcBanks; // DST banks that are expected to be found in SDMC DST files

// To time-sort and write out events from the buffer to the DST output file, outBanks
// is the initialized bank list that contains banks that should be written out
static void sortAndWriteOutBufferedEvents(vector<sdmc_dst_event>& sdmc_events, 
					  const char* outfile, sdmc_tsort_listOfOpt* opt)
{
  
  // don't need to do anything if there are no events
  if(sdmc_events.size() < 1)
    return;

  // array of pointers to event time stamps
  vector<sdmc_dst_event*> ev_ptrs;
  for(vector<sdmc_dst_event>::iterator iev = sdmc_events.begin(); iev != sdmc_events.end(); iev++)
    {
      sdmc_dst_event& ev = (*iev);
      ev_ptrs.push_back(&ev);
    }
  // sort the events in time
  sort(ev_ptrs.begin(),ev_ptrs.end(),cmp_sdmc_dst_event_ptr);
  
  // write out the events
  sd_dst_handler* ofl = new sd_dst_handler(outfile,MODE_WRITE_DST,opt->fOverwrite);
  if(opt->verbosity >=1)
    {
      fprintf(stdout,"Started DST file %s\n",outfile);
      fflush(stdout);
    }
  for(vector<sdmc_dst_event*>::iterator iev = ev_ptrs.begin(); iev != ev_ptrs.end(); iev++)
    {
      sdmc_dst_event& ev = (*(*iev));
      ev.CopyToDSTbanks();
      ofl->SetWriteBanks(sdmcBanks);
      if(!ofl->write_event())
	{
	  fprintf(stderr,"error: failed to write out the event; exiting\n");
	  exit(2);
	}
    }
  delete ofl;
  ev_ptrs.clear();
  sdmc_events.clear();
  sdmc_tsort_buffer_size = 0;
}


class sdmc_cat_tsort_tstamp_class
{
public:
  int    ifile;    // input dst file number
  int    ievent;   // event number in the dst file
  int    yymmdd;   // date of the event
  double t2000s;   // time of the event (include second fraction) since midnight of Jan 1, 2000
  sdmc_cat_tsort_tstamp_class(int file_num, int event_num, int event_yymmdd, int event_hhmmss, int usec)
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
  virtual ~sdmc_cat_tsort_tstamp_class() { ; }
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
bool operator <(const sdmc_cat_tsort_tstamp_class & left, const sdmc_cat_tsort_tstamp_class& right)
{
  return (left.t2000s < right.t2000s);
}

// combine and time sort the temporary DST files
void sdmc_cat_and_tsort(vector<string>& tmp_files, const char* outfile, sdmc_tsort_listOfOpt* opt)
{
  // hold the time stamp for every event encountered
  vector<sdmc_cat_tsort_tstamp_class> event_tstamps;
  // initialize points for the input DST file handlers
  sd_dst_handler *ofl   = 0; // output file
  vector<sd_dst_handler*> infiles; // input files as DST handlers
  // iterate over all input files, acquire dates and times for all events
  for (int ifile=0; ifile < (int)tmp_files.size(); ifile++)
    {
      infiles.push_back(new sd_dst_handler(tmp_files[ifile].c_str(), MODE_READ_DST));
      if(opt->verbosity >=1)
	{
	  fprintf(stdout,"Opened DST file %s\n",tmp_files[ifile].c_str());
	  fflush(stdout);
	}
      while(infiles[ifile]->read_event())
	{
	  int yymmdd = 0, hhmmss = 0, usec = 0;
	  if(!infiles[ifile]->get_event_time(&yymmdd,&hhmmss,&usec))
	    continue;
	  event_tstamps.push_back(sdmc_cat_tsort_tstamp_class(ifile, infiles[ifile]->GetCurEvent(),yymmdd,hhmmss,usec));
	}
    }
  // sort the event times by their time stamps
  sort(event_tstamps.begin(), event_tstamps.end());
  // write out the events into separate DST files for each date
  for(vector<sdmc_cat_tsort_tstamp_class>::iterator itr=event_tstamps.begin(); itr != event_tstamps.end(); ++itr)
    {
      const sdmc_cat_tsort_tstamp_class& tstamp = (*itr);
      int ifile  = tstamp.ifile;
      int ievent = tstamp.ievent;
      if(!infiles[ifile])
	{
	  fprintf(stderr,"error: (internal) file %d in queue was supposed to be open!\n",ifile);
	  exit(2);
	}
      // read the event program exits if can't read the expected event
      infiles[ifile]->get_event(ievent);  
      // write out the event
      // if the output file has not been initialized then start the DST output file
      // the program exits if it can't open the output file
      if(!ofl)
	{
	  ofl = new sd_dst_handler(outfile,MODE_WRITE_DST,opt->fOverwrite);
	  if(opt->verbosity >=1)
	    {
	      fprintf(stdout,"Started DST file %s\n",outfile);
	      fflush(stdout);
	    }
	}
      ofl->SetWriteBanks(infiles[ifile]->GetGotBanks());
      if(!ofl->write_event())
	{
	  fprintf(stderr,"error: failed to write out the event; exiting\n");
	  exit(2);
	}
    }
  // finilize output file if it's open
  if(ofl)
    {
      delete ofl;
      ofl = 0;
    }
  // close all input files that are open
  infiles.clear();
}

// remove temporary files and clear out the array of temporary file names
static void remove_tmp_files(vector<string>& tmp_files, sdmc_tsort_listOfOpt* opt)
{
  (void)(opt);
  for (vector<string>::iterator itr=tmp_files.begin(); itr != tmp_files.end(); ++itr)
    {
      const string& fname = (*itr);
      if(remove(fname.c_str()))
	exit(2);
      fprintf(stdout,"Removed DST file %s\n",fname.c_str());
    }
  tmp_files.clear();
}


int main(int argc, char **argv)
{
  sdmc_tsort_listOfOpt *opt = new sdmc_tsort_listOfOpt();
  if (!opt->getFromCmdLine(argc, argv))
    return 2;
  opt->printOpts();
  char* infile = 0;
  sddstio_class* dstio = new sddstio_class(opt->verbosity);
  sdmcBanks = newBankList(10);
  addBankList(sdmcBanks,RUSDMC_BANKID);
  addBankList(sdmcBanks,RUSDRAW_BANKID);
  addBankList(sdmcBanks,BSDINFO_BANKID);
  
 
  // events from the DST file (s)
  vector<sdmc_dst_event> all_events;  
  vector<sdmc_dst_event* > ev_ptrs;

  // temporary files if the buffers become too large
  vector<string> tmp_files;
  
  char* outfile = 0;
  if(opt->outfile[0])
    outfile = opt->outfile;
  else
    outfile = new char[0x400];

  while((infile=pullFile()))
    {
      
      // have an output file name ready in case if we are generating
      // output files automatically
      if(!opt->outfile[0])
	{
	  if (SDIO::makeOutFileName(infile,opt->dout,(char *)".tsorted.dst.gz",outfile) != 1)
	    return 2;
	}
      
      // open the input DST file
      if(!dstio->openDSTinFile(infile))
	return 2;
      
      int nevents_file_read = 0;
      
      while(dstio->readEvent())
	{
	  // warn in case some DST banks are missing
	  if(dstio->haveBanks(sdmcBanks,(bool)(opt->verbosity>=1)))
	    {
	      if(opt->verbosity>=2)
		fprintf(stderr,"Some expected banks were missing\n");
	    } 
	  // store the event
	  all_events.push_back(sdmc_dst_event(&sdmc_tsort_buffer_size));
	  nevents_file_read++;
	  
	  // if exceeding the buffer size
	  // start a temporary file and write out the events
	  if((double)sdmc_tsort_buffer_size/1.073741824e9 > opt->mem_usage_gb)
	    {
	      if(tmp_files.size() > N_DST_INFILES_MAX)
		{
		  fprintf(stderr,"Too many temporary files (maximum is %d); reduce the sample size to under %d * %.1e = %.1e Gb!\n",
			  N_DST_INFILES_MAX,N_DST_INFILES_MAX,opt->mem_usage_gb,(double)N_DST_INFILES_MAX*opt->mem_usage_gb);
		  exit(2);
		}
	      string tmp_file = outfile;
	      char num[5];
	      sprintf(num,"_%03d",(int)tmp_files.size());
	      tmp_file = tmp_file + num;
	      tmp_file = tmp_file + ".dst";
	      tmp_files.push_back(tmp_file);
	      sortAndWriteOutBufferedEvents(all_events,tmp_file.c_str(),opt);
	    }
	}
      
      // close the DST input file
      dstio->closeDSTinFile();
      if(opt->verbosity >=1)
	{
	  fprintf(stdout,"input file %s %d events read\n",infile,nevents_file_read);
	  fflush(stdout);
	}
      
      // if generating file names automatically and sorting each input file separately
      if(!opt->outfile[0])
	{
	  // if there are temporary files, then use a routine that combines them into the output file
	  if(tmp_files.size() > 1)
	    {
	      // sort and write out into another temporary file
	      // any events that are still in the buffer
	      // before combining and sorting the temporary files
	      if(tmp_files.size() > N_DST_INFILES_MAX)
		{
		  fprintf(stderr,"Too many temporary files (maximum is %d); reduce the sample size to under %d * %.1e = %.1e Gb!\n",
			  N_DST_INFILES_MAX,N_DST_INFILES_MAX,opt->mem_usage_gb,(double)N_DST_INFILES_MAX*opt->mem_usage_gb);
		  exit(2);
		}
	      string tmp_file = outfile;
	      char num[5];
	      sprintf(num,"_%03d",(int)tmp_files.size());
	      tmp_file = tmp_file + num;
	      tmp_file = tmp_file + ".dst";
	      tmp_files.push_back(tmp_file);
	      sortAndWriteOutBufferedEvents(all_events,tmp_file.c_str(),opt);
	      sdmc_cat_and_tsort(tmp_files,outfile,opt);
	      remove_tmp_files(tmp_files,opt);
	    }
	  // otherwise, just sort and write out the buffered events
	  else
	    sortAndWriteOutBufferedEvents(all_events,outfile,opt);
	}
    }
  // if writing all events into a single output DST file
  if(opt->outfile[0])
    {
      // if need to combine temporary files
      if(tmp_files.size() > 1)
	{
	  // sort and write out into another temporary file
	  // any events that are still in the buffer
	  // before combining and sorting the temporary files
	  if(tmp_files.size() > N_DST_INFILES_MAX)
	    {
	      fprintf(stderr,"Too many temporary files (maximum is %d); reduce the sample size to under %d * %.1e = %.1e Gb!\n",
		      N_DST_INFILES_MAX,N_DST_INFILES_MAX,opt->mem_usage_gb,(double)N_DST_INFILES_MAX*opt->mem_usage_gb);
	      exit(2);
	    }
	  string tmp_file = outfile;
	  char num[5];
	  sprintf(num,"_%03d",(int)tmp_files.size());
	  tmp_file = tmp_file + num;
	  tmp_file = tmp_file + ".dst";
	  tmp_files.push_back(tmp_file);
	  sortAndWriteOutBufferedEvents(all_events,tmp_file.c_str(),opt);
	  sdmc_cat_and_tsort(tmp_files,opt->outfile,opt);
	  remove_tmp_files(tmp_files,opt);
	}
      // otherwise, just write out the buffered events
      else
	sortAndWriteOutBufferedEvents(all_events,opt->outfile,opt);
    }
  // clean up automatic file name buffer
  else
    delete[] outfile;
  fprintf(stdout,"\nDone\n");
  return 0;
}


sdmc_tsort_listOfOpt::sdmc_tsort_listOfOpt()
{
  dout[0]          =  0;
  outfile[0]       =  0;
  fOverwrite   =  false;
  mem_usage_gb     = 2.0;
  verbosity        =  1;
}


sdmc_tsort_listOfOpt::~sdmc_tsort_listOfOpt() 
{
  
}

bool sdmc_tsort_listOfOpt::getFromCmdLine(int argc, char **argv)
{
  int i;
  char inBuf[0x400];
  char *line;
    
  sprintf(progName,"%s",argv[0]);
  
  
  // if no arguments
  if (argc <= 1)
    {
      printMan();
      return false;
    }
  
  
  // go over the arguments if there are command line arguments
  for (i = 1; i < argc; i++)
    {

      // print the manual
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
	  printMan();
	  return false;
	}
        
      // input dst file names from a list file
      else if (strcmp("-i", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      fprintf(stderr,"error: -i: specify the list file!\n");
	      return false;
	    }
	  else
	    {
	      FILE *fp = 0; // For reading the list file
	      if (!(fp = fopen(argv[i], "r")))
		{
		  fprintf(stderr, "error: -i: can't read %s\n", argv[i]);
		  return false;
		}
	      else
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
	    }
	}
      
      // input dst file names from stdin
      else if (strcmp("--tty", argv[i]) == 0)
	{
	  while (fgets(inBuf, 0x400, stdin))
	    {
	      if (((line = strtok(inBuf, " \t\r\n"))) && (strlen(line) > 0))
		{
		  if (pushFile(line) != SUCCESS)
		    return false;
		}
	    }
	}
      
      // output directory name
      else if (strcmp("-o", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: -o: specify the output directory!\n");
	      return false;
	    }
	  sscanf(argv[i], "%1023s", dout);
	}
        
      // single output file name
      else if (strcmp("-o1f", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: -o1f: specify the output file!\n");
	      return false;
	    }
	  sscanf(argv[i], "%1023s", outfile);
	}
      // force overwrite mode
      else if (strcmp("-f", argv[i]) == 0)
	fOverwrite = true;
      // event buffer size in Gb
      else if (strcmp("-m", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: -m: specify the buffer size in [Gb]\n");
	      return false;
	    }
	  sscanf(argv[i], "%lf", &mem_usage_gb);
	} 
      // verbosity mode
      else if (strcmp("-v", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	    {
	      fprintf(stderr, "error: -v: specify verbosity level\n");
	      return false;
	    }
	  sscanf(argv[i], "%d", &verbosity);
	}        
      // assume that all arguments w/o the '-' switch are input dst file names
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

  return checkOpt();
}
void sdmc_tsort_listOfOpt::printOpts()
{
  time_t now;
  struct tm *d;
  char cur_date_time[0x100];
  time(&now);
  d = localtime(&now);
  strftime(cur_date_time,255,"%Y-%m-%d %H:%M:%S %Z", d);
  fprintf(stdout,"\n\n");
  fprintf(stdout,"%s (%s):\n",progName,cur_date_time);
  if(outfile[0])
    fprintf(stdout,"OUTPUT FILE: %s\n",outfile);
  else
    fprintf(stdout, "OUTPUT DIRECTORY: %s\n", dout);
  fprintf(stdout, "OVERWRITING THE OUTPUT FILES IF EXIST: %s\n",(fOverwrite? "YES" : "NO"));
  fprintf(stdout, "MEMORY BUFFER SIZE [Gb]: %.1e\n",mem_usage_gb);
  fprintf(stdout, "VERBOSITY LEVEL: %d\n",verbosity);
  fprintf(stdout,"\n\n");
  fflush(stdout);
}

bool sdmc_tsort_listOfOpt::checkOpt()
{

  if (!dout[0])
    sprintf(dout, "./");

  // if the output file does not have a reasonable DST suffix,
  // then report a problem
  if (outfile[0] && !(SDIO::check_dst_suffix(outfile)))
    return false;
  if (countFiles() == 0)
    {
      fprintf(stderr, "error: don't have any inputs dst files!\n");
      return false;
    }
  
  
  return true;
}
void sdmc_tsort_listOfOpt::printMan()
{
  fprintf(stderr,"\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr,"Time-sort events from the DST files produced by sdmc_spctr. Events are expected to have\n");
  fprintf(stderr,"rusdmc,rusdraw,bsdinfo DST banks. This program should be executed after sdmc_spctr.\n");
  fprintf(stderr,"Author: Dmitri Ivanov <dmiivanov@gmail.com>\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr,"\nUsage: %s [in_file1 ...] and/or -i [list file]  -o [output directory]\n",progName);
  fprintf(stderr, "INPUT OPTIONS:\n");
  fprintf(stderr,"Pass input dst file names as arguments without any prefixes\n");
  fprintf(stderr, "-i         <string>  : or give an input list file (with full paths to DST files)\n");
  fprintf(stderr, "--tty                : or pipe input DST file names through stdin\n");
  fprintf(stderr, "OUTPUT OPTIONS:\n");
  fprintf(stderr, "-o         <string>  : Output directory (default is './')\n");
  fprintf(stderr, "-o1f       <string>  : Single output file mode.  All output goes to one file. Overrides the '-o' option\n");
  fprintf(stderr, "OTHER:\n");
  fprintf(stderr, "-m          <float>  : (Optional) Memory for the event buffer [Gb], default is %.1e \n",mem_usage_gb);
  fprintf(stderr, "-v          <int>    : (Optional) Verbosity level, default %d\n",verbosity);
  fprintf(stderr, "-f                   : (Optional) Don't check if output files exist, overwrite them\n\n");
}
