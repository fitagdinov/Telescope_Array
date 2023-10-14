//  Time match events from the dst files.  
//  Dmitri Ivanov <dmiivanov@gmail.com>
//  Last modified: 2020-08-27

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include "event.h"
#include "sddstio.h"
#include <algorithm>
#include <vector>
#include <map>
#include <filestack.h>
#include "sduti.h"
#include <time.h>
#include "icrr2ru.h"
#include "sdxyzclf_class.h"

using namespace std;


#define SDBSEARCH_USEC 1000               // default micrso second in time matching burst events
#define USEC_IN_SEC 1000000               // micro seconds in a second

class listOfOpt
{
public:
  char wantFile [0x400];   // list file with DST files
  char dout[0x400];        // output directory
  char outfile[0x400];     // output file in case all dst output goes to one file
  bool fOverwriteMode;     // overwrite the output files if exist
  int  usec;               // micro second for event time matching
  int  verbosity;          // verbosity mode
  bool getFromCmdLine(int argc, char **argv);
  void printOpts(); // print out the arguments
  void printMan();  // print out the manual
  listOfOpt();
  ~listOfOpt();
private:
  bool checkOpt();  // check & make sure that the options make sense
  char progName[0x400]; // save the program name
};




class ev_tstamp
{
public:
  int64_t j2000_usec;  // micro second since midnight of January 1, 2000
  int      ievent;      // event index in the DST file (from 0 to NEVENTS-1)
  ev_tstamp() 
  {
    j2000_usec = -1;
    ievent     = -1;
  }
  ~ev_tstamp() { ; }
};

// compares time of the two tubes
static bool cmp_ev_tstamp_ptr(ev_tstamp const * ev1, ev_tstamp const * ev2)
{
  return ev1->j2000_usec < ev2->j2000_usec;
}

// to merge rusdraw DST banks for the same event
static bool addEvent(rusdraw_dst_common *comevent,rusdraw_dst_common *event);

// to sort rusdraw event waveforms after merging
static void sort_event_wfms(rusdraw_dst_common* event_to_sort);

int main(int argc, char **argv)
{
  
  listOfOpt *opt = new listOfOpt();
  if (!opt->getFromCmdLine(argc, argv))
    return 2;
  opt->printOpts();
  sdxyzclf_class sdbsearch_sdxyzclf;    // settings specific to the main TASD array
  char* infile = 0;
  sd_dst_handler* ofl = 0;
  integer4 outBanks = newBankList(10);
  integer4 shwBanks = newBankList(10);
  addBankList(outBanks,RUSDRAW_BANKID); // banks to write; rusdraw supports combined burst events more easily
  addBankList(shwBanks,RUSDRAW_BANKID); // banks to print if verbosity level is high enough
  icrr2ru* icr2r   = new icrr2ru;
  icr2r->reset_event_num();
  while((infile=pullFile()))
    {
      vector<ev_tstamp> all_events;
      // read all time stamps from the DST file
      sd_dst_handler* ifl = new sd_dst_handler(infile, MODE_READ_DST);
      while(ifl->read_event())
	{ 
	  if(!tstBankList(ifl->GetGotBanks(),RUSDRAW_BANKID) && 
	     !(ifl->GetGotBanks(),TALEX00_BANKID) && 
	     !tstBankList(ifl->GetGotBanks(),TASDCALIBEV_BANKID))
	    {
	      fprintf(stderr,"WARNING: NO RUSDRAW, or TALEX00, or TASDCALIBEV BANKS in event %d; ignoring the event\n",ifl->GetCurEvent());
	      continue;
	    }
	  if(!tstBankList(ifl->GetGotBanks(),RUSDRAW_BANKID) && tstBankList(ifl->GetGotBanks(), TALEX00_BANKID))
	      sdbsearch_sdxyzclf.talex00_2_rusdraw(&talex00_,&rusdraw_);
	  if(!tstBankList(ifl->GetGotBanks(),RUSDRAW_BANKID) && !tstBankList(ifl->GetGotBanks(),TALEX00_BANKID) && 
	     tstBankList(ifl->GetGotBanks(),TASDCALIBEV_BANKID))
	    {
	      if(!icr2r->Convert())
		{
		  fprintf(stderr,"WARNING: failed to convert TASDCALIBEV to RUSDRAW in event %d; ignoring the event\n",ifl->GetCurEvent());
		  continue;
		}
	    }

	  // Skip the event if it has no waveforms at all
	  if(rusdraw_.nofwf < 1 )
	    continue;
	    
	  int yymmdd,hhmmss,usec;
	  if(!ifl->get_event_time(&yymmdd,&hhmmss,&usec))
	    {
	      fprintf(stderr,"WARNING: failed to get time for event %d\n",ifl->GetCurEvent());
	      continue;
	    }
	  ev_tstamp ev;
	  ev.j2000_usec = USEC_IN_SEC * (int64_t)SDGEN::time_in_sec_j2000(yymmdd,hhmmss)+(int64_t)usec;
	  ev.ievent = ifl->GetCurEvent();
	  all_events.push_back(ev);
	}
      
      // array of pointers to event time stamps
      vector<ev_tstamp* > ev_ptrs;
      for(vector<ev_tstamp>::iterator iev = all_events.begin(); iev != all_events.end(); iev++)
	{
	  ev_tstamp& ev = (*iev);
	  ev_ptrs.push_back(&ev);
	}
      // sort the event times
      sort(ev_ptrs.begin(),ev_ptrs.end(),cmp_ev_tstamp_ptr);
      // pick out events that are burst events, if any
      vector<vector<ev_tstamp*> > sdb_events;
      
      for (int i=0; i<(int)ev_ptrs.size(); i++)
	{
	  const ev_tstamp& ev1 =  (*ev_ptrs[i]);
	  vector<ev_tstamp*> sdb;
	  sdb.push_back(ev_ptrs[i]);
	  for (int j=i+1; j<(int)ev_ptrs.size(); j++)
	    {
	      const ev_tstamp& ev2 =  (*ev_ptrs[j]);
	      if(abs(ev1.j2000_usec-ev2.j2000_usec) <= opt->usec)
		{
		  sdb.push_back(ev_ptrs[j]);
		  ev_ptrs.erase(ev_ptrs.begin()+j);
		  j --;
		}
	    }
	  if(sdb.size() > 1)
	    sdb_events.push_back(sdb);
	  sdb.clear();
	}
      if(opt->verbosity >=1)
	{
	  fprintf(stdout,"file %s %d potential burst events found\n",infile,(int)sdb_events.size());
	  fflush(stdout);
	}
      if(sdb_events.size())
	{
	  icr2r->reset_event_num(); // will be reading again and possibly converting between TASDCALIBEV and RUSDRAW
	  if(!ofl)
	    {
	      if (opt->outfile[0])
		{
		  if(!(ofl=new sd_dst_handler(opt->outfile,MODE_WRITE_DST,opt->fOverwriteMode)))
		    return -1;
		}
	      else
		{
		  char *outfile = new char[0x400];
		  if (SDIO::makeOutFileName(infile,opt->dout,(char *)".sdbsearch.dst.gz",outfile) != 1)
		    return -1;
		  if(!(ofl=new sd_dst_handler(outfile, MODE_WRITE_DST, opt->fOverwriteMode)))
		    return -1;
		  delete[] outfile;
		}
	      ofl->SetWriteBanks(outBanks);
	    }
	  
	  for (int iburst = 0; iburst < (int)sdb_events.size(); iburst++)
	    {
	      if(sdb_events[iburst].size() < 2)
		{
		  fprintf(stderr,"ERROR: events labeled as burst but there are less than 2 of them\n");
		  exit(2);
		}
	      sort(sdb_events[iburst].begin(),sdb_events[iburst].end(),cmp_ev_tstamp_ptr);
	      rusdraw_dst_common combined_event;
	      combined_event.event_num = -1;
	      combined_event.site      = 0;
	      combined_event.nofwf     = 0;
	      for (int i=0; i < (int)sdb_events[iburst].size(); i++)
		{
		  ifl->get_event(sdb_events[iburst][i]->ievent);
		  if(!tstBankList(ifl->GetGotBanks(),RUSDRAW_BANKID) && tstBankList(ifl->GetGotBanks(),TASDCALIBEV_BANKID))
		    {
		      if(!icr2r->Convert())
			{
			  fprintf(stderr,"ERROR: failed to convert TASDCALIBEV to RUSDRAW in event %d; was able to do that before\n",ifl->GetCurEvent());
			  exit(2);
			}
		    }
		  if(opt->verbosity >= 1)
		    {
		      fprintf(stdout,"potential burst %d, %d fold, event %d, %d waveforms, time %06d %06d.%06d\n", 
			      iburst,(int)sdb_events[iburst].size(),i,rusdraw_.nofwf,
			      rusdraw_.yymmdd,rusdraw_.hhmmss,rusdraw_.usec);
		      fflush(stdout);
		    }
		  if(opt->verbosity >= 2)
		    {
		      eventSetDumpFormat(shwBanks,0);
		      if(opt->verbosity >= 3)
			eventSetDumpFormat(shwBanks,1);
		      eventDumpf(stdout,shwBanks);
		    }
		  if(!addEvent(&combined_event,&rusdraw_))
		    {
		      fprintf(stderr,"ERROR: failed to combine burst events\n");
		      return -1;
		    }
		}
	      sort_event_wfms(&combined_event);
	      memcpy(&rusdraw_,&combined_event,sizeof(rusdraw_dst_common));
	      if(!ofl->write_event())
		return -1;
	    }
	}
      delete ifl;
      if(!opt->outfile[0] && ofl)
	{
	  delete ofl;
	  ofl = 0;
	}
    }
  if(ofl)
    delete ofl;
  fprintf(stdout,"\nDone\n");
}

listOfOpt::listOfOpt()
{
  wantFile[0]      =  0;
  dout[0]          =  0;
  outfile[0]       =  0;
  fOverwriteMode   =  false;
  usec             =  SDBSEARCH_USEC;
  verbosity        =  1;
}


listOfOpt::~listOfOpt() 
{
  
}

bool listOfOpt::getFromCmdLine(int argc, char **argv)
{
  int i;
  char inBuf[0x400];
  char *line;
  FILE *wantFl; // For reading the want file
    
  sprintf(progName,"%s",argv[0]);
  
  // if no arguments and nothing on stdin then print the manual and return
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
              fprintf(stderr,"error: -i: specify the want file!\n");
              return false;
            }
          else
            {
              sscanf(argv[i], "%1023s", wantFile);
              // read the want files, put all rusdraw dst files found into a buffer.
              if ((wantFl = fopen(wantFile, "r")) == NULL)
                {
                  fprintf(stderr, "error: -i: can't read %s\n", wantFile);
                  return false;
                }
              else
                {
                  while (fgets(inBuf, 0x400, wantFl))
                    {
                      if (((line = strtok(inBuf, " \t\r\n")))
                          && (strlen(line) > 0))
                        {
                          if (pushFile(line) != SUCCESS)
                            return false;
                        }
                    }
                  fclose(wantFl);
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
        fOverwriteMode = true;
      // micro second
      else if (strcmp("-us", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              fprintf(stderr, "error: -us: specify the micro second window for matching\n");
              return false;
            }
          sscanf(argv[i], "%d", &usec);
        }
   // verbosity mode
      else if (strcmp("-v", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              fprintf(stderr, "error: -v: specify verbosity mode\n");
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
void listOfOpt::printOpts()
{
  time_t now;
  struct tm *d;
  char cur_date_time[0x100];
  const char noy_str[2][10]={"NO","YES"};
  time(&now);
  d = localtime(&now);
  strftime(cur_date_time,255,"%Y-%m-%d %H:%M:%S %Z", d);
  fprintf(stdout,"\n\n");
  fprintf(stdout,"%s (%s):\n",progName,cur_date_time);
  if (wantFile[0])
    fprintf(stdout, "WANT FILE: %s\n", wantFile);
  if(outfile[0])
    fprintf(stdout,"OUTPUT FILE: %s\n",outfile);
  else
    fprintf(stdout, "OUTPUT DIRECTORY: %s\n", dout);
  fprintf(stdout, "OVERWRITING THE OUTPUT FILES IF EXIST: %s\n",noy_str[(int)fOverwriteMode]);
  fprintf(stdout, "MICRO SECOND WINDOW FOR MATCHING BURST EVENTS: %d\n",usec);
  fprintf(stdout, "VERBOSITY LEVEL: %d\n",verbosity);
  fprintf(stdout,"\n\n");
  fflush(stdout);
}

bool listOfOpt::checkOpt()
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
void listOfOpt::printMan()
{
  fprintf(stderr,"\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr,"Find and merge together repeated SD events. Input events must have rusdraw and / or tasdcalibev DST banks.\n");
  fprintf(stderr,"This program should be called after pass0 and before any reconstruction\n");
  fprintf(stderr,"Author: Dmitri Ivanov <dmiivanov@gmail.com>\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr,"\nUsage: %s [in_file1 ...] and/or -i [list file]  -o [output directory]\n",progName);
  fprintf(stderr, "INPUT OPTIONS:\n");
  fprintf(stderr,"Pass input dst file names as arguments without any prefixes\n");
  fprintf(stderr, "-i         <string>  : or specify the input list file (with dst file name paths)\n");
  fprintf(stderr, "--tty                : or pipe input DST file names through stdin\n");
  fprintf(stderr, "OUTPUT OPTIONS:\n");
  fprintf(stderr, "-o         <string>  : Output directory (default is './')\n");
  fprintf(stderr, "-o1f       <string>  : Single output file mode.  All output goes to one file. Overrides the '-o' option\n");
  fprintf(stderr, "OTHER:\n");
  fprintf(stderr, "-us         <int>    : (Optional) Micro second time window for matching burst events, default %d\n",usec);
  fprintf(stderr, "-v          <int>    : (Optional) Verbosity level, default %d\n",verbosity);
  fprintf(stderr, "-f                   : (Optional) Don't check if output files exist, overwrite them\n\n");
}


static bool addTowerID(int *com_tid, int tid)
{
  static bool first_time = true;
  static int tower_bit_flags[RUSDRAW_BRLRSK+1];
  if(first_time)
    {
      tower_bit_flags[RUSDRAW_BR]     = 1;  // BR        [001]
      tower_bit_flags[RUSDRAW_LR]     = 2;  // LR        [010]
      tower_bit_flags[RUSDRAW_SK]     = 4;  // SK        [100]
      tower_bit_flags[RUSDRAW_BRLR]   = 3;  // BR LR     [011]
      tower_bit_flags[RUSDRAW_BRSK]   = 5;  // BR SK     [101]
      tower_bit_flags[RUSDRAW_LRSK]   = 6;  // LR SK     [110]
      tower_bit_flags[RUSDRAW_BRLRSK] = 7;  // BR LR SK  [111]
      first_time = false;
    }
  // If tower ID can't be added
  if ((tid < 0) || (tid > RUSDRAW_BRLRSK))
    {
      (*com_tid) = -1;
      fprintf(stderr,"bad tower(s) id %d!\n", tid);
      return false;
    }
  if((*com_tid < 0) || (*com_tid > RUSDRAW_BRLRSK))
    {
      fprintf(stderr,"ERROR: %s(%d): combined tower ID has unreasonable value: %d\n",
	      __FILE__,__LINE__,*com_tid);
      (*com_tid) = -1;
      return false;
    }
  // combine the bit flag and search for the index in the array
  int combined_tower_bit_flag = (tower_bit_flags[(*com_tid)] | tower_bit_flags[tid]);
  for (int i=0; i <= RUSDRAW_BRLRSK; i++)
    {
      if(tower_bit_flags[i] == combined_tower_bit_flag)
	{
	  (*com_tid) = i;
	  return true;
	}
    }
  fprintf(stderr,"error: tower IDs not readonable: %d %d\n", (*com_tid), tid);
  (*com_tid) = -1;
  return false;
}


bool addEvent(rusdraw_dst_common *comevent,rusdraw_dst_common *event)
{

  int iwfmi, iwfma, iwf, j, k, itower;
  itower=event->site;
  if (!addTowerID(&comevent->site, itower))
    return false;
  if (comevent->event_num == -1 || comevent->nofwf < 1)
    {
      comevent->event_num = event->event_num;
      comevent->event_code = event->event_code;
      comevent->errcode = event->errcode;
      comevent->yymmdd = event->yymmdd;
      comevent->hhmmss = event->hhmmss;
      comevent->usec = event->usec;
      comevent->monyymmdd = event->monyymmdd;
      comevent->monhhmmss = event->monhhmmss;
      comevent->nofwf = 0;
    }
  else
    comevent->errcode += event->errcode;
  if(itower >= RUSDRAW_BR && itower <= RUSDRAW_SK)
    {
      comevent->run_id[itower] = event->run_id[itower];
      comevent->trig_id[itower] = event->trig_id[itower];
    }
  iwfmi=comevent->nofwf;
  comevent->nofwf += event->nofwf;
  iwfma=iwfmi+event->nofwf-1;
  if (iwfma >= RUSDRAWMWF)
    {
      fprintf(stderr,"addEvent: Too many waveforms: current %d maximum %d\n", iwfma
	      +1, RUSDRAWMWF);
      comevent->nofwf = RUSDRAWMWF;
      return false;
    }
  int sec1 = SDGEN::time_in_sec_j2000(comevent->yymmdd,comevent->hhmmss);
  int sec2 = SDGEN::time_in_sec_j2000(event->yymmdd,event->hhmmss);
  for (iwf=iwfmi; iwf<=iwfma; iwf++)
    {
      comevent->nretry[iwf] = event->nretry[iwf-iwfmi];
      comevent->wf_id[iwf] = event->wf_id[iwf-iwfmi];
      comevent->trig_code[iwf] = event->trig_code[iwf-iwfmi];
      comevent->xxyy[iwf] = event->xxyy[iwf-iwfmi];
      int clkcnt_offset = (sec2 > sec1 ? (sec2-sec1) * event->mclkcnt[iwf-iwfmi] : 0);
      comevent->clkcnt[iwf] = clkcnt_offset + event->clkcnt[iwf-iwfmi];
      comevent->mclkcnt[iwf] = event->mclkcnt[iwf-iwfmi];
      for (j=0; j<2; j++)
	{
	  comevent->fadcti[iwf][j] = event->fadcti[iwf-iwfmi][j];
	  comevent->fadcav[iwf][j] = event->fadcav[iwf-iwfmi][j];
	  for (k=0; k<rusdraw_nchan_sd; k++)
	    comevent->fadc[iwf][j][k] = event->fadc[iwf-iwfmi][j][k];
	  comevent->pchmip[iwf][j] = event->pchmip[iwf-iwfmi][j];
	  comevent->pchped[iwf][j] = event->pchped[iwf-iwfmi][j];
	  comevent->lhpchmip[iwf][j] = event->lhpchmip[iwf-iwfmi][j];
	  comevent->lhpchped[iwf][j] = event->lhpchped[iwf-iwfmi][j];
	  comevent->rhpchmip[iwf][j] = event->rhpchmip[iwf-iwfmi][j];
	  comevent->rhpchped[iwf][j] = event->rhpchped[iwf-iwfmi][j];
	  comevent->mftndof[iwf][j] = event->mftndof[iwf-iwfmi][j];
	  comevent->mip[iwf][j] = event->mip[iwf-iwfmi][j];
	  comevent->mftchi2[iwf][j] = event->mftchi2[iwf-iwfmi][j];
	  for (k=0; k<4; k++)
	    {
	      comevent->mftp[iwf][j][k] = event->mftp[iwf-iwfmi][j][k];
	      comevent->mftpe[iwf][j][k] = event->mftpe[iwf-iwfmi][j][k];
	    }
	}
    }

  return true;
}

static rusdraw_dst_common _event_for_sorting_;

static bool cmp_wf_by_time(int const & i1, int const & i2)
{
  return (_event_for_sorting_.clkcnt[i1] < _event_for_sorting_.clkcnt[i2]);
}

static bool cmp_counters_by_time(vector<int> const & v1, vector<int> const & v2)
{
  return (_event_for_sorting_.clkcnt[v1[0]] < _event_for_sorting_.clkcnt[v2[0]]);
}


void sort_event_wfms(rusdraw_dst_common* event_to_sort)
{ 
  rusdraw_dst_common* event_for_sorting = &_event_for_sorting_;
  map<int,vector<int> > m_wfms;
  vector<vector<int> > wfms;
  memcpy(event_for_sorting,event_to_sort,sizeof(rusdraw_dst_common));
  for (int i=0; i<event_to_sort->nofwf; i++)
    m_wfms[event_to_sort->xxyy[i]].push_back(i);
  for (map<int,vector<int> >::iterator it=m_wfms.begin(); it!=m_wfms.end(); it++)
    wfms.push_back(it->second);
  for (int isd=0; isd < (int)wfms.size(); isd++)
    sort(wfms[isd].begin(),wfms[isd].end(),cmp_wf_by_time);
  sort(wfms.begin(),wfms.end(),cmp_counters_by_time);
  int iwf = 0;
  for (int isd = 0; isd < (int)wfms.size(); isd++)
    {
      for (int isdwf = 0; isdwf < (int)wfms[isd].size(); isdwf++)
	{
	  int jwf = wfms[isd][isdwf];
	  event_to_sort->nretry[iwf] = event_for_sorting->nretry[jwf];
	  event_to_sort->wf_id[iwf] = event_for_sorting->wf_id[jwf];
	  event_to_sort->trig_code[iwf] = event_for_sorting->trig_code[jwf];
	  event_to_sort->xxyy[iwf] = event_for_sorting->xxyy[jwf];
	  event_to_sort->clkcnt[iwf] = event_for_sorting->clkcnt[jwf];
	  event_to_sort->mclkcnt[iwf] = event_for_sorting->mclkcnt[jwf];
	  for (int j=0; j<2; j++)
	    {
	      event_to_sort->fadcti[iwf][j] = event_for_sorting->fadcti[jwf][j];
	      event_to_sort->fadcav[iwf][j] = event_for_sorting->fadcav[jwf][j];
	      for (int k=0; k<rusdraw_nchan_sd; k++)
		event_to_sort->fadc[iwf][j][k] = event_for_sorting->fadc[jwf][j][k];
	      event_to_sort->pchmip[iwf][j] = event_for_sorting->pchmip[jwf][j];
	      event_to_sort->pchped[iwf][j] = event_for_sorting->pchped[jwf][j];
	      event_to_sort->lhpchmip[iwf][j] = event_for_sorting->lhpchmip[jwf][j];
	      event_to_sort->lhpchped[iwf][j] = event_for_sorting->lhpchped[jwf][j];
	      event_to_sort->rhpchmip[iwf][j] = event_for_sorting->rhpchmip[jwf][j];
	      event_to_sort->rhpchped[iwf][j] = event_for_sorting->rhpchped[jwf][j];
	      event_to_sort->mftndof[iwf][j] = event_for_sorting->mftndof[jwf][j];
	      event_to_sort->mip[iwf][j] = event_for_sorting->mip[jwf][j];
	      event_to_sort->mftchi2[iwf][j] = event_for_sorting->mftchi2[jwf][j];
	      for (int k=0; k<4; k++)
		{
		  event_to_sort->mftp[iwf][j][k] = event_for_sorting->mftp[jwf][j][k];
		  event_to_sort->mftpe[iwf][j][k] = event_for_sorting->mftpe[jwf][j][k];
		}
	    }
	  iwf ++;
	}
    }
}
