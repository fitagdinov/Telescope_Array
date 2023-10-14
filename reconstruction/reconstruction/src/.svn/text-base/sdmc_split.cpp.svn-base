#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "event.h"
#include "sduti.h"
#include "sdrt_class.h"
#include "sddstio.h"
#include "filestack.h"
#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TMath.h"
#include <map>

#define sdmc_split_max_days 0x400 // maximum number of days than one can split the MC set into

class sdmc_split_cmdline_opt
{
  
public:
  
  char progName[0x400];
  char outbase[0x400];
  int d1; // start date, yymmdd format
  int d2; // stop  date, yymmdd format
  sdmc_split_cmdline_opt()
  {
    d1 = 80511;
    d2 = 100213;
    outbase[0] = 0;
    progName[0] = '\n';
  }
  virtual ~sdmc_split_cmdline_opt()
  {
  }
  
  bool getFromCmdLine(int argc, char **argv)
  {

    int i;
    char inBuf[0x400];
    FILE *fl;
    char *line;
    if (argc==1)
      {
	memcpy(progName,argv[0],
	       (strlen(argv[0])+1<=0x400 ? strlen(argv[0])+1 : 0x400));
	printMan();
	return false;
      }
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
                printf("error: -i: specify the input list file!\n");
                return false;
              }
            else
              {
                if ((fl = fopen(argv[i], "r")) == NULL)
                  {
                    fprintf(stderr, "error: can't open list file %s\n", argv[i]);
                    return false;
                  }
                else
                  {
                    while (fgets(inBuf, 0x400, fl))
                      {
                        if (((line = strtok(inBuf, " \t\r\n")))
                            && (strlen(line) > 0))
                          {
                            if (pushFile(line) != SUCCESS)
                              return false;
                          }
                      }
                    fclose(fl);
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
	// output base
        else if (strcmp("-o", argv[i]) == 0)
          {
            if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
              {
                fprintf(stderr, "error: -o: specify the output base!\n");
                return false;
              }
            else
              sscanf(argv[i], "%1023s", outbase);
          }
	
	
	// start date
	else if (strcmp("-d1", argv[i]) == 0)
          {
            if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
              {
                fprintf(stderr, "error: -d1: specify the start date!\n");
                return false;
              }
            else
              sscanf(argv[i], "%d", &d1);
          }
	
	// stop date
	else if (strcmp("-d2", argv[i]) == 0)
          {
            if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
              {
                fprintf(stderr, "error: -d2: specify the stop date!\n");
                return false;
              }
            else
              sscanf(argv[i], "%d", &d2);
          }
	
	// assume that all arguments w/o the '-' switch are input dst file names
        else if (argv[i][0] != '-')
          {
            if (pushFile(argv[i]) != SUCCESS)
              return false;
          }
        else
          {
            fprintf(stderr, "error: '%s': unrecognized option\n", argv[i]);
            return false;
          }
	
      }
    return checkOpt();
  }
  
  void printMan()
  {
    fprintf(stderr,"\nSplit SD MC into date-based parts (one part per day)\n");
    fprintf(stderr,"input - DST files, output - root trees for each day\n");
    fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list file]  -o [output_base]\n",progName);
    fprintf(stderr,"pass input dst file names as arguments without any prefixes\n");
    fprintf(stderr, "-i <string> : specify the want file (with dst files)\n");
    fprintf(stderr, "--tty       : or get input dst file names from stdin\n");
    fprintf(stderr, "-o <string> : output base: includes directory and name prefix (default './sdmc_split_')\n");
    fprintf(stderr, "-d1 <int>   : start date, yymmdd format ( %d default)\n",d1);
    fprintf(stderr, "-d2 <int>   : stop date,  yymmdd format ( %d default)\n",d2);
    fprintf(stderr,"\n");
  }

private:
  
  bool checkOpt()
  {

    int nd; // number of days
    
    if (d2<d1 || d1<=0 || d2<=0)
      {
	fprintf(stderr,"error: invalid date range: %d-%d\n",d1,d2);
	return false;
      }    
    nd=SDGEN::greg2jd(d2)-SDGEN::greg2jd(d1)+1;
    if (nd > sdmc_split_max_days)
      {
	fprintf(stderr, "number of days is too large (>%d); run this program for each date range separately\n",
		sdmc_split_max_days);
	return false;
      }
    if(countFiles()==0)
      {
	fprintf(stderr,"don't have any inputs!\n");
	return false;
      }
    if (!outbase[0])
      sprintf(outbase,"%s","./sdmc_split_");
    return true;
  }  
};

int main(int argc, char **argv)
{
  std::map <int,int>     run_dates; // indexing for the run dates (faster to check if exist)
  std::map <int,TString> rootfname; // root file names for each day
  std::map <int,TFile *> rootfile;  // root files for each day
  std::map <int,TTree *> mctree;    // root trees for each day

  sddstio_class *dstio = new sddstio_class(); // to handle the dst I/O
  rusdmc_class  *rusdmc  = new rusdmc_class; // handle rusdraw, rusdmc variables
  rusdraw_class *rusdraw = new rusdraw_class;
  char *infile;
  int ijd,jd1,jd2,yymmdd;
  sdmc_split_cmdline_opt *opt = new  sdmc_split_cmdline_opt;
  integer4 wantBanks;
  int events_read, events_filled;

  wantBanks = newBankList(10);
  addBankList(wantBanks,RUSDMC_BANKID);
  addBankList(wantBanks,RUSDRAW_BANKID);
  dstio->setWantBanks(wantBanks);

  if(!opt->getFromCmdLine(argc,argv))
    return 2;
  
  // prepare the output files
  jd1 = SDGEN::greg2jd(opt->d1);
  jd2 = SDGEN::greg2jd(opt->d2);  
  for(ijd=jd1; ijd<= jd2; ijd++)
    {
      yymmdd = SDGEN::jd2yymmdd((double)ijd+0.1);
      run_dates[yymmdd] = (ijd-jd1);     
      rootfname[yymmdd].Form("%s%06d.root",opt->outbase,yymmdd);
      rootfile[yymmdd] = new TFile(rootfname[yymmdd],"recreate");
      if(rootfile[yymmdd]->IsZombie()) return 2;
      mctree[yymmdd] = new TTree("mctree","SD MC");
      mctree[yymmdd]->Branch ("rusdmc","rusdmc_class",&rusdmc,64000,0);
      mctree[yymmdd]->Branch ("rusdraw","rusdraw_class",&rusdraw,64000,0);
    }
  
  
  
  // read the dst input stream and channel events into appropriate root files
  
  events_read   = 0;
  events_filled = 0;
  while((infile=pullFile()))
    {
      dstio->openDSTinFile(infile);
      while(dstio->readEvent())
	{
	  events_read ++;
	  
	  // skip events if it doesn't have the appropriate banks
	  if(!dstio->haveBanks(wantBanks,true))
	    continue;
	  
	  // skip event if its date is out of desired range
	  if(run_dates.find(rusdraw_.yymmdd) == run_dates.end())
	    continue;

	  // load the variables from dst
	  rusdraw->loadFromDST();
	  rusdmc->loadFromDST();
	  
	  // fill the appropriate root tree
	  mctree[rusdraw->yymmdd]->Fill();
	  
	  events_filled ++;
	  if (events_filled % 1000000 == 0)
	    fprintf(stdout, "events_read: %d events_filled: %d\n",
		    events_read,events_filled);
	  fflush(stdout);
	}
      dstio->closeDSTinFile();
    }
  
  fprintf(stdout, "events_read: %d events_filled: %d\n", events_read,events_filled);
  fprintf(stdout, "finalizing the root tree files ... \n");
  fflush(stdout);
  
  // finalize properly all the root trees
  for(map<int, TFile *>::iterator itr=rootfile.begin(); itr != rootfile.end(); ++itr)
    {
      (*itr).second->Write();
      (*itr).second->Close();
    }
  fprintf(stdout,"\n\nDone\n");
  return 0;
}
