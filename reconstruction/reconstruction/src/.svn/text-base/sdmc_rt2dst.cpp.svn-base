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
#include "TChain.h"
#include "TSystem.h"
#include "TMath.h"


class sdmc_rt2dst_cmdline_opt
{
  
public:
  
  char progName[0x400];
  char outfile[0x400];
  bool sortOpt;
  sdmc_rt2dst_cmdline_opt()
  {
    outfile[0]  = 0;
    progName[0] = '\n';
    sortOpt     = false;
  }
  virtual ~sdmc_rt2dst_cmdline_opt()
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
	// output file
        else if (strcmp("-o", argv[i]) == 0)
          {
            if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
              {
                fprintf(stderr, "error: -o: specify the output file!\n");
                return false;
              }
            else
              sscanf(argv[i], "%1023s", outfile);
          }
	// sorting option
        else if (strcmp("-sort", argv[i]) == 0)
	  sortOpt = true;
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
    fprintf(stderr,
	    "\nCombine and optionally time-order sd mc root trees and write events into a DST file\n");
    fprintf(stderr,"input(s) - root tree files with events on a given date, output - single dst file for that date\n");
    fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list file]  -o [output file]\n",progName);
    fprintf(stderr,"pass input root tree file names as arguments without any prefixes\n");
    fprintf(stderr, "-i <string> : specify the want file (with root tree files)\n");
    fprintf(stderr, "--tty       : or get input root tree file names from stdin\n");
    fprintf(stderr, "-o <string> : dst output file  (default is './sdmc_rt2dst.dst')\n");
    fprintf(stderr, "-sort       : time-order the MC events in each day (slow and I/O intense)\n");
    fprintf(stderr,"\n");
  }

private:
  
  bool checkOpt()
  {
    if(countFiles()==0)
      {
	fprintf(stderr,"don't have any inputs!\n");
	return false;
      }
    if (!outfile[0])
      sprintf(outfile,"./sdmc_rt2dst.dst");
    return true;
  }  
};



int main(int argc, char **argv)
{

  TChain  *mctree;
  sddstio_class *dstio = new sddstio_class(); // to handle the dst I/O
  rusdmc_class  *rusdmc  = new rusdmc_class; // handle rusdraw, rusdmc variables
  rusdraw_class *rusdraw = new rusdraw_class;
  char *infile;
  int ievent,nevents;
  sdmc_rt2dst_cmdline_opt *opt = new  sdmc_rt2dst_cmdline_opt;
  integer4 wantBanks;
  int *index;

  wantBanks = newBankList(10);
  addBankList(wantBanks,RUSDMC_BANKID);
  addBankList(wantBanks,RUSDRAW_BANKID);
  dstio->setWantBanks(wantBanks);

  if(!opt->getFromCmdLine(argc,argv))
    return 2;
  
  
  // initialize the root-tree chain and add all root tree files to it
  mctree = new TChain("mctree","SD MC");
  while((infile=pullFile()))
    mctree->AddFile(infile);
  mctree->SetBranchAddress("rusdmc",&rusdmc);
  mctree->SetBranchAddress("rusdraw",&rusdraw);

   
  // time-sort and put all events into the dst file
  nevents=mctree->GetEntries();
  if (nevents>0)
    { 
      dstio->openDSToutFile(opt->outfile);
      if (opt->sortOpt)
	{
	  mctree->SetEstimate(nevents);
	  mctree->Draw("rusdraw->hhmmss+(rusdraw->usec)/1e6","","goff");
	  index = new int[nevents];
	  TMath::Sort(nevents,mctree->GetV1(),index,false);
	  for (ievent=0; ievent<nevents; ievent++)
	    {
	      mctree->GetEntry(index[ievent]);
	      rusdmc->loadToDST();
	      rusdraw->loadToDST();
	      dstio->writeEvent(wantBanks,false);
	    }
	  delete index;
	}
      else
	{
	  for (ievent=0; ievent<nevents; ievent++)
	    {
	      mctree->GetEntry(ievent);
	      rusdmc->loadToDST();
	      rusdraw->loadToDST();
	      dstio->writeEvent(wantBanks,false);
	    }
	}
      dstio->closeDSToutFile();
    }
  fprintf(stdout, "%d events\n",nevents);
   
  fprintf(stdout,"\n\nDone\n");
  return 0;
}
