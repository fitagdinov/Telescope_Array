/*
  Select events with required DST banks: one can specify the mandatory banks,
  so that the events are writtent only if they include all of these banks or one can specify the
  optional banks, so that the evetns are written out if they have either one of these banks
  Dmitri Ivanov, <dmiivanov@gmail.com>
  Last modified: DI 20171206
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "event.h"
#include "filestack.h"

#define DSTSEL_MAXBLISTS    0x100            // maximum number of bank lists

char dstsel_outfile[0x400];                  // output file
integer4 fMode;                              // force-overwrite mode
integer4 reqBanks;                           // list of mandatory banks: events written out only if they have all of these banks
integer4 optBanks;                           // list of optional banks: events written out if they have either of these
integer4 optBankList[DSTSEL_MAXBLISTS];      // optional bank lists
integer4 optBankList_size[DSTSEL_MAXBLISTS]; // sizes of optional bank lists
integer4 reqBankList[DSTSEL_MAXBLISTS];      // required bank lists
integer4 reqBankList_size[DSTSEL_MAXBLISTS]; // sizes of optional bank lists
integer4 n_reqBanks;                         // number of required banks
integer4 n_optBanks;                         // number of optinonal banks
integer4 n_optBankList;                      // number of optional bank lists
integer4 n_reqBankList;                      // number of required bank lists

static int parseCmdLine(int argc, char **argv);

int main(int argc, char **argv)
{
  char *infile;
  integer4 eventsRead,eventsWritten;
  integer4 rc,event,mode;
  integer4 wantBanks,gotBanks;
  integer4 inUnit,outUnit;
  integer4 outfile_open;
  integer4 iblist;
  integer4 pass;
  FILE *fl;
  reqBanks = newBankList(nBanksTotal());
  optBanks = newBankList(nBanksTotal());
  if (parseCmdLine(argc, argv) != 0)
    exit(2);
  wantBanks = newBankList(nBanksTotal());
  eventAllBanks(wantBanks);
  gotBanks = newBankList(nBanksTotal());
  inUnit  = 1;
  outUnit = 2;
  eventsRead    = 0;
  eventsWritten = 0;
  outfile_open  = 0;
  while((infile = pullFile()))
    {
      if (!(fl=fopen(infile,"r")))
	{
	  fprintf(stderr,"can't open %s\n",infile);
	  exit(2);
	}
      fclose(fl);
      mode = MODE_READ_DST;
      if ((rc=dstOpenUnit(inUnit,infile,mode)) != 0)
	{
	  fprintf(stderr,"can't dst-open %s for reading\n",infile);
	  exit(2);
	}
      while ((rc = eventRead (inUnit, wantBanks, gotBanks, &event)) > 0) 
	{
	  if (!event)
	    {
	      fprintf(stderr,"corrupted event!\n");
	      exit(2);
	    }
	  eventsRead ++;
	  if(cmpBankList(gotBanks,reqBanks) < n_reqBanks) continue;
	  if((n_optBanks) && !cmpBankList(gotBanks,optBanks)) continue;
	  pass = 1;
	  for (iblist=0; iblist < n_reqBankList; iblist++)
	    {
	      if(cmpBankList(gotBanks,reqBankList[iblist])<reqBankList_size[iblist])
		pass=0;
	    }
	  if(!pass) continue;
	  if(n_optBankList) pass = 0;
	  for (iblist=0; iblist<n_optBankList; iblist++)
	    {
	      if(cmpBankList(gotBanks,optBankList[iblist])>=optBankList_size[iblist])
		pass = 1;
	    }
	  if(!pass) continue;
	  if (!outfile_open)
	    {
	      if (((fl=fopen(dstsel_outfile,"r"))) && (!fMode))
		{
		  fprintf(stderr,"%s: error: file exists; use '-f' option to overwrite files\n",dstsel_outfile);
		  exit(2);
		}
	      if (!(fl=fopen(dstsel_outfile,"w")))
		{
		  fprintf(stderr,"can't start %s\n",dstsel_outfile);
		  exit(2);
		}
	      fclose(fl);
	      mode = MODE_WRITE_DST;
	      if ((rc=dstOpenUnit(outUnit, dstsel_outfile, mode)) != 0)
		{
		  fprintf(stderr,"can't dst-open %s for writing\n",dstsel_outfile);
		  exit(2);
		}
	      outfile_open = 1;
	    }	  
	  if ((rc=eventWrite(outUnit, gotBanks, TRUE)) < 0)
	    {
	      fprintf(stderr,"failed to write an event \n");
	      exit(2);
	    }      
	  eventsWritten ++;
	}    
      dstCloseUnit(inUnit);
    }
  if(outfile_open) dstCloseUnit(outUnit);
  fprintf(stdout, "eventsRead: %d\n",eventsRead);
  fprintf(stdout, "eventsWritten: %d\n",eventsWritten);
  return 0;
}

int parseCmdLine(int argc, char **argv)
{
  int i;
  integer4 bank_id;
  char *name;
  dstsel_outfile[0] = '\0';
  clrBankList(reqBanks);
  clrBankList(optBanks);
  fMode         = 0;
  n_reqBanks    = 0;
  n_optBanks    = 0;
  n_optBankList = 0;
  n_reqBankList = 0;
  if (argc==1)
    {
      integer4 rc;
      fprintf(stderr,"\nSelect and write out events from file1,file2 ... that have required DST banks\n");
      fprintf(stderr,"\nUsage: %s -o [dst_file] file1 file2 ...  +[bank1] +[bank2] -[bank3] -[bank4]...\n",argv[0]);
      fprintf(stderr,"-bank_name1,bank_name2...  : events selected if they have either of these bank lists\n");
      fprintf(stderr,"+bank_name1,bank_name2...  : events selected if they have all of these bank lists\n");
      fprintf(stderr,"-o:    (opt) output file\n");
      fprintf(stderr,"-f:    (opt) don't check if the output file exists, just overwrite it\n");
      fprintf(stderr,"examples:\n");
      fprintf(stderr,"#1: %s -o output.dst input.dst -brraw -lrraw",argv[0]); 
      fprintf(stderr," (selects events that have either brraw or lrraw)\n");
      fprintf(stderr,"#2: %s -o output.dst input.dst +brraw +lrraw",argv[0]); 
      fprintf(stderr," (selects events that have both brraw and lrraw)\n");
      fprintf(stderr,"#3: %s -o output.dst input.dst +brraw +lrraw -rusdraw -rusdgeom",argv[0]);
      fprintf(stderr," (events with brraw and lrraw and either rusdraw or rusdgeom)\n");
      fprintf(stderr,"#4: %s -o output.dst input.dst -brraw,lrraw -brraw,hraw1 -lrraw,hraw1 ",argv[0]);
      fprintf(stderr," (with brraw,lrraw or brraw,hraw1 or lrraw,hraw1)\n");
      fprintf(stderr,"\n");
      fputs("\nCurrently recognized banks:", stderr);
      dscBankList((rc=newBankList(nBanksTotal()),eventAllBanks(rc),rc),stderr);
      fprintf(stderr,"\n\n");      
      return 2;
    }
  for(i=1; i<argc; i++)
    {
      if (strcmp ("-o", argv[i]) == 0)
        {
          if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
            {
              fprintf (stderr,"erorr: -o: specify the output file!\n");
              return -1;
            }
          else
            {
              sscanf (argv[i], "%s", dstsel_outfile);
	      continue;
            }
	}
      else if (strcmp ("-f", argv[i]) == 0)
	{
	  fMode = 1;
	  continue;
	}
      else if (argv[i][0] == '+')
	{
	  if (strchr(&argv[i][1],','))
	    {
	      if(n_reqBankList>= DSTSEL_MAXBLISTS)
		{
		  fprintf(stderr, "error: too many required bank lists\n");
		  return -1;
		}
	      reqBankList[n_reqBankList] = newBankList(nBanksTotal());
	      reqBankList_size[n_reqBankList] = 0;
	      name = strtok (&argv[i][1], ",");
	      while(name != NULL) 
		{
		  if((bank_id=eventIdFromName(name)))
		    addBankList(reqBankList[n_reqBankList], bank_id);
		  else
		    {
		      fprintf(stderr, "error: unrecognized bank: %s\n",name);
		      return -1;
		    }
		  name = strtok (NULL, ",");
		}
	      reqBankList_size[n_reqBankList] = cntBankList(reqBankList[n_reqBankList]);
	      n_reqBankList++;
	    }
	  else
	    {
	      name=&argv[i][1];
	      if((bank_id=eventIdFromName(name)))
		addBankList(reqBanks, bank_id);
	      else
		{
		  fprintf(stderr, "error: unrecognized bank: %s\n",name);
		  return -1;
		}
	    }
	}
      else if (argv[i][0] == '-')
	{
	  if (strchr(&argv[i][1],','))
	    {
	      if(n_optBankList>= DSTSEL_MAXBLISTS)
		{
		  fprintf(stderr, "error: too many optional bank lists\n");
		  return -1;
		}
	      optBankList[n_optBankList] = newBankList(nBanksTotal());
	      optBankList_size[n_optBankList] = 0;
	      name = strtok (&argv[i][1], ",");
	      while(name != NULL) 
		{
		  if((bank_id=eventIdFromName(name)))
		    addBankList(optBankList[n_optBankList], bank_id);
		  else
		    {
		      fprintf(stderr, "error: unrecognized bank: %s\n",name);
		      return -1;
		    }
		  name = strtok (NULL, ",");
		}
	      optBankList_size[n_optBankList] = cntBankList(optBankList[n_optBankList]);
	      n_optBankList++;
	    }
	  else
	    {
	      name=&argv[i][1];
	      if((bank_id=eventIdFromName(name)))
		addBankList(optBanks, bank_id);
	      else
		{
		  fprintf(stderr, "error: unrecognized bank: %s\n",name);
		  return -1;
		}
	    }
	}
      else
	pushFile(argv[i]);
    }
  n_reqBanks = cntBankList(reqBanks);
  n_optBanks = cntBankList(optBanks);
  if (!(n_reqBanks || n_optBanks || n_optBankList || n_reqBankList))
    {
      fprintf(stderr, "error: no banks selected\n");
      exit (2);
    }  
  if (dstsel_outfile[0]=='\0')
    sprintf(dstsel_outfile,"%s","dstsel.dst");
  if (countFiles() == 0)
    {
      fprintf(stderr,"error: no input files\n");
      return -1;
    }  
  return 0;
}
