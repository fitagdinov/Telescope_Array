/* 
 * Added 20100110
 * Dmitri Ivanov <dmiivanov@gmail.com>
 * Last modified: DI 20171206
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "event.h"
#include "filestack.h"

// max. number of output files
#define DSTSHUF_NOUTFILES 30

// output suffix style 
#define DSTSHUF_OUTSUFFIX "_%02d.dst.gz"

integer4 fMode;     // force overwrite mode
integer4 rseed; // random seed 
integer4 noutfiles; // number of output files

char     out_base[0x400];
char     out_suf[0x20];
char     outfiles[DSTSHUF_NOUTFILES][0x420];

int parseCmdLine(int argc, char **argv);

int main(int argc, char **argv)
{
  char *infile;
  
  unsigned long long int ievent;
  integer4 ifile,size,rc,event,mode;
  integer4 wantBanks,gotBanks;
 
  integer4 inUnit,outUnit[DSTSHUF_NOUTFILES];
  FILE *fl;

  // get the arguments
  if (parseCmdLine(argc, argv) != 0)
    exit(2);
  
  // set the random seed
  srand48(time(NULL));
  if (rseed > 0)
    srand48(rseed);
  
  size = nBanksTotal();
  wantBanks = newBankList(size);
  gotBanks  = newBankList(size);
  eventAllBanks(wantBanks);
  
  mode = MODE_WRITE_DST;
  for (ifile=0; ifile<noutfiles; ifile++)
    {
      sprintf(outfiles[ifile],"%s",out_base);
      sprintf(out_suf,DSTSHUF_OUTSUFFIX,ifile);
      strcat(outfiles[ifile],out_suf);
      outUnit[ifile] = ifile+1;
      if (((fl=fopen(outfiles[ifile],"r"))) && (!fMode))
	{
	  fprintf(stderr,"%s: file exists\n",outfiles[ifile]);
	  exit(2);
	}
      if (!(fl=fopen(outfiles[ifile],"w")))
	{
	  fprintf(stderr,"can't start %s\n",outfiles[ifile]);
	  exit(2);
	}
      fclose(fl);
      if ((rc=dstOpenUnit(outUnit[ifile],&outfiles[ifile][0],mode)) != 0)
	{
	  fprintf(stderr,"can't dst-open %s for writing\n",outfiles[ifile]);
	  exit(2);
	}
    }

  
  inUnit = noutfiles+1;
  mode   = MODE_READ_DST;
  ievent = 0;
  while ((infile=pullFile()))
    {
      if (!(fl=fopen(infile,"r")))
	{
	  fprintf(stderr,"can't open %s\n",infile);
	  exit(2);
	}
      fclose(fl);
      if ((rc=dstOpenUnit(inUnit,infile,mode)) != SUCCESS)
	{
	  fprintf(stderr,"can't dst-open %s for reading\n",infile);
	  exit(2);
	}
      while ((rc=eventRead (inUnit, wantBanks, gotBanks, &event)) > 0) 
	{
	  if (!event)
	    {
	      fprintf(stderr,"Corrupted event in %s!\n", infile);
	      exit(2);
	    }
	  ifile = ((rseed < 0) ? (int)(ievent % noutfiles) : ((int)floor(drand48() * (double)noutfiles - 1e-7)));
	  if ((rc=eventWrite(outUnit[ifile], gotBanks, TRUE)) < 0 )
	    {
	      fprintf(stderr,"failed to write an event to %s\n",outfiles[ifile]);
	      exit(2);
	    }
	  ievent++;
	}
      dstCloseUnit(inUnit); 
    }
  
  for (ifile=0; ifile < noutfiles; ifile++)
    dstCloseUnit(outUnit[ifile]);
  
  return 0;
}


int parseCmdLine(int argc, char **argv)
{

  int i;
  FILE *wantFl;
  char *line;
  char want_file[0x400];
  char inBuf[0x400];
  
  noutfiles       =  2;
 
  sprintf(out_base,"dstshuf");

  fMode           =  0;
  rseed           = -1;

  
  if (argc==1)
    {
      integer4 rc;
      fprintf(stderr,"\n\nChannel events from file1,fiel2,... to multiple output files\n");
      fprintf(stderr,"By default, i'th event goes to [i mod nparts]'th file, letting i start at 0\n");
      fprintf(stderr,"To random-shuffle events, see '-r' option\n");
      fprintf(stderr,"If the number of output files is 1, it will act as dstcat\n");
      fprintf(stderr,"Maximum %d output files allowed.\n",DSTSHUF_NOUTFILES);
      fprintf(stderr,"\nUsage: %s -o [output_base_name] -i [want_file] and/or [dst_file1] [dst_file2] ...\n",
	      argv[0]);
      fprintf(stderr,"-o:    set the output file base ('dstshuf' is default).  \n");
      fprintf(stderr,"       Will add _\?\?.dst.gz 1-digit suffixes\n");
      fprintf(stderr,"-n:    set the number of output files.  Minimum is 1, maximum is %d, default is 2\n",
	      DSTSHUF_NOUTFILES);
      fprintf(stderr,"       (if number of output files is 1, it will act as dstcat)\n");
      fprintf(stderr,"-r:    enable random shuffling and set the random seed\n");
      fprintf(stderr,"       if 0 is passed, the random seed is set equal to system time\n");
      fprintf(stderr,"-i:    to get intput files from a want_file\n");
      fprintf(stderr,"--tts: to get input files from stdin\n");
      fprintf(stderr,"       optionally, just pass input file names w/o prefixes\n");
      fprintf(stderr,"-f:    don't check if the output files exist, just overwrite them\n\n");
      fputs("\nCurrently recognized banks:", stderr);
      dscBankList((rc=newBankList(nBanksTotal()),eventAllBanks(rc),rc),stderr);
      fprintf(stderr,"\n\n");
      return -1;
    } 
    
  for(i=1; i<argc; i++)
    { 
      
      // root output file
      if (strcmp ("-o", argv[i]) == 0)
	{
	  if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"specify the output file base!\n");
	      return -1;
	    }
	  else
	    {
	      sscanf (argv[i], "%s", &out_base[0]);
	    }
	}
      
      else if (strcmp ("-n", argv[i]) == 0)
	{
	  if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"specify the number of output files!\n");
	      return -1;
	    }
	  else
	    {
	      sscanf (argv[i], "%d", &noutfiles);
	    }
	}
      
      else if (strcmp ("-r", argv[i]) == 0)
	{
	  if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"specify the random seed!\n");
	      return -1;
	    }
	  else
	    {
	      sscanf (argv[i], "%d", &rseed);
	    }
	}
      
      else if (strcmp ("-i", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printf ("specify the want file!\n");
	      return -1;
	    }
	  else
	    {
	      sscanf (argv[i], "%s", want_file);
	      if((wantFl=fopen(want_file,"r")))
		{
		  while(fgets(inBuf,0x400,wantFl))
		    {
		      if (((line=strtok (inBuf, " \t\r\n"))) && (strlen (line) > 0) 
			  && (pushFile(line) != SUCCESS))
			return -1;
		    }
		  fclose(wantFl);
		}
	      else
		{
		  fprintf(stderr,"can't open want file %s\n",want_file);
		  return -1;
		}
	    }
	}

     
      else if (strcmp ("--tts", argv[i]) == 0)
	{ 
	  while(fgets(inBuf,0x400,stdin))
	    {
	      if (((line=strtok (inBuf, " \t\r\n"))) && (strlen (line) > 0) 
		  && (pushFile(line) != SUCCESS))
		return -1;
	    }
	}
      
     
      else if (strcmp ("-f", argv[i]) == 0)
	{
	  fMode = 1;
	}
     
      else if (argv[i][0] != '-') 
	{
	  if(pushFile(argv[i]) != SUCCESS)
	    return -1;
	}
      else
	{
	  fprintf (stderr, "'%s': unrecognized option\n", argv[i]);
	  return -1;
	}
    }
  
  if (countFiles() == 0)
    {
      fprintf(stderr,"don't have any input files\n");
      return -1;
    }
  
  if (noutfiles < 1 || noutfiles > DSTSHUF_NOUTFILES)
    {
      fprintf(stderr,"minimum number of output files is %d and maximum is %d\n",
	      1,DSTSHUF_NOUTFILES);
      return -1;
    }

  

  return 0;
}


