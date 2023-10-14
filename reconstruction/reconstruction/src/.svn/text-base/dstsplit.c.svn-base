/*
 *
 * Last Modified: DI 20171206
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <string.h>

#include "dst_err_codes.h"
#include "event.h"
#include "filestack.h"
#include "bank_list.h"

integer4 inUnit = 1, outUnit = 2;
integer4 offset=0,nevent=0;
char outFileName[0x400];
integer4 f_overwrite_mode=0;

void parseCmdLine (int argc, char *argv[]);

int main (int argc, char *argv[])
{
  integer1 *inFile;
  integer4 rc, size, event;
  integer4 wantBanks, gotBanks;
  integer4 eventsRead, eventsWritten;
  integer4 outfile_open = 0;
  
  strcpy(outFileName,"dstsplit.dst");

  /* parse command line and open output unit */
  parseCmdLine (argc, argv);
  
 
   

  /* initialize bank lists */
  size = nBanksTotal();
  wantBanks = new_bank_list_ (&size);
  gotBanks = new_bank_list_ (&size);

  eventsRead    = 0;
  eventsWritten = 0;
   

  while ((inFile=pullFile())) 
    {
      if ((rc=dstOpenUnit(inUnit,inFile,MODE_READ_DST))) 
	{
	  fprintf (stderr, "%s: Error %d: failed to open for reading dst file:" 
		   " %s\n", argv[0], rc, inFile);
	  return 1;
	}
      printf ("Reading DST file: %s\n", inFile);
      while ((rc=eventRead(inUnit,wantBanks,gotBanks,&event)) > 0) 
	{
	  if(!event)
	    continue;
	  eventsRead++;
	  if((eventsRead > offset) && ((eventsWritten < nevent) || (nevent == 0))) 
	    {
	      // open the output file if it hasen't been opened yet
	      if(!outfile_open) 
		{
		  FILE* fp = 0;      
		  if(!f_overwrite_mode && (fp=fopen(outFileName,"r")))
		    {
		      fprintf(stderr,"error: %s exists; use '-f' option to overwrite the files\n",outFileName);
		      fclose(fp);
		      return (-1);
		    }
		  if(!(fp=fopen(outFileName,"w")))
		    {
		      fprintf(stderr,"error: failed to start %s\n",outFileName);
		      return (-1);
		    }
		  fclose(fp);
		  if (dstOpenUnit(outUnit,outFileName,MODE_WRITE_DST)) 
		    {
		      fprintf (stderr, "error: failed to start %s\n",outFileName);
		      return (-1);
		    }
		  outfile_open = 1;
		}
	      eventWrite(outUnit,gotBanks,event);
	      eventsWritten++;
	      // after all requested events have been written out, there is no need in
	      // reading the input dst files any further.
	      if((eventsWritten==nevent) && (nevent != 0))
		break;
	    }
	}
      dstCloseUnit(inUnit);
      // after all requested events have been written out, there is no need in
      // reading the input dst files any further.
      if((eventsWritten==nevent) && (nevent != 0))
	break;
    }
  if(outfile_open)
    dstCloseUnit(outUnit);
  return 0;
}

void parseCmdLine (int argc, char *argv[])
{
  char firstInFile[0x400], line[0x400], *name;
  FILE *listFile;
  integer4 i;

  firstInFile[0] = '\0';
  if (argc == 1) {
    integer4 rc;
    fprintf(stderr, "\nwrite n selected events into a separate dst file, starting at the (s+1)'st event\n");
    fprintf (stderr,
	     "\nusage: %s [-o output_file] [-s n_skip ] [-n number of events] [-i in_list_file] [in_files ...]\n\n", 
	     argv[0]);
    fputs ("pass input dst files w/o any prefixes or switches\n",stderr);
    fputs ("  -i <string>: or read input file names from a list file\n", stderr);
    fputs ("  --stdin:     or pipe input file names through stdin\n",stderr);
    fputs ("  -s <int>:    number of events to skip, so that the first s events are not written out\n",stderr);
    fputs ("  -n <int>:    number of events to write out, starting at (s+1)'st event\n",stderr);
    fputs ("  -f:          overwrite the output file if it exists\n\n",stderr);
    fputs("\nCurrently recognized banks:", stderr);
    dscBankList((rc=newBankList(nBanksTotal()),eventAllBanks(rc),rc),stderr);
    fprintf(stderr,"\n\n");
    exit (2);
  }
  for (i = 1; i < argc; ++i) {
    if (strcmp (argv[i], "-i") == 0) {
      if (++i >= argc || argv[i][0] == '-') {
	fprintf (stderr, "Input list option specified but no filename "
		 "given\n");
	exit (1);
      }
      else if ( (listFile=fopen (argv[i], "r")) ) {
	while (fgets (line,0x400,listFile)) {
	  name = strtok (line, " \t\r\n");
	  if (strlen (name) > 0) {
	    pushFile (name);
	    if (firstInFile[0] == '\0')
	      strcpy (firstInFile, name);
	  }
	}
	fclose (listFile);
      }
      else {
	fprintf (stderr, "Failed to open input list file %s\n", argv[i]);
	exit (1);
      }
    }
    else if (strcmp (argv[i], "--stdin") == 0) {
      while (fgets (line,0x400,stdin)) {
	name = strtok (line, " \t\r\n");
	if (strlen (name) > 0) {
	  pushFile (name);
	  if (firstInFile[0] == '\0')
	    strcpy (firstInFile, name);
	}
      }
    }
    else if(strcmp (argv[i], "-o") == 0) {
      if(++i >= argc || argv[i][0] == '-') {
	fprintf (stderr, "output filename option specified but no filename "
		 "given\n");
	exit (1);
      }
      strcpy(outFileName, argv[i]);
    }
    else if(strcmp (argv[i], "-f") == 0) {
      f_overwrite_mode = 1;
    }
    else if(strcmp (argv[i], "-s") == 0) {
      if(++i >= argc || argv[i][0] == '-') {
	fprintf (stderr, "offset option specified but no filename "
		 "given\n");
	exit (1);
      }
      offset = atoi(argv[i]);
      if(offset<0) {
	fprintf (stderr, "negative offset given, using 0 instead \n");
	offset = 0;
      }
    }
    else if(strcmp (argv[i], "-n") == 0) {
      if(++i >= argc || argv[i][0] == '-') {
	fprintf (stderr, "number of events option specified but no filename "
		 "given\n");
	exit (1);
      }
      nevent = atoi(argv[i]);
      if(nevent < 0) {
	fprintf (stderr, "negative number of events given, assuming "
		 "unlimited\n");
	nevent = 0;
      }
    }
    else if (argv[i][0] != '-') {
      pushFile (argv[i]);
      if (firstInFile[0] == '\0')
	strcpy (firstInFile, argv[i]);
    }
    else {
      fprintf (stderr, "Unrecognized option: %s\n", argv[i]);
      exit (1);
    }
  }
  if (countFiles () == 0) {
    fprintf (stderr, "Input file(s) must be specified!\n");
    exit (1);
  }
}
