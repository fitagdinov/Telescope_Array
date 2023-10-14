/* Last modified: DI 20171206 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "event.h"

// max. number of input files
#define DSTSUM_NINFILES 30


integer4 ninfiles;  // number of input files
integer4 stdin_mode; // accept input files from stdin
integer4 fMode;     // force overwrite mode
char outfile[0x400];
char infiles[DSTSUM_NINFILES][0x400];

// adds input file to the list and checks if the
// list size doesn't exceed the maximum
int add_intput_file(char *fname);

int parseCmdLine(int argc, char **argv);

int main(int argc, char **argv)
{
  integer4 ifile,size,rc,event,mode;
  integer4 wantBanks,writeBanks;
  integer4 inUnit[DSTSUM_NINFILES],gotBanks[DSTSUM_NINFILES];
  integer4 outUnit;
  FILE *fl;
  
  // get the arguments
  if (parseCmdLine(argc, argv) != 0)
    exit(2);
  
  // assign the dst input units
  for ( ifile=0; ifile < ninfiles; ifile++)
    {
      inUnit[ifile] = ifile+1;
    }
  // assign the dst output unit
  outUnit = ninfiles+1;
  
  size = nBanksTotal();

  // want banks - all possible event banks
  wantBanks = newBankList(size);
  eventAllBanks(wantBanks);
  
  // allocate gotBanks
  for (ifile=0; ifile < ninfiles; ifile++)
    gotBanks[ifile] = newBankList(size);
  
  // bank list for writing out the event
  size *= ninfiles;
  writeBanks = newBankList(size);
 
  
  // Make sure that all input files exist
  for (ifile=0; ifile < ninfiles; ifile ++ )
    {
      if (!(fl=fopen(infiles[ifile],"r")))
	{
	  fprintf(stderr,"can't open %s\n",infiles[ifile]);
	  exit(2);
	}
    }
  
  // make sure it is safe to start the output file
  if ( ((fl=fopen(outfile,"r"))) && (!fMode))
    {
      fprintf(stderr,"%s: file exists\n",outfile);
      exit(2);
    }
  if (!(fl=fopen(outfile,"w")))
    {
      fprintf(stderr,"can't start %s\n",outfile);
      exit(2);
    }
  fclose(fl);
  
  // dst - open the input files
  mode = MODE_READ_DST;
  for (ifile=0; ifile<ninfiles; ifile++)
    {
      if ((rc=dstOpenUnit(inUnit[ifile],&infiles[ifile][0],mode)) != 0)
	{
	  fprintf(stderr,"can't dst-open %s for reading\n",infiles[ifile]);
	  exit(2);
	}
    }
  
  // output file
  mode = MODE_WRITE_DST;
  if ((rc=dstOpenUnit(outUnit, outfile, mode)) != 0)
    {
      fprintf(stderr,"can't dst-open %s for writing\n",outfile);
      exit(2);
    }
  
  // read all events in the 1st file
  while ((rc = eventRead (inUnit[0], wantBanks, gotBanks[0], &event)) > 0) 
    {
      if (!event)
	{
	  fprintf(stderr,"Corrupted event!\n");
	  exit(2);
	}
      
      // reset the write bank list
      clrBankList(writeBanks);
      
      // copy all the banks from the 1st event in the 1st file
      cpyBankList(writeBanks, gotBanks[0]);
            
      // read one event from every other intput file
      for (ifile=1; ifile<ninfiles; ifile++)
	{ 
	  rc = eventRead (inUnit[ifile], wantBanks, gotBanks[ifile], &event);
	  if ( (rc <= 0) || (!event)) 
	    {
	      fprintf (stderr, "Error: '%s' has less events than '%s'\n",
		       infiles[ifile],infiles[0]);
	      exit(2);
	    }
	  // add banks from each event to the write bank list
	  sumBankList(writeBanks, gotBanks[ifile]);
	}
      
      // write out the event which now should have the sum of all banks 
      // from all other events in the input files
      if ((rc=eventWrite(outUnit, writeBanks, TRUE)) < 0)
	{
	  fprintf(stderr,"failed to write an event \n");
	  exit(2);
	}      
      
    }


  // Check the rest of the dst files.  
  // They must be on last event, if not - print an error message

  for ( ifile=1; ifile < ninfiles; ifile++)
    {
      rc = eventRead (inUnit[ifile], wantBanks, gotBanks[ifile], &event);
      if ( rc > 0 ) 
	{
	  fprintf (stderr, "Error: '%s' has more events than '%s'\n",
		   infiles[ifile],infiles[0]);
	  exit(2);
	}
    }
  
  // close all the dst units
  dstCloseUnit(outUnit);
  for (ifile=0; ifile < ninfiles; ifile++)
    dstCloseUnit(inUnit[ifile]);
  
  return 0;
}

int parseCmdLine(int argc, char **argv)
{

  int i;
  FILE *wantFl;
  char *line;
  char want_file[0x400];
  char inBuf[0x400];
  
  ninfiles   = 0;
  stdin_mode = 0;
  outfile[0] = '\0';
  fMode      = 0;
  
  if (argc==1)
    {
      integer4 rc;
      fprintf(stderr,"\n\nSum over all dst banks from each event in dst_file1,dst_fiel2,... \n");
      fprintf(stderr,"and write compound events containing all banks into a separate file.\n");
      fprintf(stderr,"Repeating banks are written out once.\n");
      fprintf(stderr,"Input files dst_file1,dst_file2... must have same numbers of events.\n");
      fprintf(stderr,"Maximum %d input files are allowed.\n",DSTSUM_NINFILES);
      fprintf(stderr,"\nUsage: %s -o [dst_file] -i [want_file] and/or [dst_file1] [dst_file2] ...\n",
	      argv[0]);
      fprintf(stderr,"-o:    output file\n");
      fprintf(stderr,"-i:    (opt) get intput files from a want_file\n");
      fprintf(stderr,"--tts: (opt) get input files from stdin\n");
      fprintf(stderr,"       optionally, just pass input file names w/o prefixes\n");
      fprintf(stderr,"-f:    (opt) don't check if the output file exists, just overwrite it\n\n");
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
              fprintf (stderr,"specify the output file!\n");
              return -1;
            }
          else
            {
              sscanf (argv[i], "%s", &outfile[0]);
            }
        }
      
      // want file
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
	      // read the want files, put all dst files found into a buffer.
	      if((wantFl=fopen(want_file,"r")))
		{
		  while(fgets(inBuf,0x400,wantFl))
		    {
		      if (((line=strtok (inBuf, " \t\r\n"))) && (strlen (line) > 0) 
			  && (add_intput_file(line) != 0))
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

      // getting the input files for stdin
      else if (strcmp ("--tts", argv[i]) == 0)
	{ 
	  while(fgets(inBuf,0x400,stdin))
	    {
	      if (((line=strtok (inBuf, " \t\r\n"))) && (strlen (line) > 0) 
		  && (add_intput_file(line) != 0))
		return -1;
	    }
	}
      
      // fore overwrite mode ?
      else if (strcmp ("-f", argv[i]) == 0)
	{
	  fMode = 1;
	}
      // Assume that any other arguments are the input files
      else if (argv[i][0] != '-') 
	{
	  // Just add the file name into a buffer.
	  if(add_intput_file(argv[i]) != 0)
	    return -1;
	}
      else
	{
	  fprintf (stderr, "'%s': unrecognized option\n", argv[i]);
	  return -1;
	}
    }

  if (ninfiles == 0)
    {
      fprintf(stderr,"don't have any input files\n");
      return -1;
    }
  
  if (outfile[0]=='\0')
    sprintf(outfile,"%s","dstsum.dst");
  

  return 0;
}

// adds input file to the list and checks if the
// list size doesn't exceed the maximum
int add_intput_file(char *fname)
{
  if (ninfiles >= DSTSUM_NINFILES)
    {
      fprintf(stderr,"too many input files, maximum is %d\n", DSTSUM_NINFILES);
      return -1;
    }
  sprintf(infiles[ninfiles],"%s",fname);
  ninfiles++;
  return 0;
}
