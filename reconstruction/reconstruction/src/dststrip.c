/* Last modified: DI 20171206 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "event.h"
#include "filestack.h"


static char outfile[0x400];
integer4 fMode;     // force-overwrite mode
integer4 stripBanks; // list of banks to be removed from each event
static int parseCmdLine(int argc, char **argv);

int main(int argc, char **argv)
{
  char *infile;
  integer4 ievent,size,rc,event,mode;
  integer4 wantBanks,gotBanks,writeBanks;
  integer4 inUnit,outUnit;
  FILE *fl;
  size = nBanksTotal();  
  stripBanks = newBankList(size);
  if (parseCmdLine(argc, argv) != 0)
    exit(2);
  wantBanks = newBankList(size);
  eventAllBanks(wantBanks);
  gotBanks = newBankList(size);
  writeBanks = newBankList(size);
  inUnit  = 1;
  outUnit = 2;
  if (((fl=fopen(outfile,"r"))) && (!fMode))
    {
      fprintf(stderr,"%s: file exists\n",outfile);
      exit(2);
    }
  if(fl)
    {
      fclose(fl);
      fl = 0;
    }
  if (!(fl=fopen(outfile,"w")))
    {
      fprintf(stderr,"can't start %s\n",outfile);
      exit(2);
    }
  fclose(fl);
  mode = MODE_WRITE_DST;
  if ((rc=dstOpenUnit(outUnit, outfile, mode)) != 0)
    {
      fprintf(stderr,"can't dst-open %s for writing\n",outfile);
      exit(2);
    } 
  ievent=0;
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
	  cpyBankList(writeBanks, gotBanks);
	  difBankList(writeBanks, stripBanks);
	  if ((rc=eventWrite(outUnit, writeBanks, TRUE)) < 0)
	    {
	      fprintf(stderr,"failed to write an event \n");
	      exit(2);
	    }      
	  ievent ++;
	}    
      dstCloseUnit(inUnit);
    }
  fprintf(stdout, "%d events\n",ievent);
  dstCloseUnit(outUnit);
  return 0;
}

int parseCmdLine(int argc, char **argv)
{

  int i;
  outfile[0] = '\0';
  fMode      = 0;  
  if (argc==1)
    {
      integer4 rc;
      fprintf(stderr,"\nStrip specified dst banks from events in file1,file2 ...\n");
      fprintf(stderr,"\nUsage: %s -o [dst_file] file1 file2 ...  -bank1 -bank2 or -bank3,bank4 ...\n",
	      argv[0]);
      fprintf(stderr,"-o:    (opt) output file\n");
      fprintf(stderr,"-f:    (opt) don't check if the output file exists, just overwrite it\n\n");
      fputs("\nCurrently recognized banks:", stderr);
      dscBankList((rc=newBankList(nBanksTotal()),eventAllBanks(rc),rc),stderr);
      fprintf(stderr,"\n\n");
      return -1;
    }
  clrBankList(stripBanks);
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
              sscanf (argv[i], "%s", outfile);
	      continue;
            }
	}
      
      // fore overwrite mode ?
      else if (strcmp ("-f", argv[i]) == 0)
	{
	  fMode = 1;
	  continue;
	}
      // assume that the bank name has been given
      // after a dash (if the argument is not an option)
      else if (argv[i][0] == '-')
	{
	  integer4 bank_id = 0;
	  char* name       = 0;
	  if (strchr(&argv[i][1],','))
            {
              name = strtok (&argv[i][1], ",");
              while(name != NULL) 
                {
                  if((bank_id=eventIdFromName(name)))
		    addBankList(stripBanks, bank_id);
                  else
		    fprintf(stderr, "unrecognized bank: %s\n",name);
                  name = strtok (NULL, ",");
                }
            }
	  else
	    {
	      if((bank_id=eventIdFromName(&argv[i][1])))
		addBankList(stripBanks, bank_id);
	      else
		fprintf(stderr, "unrecognized bank: %s\n",&argv[i][1]);
	    }
	}
      // everything else is assumed to be dst file names
      else
	pushFile(argv[i]);
    }
  
  if (outfile[0]=='\0')
    sprintf(outfile,"%s","dststrip.dst");
  
  if (countFiles() == 0)
    {
      fprintf(stderr,"no input files\n");
      return -1;
    }  
  return 0;
}
