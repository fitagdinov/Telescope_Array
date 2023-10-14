#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sddstio.h"
#include "filestack.h"


static bool parseCmdLine(int argc, char **argv);



int main(int argc, char **argv)
{
  sddstio_class *dstio;
  char *infile;
  FILE *fp;
  int yymmdd;
  int hhmmss;
  int usec;
  int igevent;
  if (!parseCmdLine(argc,argv))
    return -1; 
  dstio = new sddstio_class();
  fp = stdout;
  while((infile=pullFile()))
    {
      if(!dstio->openDSTinFile(infile))
	return -1;
      while(dstio->readEvent())
	{
	  if (!dstio->haveBank(SDTRGBK_BANKID))
	    {
	      fprintf(stderr,"sdtrgbk bank not found\n");
	      return -1;
	    }
	  igevent = (int)sdtrgbk_.igevent;
	  if (dstio->haveBank(TASDCALIBEV_BANKID))
	    {
	      yymmdd = tasdcalibev_.date;
	      hhmmss = tasdcalibev_.time;
	      usec   = tasdcalibev_.usec;
	    }
	  else if ((dstio->haveBank(RUSDRAW_BANKID)) && !(dstio->haveBank(TASDCALIBEV_BANKID)))
	    {
	      yymmdd = rusdraw_.yymmdd;
	      hhmmss = rusdraw_.hhmmss;
	      usec   = rusdraw_.usec;
	    }
	  else if (dstio->haveBank(TASDEVENT_BANKID))
	    {
	      yymmdd = tasdevent_.date;
	      hhmmss = tasdevent_.time;
	      usec   = tasdevent_.usec;
	    }
	  else
	    {
	      fprintf(stderr,"must have either tasdcalibev or rusdraw or tasdevent banks\n");
	      return -1;
	    }
	  if (igevent > 1)
	    continue;

	  fprintf(fp, "%06d %06d.%06d %d\n",yymmdd,hhmmss,usec,igevent);
	  
	}
      dstio->closeDSTinFile();
    }
  fprintf(stdout,"\nDone\n");
  return 0;
}

bool parseCmdLine(int argc, char **argv)
{

  int i;
  char *line;
  char inBuf[0x400];
  FILE *fp;
  
  if(argc == 1) 
    {
      fprintf(stdout, "\nTo print out the events that did not pass the SD trigger backup\n");
      fprintf(stdout, "\nUsage: %s [in_file1 ...] and/or -i [list file] \n", argv[0]);
      fprintf(stderr, "Pass input dst file names as arguments without any prefixes\n");
      fprintf(stderr, "-i:    Specify the want file (with dst files)\n");
      fprintf(stderr, "--tts: Or get input dst file names from stdin\n");
      return false;
    }
  for (i = 1; i < argc; i++)
    {
      if (strcmp("-i", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              printf("Specify the list file!\n");
              return false;
            }
          else
            {
              if ((fp = fopen(argv[i], "r")) == NULL)
                {
                  fprintf(stderr, "can't open %s\n", argv[i]);
                  return false;
                }
              else
                {
                  while (fgets(inBuf, 0x400, fp))
                    {
                      if (((line = strtok(inBuf, " \t\r\n"))) && (strlen(line) > 0))
                        {
                          if (pushFile(line) != SUCCESS)
                            return false;
                        }
                    }
                  fclose(fp);
                }
            }
        }
      else if (strcmp("--tts", argv[i]) == 0)
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
      else if (argv[i][0] != '-')
        {
          if (pushFile(argv[i]) != SUCCESS)
            return false;
        }
      else
        {
          fprintf(stderr, "'%s': unrecognized option\n", argv[i]);
          return false;
        }
    }
  if (!countFiles())
    {
      fprintf(stderr,"no input files\n");
      return false;
    }
  return true;
}

