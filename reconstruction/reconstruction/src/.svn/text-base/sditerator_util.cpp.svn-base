#include "sditerator.h"
// c++ - utility function for rusdhist program


//////////////////////////////////////////////////////////////////////////


/************  CLASS FOR HANDLING THE PROGRAM ARGUMENT ******************/

listOfOpt::listOfOpt()
{
  wantFile[0] = '\0';
  outFile[0]  = '\0';
  verbose = false;
}

// destructor does nothing
listOfOpt::~listOfOpt() {}

bool listOfOpt::getFromCmdLine(int argc, char **argv)
{  
  int i;
  char inBuf[sd_fname_size];
  char *line;
  FILE *wantFl;                  // For reading the want file
  
  if(argc==1) 
    {    
      fprintf(stdout, 
	      "\nUsage: %s -i [DST file list file (ASCII)] -o [output file]\n",
	      argv[0]);
      fprintf(stdout,"Pass input DST file names as arguments w/o any prefixes\n");
      fprintf(stdout,"-i:    Specify the want file with sd dst files\n");
      fprintf(stdout,"--tts: Get dst file names from stdin\n");
      fprintf(stdout,"-o:    Specify the output file\n");
      fprintf(stdout,"-v:    Sets the verbose mode on\n\n");
      return false;
    }
  for(i=1; i<argc; i++)
    {      
      // Get intput from a list file
      if (strcmp ("-i", argv[i]) == 0)
	{
	  if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printf ("Specify the want file!\n");
	      exit (1);
	    }
	  else
	    {
	      sscanf (argv[i], "%1023s", wantFile);
	      // read the want files, put all pass1 dst files found into a buffer.
	      if((wantFl=fopen(wantFile,"r"))==NULL)
		{
		  fprintf(stderr,"Can't open list file %s\n",wantFile);
		  return false;
		}
	      else
		{
		  while(fgets(inBuf,sd_fname_size,wantFl))
                    {
                      if ((line=strtok (inBuf, " \t\r\n")) && strlen (line) > 0 
                          && pushFile(line) != SUCCESS)
                        return false;
                    }
		  fclose(wantFl);
		}
	    }
	}
      // input file names from stdin
      else if (strcmp ("--tts", argv[i]) == 0)
        {
	  while(fgets(inBuf,sd_fname_size,stdin))
	    {
	      if ((line=strtok (inBuf, " \t\r\n")) && strlen (line) > 0 
		  && pushFile(line) != SUCCESS)
		return false;
	    }
        }
      // output file name
      else if (strcmp ("-o", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"Specify the output file!\n");
	      exit (1);
	    }
	  else
	    {
	      sscanf (argv[i], "%1023s", outFile);
	    }
	}
      // verbose mode option
      else if (strcmp ("-v", argv[i]) == 0)
        {
          verbose = true;
        }
      // assume that all arguments w/o the '-' switch are input dst file names
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
  
  return checkOpt();
}


void listOfOpt::printOpts()
{
  fprintf(stdout,"INPUT LIST FILE: %s\n",wantFile);
  fprintf(stdout,"OUTPUT FILE: %s\n",outFile);
}

bool listOfOpt::checkOpt()
{
  bool chkFlag;
  FILE *fl;
  chkFlag = true;
  if((wantFile[0] == '\0') && (countFiles()==0))
    {
      fprintf(stderr, "Don't have any inputs!\n");
      chkFlag = false;
    }
  if(outFile[0] == '\0')
    {
      fprintf(stderr,"Output file was not given!\n");
      chkFlag = false;
    } 
  else 
    {
      // Make sure it's possible to start a new RZ files
      if((fl=fopen(outFile,"w"))==NULL)
	{
	  fprintf(stderr,"Can't start %s\n",outFile);
	  chkFlag = false;
	}
      else
	{
	  fclose(fl);
	}
    }
  return chkFlag;
  
}



