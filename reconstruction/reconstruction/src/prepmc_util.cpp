#include "prepmc.h"
// c++ - utility function for prepmc program

//////////////////////////////////////////////////////////////////////////


/************  CLASS FOR HANDLING THE PROGRAM ARGUMENTS *****************/

listOfOpt::listOfOpt()
{
  wantFile[0]   =  0;
  dout[0]       =  0;
  outpr[0]      =  0;
  verbose       =  false;
}

// destructor does nothing
listOfOpt::~listOfOpt() {}

bool listOfOpt::getFromCmdLine(int argc, char **argv)
{
  int i,l;
  char inBuf[sd_fname_size];
  char *line;
  FILE *wantFl;                  // For reading the want file
  strcpy(progName,argv[0]);
  if(argc==1)
    {
      printMan();
      return false;
    }
  for(i=1; i<argc; i++)
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
      
      // want file
      else if (strcmp ("-i", argv[i]) == 0)
	{
	  if (argv[++i][0] == 0 || argv[i][0]=='-')
	    {
	      printf ("Specify the want file!\n");
	      exit (1);
	    }
	  else
	    {
	      sscanf (argv[i], "%1023s", wantFile);
	      
	      // read the want files, put all dst files found into a buffer.
	      if((wantFl=fopen(wantFile,"r"))==NULL)
		{
		  fprintf(stderr,"Can't open wantFile %s\n",wantFile);
		  return false;
		}
	      else
		{
		  while(fgets(inBuf,sd_fname_size,wantFl))
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
      else if (strcmp ("--tts", argv[i]) == 0)
	{
	  while(fgets(inBuf,sd_fname_size,stdin))
	    {
	      if (((line = strtok(inBuf, " \t\r\n")))
		  && (strlen(line) > 0))
		{
		  if (pushFile(line) != SUCCESS)
		    return false;
		}
	    }
	}
      
      // output directory
      else if (strcmp ("-o", argv[i]) == 0)
        {
          if (argv[++i][0] == 0 || argv[i][0]=='-')
            {
              fprintf (stderr,"Specify the DST output directory!\n");
              exit (1);
            }
          else
            {
              sscanf (argv[i], "%1023s", &dout[0]);
	      // Make sure that the output directory ends wiht '/'
	      l = (int)strlen(dout);
	      if(dout[l-1]!='/')
		{
		  dout[l] = '/';
		  dout[l+1] = '\0';
		}
            }
        }
      // output file prefix
      else if (strcmp ("-pr", argv[i]) == 0)
        {
          if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
            {
              fprintf (stderr,"Specify the output file prefix!\n");
              exit (1);
            }
          else
            {
              sscanf (argv[i], "%1023s", &outpr[0]);
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

      // Assume that any other arguments are the input files
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
    fprintf(stdout, "WANT FILE: %s\n", wantFile);
    fprintf(stdout, "OUTPUT DIRECTORY: %s\n", dout);
    fprintf(stdout, "OUTPUT FILE PREFIX: ");
    if (outpr[0])
      {
        fprintf(stdout, "%s\n", outpr);
      }
    else
      {
        fprintf(stdout, "NOT USED;  ");
        fprintf(stdout, "USING DEFAULT OUTPUT NAME: 'thrown.root'\n");
      }
  }

void listOfOpt::printMan()
{
  fprintf(stderr,
	  "\nUsage: %s [dst_file1] [dst_file2 ...] and/or -i [list_file] -o [output directory]\n",
	  progName);
  fprintf(stderr, "Pass DST input file names w/o any prefixes\n");
  fprintf(stderr, "-i:    Specify the want-file with dst files\n");
  fprintf(stderr, "--tts: Get input dst file names from stdin\n");
  fprintf(stdout, "-o:    Output directory, default is './'\n");
  fprintf(stderr, "-pr:   Optional: prefix for the output files ( output file will end with '.thrown.root' )\n");
  fprintf(stderr, "-v:    Optional: Sets the verbose mode on\n");
  fprintf(stderr,"\n");
}

bool listOfOpt::checkOpt()
{
  if (!dout[0])
    sprintf(dout,"./");
  if(countFiles()==0)
    {
      fprintf(stderr, "Don't have any inputs!\n");
      return false;
    }
  return true;
}
