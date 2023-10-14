#include "rufldf.h"


listOfOpt::listOfOpt()
{
  wantFile[0]      =  0;
  dout[0]          =  0;
  outfile[0]       =  0;
  fOverwriteMode   =  false;
  verbose          =  0;
  bank_warning_opt =  true;
}

listOfOpt::~listOfOpt() {;}

bool listOfOpt::getFromCmdLine(int argc, char **argv)
  {
    int i;
    char inBuf[0x400];
    char *line;
    FILE *wantFl; // For reading the want file

    sprintf(progName,"%s",argv[0]);

    if (argc == 1)
      {
        printMan();
        return false;
      }
    for (i = 1; i < argc; i++)
      {

        // manual
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
                fprintf(stderr,"error: -i: specify the want file!\n");
                return false;
              }
            else
              {
                sscanf(argv[i], "%1023s", wantFile);
                // read the want files, put all rusdraw dst files found into a buffer.
                if ((wantFl = fopen(wantFile, "r")) == NULL)
                  {
                    fprintf(stderr, "error: -i: can't read %s\n", wantFile);
		    return false;
                  }
                else
                  {
                    while (fgets(inBuf, 0x400, wantFl))
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

        // output directory name
        else if (strcmp("-o", argv[i]) == 0)
          {
            if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
              {
                fprintf(stderr, "error: -o: specify the output directory!\n");
                return false;
              }
            else
	      sscanf(argv[i], "%1023s", dout);
          }
	
        // single output file name
        else if (strcmp("-o1f", argv[i]) == 0)
          {
            if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
              {
                fprintf(stderr, "error: -o1f: specify the output file!\n");
                return false;
              }
            else
	      sscanf(argv[i], "%1023s", outfile);
          }

        // force overwrite mode
	else if (strcmp("-f", argv[i]) == 0)
	  fOverwriteMode = true;
	
	// verbosity level
	else if (strcmp("-v", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -v: specify the verbosity level (>= 0)\n");
		return false;
	      }
	    else
	      sscanf(argv[i], "%d", &verbose);
	  }


	// bank warning option
        else if (strcmp("-no_bw", argv[i]) == 0)
	  bank_warning_opt = false;
	
        // assume that all arguments w/o the '-' switch are input dst file names
        else if (argv[i][0] != '-')
          {
            if (pushFile(argv[i]) != SUCCESS)
              return false;
          }
        else
          {
            fprintf(stderr, "error: %s: unrecognized option\n", argv[i]);
            return false;
          }
      }

    return checkOpt();
  }

void listOfOpt::printOpts()
  {
    const char noy_str[2][10]={"NO","YES"};
    time_t now;
    struct tm *d;
    char cur_date_time[0x100];
    time(&now);
    d = localtime(&now);
    strftime(cur_date_time,255,"%Y-%m-%d %H:%M:%S %Z", d);
    fprintf(stdout,"\n\n");
    fprintf(stdout,"%s (%s):\n",progName,cur_date_time);
    if (wantFile[0])
      fprintf(stdout,"WANT FILE: %s\n", wantFile);
    if(outfile[0])
      fprintf(stdout,"OUTPUT FILE: %s\n",outfile);
    else
      fprintf(stdout,"OUTPUT DIRECTORY: %s\n", dout);
    fprintf(stdout,"OVERWRITING  THE OUTPUT FILES IF EXIST: %s\n",noy_str[(int)fOverwriteMode]);
    fprintf(stdout,"VERBOSITY LEVEL: %d\n",verbose);
    fprintf(stdout,"DISPLAY MISSING BANK WARNINGS: %s\n",noy_str[(int)bank_warning_opt]);
    fprintf(stdout,"\n\n");
    fflush(stdout);
  }

bool listOfOpt::checkOpt()
  {

    if (!dout[0])
       sprintf(dout, "./");

    // if the output file does not have a reasonable DST suffix,
    // then report a problem
    if (outfile[0] && !(SDIO::check_dst_suffix(outfile)))
      return false;
    if (countFiles() == 0)
      {
        fprintf(stderr, "Don't have any inputs dst files!\n");
        return false;
      }
    return true;
  }

void listOfOpt::printMan()
{
  fprintf(stderr,"\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr,"TA SD LDF and Geometry fitting program. Also determines the event energy\n");
  fprintf(stderr,"Next step in the chain after rufptn.run program. Runs on dst files produced by rufptn.run\n");
  fprintf(stderr,"Adds: rufldf DST bank (LDF fit and energy information)\n");
  fprintf(stderr,"Updates rusdgeom DST bank with better gometry fit values\n");
  fprintf(stderr,"Does not filter out any events. Quality cuts should be applied after the analysis\n");
  fprintf(stderr,"Author: Dmitri Ivanov <ivanov@physics.rutgers.edu>\n");
  fprintf(stderr,"****************************************************************************************\n");

  fprintf(
          stderr,
          "\nUsage: %s [in_file1 ...] and/or -i [list file]  -o [output directory]\n",
          progName);
  fprintf(stderr, "INPUT:\n");
  fprintf(stderr,"pass input dst file names as arguments without any prefixes or switches\n");
  fprintf(stderr, "-i <string>   : or specify a list file with dst files\n");
  fprintf(stderr, "--tty:        : or get input dst file names from stdin\n");
  fprintf(stderr, "OUTPUT:\n");
  fprintf(stderr, "-o <string>   : output directory (default is './')\n");
  fprintf(stderr, "-o1f <string> : specify a single output file.  All dst output will go to this file. Overrides the '-o' option.\n");
  fprintf(stderr, "OTHER:\n");
  fprintf(stderr, "-f            : don't check if output files exist, overwrite them\n");
  fprintf(stderr, "-v <int>      : set the verbosity level (integer number, >= 0)\n");
  fprintf(stderr, "-no_bw <int>  : disable warnings about the missing banks\n");
  fprintf(stderr, "\n\n");
}
