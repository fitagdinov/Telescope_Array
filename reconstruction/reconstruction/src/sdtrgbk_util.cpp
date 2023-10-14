#include "sdtrgbk.h"
// c++ - utility function for sdtrgbk program


//////////////////////////////////////////////////////////////////////////////


/**** CLASS FOR HANDLING THE PROGRAM ARGUMENTS ******************************/

listOfOpt::listOfOpt()
{
  wantFile[0] = 0;
  dout[0] = 0;
  outfile[0] = 0;
  ignore_bsdinfo =  false; // by default, always exlcude bad SDs in bsdinfo DST bank if it's present
  icrrbankoption = -1;
  write_trig_only = false;
  write_notrig_only = false;
  fOverwriteMode = false;
  verbosity = 0;
}

// destructor does nothing
listOfOpt::~listOfOpt()
{
}

bool listOfOpt::getFromCmdLine(int argc, char **argv)
{
  int i;
  char inBuf[sd_fname_size];
  char *line;
  FILE *wantFl; // For reading the want file

  sprintf(progName, "%s", argv[0]);

  if (argc == 1)
    {
      printMan();
      return false;
    }
  for (i = 1; i < argc; i++)
    {

      // print the manual
      if ((strcmp("-h", argv[i]) == 0) || (strcmp("--h", argv[i]) == 0) || (strcmp("-help", argv[i]) == 0) || (strcmp(
          "--help", argv[i]) == 0) || (strcmp("-?", argv[i]) == 0) || (strcmp("--?", argv[i]) == 0) || (strcmp("/?",
          argv[i]) == 0))
        {
          printMan();
          return false;
        }

      // input dst file names from a list file
      else if (strcmp("-i", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              printf("Specify the want file!\n");
              exit(1);
            }
          else
            {
              sscanf(argv[i], "%1023s", wantFile);
              // read the want files, put all rusdraw dst files found into a buffer.
              if ((wantFl = fopen(wantFile, "r")) == NULL)
                {
                  fprintf(stderr, "Can't open wantFile %s\n", wantFile);
                  return false;
                }
              else
                {
                  while (fgets(inBuf, 0x400, wantFl))
                    {
                      if (((line = strtok(inBuf, " \t\r\n"))) && (strlen(line) > 0))
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

      // output directory name
      else if (strcmp("-o", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              fprintf(stderr, "Specify the output directory!\n");
              exit(1);
            }
          else
            {
              sscanf(argv[i], "%1023s", dout);
            }
        }

      // single output file name
      else if (strcmp("-o1f", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              fprintf(stderr, "Specify the output file!\n");
              exit(1);
            }
          else
            {
              sscanf(argv[i], "%1023s", outfile);
            }
        }

      // ICRR bank data option
      else if (strcmp("-icrr", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              fprintf(stderr, "Specify the icrr bank option (0 - tasdevent, 2 - tasdcalibev)\n");
              exit(1);
            }
          else
            {
              if (isdigit(argv[i][0]))
                sscanf(argv[i], "%d", &icrrbankoption);
              else
                {
                  fprintf(stderr, "-icrr: must pass an integer flag\n");
                  return false;
                }
            }
        }

      // option to use bsdinfo DST bank to exclude counters
      // from the trigger
      else if (strcmp("-ignore_bsdinfo", argv[i]) == 0)
	ignore_bsdinfo = true;
      
      // write only triggered events

      else if (strcmp("-t", argv[i]) == 0)
        {
          write_trig_only = true;
        }

      // write only non-triggered events

      else if (strcmp("-nt", argv[i]) == 0)
        {
          write_notrig_only = true;
        }

      // force overwrite mode

      else if (strcmp("-f", argv[i]) == 0)
        {
          fOverwriteMode = true;
        }

      // verbosity mode option
      else if (strcmp ("-v", argv[i]) == 0)
        {
	  verbosity = 1;
	  if (i+1 < argc && argv[i+1][0])
	    {
	      if(isdigit(argv[i+1][0]) || (argv[i+1][0] == '-' && isdigit(argv[i+1][1])))
		sscanf (argv[++i], "%d", &verbosity);
	    }
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
  time_t now;
  struct tm *d;
  char cur_date_time[0x100];
  time(&now);
  d = localtime(&now);
  strftime(cur_date_time, 255, "%Y-%m-%d %H:%M:%S %Z", d);
  fprintf(stdout, "\n\n");
  fprintf(stdout, "%s (%s):\n", progName, cur_date_time);

  if (wantFile[0])
    fprintf(stdout, "WANT FILE: %s\n", wantFile);

  if (outfile[0])
    fprintf(stdout, "OUTPUT FILE: %s\n", outfile);
  else
    fprintf(stdout, "OUTPUT DIRECTORY: %s\n", dout);

  
  fprintf(stdout, "Ignore bsdinfo DST bank that excludes bad SDs from trigger: %s\n",(ignore_bsdinfo ? "YES" : "NO"));

  fprintf(stdout, "Use ICRR bank: ");
  if (icrrbankoption != -1)
    {
      fprintf(stdout, "YES, ");
      if (icrrbankoption == 0)
        fprintf(stdout, "tasdevent");
      else if (icrrbankoption == 2)
        fprintf(stdout, "tasdcalibev");
      else
        fprintf(stdout, "INVALID BANK OPTION");
      fprintf(stdout, "\n");
    }
  else
    fprintf(stdout, "NO\n");

  if (write_trig_only)
    fprintf(stdout, "Writing only events that pass the trigger backup\n");
  if (write_notrig_only)
    fprintf(stdout, "Writing only events that do not pass the trigger backup\n");
  
  fprintf(stdout, "Overwriting the output files, if exist: %s\n",(fOverwriteMode ? "YES" : "NO"));
  
  fprintf(stdout, "Verbosity level: %d\n", verbosity);
  
  fprintf(stdout, "\n\n");
}

bool listOfOpt::checkOpt()
{

  if ((icrrbankoption != -1) && (icrrbankoption != 0) && (icrrbankoption != 2))
    {
      fprintf(stderr,
          "Wrong ICRR bank option: -icrr 0 - tasdevent (raw event format), -icrr 2 - for tasdcalibev (calibrated event format)\n");
      return false;
    }

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
  if (write_trig_only && write_notrig_only)
    {
      fprintf(stderr, "-t and -nt options can't be used simultaneously\n");
      return false;
    }

  return true;
}

void listOfOpt::printMan()
{
  fprintf(stderr,"\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr,"TA SD Trigger verification program. Checks if TA SD trigger logic works as expected\n");
  fprintf(stderr,"Runs on TA SD data (events that have tasdevent or tasdcalibev or rusdraw banks)\n");
  fprintf(stderr,"Also runs on TA SD MC (with tasdcalibev or rusdraw banks)\n");
  fprintf(stderr,"Adds sdtrgbk DST banks which contains the trigger verification information\n");
  fprintf(stderr,"Does not filter out any events\n");
  fprintf(stderr,"Author: Dmitri Ivanov <dmiivanov@gmail.com>\n");
  fprintf(stderr,"****************************************************************************************\n");
  
  fprintf(stderr, "\nUsage: %s [in_file1 ...] and/or -i [list file]  -o [output directory]\n", progName);
  fprintf(stderr, "INPUT:\n");
  fprintf(stderr, "pass input dst file names as arguments without any prefixes\n");
  fprintf(stderr, "-i <int>        : specify the want file (with dst files)\n");
  fprintf(stderr, "--tts           : or get input dst file names from stdin\n");
  fprintf(stderr, "OUTPUT:\n");
  fprintf(stderr, "-o <string>     : output directory (default is './')\n");
  fprintf(stderr, "-o1f <string>   : one output file for all events. This overrides the '-o' option\n");
  fprintf(stderr, "OTHER:\n");
  fprintf(stderr, "-icrr <int>     : to use ICRR banks: 0 - tasdevent, 2 - tasdcalibev\n");
  fprintf(stderr, "-ignore_bsdinfo : ignore bsdinfo DST bank if present, otherwise exclude bad SDs in bsdinfo from trigger\n");
  fprintf(stderr, "-t:             : write only events that pass the trigger backup\n");
  fprintf(stderr, "-nt:            : write only events that do not pass the trigger backup\n");
  fprintf(stderr, "-f:             : don't check if output files exist, overwrite them\n");
  fprintf(stderr, "-v:             : sets the verbosity level to 1\n");
  fprintf(stderr, "-v <int>:       : sets the verbosity level to a desired integer number, e.g. 2 prints more\n\n");
}

