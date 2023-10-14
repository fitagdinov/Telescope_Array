#include "rusdhist.h"

listOfOpt::listOfOpt()
{
  wantFile[0] = 0;
  outfile[0] = 0;
  verbose = false;
  tbopt = false;
  tbflag = 0;
  yymmdd_start = 80511;
  yymmdd_stop  = 401231;
  e3wopt = 0;
  bank_warning_opt = true;
}


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

      // output file name
      else if (strcmp("-o", argv[i]) == 0)
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

      // start date option
      else if (strcmp("-d1", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              fprintf(stderr, "Specify the start date !\n");
              return false;
            }
          else
	    sscanf(argv[i], "%d", &yymmdd_start);
        }
      // stop date option
      else if (strcmp("-d2", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              fprintf(stderr, "Specify the stop date !\n");
              return false;
            }
          else
	    sscanf(argv[i], "%d", &yymmdd_stop);
        }
      // trigger backup option
      else if (strcmp("-tb", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              fprintf(stderr, "Specify the trigger backup flag !\n");
              return false;
            }
          else
            {
              tbopt = true;
              sscanf(argv[i], "%d", &tbflag);
              if (tbflag != 1 && tbflag != 2)
                {
                  fprintf(stderr, "trigger backup option can be either 1 or 2\n");
                  return false;
                }
            }
        }


      // weight E^-3 MC to get the ankle
      else if (strcmp("-w", argv[i]) == 0)
        {
          if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
            {
              fprintf(stderr, "Specify E^-3 MC weight option (0, 1, 2, 3) !\n");
              return false;
            }
          else
	    sscanf(argv[i], "%d", &e3wopt);
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
          fprintf(stderr, "'%s': unrecognized option\n", argv[i]);
          return false;
        }
    }

  return checkOpt();
}

void listOfOpt::printOpts()
{
  const char ynstr[2][10] =
    {
    "NO", "YES"
    };
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
  fprintf(stdout, "OUTPUT FILE: %s\n", outfile);
  fprintf(stdout, "START DATE(YYMMDD): %06d\n",yymmdd_start);
  fprintf(stdout, "STOP  DATE(YYMMDD): %06d\n",yymmdd_stop);
  fprintf(stdout, "TRIGGER BACKUP CUT: %s\n", ynstr[(tbopt ? 1 : 0)]);
  if (tbopt)
    fprintf(stdout, "TRIGGER BACKUP FLAG: %d\n", tbflag);
  fprintf(stdout, "WEIGHT E^-3 MC TO GET THE ANKLE: %s",ynstr[(e3wopt>0 ? 1 : 0)]);
  if (e3wopt==1)
    fprintf(stdout," - use HiRes p-law and HiRes 18.65 eV ankle\n");
  else if(e3wopt==2)
    fprintf(stdout, " - use HiRes p-law and TASD 18.75 eV ankle\n");
  else
    fprintf(stdout,"\n");
  fprintf(stdout,"DISPLAY MISSING BANK WARNINGS: %s\n",(bank_warning_opt ? "YES" : "NO" ));
  fflush(stdout);
}

bool listOfOpt::checkOpt()
{

  if (!outfile[0])
    sprintf(outfile, "rusdhist.root");

  if (countFiles() == 0)
    {
      fprintf(stderr, "Don't have any inputs dst files!\n");
      return false;
    }
  if (yymmdd_start > yymmdd_stop)
    {
      fprintf(stderr,
	      "Start date(%06d) is geater than the stop date (%06d)!\n",
	      yymmdd_start,yymmdd_stop);
      return false;
    }
  if (e3wopt != 0 && e3wopt != 1 && e3wopt != 2)
    {
      fprintf(stderr,"E^-3 MC weight option can be either 0 or 1 or 2\n");
      return false;
    }
  return true;
}

void listOfOpt::printMan()
{
  fprintf(stdout, "\nUsage: %s [in_file1 ...] and/or -i [list file]  -o [output file (.root-file)]\n", progName);
  fprintf(stderr, "INPUT:\n");
  fprintf(stderr, "pass input dst file names as arguments without any prefixes or switches\n");
  fprintf(stderr, "-i <string>  : or specify a list file (with dst files)\n");
  fprintf(stderr, "--tts        : or get input dst file names from stdin\n");
  fprintf(stderr, "OUTPUT:\n");
  fprintf(stderr, "-o <string>  : output file (default is 'rusdhist.root')\n");
  fprintf(stderr, "OTHER:\n");
  fprintf(stderr, "-d1 <int>    : start date, yymmdd format; default = %06d\n",yymmdd_start);
  fprintf(stderr, "-d2 <int>    : stop date, yymmdd format;  default = %06d\n",yymmdd_stop);
  fprintf(stderr, "-tb <int>    : set the trigger backup flag (must have sdtrgbk bank)\n");
  fprintf(stderr, "                1 - histograms over events that pass the trigger\n");
  fprintf(stderr, "                2 - histograms over events that don't pass the trigger\n");
  fprintf(stderr, "-w <int>     : apply ankle weights to E^-3 MC: \n");
  fprintf(stderr, "                1 - HiRes ankle, 2 - TASD ankle\n");
  fprintf(stderr, "-no_bw <int> : disable warnings about the missing banks\n");
  fprintf(stderr,"\n\n");
}

