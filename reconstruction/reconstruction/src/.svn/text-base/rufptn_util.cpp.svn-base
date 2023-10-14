#include "rufptn.h"
#include "rufptn_constants.h"

// c++ - utility function for rufptn program

/*****  CLASS FOR HANDLING THE PROGRAM ARGUMENTS ******************/

listOfOpt::listOfOpt()
{
  wantFile[0]      =  0;
  dout[0]          =  0;
  outfile[0]       =  0;
  useICRRbank      =  false;
  fOverwriteMode   =  false;
  verbose          =  0;
  ignore_bsdinfo   =  false; // by default, always use bsdinfo DST bank if it's available
  bank_warning_opt =  true;  // by default, print out warning messages
                             // when the event banks are missing, etc
  bad_sd_file[0] =  0;
  bad_sd_fp      =  0;     // by default, information on bad SDs is not written out
  
  // default value of speed of light in space-time
  // pattern recognition
  stc            =  1.0;
  
}


listOfOpt::~listOfOpt() 
{
  // use desctructor to finish the output file on bad SDs if it's used
  if(bad_sd_fp)
    fclose(bad_sd_fp);
}

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
	
	// output file for information on bad SDs
        else if (strcmp("-bad_sd", argv[i]) == 0)
          {
            if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
              {
                fprintf(stderr, "error: -bad_sd: specify the output file for bad SD information!\n");
                return false;
              }
            else
	      sscanf(argv[i], "%1023s", bad_sd_file);
          }

        // ICRR bank data option
        else if (strcmp("-icrr", argv[i]) == 0)
	  useICRRbank = true;

        // force overwrite mode
         else if (strcmp("-f", argv[i]) == 0)
	   fOverwriteMode = true;
	
	// speed of light to use in the time pattern recognition
         else if (strcmp("-stc", argv[i]) == 0)
           {
             if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
               {
                 fprintf(stderr, "error: -stc: specify the speed of light value!\n");
                 return false;
               }
             else
               sscanf(argv[i], "%lf", &stc);
           }
	
	// option to not use bsdinfo DST bank and exclude bad counters from the 
	// reconstruction event if bsdinfo DST bank is present
	else if (strcmp("-ignore_bsdinfo", argv[i]) == 0)
	  ignore_bsdinfo = true;

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

	// option to disable warnings about missing banks
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
    time_t now;
    struct tm *d;
    char cur_date_time[0x100];
    time(&now);
    d = localtime(&now);
    strftime(cur_date_time,255,"%Y-%m-%d %H:%M:%S %Z", d);
    fprintf(stdout,"\n\n");
    fprintf(stdout,"%s (%s):\n",progName,cur_date_time);
    if (wantFile[0])
      fprintf(stdout, "WANT FILE: %s\n", wantFile);
    if(outfile[0])
      fprintf(stdout,"OUTPUT FILE: %s\n",outfile);
    else
      fprintf(stdout, "OUTPUT DIRECTORY: %s\n", dout);
    fprintf(stdout, "FORCING THE USE OF ICRR BANK: %s\n",(useICRRbank ? "YES" : "NO"));
    fprintf(stdout, "OVERWRITING THE OUTPUT FILES IF EXIST: %s\n",(fOverwriteMode ? "YES" : "NO"));
    fprintf(stdout, "VERBOSITY LEVEL: %d\n",verbose);
    fprintf(stdout, "DISPLAY MISSING BANK WARNINGS: %s\n",(bank_warning_opt ? "YES" : "NO"));
    if(bad_sd_file[0])
      fprintf(stdout,"INFORMATION ON BAD SDs: %s\n", bad_sd_file);
    fprintf(stdout, "IGNORE bsdinfo DST BANK WHEN AVAILABLE: %s\n",(ignore_bsdinfo ? "YES" : "NO"));
    fprintf(stdout, "SPEED OF LIGHT USED BY SPACE-TIME PATTERN RECOGNITION: %.1f c\n",stc);
    fprintf(stdout,"\n");
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
        fprintf(stderr, "error: don't have any inputs dst files!\n");
        return false;
      }

    if(bad_sd_file[0])
      {
	if((bad_sd_fp=fopen(bad_sd_file,"r")))
	  {
	    fclose(bad_sd_fp);
	    bad_sd_fp = 0;
	    if(!fOverwriteMode)
	      {
		fprintf(stderr,"error: %s exists; use -f option to overwrite the output files\n",
			bad_sd_file);
		return false;
	      }
	  }
	if(!(bad_sd_fp=fopen(bad_sd_file,"w")))
	  {
	    fprintf(stderr,"error: can't open %s for writing\n",bad_sd_file);
	    bad_sd_fp = 0;
	    return false;
	  }
      }
    return true;
  }

void listOfOpt::printMan()
{
  fprintf(stderr,"\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr,"TA SD signal analysis, pattern recognition, and simplified geometry fitting program\n");
  fprintf(stderr,"Uses TA SD DATA OR MC that have either tasdcalibev or rusdraw DST banks\n");
  fprintf(stderr,"Adds: rufptn (signal analysis+pattern recognition), rusdgeom (simplified geometry) banks\n");
  fprintf(stderr,"For MC, adds rusdmc1 bank (additional variables calculated from the thrown values)\n");
  fprintf(stderr,"If starts from tasdcalibev then adds rusdraw for real data, rusdraw,rusdmc,rusdmc1 (MC)\n");
  fprintf(stderr,"Does not filter out any events. Quality cuts should be applied after the analysis\n");
  fprintf(stderr,"Author: Dmitri Ivanov <dmiivanov@gmail.com>\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr,"\nUsage: %s [in_file1 ...] and/or -i [list file]  -o [output directory]\n",progName);
  fprintf(stderr, "\nINPUT:\n");
  fprintf(stderr,"pass input dst file names simply as arguments without any prefixes or switches\n");
  fprintf(stderr, "-i <string>      : or specify a list file with dst files\n");
  fprintf(stderr, "--tty            : or get input dst file names from stdin\n");
  fprintf(stderr, "\nOUTPUT:\n");
  fprintf(stderr, "-o <string>      : output directory (default is './').  Output files are generated by adding .rufptn.dst.gz suffixes\n");
  fprintf(stderr, "-o1f <string>    : or specify a single output file.  All dst output will go to one file. Overrides the '-o' option\n");
  fprintf(stderr, "\nOTHER:\n");
  fprintf(stderr, "-icrr            : force the use of tasdcalibev bank, relevant if you have DST files with both rusdraw and tasdcalibev banks\n");
  fprintf(stderr, "-f               : don't check if output files exist, overwrite them\n");
  fprintf(stderr, "-bad_sd <string> : specify the ASCII file to write out the information on bad SDs; by default, no bad SD info is written\n");
  fprintf(stderr, "-v <int>         : verbosity level (integer number, >= 0)\n");
  fprintf(stderr, "-no_bw           : disable warnings about the missing banks\n");
  fprintf(stderr, "\nEXPERTS ONLY:\n");
  fprintf(stderr, "-ignore_bsdinfo  : do not exclude bad counters in bsdinfo DST bank from the reconstruction (when bsdinfo bank is present)\n");
  fprintf(stderr, "-stc <float>     : set the speed of light to use in space-time pattern recognition (units in [c])\n");
  fprintf(stderr, "                   (default is %.1f; set it to a value less than 1.0 to relax space-time clustering requirements)\n",stc);
  fprintf(stderr, "\n\n");
}
