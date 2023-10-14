
#include "rusdpass0io.h"

// check that the acceptable SD data suffix: 0 = wrong, 1 = .Y????, 2 = .tar.bz2
static int rusdpass0io_chk_sd_suf(const char* fname)
{
  const char* ch = 0;
  int  d = 0;
  if(!(ch=strrchr(fname,'.')))
    return 0;
  // .Y???? case
  if(strlen(ch)==6)
    {
      if(ch[0] != '.' || ch[1] != 'Y')
	return 0;
      if(!sscanf(&ch[2],"%d",&d))
	return 0;
      return 1;
    }
  // testing if this is a .tar.bz2 case
  ch = fname - 1;
  do
    {
      ch = strstr(ch + 1, ".tar.bz2");
    } while (ch && (strstr(ch + 1, ".tar.bz2")));
  if (!ch || (*(ch + strlen(".tar.bz2"))))
    return 0;
  return 2;
}

/************  CLASS FOR HANDLING THE PROGRAM ARGUMENTS *****************/

listOfOpt::listOfOpt()
{
  yymmdd = 0;           // initialize the (mandatory) date
  sprintf(dout,"./");   // default output directory
  sprintf(prodir,"./"); // default processing directory for intermediate files
  evtfile[0] = 0;       // initialize event output DST file
  monfile[0] = 0;       // initialize monitoring output DST file
  wevt = 1;             // write out the event information
  wmon = 1;             // write out the monitoring information
  rem_tmp = 1;          // remove any temporary files at the end of the processing
  fOverwrite = false;   // don't overwrite any files by default
  fIncomplete = false;  // don't allow incomplete data sets by default
  tmatch_usec = TMATCH_USEC; // set up default value for time matching events
  dup_usec = DUP_USEC;       // set up default value for removing duplicated events
  verbose = 0;          // default level of verbosity
}

// destructor does nothing
listOfOpt::~listOfOpt()
{
}

bool listOfOpt::getFromCmdLine(int argc, char **argv)
{
  int i, l;
  char inBuf[0x400];
  char *line;
  if (argc==1)
    {
      fprintf(stderr,"\nProgram to parse raw TA SD data\n");
      fprintf(stderr,"Inputs are the SD tower run files, either unpacked ascii files (ending with .Y\?\?\?\?) and/or\n");
      fprintf(stderr,"the corresponding *.tar.bz2 files (*.tar.bz2 files can be processed only on unix-like systems).\n");
      fprintf(stderr,"The outputs are two DST files / day, one with rusdraw bank (calibrated events) and one with\n");
      fprintf(stderr,"sdmon bank (calibration and monitoring information).\n");
      fprintf(stderr,"\nUsage: %s -d [yymmdd] .Y\?\?\?\?-file1 and/or .tar.bz2-file2 ... ",argv[0]);
      fprintf(stderr,"and/or -i [ascii list of run files] -d [yymmdd] -o [outdir]\n");
      fprintf(stderr, "\nINPUT:\n");
      fprintf(stderr, "pass .Y\?\?\?\? and/or .tar.bz2 SD tower files on the command line w/o any prefixes or switches\n");
      fprintf(stderr, "-i <string>    : or provide an ascii list file with full paths to raw ascii files ");
      fprintf(stderr,                   "from SD towers (files ending with .Y\?\?\?\?)\n");
      fprintf(stderr, "--tty          : or pipe the .Y\?\?\?\? and/or .tar.bz2 SD tower file names from stdin\n");
      fprintf(stderr, "-d <int>       : readout date in yymmdd format \n");
      fprintf(stderr, "\nOUTPUT:\n");
      fprintf(stderr, "-o <string>    : output DST directory, default: '%s'. If event and/or monitoring DST\n",dout);
      fprintf(stderr, "                 files are not specified, 'SDYYMMDD.rusdraw.dst.gz', 'SDYYMMDD.sdmon.dst.gz'\n");
      fprintf(stderr, "                 files are created in this output directory\n");
      fprintf(stderr, "-evt <string>  : event output DST file, overrides the '-o' option for events\n");
      fprintf(stderr, "-mon <string>  : monitoring output DST file, overrides the '-o' option for monitoring\n");
      fprintf(stderr, "-wevt <int>    : write out the event information (1/0=YES/NO), defaut: %d\n",wevt);
      fprintf(stderr, "-wmon <int>    : write out the monitoring information (1/0=YES/NO), defaut: %d\n",wmon);
      fprintf(stderr, "-rem_tmp <int> : remove the temporary files at the end of the proceesing (1/0=YES/NO), defaut: %d\n",rem_tmp);
      fprintf(stderr, "-pro <string>  : processing directory for temporary/intermediate files, default is %s\n",prodir);
      fprintf(stderr, "-f             : overwrite the output files if they exist\n");
      fprintf(stderr, "\nOTHER:\n");
      fprintf(stderr, "-incomplete        : allow processing when the file list is incomplete (have missing SD tower files)\n");
      fprintf(stderr, "-tmatch_usec <int> : time match events and combine them into one if they occur within this many microseconds\n");
      fprintf(stderr, "                     default: %d\n", tmatch_usec);
      fprintf(stderr, "-dup_usec <int>    : remove duplicated events if you have any two events within this many microseconds\n");
      fprintf(stderr, "                     default: %d\n",   dup_usec);
      fprintf(stderr, "-v <int>           : verbosity level (>0 prints more), default: %d\n",verbose);
      fprintf(stderr,"\n");
      return false;
    }
  for (i=1; i<argc; i++)
    {
      // raw data file names from a list file
      if (strcmp("-i", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("error: -i: specify the list file");
	      return false;
	    }
	  FILE *fp = fopen(argv[i],"r");
	  if(!fp)
	    {
	      printErr("can't open %s for reading",argv[i]);
	      return false;
	    }
	  while (fgets(inBuf, 0x400, fp))
	    {
	      if (((line = strtok(inBuf," \t\r\n")))
		  && (strlen(line) > 0) && rusdpass0io_chk_sd_suf(line))
		{
		  if (pushFile(line) != SUCCESS)
		    return false;
		}
	    }
	  fclose(fp);
	}
      
      // raw data file names from stdin
      else if (strcmp("--tty", argv[i]) == 0)
          {
            while (fgets(inBuf, 0x400, stdin))
              {
                if (((line=strtok(inBuf," \t\r\n")))
		    && (strlen(line) > 0) && rusdpass0io_chk_sd_suf(line))
                  {
                    if (pushFile(line) != SUCCESS)
                      return false;
                  }
              }
          }
      
      // Date
      else if (strcmp("-d", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-d: specify the date!");
	      return false;
	    }
	  else
	    sscanf(argv[i], "%d", &yymmdd);
	}
	
      // output directory
      else if (strcmp("-o", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-o: specify the DST output directory!");
	      return false;
	    }
	  sscanf(argv[i], "%1023s", &dout[0]);
	  // Make sure that the output directory ends wiht '/'
	  l = (int)strlen(dout);
	  if (dout[l-1]!='/')
	    {
	      dout[l] = '/';
	      dout[l+1] = '\0';
	    }
	}
      
      // processing directory
      else if (strcmp("-pro", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-pro: specify the processing directory");
	      return false;
	    }
	  sscanf(argv[i], "%1023s", &prodir[0]);
	  // Make sure that the directory ends wiht '/'
	  l = (int)strlen(prodir);
	  if (prodir[l-1]!='/')
	    {
	      prodir[l] = '/';
	      prodir[l+1] = '\0';
	    }
	}
      
      // event output DST file
      else if (strcmp("-evt", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-evt: specify the event output DST file!");
	      return false;
	    }
	  else
	    {
	      sscanf(argv[i], "%1023s", &evtfile[0]);
	      if(!(SDIO::check_dst_suffix(evtfile)))
		return false;
	    }
	}
      
      // monitoring information output DST file
      else if (strcmp("-mon", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-mon: specify the monitoring information output DST file!");
	      return false;
	    }
	  else
	    {
	      sscanf(argv[i], "%1023s", &monfile[0]);
	      if(!(SDIO::check_dst_suffix(monfile)))
		return false;
	    }
	}
      
      // write out the event information
      else if (strcmp("-wevt", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-wevt: specify the writing event option (1=YES, 0=NO)");
	      return false;
	    }
	  else
	    sscanf(argv[i], "%d", &wevt);
	}
      
      // write out the monitoring information
      else if (strcmp("-wmon", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-wmon: specify the writing monitoring information option (1=YES, 0=NO)");
	      return false;
	    }
	  else
	    sscanf(argv[i], "%d", &wmon);
	}
      
      // remove the temporary files at the end of the processing
      else if (strcmp("-rem_tmp", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-rem_tmp: specify the flag remove temporary files (1=YES, 0=NO)");
	      return false;
	    }
	  else
	    sscanf(argv[i], "%d", &rem_tmp);
	}
      
      // force overwrite mode
      else if (strcmp("-f", argv[i]) == 0)
	fOverwrite = true;
     
      // allow incomplete datesets 
      else if (strcmp("-incomplete", argv[i]) == 0) 
	fIncomplete = true;

      // time matching and combining events time window
      else if (strcmp("-tmatch_usec", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-tmatch_usec: specify the time window using integer number of micro seconds");
	      return false;
	    }
	  else
	    sscanf(argv[i], "%d", &tmatch_usec);
	}
      
      // time matching and combining events time window
      else if (strcmp("-dup_usec", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-dup_usec: specify the time window using integer number of micro seconds");
	      return false;
	    }
	  else
	    sscanf(argv[i], "%d", &dup_usec);
	}
      
      // verbose mode option
      else if (strcmp("-v", argv[i]) == 0)
	{
	  if ( (++i>=argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      printErr("-v: specify the verbose mode flag (integer number >= 0)!");
	      return false;
	    }
	  else
	    sscanf(argv[i], "%d", &verbose);
	}
      
      // assume that all arguments w/o the '-' switch are the input raw data file names
      else if (argv[i][0] != '-')
	{
	  if(rusdpass0io_chk_sd_suf(argv[i]))
	    {
	      if (pushFile(argv[i]) != SUCCESS)
		return false;
	    }
	}
      
      // junk option
      else
	{
	  printErr("'%s': option not recognized", argv[i]);
	  return false;
	}
    }
  return checkOpt();
}

void listOfOpt::printOpts()
{
  fprintf(stdout,"DATE: %06d\n",yymmdd);
  fprintf(stdout,"OUTPUT DIRECTORY: %s\n",dout);
  fprintf(stdout,"PROCESSING DIRECTORY: %s\n",prodir);
  if(wevt)
    fprintf(stdout,"EVENT OUTPUT FILE: %s\n",evtfile);
  if(wmon)
    fprintf(stdout,"MONITORING OUTPUT FILE: %s\n",monfile);
  fprintf(stdout,"REMOVE TEMPORARY FILES: %s\n",(rem_tmp ? "YES" : "NO"));
  fprintf(stdout,"OVERWRITING OUTPUT FILES: %s\n", (fOverwrite ? "YES" : "NO"));
  fprintf(stdout,"ALLOW INCOMPLETE PROCESSING: %s\n", (fIncomplete ? "YES" : "NO"));
  fprintf(stdout,"EVENT RECOMBINATION TIME MATCHING WINDOW: %d us\n",tmatch_usec);
  fprintf(stdout,"DUPLICATE EVENT REMOVAL TIME WINDOW: %d us\n",dup_usec);
  fprintf(stdout,"VERBOSE: %d\n",verbose);
  fprintf(stdout,"\n");
  fflush(stdout);
}
bool listOfOpt::checkOpt()
{
  bool chkFlag;
  chkFlag = true;
  
  if (!countFiles())
    {
      printErr("no input files");
      chkFlag = false;
    }
  
  if (yymmdd == 0)
    {
      printErr("Readout date not specified. Use '-d' option.");
      chkFlag = false;
    }
  
  if(!chkFlag)
    return false;
  
  // default event output file, if needed
  if(!evtfile[0])
    {
      if(!fIncomplete)
	sprintf(evtfile, "%s%s%06d%s", dout, "SD", yymmdd, RUSDRAW_DST_GZ);
      else
	sprintf(evtfile, "%s%s%06d.incomplete%s", dout, "SD", yymmdd, RUSDRAW_DST_GZ);
    }
  
  // default monitoring output file, if needed
  if(!monfile[0])
    {
      if(!fIncomplete)
	sprintf(monfile, "%s%s%06d%s", dout, "SD", yymmdd, SDMON_DST_GZ);
      else
	sprintf(monfile, "%s%s%06d.incomplete%s", dout, "SD", yymmdd, SDMON_DST_GZ);
    }
  if(!fOverwrite)
    {
      FILE* fp = 0;
      if(wevt)
	{
	  if((fp=fopen(evtfile, "r")))
	    {
	      printErr("%s exists; use '-f' option to overwrite files",evtfile);
	      fclose(fp);
	      chkFlag = false;
	    }
	}
      if(wmon)
	{
	  if((fp=fopen(monfile, "r")))
	    {
	      printErr("%s exists; use '-f' option to overwrite files",monfile);
	      fclose(fp);
	      chkFlag = false;
	    }
	}
    }
  return chkFlag;
}
void listOfOpt::printErr(const char *form, ...)
{
  char mess[0x400];
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fprintf(stderr, "listOfOpt:  %s\n", mess);
}

rusdpass0io::rusdpass0io(listOfOpt& passed_opt): opt(passed_opt)
{
  int itower, irun, run_id;
  bool got_earlier, got_later;
  int yy, mm, dd, want_jd;
  int yymmdd, hhmmss, got_jd;
  int jd1[3], jd2[3]; // earliest and latest julian days
  char towerName[3][3];
  bool got_start_date = false;
  char* fname_in_list = 0;
  const char* fname = 0;
  FILE *fp = 0;
  bool enough_files = true;
  int sufid = 0; // suffix identification number (.Y???? = 1, .tar.bz2 = 2)

  sprintf(towerName[0], "%s", "BR");
  sprintf(towerName[1], "%s", "LR");
  sprintf(towerName[2], "%s", "SK");

  SDGEN::parseAABBCC(opt.yymmdd, &yy, &mm, &dd);
  want_jd=SDGEN::greg2jd((2000+yy), mm, dd);
  
  for (itower=0; itower<3; itower++)
    {
      nrawfiles[itower]=0;
      currawfile[itower]=0;
      jd1[itower]=(int)1e9;
      jd2[itower]=0;
      saved_irawfile[itower] = -1;
      saved_iline[itower] = -1;
    }

  // Load the list of BR, LR, SK raw data files into the array
  // so that can keep them throughout the processing
  got_jd = 0;
  while ((fname_in_list=pullFile()))
    {
      sufid = rusdpass0io_chk_sd_suf(fname_in_list);
      switch(sufid)
	{
	  // .Y???? - file
	case 1:
	  fname = fname_in_list;
	  break;
	  
	  // .tar.bz2 - file
	case 2:
	  if(!(fname=unpack_run_file(fname_in_list,opt.prodir)))
	    {
	      printErr("warning: rusdpass0io: failed to unpack %s",fname_in_list);
	      continue;
	    }
	  break;
	  
	  // junk
	default:
	  printErr("warning: rusdpass0io: '%s' has wrong suffix. Allowed suffixes are .Y???? or .tar.bz2",
		   fname);
	}
      
      
      // determine the tower id from the form of the file name
      // (.Y???? - file by now)
      if (!getRunInfo(fname, &itower, &run_id))
	{
	  printErr("error: can't get tower/run_id from '%s'", fname);
	  if(opt.fIncomplete)
	    continue;
	  fprintf(stderr,"use '-incomplete' option to proceed with such files\n");
	  exit(2);
	}

      // by now, it's a .Y???? -file.  make sure it can be opened for reading
      if (!(fp=fopen(fname, "r")))
	{
	  printErr("error: can't open '%s' for reading", fname);
	  if(opt.fIncomplete)
	    continue;
	  fprintf(stderr,"use '-incomplete' option to proceed with such files\n");
	  exit(2);
	}
      else
	fclose(fp);
      
      // Don't want runs in the list which are earlier or later by more than 1 day
      // than the date that we want to process.
      if ((got_start_date=getRunStartDate(fname,&yymmdd,&hhmmss)))
	{
	  SDGEN::parseAABBCC(yymmdd, &yy, &mm, &dd);
	  got_jd=SDGEN::greg2jd((2000+yy), mm, dd);
	  if (abs(want_jd-got_jd)>1)
	    continue;
	  // Get earliest and latest julian day numbers
	  if (jd1[itower] > got_jd)
	    jd1[itower]=got_jd;
	  if (jd2[itower] < got_jd)
	    jd2[itower] = got_jd;
	}
      else
	printErr("can't get run start date from %s", fname);
      
      // make sure that the number of files in the list for each tower doesn't exceed the maximum
      if (nrawfiles[itower]==NRAWFILESPT)
	{
	  printErr("Fatal Error: too many runs (>%d) for %s", NRAWFILESPT,
		   towerName[itower]);
	  fprintf(stdout, "DATE: %06d SUCCESS-ALL: %d\n", opt.yymmdd, 0);
	  exit(3);
	}
      
      // if the file was unpacked by the program then it's a temporary file, so label it accordingly
      is_rawfname_tmp[nrawfiles[itower]][itower] = (sufid==2);
      sprintf(rawfname[nrawfiles[itower]][itower], "%s", fname);
      full_run_id[nrawfiles[itower]][itower]= (2000 + yy) * 1000000 + run_id;
      if (got_start_date)
	run_jd[nrawfiles[itower]][itower] = got_jd;
      // If couldn't get the julian date for some run, then
      // treat the run as if its julian date is the same as the wanted one,
      // so that the run can participate in the parsing process.  If its events / monitoring
      // cycles are all out of date/time then they will be be used (tower_parser will decide that)
      else
	run_jd[nrawfiles[itower]][itower] = want_jd;
      has_date[nrawfiles[itower]][itower] = got_start_date;
      run_needed[nrawfiles[itower]][itower] = true;
      nrawfiles[itower]++;
    }
  enough_files=true;
  for (itower=0; itower<3; itower++)
    {
      if (nrawfiles[itower]==0)
	{
	  printErr("no files for tower id = %d", itower);
	  enough_files = false;
	}
      if (jd1[itower]>=want_jd)
	{
	  printErr("should include at least one %s run 1 day before %06d",
		   towerName[itower], opt.yymmdd);
	  enough_files = false;
	}
      if (jd2[itower]<=want_jd)
	{
	  printErr("should include at least one %s run 1 day after %06d",
		   towerName[itower], opt.yymmdd);
	  enough_files = false;
	}
    }

  // Check if we have enough files (1 for BR, 1 for LR, 1 for SK)
  if (!enough_files && !opt.fIncomplete)
    {
      fprintf(stderr,"use '-incomplete' option if you don't have these files\n");
      exit(3);
    }
  // Sort the files for BR,LR,SK in order of increasing full_run_id
  sort_by_full_run_id();
  
  // Check if any full_run_id's are missing in between.  This means that
  // data is corrputed: did not unpack properly.
  if (!check_missing() && !opt.fIncomplete)
    {
      fprintf(stderr,"use '-incomplete' option if you can't get the missing files\n");
      exit(3);
    }
  // Label off runs that are not necessary

  for (itower=0; itower < 3; itower ++)
    {
      // First pick out only 1 run that starts one day earlier than the wanted date
      got_earlier = false;
      for (irun=(nrawfiles[itower]-1); irun >=0; irun--)
	{
	  if (run_jd[irun][itower] < want_jd)
	    {
	      if (got_earlier)
		{
		  run_needed[irun][itower] = false;
		  continue;
		}
	      run_needed[irun][itower] = true;
	      got_earlier = true;
	    }
	}
      // Pick out only 1 run that starts 1 day after than the wanted date
      got_later = false;
      for (irun=0; irun < nrawfiles[itower]; irun++)
	{
	  if (run_jd[irun][itower] > want_jd)
	    {
	      if (got_later)
		{
		  run_needed[irun][itower] = false;
		  continue;
		}
	      run_needed[irun][itower] = true;
	      got_later = true;
	    }
	}

    }

  // Initialize units and banks for DST writing DST files will be
  // started when 1st events / monitoring cycles are written
  
  // for events
  evt_outUnit = 1;
  evt_outBanks = newBankList(10);
  addBankList(evt_outBanks,RUSDRAW_BANKID);
  evt_file_open = false;
  
  
  // for monitoring information
  mon_outUnit = 2;
  mon_outBanks = newBankList(10);
  addBankList(mon_outBanks,SDMON_BANKID);
  mon_file_open = false;
  
  // Open first 3 files in the list
  for (itower=0; itower<3; itower++)
    {
      
      // don't bother with a tower that doesn't have any run files
      if(nrawfiles[itower] < 1)
	continue;
      
      // Set on the 1st run in the list that's needed for parsing the wanted date
      irawfile[itower] = 0;
      while ( (!run_needed[irawfile[itower]][itower]) && 
	      (irawfile[itower] < (nrawfiles[itower]-1)))
	{
	  irawfile[itower]++;
	}
      if ((currawfile[itower]=fopen(rawfname[irawfile[itower]][itower], "r"))==0)
	{
	  printErr("Fatal Error: can't open '%s'",
		   rawfname[irawfile[itower]][itower]);
	  fprintf(stdout, "DATE: %06d SUCCESS-ALL: %d\n", opt.yymmdd, 0);
	  exit(2);
	}
      if (opt.verbose>=2)
	{
	  fprintf(stdout, "OPENED: %s\n",rawfname[irawfile[itower]][itower]);
	  fflush(stdout);
	}
      iline[itower] = 0;
    }
}
rusdpass0io::~rusdpass0io()
{
  if(evt_file_open)
    {
      dstCloseUnit(evt_outUnit);
      evt_file_open = false;
    }
  if(mon_file_open)
    {
      dstCloseUnit(mon_outUnit);
      mon_file_open = false;
    }
  
  // remove the temporary files if the option flag has been is set to
  // something other than zero
  if(opt.rem_tmp)
    {
      for (int isite=0; isite < 3; isite++)
	{
	  for (int ifile=0; ifile < nrawfiles[isite]; ifile++)
	    {
	      if(is_rawfname_tmp[ifile][isite])
		{
		  if (remove(rawfname[ifile][isite]))
		    printErr("error: failed to remove a temporary file %s",rawfname[ifile][isite]);
		}
	    }
	}
    }
}
int rusdpass0io::GetNfiles(int itower)
{
  if (!chk_tower_id(itower))
    return 0;
  return nrawfiles[itower];
}
int rusdpass0io::GetSetDate()
{
  return opt.yymmdd;
}
int rusdpass0io::GetReadFileNum(int itower)
{
  if (!chk_tower_id(itower))
    return 0;
  return irawfile[itower];
}
const char* rusdpass0io::GetReadFile(int itower)
{
  if (!chk_tower_id(itower))
    return 0;
  return (const char *)rawfname[irawfile[itower]][itower];
}
int rusdpass0io::GetReadFileRunID(int itower)
{
  if (!chk_tower_id(itower))
    return 0;
  return (int)(full_run_id[irawfile[itower]][itower] % 1000000);
}
const char* rusdpass0io::GetReadFile(int ifile, int itower)
{
  if (!chk_tower_id(itower))
    return 0;
  if (ifile < 0 || ifile>=nrawfiles[itower])
    {
      printErr("ifile must be in 0-%d range", nrawfiles[itower]-1);
      return 0;
    }
  return (const char *)rawfname[ifile][itower];
}
int rusdpass0io::GetReadFileRunID(int ifile, int itower)
{
  if (!chk_tower_id(itower))
    return 0;
  if (ifile < 0 || ifile>=nrawfiles[itower])
    {
      printErr("ifile must be in 0-%d range", nrawfiles[itower]-1);
      return 0;
    }
  return (int)(full_run_id[ifile][itower] % 1000000);
}
int rusdpass0io::GetReadLine(int itower)
{
  if (!chk_tower_id(itower))
    return 0;
  return iline[itower];
}

bool rusdpass0io::get_line(int itower, char *line)
{
  if (!chk_tower_id(itower))
    exit(2);

  // make sure that the tower has files
  if(nrawfiles[itower] < 1)
    {
      printErr("Warning: rusdpass0io::get_line: method called but no files for tower id = %d",
	       itower);
      return false;
    }

  // Successfuly read a line from the current ascii file
  if (fgets(line, ASCII_LINE_LEN, currawfile[itower]))
    {
      iline[itower]++;
      return true;
    }
  else
    {
      fclose(currawfile[itower]);
      currawfile[itower] = 0;
    }

  // Return false if there are no more files for a given tower
  if (irawfile[itower]==(nrawfiles[itower]-1))
    return false;
 
  irawfile[itower]++; // increment the current file index
    
  // If the next run is not needed, so will be the all later ones. We are at the end of the data needed for
  // parsing data on specified date. 
  if (!run_needed[irawfile[itower]][itower])
    return false;

  // Open the next needed raw ascii file for a given tower
  if ((currawfile[itower]=fopen(rawfname[irawfile[itower]][itower], "r"))==0)
    {
      printErr("Fatal Error: can't open %s", rawfname[irawfile[itower]][itower]);
      fprintf(stdout, "DATE: %06d SUCCESS-ALL: %d\n", opt.yymmdd, 0);
      exit(2);
    }
  if (opt.verbose>=2)
    {
      fprintf(stdout, "OPENED: %s\n",rawfname[irawfile[itower]][itower]);
      fflush(stdout);
    }
  iline[itower]=0; // reset current read line
  return (get_line(itower, line));
}

bool rusdpass0io::chk_tower_id(int itower)
{
  if ( (itower<0) || (itower>2))
    {
      printErr("%d is an invalid tower id value", itower);
      return false;
    }
  return true;
}

bool rusdpass0io::getRunInfo(const char* fname, int *itower, int *run_id)
{
  const char *ch1, *ch2;
  char sname[3], runno[10];
  if (strlen(fname)<14)
    return false;

  if ((ch1=strrchr(fname, '/'))==NULL)
    ch1=fname-1;
  if ((ch2=strrchr(fname, 'Y'))==NULL)
    return false;

  if ((strlen(fname)-(ch1-fname)-1)<14)
    return false;
  memcpy(sname, ch1+1, 2);
  sname[2]='\0';

  if ((strcmp(sname, "BR"))==0)
    {
      (*itower) = RUSDRAW_BR;
    }
  else if ((strcmp(sname, "LR"))==0)
    {
      (*itower) = RUSDRAW_LR;
    }
  else if ((strcmp(sname, "SK"))==0)
    {
      (*itower) = RUSDRAW_SK;
    }
  else
    {
      return false;
    }
  if ((ch2-ch1-4) < 1)
    return false;
  memcpy(runno, ch1+3, (ch2-ch1)-4);
  runno[(ch2-ch1)-4] = '\0';
  sscanf(runno, "%d", run_id);
  return true;
}

bool rusdpass0io::getRunStartDate(const char* fname, 
				  int *yymmdd_start,
				  int *hhmmss_start)
{
  char line[ASCII_LINE_LEN];
  bool got_date;
  int repno;
  got_date = false;
  FILE* fp = fopen(fname,"r");
  if(!fp)
    {
      printErr("getRunStartDate: failed to open %s for reading",fname);
      return false;
    }
  while (fgets(line, ASCII_LINE_LEN, fp))
    {
      /* REPETITION HEADER, provides the absolute time ...
         [REPETITION] [YYMMDD] [HHMMSS] [SUBSEC] [SECNUM]
         #T 00005850 080313 051025 4340 599*/
      /* rep. number is in HEX, everything else is in decimal */
      if ((line[0]=='#') && (line[1] == 'T') && isxdigit(line[3])
	  && isdigit(line[12]) && isdigit(line[19]))
	{
	  sscanf(line, "#T %8x %6d %6d", &repno, yymmdd_start, hhmmss_start);
	  got_date=true;
	  break;
	}
    }
  fclose(fp);
  return got_date;
}

void rusdpass0io::swap_run_files(int itower, int index1, int index2)
{
  int irun_jd;
  uint64_t ifull_run_id;
  bool ihas_date;
  bool irun_needed;
  char irawfname[ASCII_NAME_LEN];
  bool iis_rawfname_tmp;
  memcpy(irawfname, rawfname[index1][itower], ASCII_NAME_LEN);
  ifull_run_id = full_run_id[index1][itower];
  irun_jd = run_jd[index1][itower];
  ihas_date = has_date[index1][itower];
  irun_needed = run_needed[index1][itower];
  iis_rawfname_tmp = is_rawfname_tmp[index1][itower];
  memcpy(rawfname[index1][itower], rawfname[index2][itower], ASCII_NAME_LEN);
  full_run_id[index1][itower] = full_run_id[index2][itower];
  run_jd[index1][itower] = run_jd[index2][itower];
  has_date[index1][itower] = has_date[index2][itower];
  run_needed[index1][itower] = run_needed[index2][itower];
  is_rawfname_tmp[index1][itower] = is_rawfname_tmp[index2][itower];
  memcpy(rawfname[index2][itower], irawfname, ASCII_NAME_LEN);
  full_run_id[index2][itower] = ifull_run_id;
  run_jd[index2][itower] = irun_jd;
  has_date[index2][itower] = ihas_date;
  run_needed[index2][itower] = irun_needed;
  is_rawfname_tmp[index2][itower] = iis_rawfname_tmp;
}
void rusdpass0io::sort_by_full_run_id()
{
  int i, j, itower;
  for (itower=0; itower<3; itower++)
    {
      for (i=0; i<nrawfiles[itower]; i++)
	{
	  for (j=i+1; j<nrawfiles[itower]; j++)
	    {
	      if (full_run_id[j][itower]<full_run_id[i][itower])
		{
		  swap_run_files(itower, i, j);
		  j=i;
		}
	    }
	}
    }
}

bool rusdpass0io::check_missing()
{
  int ifile, itower;
  uint64_t rndiff, idiff;
  bool have_missing;
  char towerName[3][3];
  sprintf(towerName[0], "%s", "BR");
  sprintf(towerName[1], "%s", "LR");
  sprintf(towerName[2], "%s", "SK");
  have_missing = false;
  for (itower=0; itower<3; itower++)
    {
      for (ifile=0; ifile<(nrawfiles[itower]-1); ifile++)
	{
	  // if one of the runs in the pair did not have a valid date
	  // in it then it is hard to do the continuity check.
	  if(!has_date[ifile][itower] || !has_date[ifile+1][itower])
	    {
	      if(opt.verbose >= 1)
		printErr("warning: not checking run id continuity between %s%06d and %s%06d",
			 towerName[itower],(int)(full_run_id[ifile][itower] % 1000000),
			 towerName[itower],(int)(full_run_id[ifile+1][itower] % 1000000));
	      continue;
	    }
	  rndiff = full_run_id[ifile+1][itower] - full_run_id[ifile][itower] - 1;
	  // if the full id of the next run is different from the previous by more than 1
	  // then check if this is a year change (year of the next run is more by 1 and the 
	  // abbreviated run ID is reset to 1)
	  if (rndiff > 0)
	    {
	      if (full_run_id[ifile+1][itower] / 1000000 - full_run_id[ifile][itower] / 1000000 != 1
		  || full_run_id[ifile+1][itower] % 1000000 != 1)
		have_missing = true;
	      else
		rndiff = 0;
	    }
	  for (idiff=0; idiff<rndiff; idiff++)
	    {
	      printErr("%s%06d is missing", towerName[itower],
		       (((int)full_run_id[ifile][itower]%1000000)+1+(int)idiff));
	      if(idiff > 1000)
		{
		  fprintf(stderr,"... there seems to be a problem; not reporting any more\n");
		  break;
		}
	    }
	}
    }
  return (!have_missing);
}

bool rusdpass0io::writeEvent(rusdraw_dst_common *event)
{
  if(!evt_file_open)
    {
      FILE* fp = fopen(opt.evtfile, "w");
      if (!fp)
	{
	  printErr("Fatal Error: can't start %s", opt.evtfile);
	  fprintf(stdout, "DATE: %06d SUCCESS-ALL: %d\n", opt.yymmdd, 0);
	  return false;
	}
      fclose(fp);
      char* dstname = new char[strlen(opt.evtfile)+1]; // because DST opener doesn't accept const char ...
      memcpy(dstname,opt.evtfile,strlen(opt.evtfile)+1);
      if (dstOpenUnit(evt_outUnit,dstname,MODE_WRITE_DST) != 0)
	{
	  printErr("Fatal Error: can't start %s", opt.evtfile);
	  fprintf(stdout, "DATE: %06d SUCCESS-ALL: %d\n", opt.yymmdd, 0);
	  exit(2);
	}
      delete[] 
	dstname;
      if(opt.verbose >= 1)
	{
	  fprintf(stdout,"STARTED EVENT DST FILE: %s\n",opt.evtfile);
	  fflush(stdout);
	}
      evt_file_open = true;
    }
  memcpy(&rusdraw_, event, sizeof(rusdraw_dst_common));
  if (eventWrite(evt_outUnit, evt_outBanks, TRUE) < 0)
    {
      printErr("Couldn't write event into DST file!");
      return false;
    }
  return true;
}

bool rusdpass0io::writeMon(sdmon_dst_common *mon)
{
  if(!mon_file_open)
    {
      FILE* fp = fopen(opt.monfile, "w");
      if (!fp)
	{
	  printErr("Fatal Error: can't start %s", opt.monfile);
	  fprintf(stdout, "DATE: %06d SUCCESS-ALL: %d\n", opt.yymmdd, 0);
	  return false;
	}
      fclose(fp);
      char* dstname = new char[strlen(opt.monfile)+1]; // because DST opener doesn't accept const char ...
      memcpy(dstname,opt.monfile,strlen(opt.monfile)+1);
      if (dstOpenUnit(mon_outUnit,dstname,MODE_WRITE_DST) != 0)
	{
	  printErr("Fatal Error: can't start %s", opt.monfile);
	  fprintf(stdout, "DATE: %06d SUCCESS-ALL: %d\n", opt.yymmdd, 0);
	  exit(2);
	}
      delete[] 
	dstname;
      if(opt.verbose >= 1)
	{
	  fprintf(stdout,"STARTED MONITORING DST FILE: %s\n",opt.monfile);
	  fflush(stdout);
	}
      mon_file_open = true;
    }
  memcpy(&sdmon_, mon, sizeof(sdmon_dst_common));
  if (eventWrite(mon_outUnit, mon_outBanks, TRUE) < 0)
    {
      printErr("Couldn't write monitoring cycle into DST file!");
      return false;
    }
  return true;
}

bool rusdpass0io::save_current_pos(int itower)
{
  if (!chk_tower_id(itower))
    return false;
  if(!currawfile[itower])
    {
      printErr("save_current_pos: itower=%d input stream was expected to be open",itower);
      return false;
    } 
  saved_irawfile[itower] =  irawfile[itower];
  if(fgetpos(currawfile[itower],&saved_fpos[itower]) != 0)
    {
      printErr("save_current_pos: can't save file position for irawfile[%d]=%d",
	       itower,irawfile[itower]);
      return false;
    }
  saved_iline[itower] = iline[itower];
  return true;
}

bool rusdpass0io::goto_saved_pos(int itower)
{
  if (!chk_tower_id(itower))
    return false;
  if((saved_irawfile[itower] == -1) || (saved_iline[itower] == -1))
    {
      printErr("goto_saved_pos: itower=%d: file position wasn't saved",itower);
      return false;
    }
  // recover the raw file index, close the current file 
  // and re-open the one that corresponds to the saved file index,
  // if necessary
  if(saved_irawfile[itower] != irawfile[itower])
    {
      if(currawfile[itower])
	{
	  fclose(currawfile[itower]);
	  currawfile[itower] = 0;
	}
      irawfile[itower] = saved_irawfile[itower];
    }
  if(!currawfile[itower])
    {
      if (!(currawfile[itower]=fopen(rawfname[irawfile[itower]][itower], "r")))
	{
	  printErr("goto_saved_pos: can't open %s", rawfname[irawfile[itower]][itower]);
	  return false;
	}
      if (opt.verbose>=2)
	{
	  fprintf(stdout, "RE-OPENED: %s\n",rawfname[irawfile[itower]][itower]);
	  fflush(stdout);
	}
    }
  if(fsetpos(currawfile[itower],&saved_fpos[itower]) != 0)
    {
      printErr("goto_saved_pos: can't recover file position for irawfile[%d]=%d",
	       itower,irawfile[itower]);
      return false;
    }
  iline[itower] = saved_iline[itower]; // recover the line number
  // reset the variables which indicate that the file 
  // position for a given itower has been saved
  saved_irawfile[itower] = -1;
  saved_iline[itower]    = -1;
  return true;
}

void rusdpass0io::printErr(const char *form, ...)
{
  char mess[0x400];
  va_list args;
  va_start(args, form);
  vsprintf(mess, form, args);
  va_end(args);
  fprintf(stderr, "rusdpass0io:  %s\n", mess);
}

const char* rusdpass0io::unpack_run_file(const char* bz2file, const char* prodir)
{
  // unpack the .tar.bz2 file and return the path to the unpacked .Y???? - file
  // works only on unix systems.  On other systems, will return null and print
  // an error messsage. Returns a pointer to the unpacked file name string in
  // case of success.
  
  static char fname[0x400];
#if defined(unix) || defined(__unix__)
  char *cmd = new char[0x1200];
  FILE *fp = 0;
  // un-tar and un-bzip the file
  //  sprintf(cmd,"tar -C %s -xjvf %s ????????.Y????",prodir,bz2file);
  sprintf(cmd,"tar --wildcards -C %s -xjvf %s ????????.Y????",prodir,bz2file);
  fp=popen(cmd,"r");
  if(fscanf(fp,"%s",fname) != 1)
    fname[0] = 0;
  fclose(fp);
  // get the location of the unpacked file
  if(fname[0])
    {
      sprintf(cmd,"ls %s%s",prodir,fname);
      fp  = popen(cmd,"r");
      if(fscanf(fp,"%s",fname) != 1)
	fname[0] = 0;
      fclose(fp);
    }
  delete cmd;
#else
  fprintf(stderr,"error: use .tar.bz2 files only on Unix systems. ");
  fprintf(stderr,"Otherwise, unpack the *.Y???? - files from .tar.bz2 - files first.\n");
#endif
  if(!fname[0])
    return 0;
  return fname;
}

