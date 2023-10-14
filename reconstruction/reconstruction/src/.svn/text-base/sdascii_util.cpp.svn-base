#include "sdascii.h"
#include "tacoortrans.h"
#include <time.h>
#include <ctype.h>

listOfOpt::listOfOpt()
{
  wantfile[0]      = 0;        // list file variable initialized
  outfile[0]       = 0;        // output file variable initialized
  sfoutfile[0]     = 0;        // shower front structure ASCII output file initialized
  dstoutfile[0]    = 0;        // output dst file variable initialized
  f_etrack         = false;    // don't fill and add etrack dst bank unless this option is used
  format           = 1;        // output format set to default (TA-Wiki) format
  stdout_opt       = false;    // by default output expected to go to a file
  fOverwrite       = false;    // by default, don't allow the output file overwriting
  bank_warning_opt = true;     // by default, print warnings about missing banks
  rescale_err_opt  = false;    // by default, do not rescale theta,phi errors to get  the 68% c.l.
  tb_opt           = false;    // don't do trigger back up by default
  tb_delta_ped     = 0;        // don't raise/lowe pedestal requirements for trigger back up by default
  za_cut           = 45.0;     // default value for the zenith angle cut
  brd_cut          = 0;        // by default, the D_border > 1200m cut is used
  enscale          = 1.0/1.27; // default value for the SD energy scale
  emin             = 1.0;      // default value for minimum energy cut in EeV
  progName[0]      = '\0';
}

// destructor does nothing
listOfOpt::~listOfOpt() {}

bool listOfOpt::getFromCmdLine(int argc, char **argv)
{  
  int i;
  char inBuf[0x400];
  char *line;
  FILE *wantfl;                  // For reading the want file
  
  memcpy(progName,argv[0],0x400);
  progName[0x400-1]='\0';
  if(argc==1) 
    {
      printMan();
      return false;
    }
  for(i=1; i<argc; i++)
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
      
      // intput from a list file
      else if (strcmp ("-i", argv[i]) == 0)
	{
	  if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf(stderr,"error: -i: specify the list file!\n");
	      return false;
	    }
	  else
	    {
	      sscanf (argv[i], "%1023s", wantfile);
	      // read the want files, put all pass1 dst files found into a buffer.
	      if((wantfl=fopen(wantfile,"r"))==NULL)
		{
		  fprintf(stderr,"error: -i: can't open the list file %s\n",wantfile);
		  return false;
		}
	      else
		{
		  while(fgets(inBuf,0x400,wantfl))
                    {
                      if ((line=strtok (inBuf, " \t\r\n")) && strlen (line) > 0 
                          && pushFile(line) != SUCCESS)
                        return false;
                    }
		  fclose(wantfl);
		}
	    }
	}
      
      // input file names from stdin
      else if (strcmp ("--tty", argv[i]) == 0)
        {
	  while(fgets(inBuf,0x400,stdin))
	    {
	      if ((line=strtok (inBuf, " \t\r\n")) && strlen (line) > 0 
		  && pushFile(line) != SUCCESS)
		return false;
	    }
        }
      
      // output file
      else if (strcmp ("-o", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"error: -o: specify the output file!\n");
	      return false;
	    }
	  sscanf (argv[i], "%1023s", outfile);
	}

      // output shower front ASCII file
      else if (strcmp ("-sfo", argv[i]) == 0)
	{
	  if ((++i >= argc) || !argv[i] || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"error: -sfo: specify the output shower front ASCII file!\n");
	      return false;
	    }
	  sscanf (argv[i], "%1023s", sfoutfile);
	}

      // output file
      else if (strcmp ("-dsto", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"error: -dsto: specify the output dst file!\n");
	      return false;
	    }
	  sscanf (argv[i], "%1023s", dstoutfile);
	}
      
      // to fill etrack dst bank
      else if (strcmp ("-etrack", argv[i]) == 0)
	f_etrack = true;
      
      // output format flag
      else if (strcmp ("-form", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"error: -form: specify the output format flag!\n");
	      return false;
	    }
	  sscanf (argv[i], "%d", &format);
	}
      
      // do the trigger backup cut?
      else if (strcmp ("-tb", argv[i]) == 0)
        {
	  tb_opt = true;
	  if (i+1 < argc && argv[i+1][0])
	    {
	      if(isdigit(argv[i+1][0]) || (argv[i+1][0] == '-' && isdigit(argv[i+1][1])))
		sscanf (argv[++i], "%d", &tb_delta_ped);
	    }
        }

      // zenith angle cut
      else if (strcmp ("-za", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"error: -za: specify the zenith angle cut!\n");
	      return false;
	    }
	  sscanf (argv[i], "%lf", &za_cut);
	}
      
      // energy scale
      else if (strcmp ("-enscale", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"error: -enscale: specify the energy scale constant!\n");
	      return false;
	    }
	  sscanf (argv[i], "%lf", &enscale);
	}
      
      // minimum energy cut
      else if (strcmp ("-emin", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"error: -emin: specify the minimum energy cut!\n");
	      return false;
	    }
	  sscanf (argv[i], "%lf", &emin);
	}

      // border cut flag
      else if (strcmp ("-brd", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"error: -brd: specify the border cut flag!\n");
	      return false;
	    }
	  sscanf (argv[i], "%d", &brd_cut);
	}
      
      // rescale theta, phi errors so that they are 68% C.L. 
      else if (strcmp ("-rse", argv[i]) == 0)
	rescale_err_opt = true;
      
      // stdout option
      else if (strcmp ("-O", argv[i]) == 0)
	stdout_opt = true;
      
      // force overwrite mode
      else if (strcmp ("-f", argv[i]) == 0)
	fOverwrite = true;
      
      // missing bank warning option
      else if (strcmp ("-no_bw", argv[i]) == 0)
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
  time_t now;
  struct tm *d;
  char cur_date_time[0x100];
  time(&now);
  d = localtime(&now);
  strftime(cur_date_time,255,"%Y-%m-%d %H:%M:%S %Z", d);
  fprintf(stdout,"\n\n");
  fprintf(stdout,"%s (%s):\n",progName,cur_date_time);
  const char format_name[2][0x40]={"FULL-ANISOTROPY","TA-WIKI"};
  if(wantfile[0])
    fprintf(stdout,"INPUT LIST FILE: %s\n",wantfile);
  if(outfile[0])
    fprintf(stdout,"OUTPUT FILE: %s\n",outfile);
  if(sfoutfile[0])
    fprintf(stdout,"SHOWER FRONT INFORMATION OUTPUT FILE: %s\n",sfoutfile);
  if(dstoutfile[0])
    fprintf(stdout,"DST OUTPUT FILE: %s\n",dstoutfile);
  fprintf(stdout,"ASCII OUTPUT FORMAT: %s\n",(char *)format_name[format]);
  fprintf(stdout,"USE TRIGGER BACKUP INFORMATION: %s\n",(tb_opt ? "YES" : "NO"));
  if(tb_opt)
    fprintf(stdout,"TRIGGER BACKUP RAISING/LOWERING PEDESTAL BY: %d\n", tb_delta_ped);
  fprintf(stdout,"ZENITH ANGLE CUT: <= %.2f degrees\n",za_cut);
  fprintf(stdout,"RESCALE THETA/PHI ERRORS: %s\n",(char *)(fOverwrite ? "YES" : "NO"));
  fprintf(stdout,"ENERGY SCALE: %.2f  (1/%.2f)\n",enscale,1.0/enscale);
  fprintf(stdout,"MINIMUM ENERGY: %.3f EeV\n",emin);
  fprintf(stdout,"FORCE-OVERWRITE MODE: %s\n", (char *)(fOverwrite ? "YES" : "NO"));
  fprintf(stdout,"DISPLAY MISSING BANK WARNINGS: %s\n", (char *)(bank_warning_opt ? "YES" : "NO"));
  fprintf(stdout,"\n\n");
  fflush(stdout);
}

bool listOfOpt::checkOpt()
{
  if(format < 0 || format > 1)
    {
      fprintf(stderr,"error: invalid format flag: -form takes on 0 or 1 values\n");
      return false;
    }
  if(brd_cut < 0 || brd_cut > 1)
    {
      fprintf(stderr,"error: invalid border cut flag: -brd takes on 0 or 1 values\n");
      return false;
    }
  if(countFiles()==0)
    {
      fprintf(stderr, "error: don't have any inputs!\n");
      return false;
    } 
  return true;
}

void listOfOpt::printMan()
{
  FILE *fp = stdout;
  
  fprintf(fp,"\n");
  fprintf(fp,"****************************************************************************************\n");
  fprintf(fp,"TA SD event list / anisotropy program\n");
  fprintf(fp,"Runs on analyzed TA SD data or MC\n");
  fprintf(fp,"Typically writes out the event lists in ascii format (main use)\n");
  fprintf(fp,"However, one may also save selected events into a DST file by using -dsto option\n");
  fprintf(fp,"Events must have rusdraw,rufptn,rusdgeom,rufldf DST banks\n");
  fprintf(fp,"All of these DST banks are available after rufldf.run analysis\n");
  fprintf(fp,"Applies quality cuts\n");
  fprintf(fp,"Author: Dmitri Ivanov <dmiivanov@gmail.com>\n");
  fprintf(fp,"****************************************************************************************\n");
  fprintf(fp, "Write an ascii file of SD reconstructed events using correct energy scale and quality cuts\n");
  fprintf(fp, "Optionally, one may also save the selected events into a dst file\n");
  fprintf(fp, "LOCAL COORDINATE SYSTEM: CLF LAT,LON=%.3f,%.3f degree; ",
	  tacoortrans_CLF_Latitude,tacoortrans_CLF_Longitude);
  fprintf(fp, "X=EAST, Y=NORTH, Z=UP\n");
  fprintf(fp, "R = | sin(zenith angle)*cos(azimuthal angle) |\n");
  fprintf(fp, "    | sin(zenith angle)*sin(azimuthal angle) |\n");
  fprintf(fp, "    |         cos(zenith angle)              |\n");
  fprintf(fp, "is where events come from in this coordinate system\n");
      
  // ANISOTROPY FORMAT
  fprintf(fp, "\nFULL-ANISOTROPY FORMAT ('-form 0' option):\n");
  fprintf(fp, "col01:  utc date, [yyyymmdd format]\n");
  fprintf(fp, "col02:  utc time, [hhmmss + second fraction format]\n");
  fprintf(fp, "col03:  full julian date+time [days]\n");
  fprintf(fp, "col04:  J2000 local mean sidereal time [radians]\n");
  fprintf(fp, "col05:  energy [EeV, set to FD energy scale by default]\n");
  fprintf(fp, "col06:  zenith angle [radians]\n");
  fprintf(fp, "col07:  error on zenith angle [radians]\n");
  fprintf(fp, "col08:  azimuthal angle [radians]\n");
  fprintf(fp, "col09:  error on azimuthal angle [radians]\n");
  fprintf(fp, "col10:  hour angle [radians]\n");
  fprintf(fp, "col11:  J2000 right ascension [radians]\n");
  fprintf(fp, "col12:  declination [radians]\n");
  fprintf(fp, "col13:  J2000 galactic longitude [radians]\n");
  fprintf(fp, "col14:  J2000 galactic latitude [radians]\n");
  fprintf(fp, "col15:  J2000 supergalactic longitude [radians]\n");
  fprintf(fp, "col16:  J2000 supergalactic latitude [radians]\n");
      
  // The "TA-Wiki" format
  fprintf(fp, "\nTA-WIKI FORMAT, USED BY DEFAULT ('-form 1' option)\n");      
  fprintf(fp, "col01:  date - YYYYMMDD\n");
  fprintf(fp, "col02:  time - UT, 24 hr clock, time to nearest microsecond - HHMMSS.xxxxxx\n");
  fprintf(fp, "col03:  x core (e-core)- km E of CLF\n");
  fprintf(fp, "col04:  y core (n-core)- km N of CLF\n");
  fprintf(fp, "col05:  s800 - (0's for FDs) - VEM/m^2\n");
  fprintf(fp, "col06:  zenith angle - degrees\n");
  fprintf(fp, "col07:  azimuthal angle - degrees N of E (pointing back to source)\n");
  fprintf(fp, "col08:  rp - (0's for SDs) - impact parameter - km (relative to FD detector) (ZERO)\n");
  fprintf(fp, "col09:  psi - (0's for SDs) - angle in FD shower detector plane - degrees (ZERO)\n");
  fprintf(fp, "col10:  reconstructed energy - EeV\n");
    
  fprintf(fp, "\nUsage: %s dst_file1 dst_file2 ... or -i want_file -o [output file]\n",progName);
  fprintf(fp,"\nINPUT: \n");
  fprintf(fp,"pass input DST file names as arguments without any prefixes or switches\n");
  fprintf(fp,"-i <string>      : and/or specify the list file which contains the dst file name paths\n");
  fprintf(fp,"--tty            : and/or get dst file name paths from stdin\n");
  fprintf(fp,"\nOUTPUT: \n");
  fprintf(fp,"-o <string>      : specify the output ASCII file for events\n");
  fprintf(fp,"-sfo <string>    : specify the output ASCII file for information for the shower front studies\n");
  fprintf(fp,"-dsto <string>   : and/or specify the output dst file for the selected events\n");
  fprintf(fp,"-etrack          : and/or fill etrack dst bank (see dst2k-ta/inc/etrack_dst.h for more info)\n");
  fprintf(fp,"-form <int>      : set the output format flag, in [0,1] range, default is 1 (TA-Wiki format)\n");
  fprintf(fp,"-O               : pour event information into stdout with 'EVT' prefix ignoring the '-o' option\n");
  fprintf(fp,"-f               : force overwrite mode on all output files, off by default\n");
  fprintf(fp,"-no_bw           : disable warnings about the missing banks\n");
  fprintf(fp,"-h               : show this manual and quit\n");
  fprintf(fp,"\nCUTS:\n");
  fprintf(fp,"-tb              : to do the trigger backup cut (must have sdtrgbk DST bank)\n");
  fprintf(fp,"-tb <int>        : to do trigger backup with raised (positive) or lowered (negative) pedestal\n");
  fprintf(fp,"-za <float>      : max. zenith angle cut, degree (default %.2f)\n",za_cut);
  fprintf(fp,"-emin <float>    : minimum enegy cut, EeV (default %.3f)\n",emin);
  fprintf(fp,"-brd <int>       : border cut flag (default %d):\n",brd_cut);
  fprintf(fp,"                   0: D_border > 1200m\n");
  fprintf(fp,"                   1: largest signal counter that's part of event is surrounded by 4 working counters\n");
  fprintf(fp,"                   2: largest signal counter that's part of event is surrounded by 4 working counters\n");
  fprintf(fp,"                      that are immediate neighbors of the largest signal counter\n");
  fprintf(fp,"\nADDITIONAL STEERING:\n");
  fprintf(fp,"-enscale <float> : multiply MC-derived SD energy by this constant. default value is %.2f (FD energy scale)\n",enscale);
  fprintf(fp,"-rse             : rescale the zenith/azimuth errors so that they are true 68%c C.L. (off by default)\n",'%');
  fprintf(fp,"\n\n");
  fprintf(fp,"NOTE: do '%s -h | more' to see everything\n\n",progName);
  fflush(fp);
}
