#include "dst2rt_sd.h"
#include "time.h"
// c++ - utility function for dst2rt_sd program

//////////////////////////////////////////////////////////////////////////


/************  CLASS FOR HANDLING THE PROGRAM ARGUMENTS *****************/

listOfOpt::listOfOpt()
{
  dout[0]       =  0;
  outpr[0]      =  0;
  wt            =  0;
  rtof[0]       =  0;
  dtof[0]       =  0;
  verbose       =  true;
  fOverwrite    =  false;
  atmparopt     =  false;
  gdasopt       =  false;
  etrackopt     =  false;
  sdopt         =  true;
  tasdevent     =  false;
  tasdcalibev   =  false;
  bsdinfo       =  false;
  tbopt         =  false;
  mcopt         =  false;
  mdopt         =  false;
  fdplane_opt   =  false;
  fdprofile_opt =  false;
  progName[0]   = 0;
}

// destructor does nothing
listOfOpt::~listOfOpt() {}

bool listOfOpt::getFromCmdLine(int argc, char **argv)
{  
  int i,l;
  char inBuf[SD_INBUF_SIZE];
  FILE *fp;                  // For reading the want file
  
  char *line;

  sprintf(progName,"%s",argv[0]);
  
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
      
      // getting inputs from a want file
      else if (strcmp ("-i", argv[i]) == 0)
	{
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr, "error: -i: specify the list file\n");
	      return false;
	    }
	  else
	    {
	      sscanf (argv[i], "%1023s", argv[i]);
	      // read the want files, put all dst files found into a buffer.
	      if(!(fp=fopen(argv[i],"r")))
		{
		  fprintf(stderr,"error: can't open the list file %s\n",argv[i]);
		  return false;
		}
	      while(fgets(inBuf,SD_INBUF_SIZE,fp))
		{
		  if (((line=strtok (inBuf, " \t\r\n"))) && (strlen (line) > 0) 
		      && (pushFile(line) != SUCCESS))
		    return false;
		}
	      fclose(fp);
	    }
	}
      
      // getting inputs from stdin
      else if (strcmp ("--tty", argv[i]) == 0)
	{
	  while(fgets(inBuf,SD_INBUF_SIZE,stdin))
	    {
	      if (((line=strtok (inBuf, " \t\r\n"))) && (strlen (line) > 0) 
		  && (pushFile(line) != SUCCESS))
		return false;
	    }
	}
      
      // output directory
      else if (strcmp ("-o", argv[i]) == 0)
        {
          if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
            {
              fprintf (stderr,"error: -o: specify the root tree output directory\n");
              return false;
            }
          else
            {
	      if(strlen(argv[i]) + 1 <= SD_DOUT_SIZE)
		sscanf (argv[i], "%1023s", &dout[0]);
	      else
		{
		  fprintf(stderr,"error: -o: output directory name cannot exceed %d characters\n",
			  SD_DOUT_SIZE-1);
		  return false;
		}
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
              fprintf (stderr,"error: -pr: specify the output file prefix\n");
              return false;
            }
	  if(strlen(argv[i]) + 1 <= SD_PREF_SIZE)
	    sscanf (argv[i], "%1023s", &outpr[0]);
	  else
	    {
	      fprintf (stderr,"error: -pr: prefix cannot exceed %d characters\n",
		       SD_PREF_SIZE-1);
              return false;
	    }
        }

      // result root tree output file
      else if (strcmp ("-rtof", argv[i]) == 0)
        {
          if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
            {
              fprintf (stderr,"error: -rtof: specify the result root tree output file\n");
              return false;
            }
	  sscanf (argv[i], "%1023s", &rtof[0]);
        }
      
      // detailed root tree output file
      else if (strcmp ("-dtof", argv[i]) == 0)
        {
          if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
            {
              fprintf (stderr,"error: -dtof: specify the detailed root tree output file\n");
              return false;
            }
	  sscanf (argv[i], "%1023s", &dtof[0]);
        }
      
      // root tree writing mode
      else if (strcmp ("-wt", argv[i]) == 0)
        {
          if ((++i >= argc ) || (argv[i][0] == 0) || (argv[i][0]=='-'))
            {
              fprintf (stderr,"error: -wt: specify the root tree writing flag\n");
	      return false;
            }
	  sscanf (argv[i], "%d", &wt);
	  if (wt < 0 || wt > 2)
	    {
	      fprintf(stderr,"error: -wt: specify an integer flag in 0 to 2 range\n");
	      return false;
	    }
        }

      // atmospheric parameter option
      else if (strcmp ("-atmpar", argv[i]) == 0)
	atmparopt = true;

      // gdas option
      else if (strcmp ("-gdas", argv[i]) == 0)
	gdasopt = true;
      
      // event track option
      else if (strcmp ("-etrack", argv[i]) == 0)
	etrackopt = true;
      
      // MC branches option
      else if (strcmp ("-mc", argv[i]) == 0)
	mcopt = true;
      
      // MD branches option
      else if (strcmp ("-md", argv[i]) == 0)
	mdopt = true;
      
      // trigger backup option
      else if (strcmp ("-tb", argv[i]) == 0)
	tbopt = true;
      
      // fdplanes branches option (brplane and lrplane)
      else if (strcmp ("-fdplane", argv[i]) == 0)
	fdplane_opt = true;
      
      // fdprofile branches option (brprofile and lrprofile)
      else if (strcmp ("-fdprofile", argv[i]) == 0)
	fdprofile_opt = true;
      
      // tasdevent branch option
      else if (strcmp ("-tasdevent", argv[i]) == 0)
	tasdevent = true;
      
      // sd passes branches shutoff option
      else if (strcmp ("-bsdinfo", argv[i]) == 0)
	bsdinfo = true;

      // tasdcalibev branch option
      else if (strcmp ("-tasdcalibev", argv[i]) == 0)
	tasdcalibev = true;
      
      // sd passes branches shutoff option
      else if (strcmp ("-no_sd", argv[i]) == 0)
	sdopt = false;
      
      // verbose mode shut off option
      else if (strcmp ("-no_v", argv[i]) == 0)
	verbose = false;
      
      // force-overwrite mode
      else if (strcmp ("-f", argv[i]) == 0)
	fOverwrite = true;
      
      // assume that all arguments w/o the '-' switch are input dst file names
      else if (argv[i][0] != '-')
        {
          if (pushFile(argv[i]) != SUCCESS)
            return false;
        }
      else
        {
          fprintf(stderr, "error: '%s': unrecognized option\n", argv[i]);
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
  fprintf(stdout,"WRITING ROOT TREE FILE(S): \n");
  // result root tree
  if (wt == 0 || wt == 1)
    fprintf(stdout,"%s\n", rtof);
  // detailed root tree
  if (wt == 0 || wt == 2)
    fprintf(stdout,"%s\n", dtof);
  fprintf (stdout, "OVERWRITING THE OUTPUT FILES: %s\n", (fOverwrite ? "YES" : "NO")    );
  fprintf (stdout, "ADD etrack BRANCH: %s\n",            (etrackopt ? "YES" : "NO")     );  
  fprintf (stdout, "ADD atmpar BRANCH: %s\n",            (atmparopt ? "YES" : "NO")     );
  fprintf (stdout, "ADD gdas   BRANCH: %s\n",            (gdasopt   ? "YES" : "NO")     );
  fprintf (stdout, "ADD SD PASSES BRANCHES: %s\n",       (sdopt ? "YES" : "NO")         );
  fprintf (stdout, "ADD TASDEVENT BRANCH: %s\n",         (tasdevent ? "YES" : "NO")     );
  fprintf (stdout, "ADD BSDINFO BRANCH: %s\n",           (bsdinfo ? "YES" : "NO")       );
  fprintf (stdout, "ADD TASDCALIBEV BRANCH: %s\n",       (tasdcalibev ? "YES" : "NO")   );
  fprintf (stdout, "ADD MC BRANCHES: %s\n",              (mcopt ? "YES" : "NO")         );
  fprintf (stdout, "ADD TRIGGER BACKUP BRANCHES: %s\n",  (tbopt ? "YES" : "NO")         );
  fprintf (stdout, "ADD MD BRANCHES: %s\n",              (mdopt ? "YES" : "NO")         );
  fprintf (stdout, "ADD fdplane BRANCHES: %s\n",         (fdplane_opt ? "YES" : "NO")   );
  fprintf (stdout, "ADD fdprofile BRANCHES: %s\n",       (fdprofile_opt ? "YES" : "NO") );
  fprintf(stdout, "\n\n");
  fflush(stdout);
}

void listOfOpt::printMan()
{
  fprintf(stderr, 
	  "\nUsage: %s [dstfile1] [dstfile2] ... or -i [list of dst files (ASCII)] -o [output directory]\n",
	  progName);
  fprintf(stderr, "------------------------- INPUT OPTIONS: -------------------------\n");
  fprintf(stderr, "pass DST files  as arguments w/o any swithces or prefixes\n");
  fprintf(stderr, "-i <string>:    specify the want-file with dst files\n");
  fprintf(stderr, "--tty:          read input dst file names from stdin\n");
  fprintf(stderr, "------------------------- OUTPUT OPTIONS: ------------------------\n");
  fprintf(stdout, "-o <string>:    output directory, default is './'\n");
  fprintf(stderr, "-pr <string>:   output files prefix (endings will be '.result.root' and '.details.root')\n");
  fprintf(stderr, "-wt <int> :     make trees: 0(default)=both result and detailed trees, 1=only result tree, 2= only detailed tree\n");
  fprintf(stderr, "-rtof <string>: output file name for result tree, overrides -o and -pr options\n");
  fprintf(stderr, "-dtof <string>: output file name for detailed tree, overrides -o and -pr options\n");
  fprintf(stderr, "------------------------- RUNNING OPTIONS: --------------------------------------\n");
  fprintf(stderr, "-mc:           add MC branches\n");
  fprintf(stderr, "-tb:           add SD trigger backup branch\n");
  fprintf(stderr, "-md:           add MD branches\n");
  fprintf(stderr, "-atmpar:       add atmospheric parameter branches\n");
  fprintf(stderr, "-gdas:         add gdas variable branches\n");
  fprintf(stderr, "-etrack:       add event track branch\n");
  fprintf(stderr, "-fdplane:      add fdplane branches (should have either brplane or lrplane banks or both)\n");
  fprintf(stderr, "-fdprofile:    add fdprofile branch (should have either brprofile or lrprofile banks or both)\n");
  fprintf(stderr, "-tasdevent:    add tasdevent (ICRR) branch\n");
  fprintf(stderr, "-bsdinfo:      add bsdinfo branch\n");
  fprintf(stderr, "-tasdcalibev:  add tasdcalibev (ICRR) branch\n");
  fprintf(stderr, "-no_sd:        do not add branches for SD passes\n");
  fprintf(stderr, "-f:            overwrite the output files if they exist\n");
  fprintf(stderr, "-no_v:         shut off verbose mode\n");
  fprintf(stderr,"\n" );
}


bool listOfOpt::checkOpt()
{
  
  if (!dout[0])
    sprintf(dout,"./");
  
  // make file names for result and detailed root trees
  // if they have not been specified
  if (!rtof[0])
    {
      if (outpr[0])
	sprintf(rtof,"%s%s.result.root",dout,outpr);
      else
	sprintf(rtof,"%sresult.root",dout);
    }
  if (!dtof[0])
    {
      if (outpr[0])
	sprintf(dtof,"%s%s.detailed.root",dout,outpr);
      else
	sprintf(dtof,"%sdetailed.root",dout);
    }
  if(countFiles()==0)
    {
      fprintf(stderr, "Don't have any inputs!\n");
      return false;
    }
  
  return true;
}
