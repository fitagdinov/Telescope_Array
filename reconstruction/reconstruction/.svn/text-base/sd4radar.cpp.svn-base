#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "event.h"
#include "filestack.h"
#include "sduti.h"
#include "sdenergy.h"
#include "sddstio.h"
#include "tacoortrans.h"
#include "TMath.h"

using namespace TMath;


class sd4radar_listOfOpt
{
public:
  char   wantfile[0x400];     // input dst list file
  char   outfile[0x400];      // ascii output file
  bool   stdout_opt;          // dump event information into stdout
  bool   w_cuts_opt;          // apply the quality cuts
  bool   fOverwrite;          // force-overwrite mode
  bool   bank_warning_opt;    // print warnings about missing banks

  bool getFromCmdLine(int argc, char **argv);
  void printOpts();
  sd4radar_listOfOpt();                // show the options
  void printMan();            // show the manual
  ~sd4radar_listOfOpt();
private:
  char progName[0x400];       // name of the program
  bool checkOpt();
};



int main(int argc, char **argv)
{
  sd4radar_listOfOpt opt;
  sddstio_class dstio;
  char *infile;
  FILE *outfl;

  integer4 reqBanks;        // list of required banks


  /////// output variables /////////////
  int yymmdd,hour,minute,second,usec; // date and time, UTC
  int nstclust;             // number of good SDs
  double energy;            // energy, EeV
  double aenergy;           // energy by old AGASA formula, EeV
  double theta;             // event zenith angle, [Degree]
  double phi;               // event azimuth angle, [Degree]
  double xcore;             // core X position in CLF frame, [m]
  double ycore;             // core Y position in CLF frame, [m]
  
  
 

  // additional variables needed for making cuts
  double gfchi2pdof;        // Goeom. chi2 / dof
  double ldfchi2pdof;       // LDF chi2 / dof
  double pderr;             // pointing direction error, degree
  double ds800os800;        // sigma_S800 / S800
  
  
  int events_read;
  int events_written;
  
 
  if(!opt.getFromCmdLine(argc,argv)) 
    return 2;
  
  opt.printOpts();
  outfl = (opt.stdout_opt ? stdout : 0);
  
  events_read = 0;
  events_written = 0;

  reqBanks = newBankList(10);
  addBankList(reqBanks,RUSDRAW_BANKID);
  addBankList(reqBanks,RUFPTN_BANKID);
  addBankList(reqBanks,RUSDGEOM_BANKID);
  addBankList(reqBanks,RUFLDF_BANKID);
  
  while((infile=pullFile()))
    {
      
      // Open DST files
      if(!dstio.openDSTinFile(infile)) 
	return 2;
      
      while(dstio.readEvent())
	{

	  events_read ++;

	  // check the required banks
	  if(!dstio.haveBanks(reqBanks,opt.bank_warning_opt))
	    continue;
	  
	  // date and time
	  yymmdd = rusdraw_.yymmdd;
	  usec     = rusdraw_.usec;
	  hour   = (rusdraw_.hhmmss / 10000);
	  minute = (rusdraw_.hhmmss % 10000)/100;
	  second = rusdraw_.hhmmss % 100;
	  usec   = rusdraw_.usec;

	  // number of good SDs
	  nstclust = rufptn_.nstclust;
	  
	  // zenith angle (pointing back to the source)
	  theta  = (rusdgeom_.theta[2] + 0.5);
	  
	  // azimuthal angle, X=East, pointing back to the source
	  phi    = tacoortrans::range(rusdgeom_.phi[2] + 180.0,360.0);
	  
	  
	  // Cores position in SD coordinate system, [1200m] units
	  xcore = rufldf_.xcore[0];
	  ycore = rufldf_.ycore[0];
	  
	  // geometry and LDF chi2 per degree of freedom
	  gfchi2pdof = (rusdgeom_.ndof[2] < 1 ? rusdgeom_.chi2[2] : rusdgeom_.chi2[2]/(double)rusdgeom_.ndof[2]);
	  ldfchi2pdof = (rufldf_.ndof[0] < 1 ? rufldf_.chi2[0] : rufldf_.chi2[0] / (double) rufldf_.ndof[0]);
	  
	  // sigma_s800 over s800
	  ds800os800 = rufldf_.dsc[0] / rufldf_.sc[0];
	  
	  // pointing direction error
	  pderr = sqrt(sin(rusdgeom_.theta[2] * DegToRad()) * sin(rusdgeom_.theta[2] * DegToRad()) * rusdgeom_.dphi[2]
		       * rusdgeom_.dphi[2] + rusdgeom_.dtheta[2] * rusdgeom_.dtheta[2]);

	  // energy
	  energy  = rufldf_.energy[0]/1.27;  // June 2010 energy estimation 
	  aenergy = rufldf_.aenergy[0];      // energy by old agasa formula
	  
	  // can't tell the event energy if there is not enough counters 
	  // or if zenith angle is too large
	  if (nstclust <= 3 || theta > 60.0)
	    {
	      energy  = 0.0;
	      aenergy = 0.0;
	    }
	  
	  ///////////////////// CUTS (BELOW) /////////////////////////
	  
	  if (opt.w_cuts_opt)
	    {
	      // number of good SDs
	      if (rufptn_.nstclust < 5)
		continue;
	  
	      // border
	      if ((rufldf_.bdist < 1.0) || ((rusdraw_.yymmdd <= 81110) && (rufldf_.tdist < 1.0)))
		continue;
	  
	      // chi2
	      if (gfchi2pdof > 4.0 || ldfchi2pdof > 4.0)
		continue;
	  
	      // sigma_s800 / s800
	      if (ds800os800 > 0.25)
		continue;
	  
	      // pointing direction error
	      if (pderr > 5.0)
		continue;
	  
	      // zentith angle
	      if (theta > 45.0)
		continue;
	  
	      // minimum energy
	      if (energy < 1.0)
		continue;
	    }
	  ///////////////////// CUTS (ABOVE) /////////////////////////
	  
	  
	  
	  
	  // open the output ascii file if it hasn't been opened yet
	  if(!outfl)
	    {
	      if (!opt.fOverwrite)
		{
		  if ((outfl=fopen(opt.outfile,"r")))
		    {
		      fprintf(stderr,"error: %s exists; use -f option\n",
			      opt.outfile);
		      fclose(outfl);
		      return 2;
		    }
		}
	      if(!(outfl=fopen(opt.outfile,"w")))
		{
		  fprintf(stderr,"error: failed to start %s\n",opt.outfile);
		  return 2;
		}
	    }
	  // if we're writing into stdout ( -O option) then prefix event lines with EVT so that 
	  // it's easy to grep for the them and separate them from the other outputs
	  if(outfl == stdout)
	    fprintf(outfl,"EVT ");
	  
	  
	  /////////////  WRITE OUT THE EVENT INFORMATION (BELOW) //////

	  // date and time
	  fprintf(outfl,"%d %02d %02d %02d.%06d ",yymmdd,hour,minute,second,usec);
	  
	  // number of good sds, energy ( current and old agasa), and core X and Y
	  fprintf(outfl,"%03d %5.2f %5.2f %5.2f %5.2f ",nstclust,energy,aenergy,xcore,ycore);
	  
	  // zenith angle and azimuthal angle, geom. chi2, geom. ndof, ldf chi2, ldf ndof
	  fprintf(outfl,"%5.2f %5.2f %5.2f %02d %5.2f %02d",
		  theta,phi,rusdgeom_.chi2[2], rusdgeom_.ndof[2],
		  rufldf_.chi2[0],rufldf_.ndof[0]);
	  
	  fprintf(outfl,"\n");
	  
	  fflush(outfl); 
	  events_written ++;

	  
	} // while(dstio.ReadEvent ...
      
      dstio.closeDSTinFile();
    
    }
  
  // finalize the output
  if((outfl != 0) && (outfl != stdout))
    fclose(outfl);
  
  fprintf(stdout,"events_read: %d events_written: %d cuts_applied: %s\n",
	  events_read,events_written, (opt.w_cuts_opt ? "yes" : "no"));
  fprintf(stdout,"\n\nDone\n");
  fflush(stdout);
  return 0;
}

sd4radar_listOfOpt::sd4radar_listOfOpt()
{
  wantfile[0]      = 0;        // list file variable initialized
  outfile[0]       = 0;        // output file variable initialized
  stdout_opt       = false;    // by default output expected to go to a file
  fOverwrite       = false;    // by default, don't allow the output file overwriting
  w_cuts_opt       = false;    // by default, don't apply the quality cuts
  bank_warning_opt = true;     // by default, print warnings about missing banks
  progName[0]      = '\0';
}

// destructor does nothing
sd4radar_listOfOpt::~sd4radar_listOfOpt() {}

bool sd4radar_listOfOpt::getFromCmdLine(int argc, char **argv)
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
	      printf ("specify the want file!\n");
	      return false;
	    }
	  else
	    {
	      sscanf (argv[i], "%1023s", wantfile);
	      // read the want files, put all pass1 dst files found into a buffer.
	      if((wantfl=fopen(wantfile,"r"))==NULL)
		{
		  fprintf(stderr,"can't open list file %s\n",wantfile);
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
	  if ((++i >= argc) || (argv[i][0] == 0) || (argv[i][0]=='-'))
	    {
	      fprintf (stderr,"error: -o: specify the output file!\n");
	      return false;
	    }
	  else
	    sscanf (argv[i], "%1023s", outfile);
	}
      
      // stdout option
      else if (strcmp ("-O", argv[i]) == 0)
	stdout_opt = true;
      
      // apply the quality cuts
      else if (strcmp ("-w_cuts", argv[i]) == 0)
	w_cuts_opt = true;
      
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


void sd4radar_listOfOpt::printOpts()
{
  if(wantfile[0])
    fprintf(stdout,"INPUT LIST FILE: %s\n",wantfile);
  if(!stdout_opt && outfile[0])
    fprintf(stdout,"OUTPUT FILE: %s\n",outfile);
  fprintf(stdout,"APPLY QUALITY CUTS: %s\n",(char* )(w_cuts_opt ? "YES" : "NO"));
  fprintf(stdout,"FORCE-OVERWRITE MODE: %s\n", (char* )(fOverwrite ? "YES" : "NO"));
  fprintf(stdout,"DISPLAY MISSING BANK WARNINGS: %s\n", (char* )(bank_warning_opt ? "YES" : "NO"));
  fprintf(stdout,"\n\n");
  fflush(stdout);
}

bool sd4radar_listOfOpt::checkOpt()
{
  if(outfile[0] == 0)
    sprintf(outfile,"./sd4radar.txt");
  if(countFiles()==0)
    {
      fprintf(stderr, "error: don't have any inputs!\n");
      return false;
    } 
  return true;
}

void sd4radar_listOfOpt::printMan()
{
  fprintf(stderr,"\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr,"Make TA SD event list for radar studies\n");
  fprintf(stderr,"Runs on analysed TA SD data\n");
  fprintf(stderr,"Writes out the event lists in ascii format\n");
  fprintf(stderr,"Events must have rusdraw,rufptn,rusdgeom,rufldf DST banks\n");
  fprintf(stderr,"All of these DST banks are available after rufptn.run and rufldf.run analyses\n");
  fprintf(stderr,"May or may not apply the quality cuts, depending on the options used\n");
  fprintf(stderr,"\n");
  fprintf(stderr,"****************************************************************************************\n");
  fprintf(stderr, "X=EAST, Y=NORTH, Z=UP\n");
  fprintf(stderr, "R = | sin(zenith angle)*cos(azimuthal angle) |\n");
  fprintf(stderr, "    | sin(zenith angle)*sin(azimuthal angle) |\n");
  fprintf(stderr, "    |         cos(zenith angle)              |\n");
  fprintf(stderr, "is where events come from in the local sky\n");
      
  
  // The "Radar" format
  fprintf(stderr, "\nRADAR FORMAT\n");      
  fprintf(stderr, "col01:  date - YYMMDD\n");
  fprintf(stderr, "col02:  hour\n");
  fprintf(stderr, "col03:  minute\n");
  fprintf(stderr, "col04:  second.scond_fraction\n");
  fprintf(stderr, "col05:  number of good SDs\n");
  fprintf(stderr, "col06:  energy, EeV\n");
  fprintf(stderr, "col07:  energy, EeV, by old AGASA formula\n");
  fprintf(stderr, "col08:  x core, [1200m] units, E of SD origin\n");
  fprintf(stderr, "col09:  y core [1200m] units,  N of SD origin\n");
  fprintf(stderr, "col10:  zenith angle, [Degree]\n");
  fprintf(stderr, "col11:  azimuthal angle, [Degree]\n");
  fprintf(stderr, "col12:  geometry fit chi2\n");
  fprintf(stderr, "col13:  geometry fit number of degrees of freedom\n");
  fprintf(stderr, "col14:  LDF fit chi2\n");
  fprintf(stderr, "col15:  LDF fit number of degrees of freedom\n");

  
  fprintf(stderr, "\nUsage: %s dst_file1 dst_file2 ... or -i want_file -o [output file]\n",progName);
  fprintf(stderr, "pass input DST file names as arguments without any prefixes or switches\n");
  fprintf(stderr, "-i <string>      : or specify the want file with sd dst files\n");
  fprintf(stderr, "--tty            : or get piped dst file names from stdin\n");
  fprintf(stderr, "-o <string>      : specify the output ASCII file, default is './sd4radar.txt'\n");
  fprintf(stderr, "-O               : pour event information into stdout with 'EVT' prefix ignoring the '-o' option\n");
  fprintf(stderr, "-w_cuts          : apply the quality cuts, off by default\n");
  fprintf(stderr, "-f               : force overwrite mode on all output files, off by default\n");
  fprintf(stderr, "-no_bw           : disable warnings about the missing banks; quietly skip events\n");
  fprintf(stderr, "-h               : show this manual\n");
  fprintf(stderr, "\n\n");
  fflush(stderr);
}
