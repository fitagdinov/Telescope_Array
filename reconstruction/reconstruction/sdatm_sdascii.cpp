#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "TMath.h"
#include "event.h"
#include "filestack.h"
#include "sddstio.h"
#include "fdcalib_util.h"
#include "sduti.h"
#include <vector>
#include <algorithm>

using namespace std;


class sdascii_event_class
{
public:
  int yyyymmdd,hhmmss,usec;
  double xcore,ycore,s800;
  double theta,phi;
  double rp,psi;
  double energy;
  double temperature;
  double density;
  double cfactor;

  sdascii_event_class()
  {
    yyyymmdd    = 0;
    hhmmss      = 0;
    usec        = 0;
    xcore       = 0;
    ycore       = 0;
    s800        = 0;
    theta       = 0;
    phi         = 0;
    rp          = 0;
    psi         = 0;
    energy      = 0;
    temperature = 0;
    density     = 0;
    cfactor     = 0;
  }
  
  ~sdascii_event_class() { ; }
  
  bool parse_sdascii_event(const char* inBuf)
  {
    if(sscanf(inBuf,"%d, %d.%d, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf",
	      &yyyymmdd,&hhmmss,&usec,&xcore,&ycore,&s800,
	      &theta,
	      &phi,
	      &rp,&psi,
	      &energy) != 11)
      {
	fprintf(stderr,"error: failed to parse buffer:\n");
	fprintf(stderr,"%s",inBuf);
	return false;
      }
    return true;
  }
  

  void apply_atm_correction(int cor)
  {
    if(fabs(cfactor) > 1e-7)
      {
        fprintf(stderr,"warning: trying to apply correction more than once!\n");
        return;
      }
    if(cor==0)
      {
	const double density_0 = 1.042e-3; // Mean air density at TA (1400m) in g/cm^3
	const double eslope    = 1.7;      // approximate slope of integral flux versus energy above 10 EeV
	const double corslope  = -3.5;     // slope of the correction expression in the denominator
	cfactor                = 1.0 / TMath::Power((1.0 + corslope * (density / density_0 - 1.0)), 1.0/eslope);
	energy                 *= cfactor;
      }
    else if(cor==1)
      {
	const double temperature_0 = 287.3; // Mean temperature at TA (1400m) in K
	const double eslope    = 1.7;       // approximate slope of integral flux versus energy above 10 EeV
	const double corslope  = 2.9;       // slope of the correction expression in the denominator
	double temperature = SDGEN::get_gdas_temp(1.4e5);
	double cfactor = 1.0 / TMath::Power((1.0 + corslope * (temperature / temperature_0 - 1.0)), 1.0/eslope);
	energy                 *= cfactor;
      }
    else
      {
	fprintf(stderr,"error: wrong correction flag %d, only 0 (density based) or 1(temperature based) supported;\n",cor);
	exit(2);
      }
  }
  bool format_sdascii_event(char* outBuf)
  {
    sprintf(outBuf," %d, %06d.%06d, %8.3f, %8.3f, %8.2f, %8.2f, %8.2f, %8.3f, %8.2f, %8.2f",
            yyyymmdd,hhmmss,usec,xcore,ycore,s800,
            theta,
            phi,
            rp,psi,
            energy);
    return true;
  }
  bool format_sdascii_event_w_atm(char* outBuf)
  {
    sprintf(outBuf," %d, %06d.%06d, %8.3f, %8.3f, %8.2f, %8.2f, %8.2f, %8.3f, %8.2f, %8.2f %8.2f %8.3e",
	    yyyymmdd,hhmmss,usec,xcore,ycore,s800,
	    theta,
	    phi,
	    rp,psi,
	    energy,
	    temperature,
	    density);
    return true;
  }

};


vector<sdascii_event_class> sdascii_events;

vector<gdas_dst_common>  gdas_all;
vector<gdas_dst_common*> gdas_sorted;


struct gdas_less_than_key
{
  inline bool operator() (const gdas_dst_common* gdas1, const gdas_dst_common* gdas2)
  {
    return (gdas1->dateFrom < gdas2->dateFrom);
  }
};

static int find_closest_gdas_before_event(int j2000sec)
{
  int n=(int)gdas_sorted.size();
  int i=n/2,ilo=0,iup=n;
  while (i > 0 && i < n - 1)
    {
      int j2000sec_rec = (int) ((time_t)gdas_sorted[i]->dateFrom-convertDate2Sec(2000,1,1,0,0,0));
      if(j2000sec_rec == j2000sec)
	break;
      if(j2000sec_rec < j2000sec)
	{
	  ilo=i;
	  i=(i+iup)/2;
	  continue;
	}
      int j2000sec_rec1 = (int) ((time_t)gdas_sorted[i-1]->dateFrom - convertDate2Sec(2000,1,1,0,0,0));
      if(j2000sec_rec1 < j2000sec)
	break;
      iup=i;
      i=(ilo+i)/2;
    }
  i -= 1;
  if (i < 0)
    i = 0;
  return i;
}


class sdatm_sdascii_listOfOpt
{
  
public:

  char   gdasfile[0x400];
  char   outfile[0x400];
  char   dout[0x400];
  int    cor;
  bool   fOverwrite;
  int    verbosity;
  
  sdatm_sdascii_listOfOpt()
  {
    gdasfile[0] = 0;            // user must specify atmospheric data base DST file
    outfile[0] = 0;               // output file initialized
    sprintf(dout,"./");           // default output directory is the current working directory
    cor        = -1;              // by default, don't do correction, just print the density and temperature
    fOverwrite = false;           // don't overwrite the output files if they exist by default
    verbosity  = 1;               // print minimum stuff by default
  }
  ~sdatm_sdascii_listOfOpt() { ; }
  

  bool parseCmdLine(int argc, char **argv)
  {
    int i;
    char inBuf[0x400];
    
    if (argc <= 1)
      {
	printMan(argv[0]);
	return false;
      }

    for (i = 1; i < argc; i++)
      {
	// man
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
	    printMan(argv[0]);
	    return false;
	  }
	
	// gdas DST file
	else if (strcmp("-a", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -a: specify the gdas data base DST file!\n");
		return false;
	      }
	    sscanf(argv[i], "%1023s", gdasfile);
	  }
	
	// list file
	else if (strcmp("-i", argv[i]) == 0)
	  {
	    FILE *fp = 0;
	    char* line = 0;
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr,"error: -i: specify the list file\n");
		return false;
	      }
	    if (!(fp = fopen(argv[i], "r")))
	      {
		fprintf(stderr, "error: can't open %s\n", argv[i]);
		return false;
	      }
	    while (fgets(inBuf, 0x400, fp))
	      {
		if (!((line = strtok(inBuf, " \t\r\n"))) || !strlen(line))
		  continue;
		if (pushFile(line) != SUCCESS)
		  return false;
	      }
	    fclose(fp); 
	  }
	// standard input
	else if (strcmp("--tty", argv[i]) == 0)
	  {
	    char* line = 0;
	    while (fgets(inBuf, 0x400, stdin))
	      {
		if (!((line = strtok(inBuf, " \t\r\n"))) || !strlen(line))
		  continue;
		if (pushFile(line) != SUCCESS)
		  return false;
	      }
	  }
	// output dst file
	else if (strcmp("-o1f", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -o1f: specify the output dst file\n");
		return false;
	      }
	    sscanf(argv[i], "%1023s", outfile);
	  }
	// output directory
	else if (strcmp("-o", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -o: specify the output directory\n");
		return false;
	      }
	    sscanf(argv[i], "%1023s", dout);
	  }
	// correction flag
	else if (strcmp("-cor", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -cor: specify the correction flag\n");
		return false;
	      }
	    sscanf(argv[i], "%d", &cor);
	  }
	// force overwrite mode
	else if (strcmp("-f", argv[i]) == 0)
	  fOverwrite = true;
	// verbosity flag
	else if (strcmp("-v", argv[i]) == 0)
	  {
	    if ((++i >= argc) || !argv[i] || (argv[i][0] == '-'))
	      {
		fprintf(stderr, "error: -v: specify the verbosity flag\n");
		return false;
	      }
	    sscanf(argv[i], "%d", &verbosity);
	  }
	// all arguments w/o the '-' switch should be the input files
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
    if (countFiles() < 1)
      {
	fprintf(stderr,"error: no input files\n");
	return false;
      }
    if(!gdasfile[0])
      {
	fprintf(stderr,"error: gdas DST file is not given; use -a option\n");
	fprintf(stderr,"to specify the gdas database DST file\n");
	return false;
      }
    return true;
  }
 
  void printMan(char* progName)
  {
    fprintf(stderr,"\nAdd atmospheric information, that corresponds in date and time, to each event in the .txt files of TA-Wiki format\n");
    fprintf(stderr,"\nusage: %s [in_file1 ...] and/or -i [list file]  -o [output directory]\n",progName);
    fprintf(stderr,"pass input event txt file names as arguments without any prefixes or switches\n");
    fprintf(stderr, "-i <string>    : specify the want file (with txt files)\n");
    fprintf(stderr, "-a <string>    : (mandatory) DST file that contains gdas bank for the most recent TA period\n");
    fprintf(stderr, "--tty <string> : or get input txt file names from stdin\n");
    fprintf(stderr, "-o <string>    : directory for output .txt files with automatically\n");
    fprintf(stderr, "                 generated names for each input .txt file name\n");
    fprintf(stderr, "                 default output directory is %s\n",dout);
    fprintf(stderr, "-o1f <string>  : single output .txt file name, overrides -o option\n");
    fprintf(stderr, "-cor <int>     : correction flag, 0=using density, 1=using temperature; by default, no correction\n");
    fprintf(stderr, "                 just add the atmospheric information at the end of the original event string\n");
    fprintf(stderr, "-f             : overwrite the output files if they exist\n");
    fprintf(stderr, "-v             : verbosity flag, default %d\n",verbosity);
    fprintf(stderr,"\n");
  }
  
};


int main(int argc, char *argv[])
{
   
  // cmd line options
  sdatm_sdascii_listOfOpt opt;
  if (!opt.parseCmdLine(argc,argv))
    return 2;
  
  // dst iterator
  sddstio_class dstio((opt.verbosity>=1));


  // load atmospheric parameter DST file
  if(!dstio.openDSTinFile(opt.gdasfile))
    return 2;

  if(opt.verbosity)
    {
      fprintf(stdout,"Reading atmospheric parameter database DST file %s:\n",opt.gdasfile);
      fflush(stdout);
    }
  int recordNo = 0;

  while(dstio.readEvent())
    {
      recordNo ++;
      if(!dstio.haveBank(GDAS_BANKID))
	{
	  fprintf(stderr,"error: gdas DST bank absent for the record number %d\n", recordNo);
	  continue;
	}
      if(!SDGEN::check_gdas())
	{
	  if(opt.verbosity>=1)
	    {
	      char d1[0x100];
	      char d2[0x100];
	      convertSec2DateLine((time_t)gdas_.dateFrom,d1);
	      convertSec2DateLine((time_t)gdas_.dateTo,d2);
	      fprintf(stderr,"Rejecting gdas recordNo %d FROM %s TO %s: bad gdas cycle\n",recordNo,d1,d2);
	    }
	  continue;
	}
      gdas_all.push_back(gdas_);
    }
  dstio.closeDSTinFile();
  gdas_sorted.resize(gdas_all.size());
  for (int i=0; i < (int)gdas_all.size(); i++)
    gdas_sorted[i] = &gdas_all[i];
  sort(gdas_sorted.begin(),gdas_sorted.end(),gdas_less_than_key());
  if(opt.verbosity>=1)
    {
      char d1[0x100];
      char d2[0x100];
      convertSec2DateLine((time_t)gdas_sorted.front()->dateFrom,d1);
      convertSec2DateLine((time_t)gdas_sorted.back()->dateTo,d2);
      fprintf(stdout,"%d gdas data base records loaded, from %s to %s\n",
	      (int)gdas_sorted.size(),d1,d2);
      fflush(stdout);
    }
  
  
  char* infile  = 0;
  char* outfile = 0;
  FILE *fin  = 0;
  FILE *fout = 0;
  const int ioBuf_len = 1024;
  char ioBuf[ioBuf_len];
  
  while((infile=pullFile()))
    {
      if(!(fin=fopen(infile,"r")))
	{
	  fprintf(stderr,"error: failed to open '%s' for reading\n",infile);
	  return 2;
	}
      sdascii_events.clear();
      if(!opt.outfile[0])
	{
	  char suf[0x20];
	  if(opt.cor == -1)
	    sprintf(suf,".sdatm_calib.txt");
	  else
	    sprintf(suf,".sdatm_corr.txt");
	  outfile = new char[strlen(infile)+strlen(suf)+1];
	  if(SDIO::GetOutFileName(infile,opt.dout,suf, outfile) != 1)
	    return 2;
	}
      else
	outfile = opt.outfile;
      while(fgets(ioBuf,ioBuf_len,fin))
	{
	  sdascii_event_class evt;
	  if(!evt.parse_sdascii_event(ioBuf))
	    continue;
	  sdascii_events.push_back(evt);
	}
      fclose(fin);
      if(!fout)
	{
	  if(!opt.fOverwrite)
	    {
	      if((fout=fopen(outfile,"r")))
		{
		  fprintf(stderr,"error: '%s' exists; use -f to overwrite files\n",outfile);
		  return 2;
		}
	    }
	  if(!(fout=fopen(outfile,"w")))
	    {
	      fprintf(stderr,"error: failed to start '%s'\n",outfile);
	      return 2;
	    }
	}
      for (vector<sdascii_event_class>::iterator iev=sdascii_events.begin(); iev != sdascii_events.end(); iev++)
	{
	  sdascii_event_class& evt = (*iev);
	  int j2000sec = SDGEN::time_in_sec_j2000(evt.yyyymmdd-20000000,evt.hhmmss);
	  int ical = find_closest_gdas_before_event(j2000sec);
	  // Get the density and temperature at the (1400m) level of the TA SD detector.
	  memcpy(&gdas_,gdas_sorted[ical],sizeof(gdas_dst_common));
	  evt.temperature = SDGEN::get_gdas_temp(1.4e5);
	  evt.density     = SDGEN::get_gdas_rho_numerically(1.4e5);
	  // if correcting, then do the correction and format
	  // the event in the usual way
	  if(opt.cor != -1)
	    {
	      evt.apply_atm_correction(opt.cor);
	      evt.format_sdascii_event(ioBuf);
	    }
	  // otherwise print the same event and add the atmospheric information
	  else
	    evt.format_sdascii_event_w_atm(ioBuf);
	  fprintf(fout,"%s\n",ioBuf);
	}
      if(!opt.outfile[0])
	{
	  fclose(fout);
	  fout = 0;
	}
    }
  
  if(fout)
    {
      fclose(fout);
      fout = 0;
    }
  
  fprintf(stdout,"\nDone\n");
  
  return 0; 
}

